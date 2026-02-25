"""
Triton kernels for modded-nanoLOOP — A100 compatible.

Kept from modded-nanogpt:
  - XXT, XTX, ba_plus_cAA  — symmetric matmuls for Polar Express (Muon)
  - transpose_copy, transpose_add — tiled coalesced transpose operations

New / rewritten for A100:
  - relu_squared_mlp — ReLU²(x @ W1) @ W2 without TMA TensorDescriptors
  - softcapped_cross_entropy — softcapped loss without FP8

All block sizes should be re-tuned on A100 via Triton autotuning;
the hardcoded values here are from H100 and will work but may be suboptimal.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Symmetric matrix multiplication kernels (for Polar Express)
# ---------------------------------------------------------------------------


@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)
    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx


# --- XXT: C = A @ A.T ---


@triton.jit
def XXT_kernel(
    A_ptr,
    C_ptr,
    M,
    K,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_n[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at_temp = tl.load(at_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at = tl.trans(at_temp)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def XXT(A: torch.Tensor, out: torch.Tensor):
    """Compute C = A @ A.T using a Triton kernel with symmetry optimization."""
    assert A.ndim in (2, 3)
    M, K = A.shape[-2:]
    assert out.size(-2) == M and out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    # Block sizes from H100 autotuning — functional on A100, re-tune for perf
    if K == 768:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64
    else:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 128
    num_stages, num_warps = 4, 8

    grid = (batch_size * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),)
    XXT_kernel[grid](
        A,
        out,
        M,
        K,
        input_batch_stride,
        A.stride(-2),
        A.stride(-1),
        output_batch_stride,
        out.stride(-2),
        out.stride(-1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        LOWER_UPPER=1,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return out


# --- XTX: C = A.T @ A ---


@triton.jit
def XTX_kernel(
    A_ptr,
    C_ptr,
    M,
    K,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, k_idx, n_idx = _pid_to_block(pid, K, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= k_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (k_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    offs_k = (k_idx + tl.arange(0, BLOCK_SIZE_M)) % K
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % K
    offs_m = tl.arange(0, BLOCK_SIZE_K)

    at_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_n[None, :] * a_stride_c)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for m in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        m_remaining = M - m * BLOCK_SIZE_K
        at = tl.load(at_ptrs, mask=offs_m[:, None] < m_remaining, other=0.0)
        a = tl.load(a_ptrs, mask=offs_m[:, None] < m_remaining, other=0.0)
        accumulator = tl.dot(at.T, a, accumulator)
        at_ptrs += BLOCK_SIZE_K * a_stride_r
        a_ptrs += BLOCK_SIZE_K * a_stride_r

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    offs_ck = k_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_ck[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_ck[:, None] < K) & (offs_cn[None, :] < K)
    tl.store(c_ptrs, output, mask=c_mask)

    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_ck[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < K) & (offs_ck[None, :] < K)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def XTX(A: torch.Tensor, out: torch.Tensor):
    """Compute C = A.T @ A for tall matrices (M > K)."""
    assert A.ndim in (2, 3)
    M, K = A.shape[-2:]
    assert out.size(-2) == K and out.size(-1) == K

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    if K == 768:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64
    else:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 64, 128, 128
    num_stages, num_warps = 4, 8

    grid = (batch_size * triton.cdiv(K, BLOCK_SIZE_M) * triton.cdiv(K, BLOCK_SIZE_N),)
    XTX_kernel[grid](
        A,
        out,
        M,
        K,
        input_batch_stride,
        A.stride(-2),
        A.stride(-1),
        output_batch_stride,
        out.stride(-2),
        out.stride(-1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        LOWER_UPPER=1,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return out


# --- ba_plus_cAA: C = beta*A + alpha*(A @ A.T) ---


@triton.jit
def ba_plus_cAA_kernel(
    A_ptr,
    C_ptr,
    M,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)

    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_n[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        k_remaining = M - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at_temp = tl.load(at_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        at = tl.trans(at_temp)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A and fuse: C = alpha * (A @ A.T) + beta * A
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


def ba_plus_cAA(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """Compute C = beta * A + alpha * (A @ A.T).  A must be square."""
    assert A.ndim in (2, 3)
    M, K = A.shape[-2:]
    assert M == K and out.size(-2) == M and out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 64
    num_stages, num_warps = 4, 8

    grid = (batch_size * triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(M, BLOCK_SIZE_N),)
    ba_plus_cAA_kernel[grid](
        A,
        out,
        M,
        input_batch_stride,
        A.stride(-2),
        A.stride(-1),
        output_batch_stride,
        out.stride(-2),
        out.stride(-1),
        alpha=alpha,
        beta=beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=8,
        LOWER_UPPER=1,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return out


# ---------------------------------------------------------------------------
# Tiled transpose kernels
# ---------------------------------------------------------------------------


@triton.jit
def _transpose_copy_kernel(
    src_ptr,
    dst_ptr,
    M,
    N,
    src_stride_m,
    src_stride_n,
    dst_stride_0,
    dst_stride_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tile = tl.load(
        src_ptr + offs_m[:, None] * src_stride_m + offs_n[None, :] * src_stride_n,
        mask=mask,
        other=0.0,
    )
    mask_T = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(
        dst_ptr + offs_n[:, None] * dst_stride_0 + offs_m[None, :] * dst_stride_1,
        tl.trans(tile),
        mask=mask_T,
    )


def transpose_copy(src: torch.Tensor, dst: torch.Tensor):
    """Tiled transpose copy: dst = src.T."""
    assert src.ndim == 2 and dst.ndim == 2
    M, N = src.shape
    assert dst.shape == (N, M)
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _transpose_copy_kernel[grid](
        src,
        dst,
        M,
        N,
        src.stride(0),
        src.stride(1),
        dst.stride(0),
        dst.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def _transpose_add_kernel(
    src_ptr,
    dst_ptr,
    M,
    N,
    src_stride_m,
    src_stride_n,
    dst_stride_0,
    dst_stride_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    src_tile = tl.load(
        src_ptr + offs_m[:, None] * src_stride_m + offs_n[None, :] * src_stride_n,
        mask=mask,
        other=0.0,
    )
    mask_T = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    dst_ptrs = dst_ptr + offs_n[:, None] * dst_stride_0 + offs_m[None, :] * dst_stride_1
    dst_tile = tl.load(dst_ptrs, mask=mask_T, other=0.0)
    tl.store(dst_ptrs, dst_tile + tl.trans(src_tile), mask=mask_T)


def transpose_add(src: torch.Tensor, dst: torch.Tensor):
    """Tiled transpose-add: dst += src.T."""
    assert src.ndim == 2 and dst.ndim == 2
    M, N = src.shape
    assert dst.shape == (N, M)
    BLOCK_M, BLOCK_N = 32, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _transpose_add_kernel[grid](
        src,
        dst,
        M,
        N,
        src.stride(0),
        src.stride(1),
        dst.stride(0),
        dst.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2,
    )
