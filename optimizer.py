"""
NorMuon + Adam optimizer — extracted from modded-nanogpt speedrun.

NorMuon applies Nesterov momentum followed by Polar Express orthogonalization
to 2D weight matrices (attention/MLP projections).  Adam handles everything
else (embeddings, scalars, biases, gates).

Both branches use *cautious weight decay* (only decay when gradient and
parameter agree in sign) and NorMuon tracks mantissa bits for higher-
precision bf16 parameter updates.

The Polar Express step is the key differentiator: it replaces each gradient
update with the nearest orthogonal matrix, which stabilises training and
allows larger learning rates.  It runs entirely in bf16 on-GPU using fused
Triton kernels (XXT / XTX / ba_plus_cAA) from ``triton_kernels.py``.

References
----------
- Muon: https://kellerjordan.github.io/posts/muon/
- Polar Express: https://arxiv.org/pdf/2505.16932
- NorMuon (variance reduction): https://arxiv.org/pdf/2510.05491
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor, nn

from kernels_triton import XTX, XXT, ba_plus_cAA

# ---------------------------------------------------------------------------
# Polar Express orthogonalization
# ---------------------------------------------------------------------------

# Precomputed polynomial coefficients for 5 iterations with
# safety_factor=2e-2 and cushion=2.
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@torch.compile(dynamic=False, fullgraph=True)
def polar_express(
    grad_chunk: torch.Tensor,
    momentum_buffer: torch.Tensor,
    momentum_t: torch.Tensor,
    split_baddbmm: bool = False,
) -> torch.Tensor:
    """Fused Nesterov momentum + Polar Express orthogonalization.

    Nesterov momentum is computed in FP32, then cast to BF16 for the polar
    decomposition.  ``momentum_t`` is a 0-D *CPU* tensor so that changing its
    value does not trigger ``torch.compile`` recompilation.
    """
    # Nesterov momentum (FP32)
    momentum = momentum_t.to(grad_chunk.dtype)
    momentum_buffer.lerp_(grad_chunk, 1 - momentum)
    g = grad_chunk.lerp_(momentum_buffer, momentum)

    X = g.bfloat16()
    is_tall = g.size(-2) > g.size(-1)

    # Ensure spectral norm ≤ 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + 2e-2) + 1e-6)
    X = X.contiguous()

    if is_tall:
        # Tall: use X^T @ X (small) and right-multiply
        A = torch.empty((*X.shape[:-2], X.size(-1), X.size(-1)), device=X.device, dtype=X.dtype)
        B = torch.empty_like(A)
        C = torch.empty_like(X)

        if split_baddbmm:
            XB_matmul = torch.bmm if X.ndim > 2 else torch.mm
        else:
            aX_plus_XB = torch.baddbmm if X.ndim > 2 else torch.addmm

        for a, b, c in POLAR_EXPRESS_COEFFS:
            XTX(X, out=A)
            ba_plus_cAA(A, alpha=c, beta=b, out=B)
            if split_baddbmm:
                XB_matmul(X, B, out=C)
                C.add_(X, alpha=a)
            else:
                aX_plus_XB(X, X, B, beta=a, out=C)
            X, C = C, X
    else:
        # Wide: use X @ X^T (small) and left-multiply
        A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
        B = torch.empty_like(A)
        C = torch.empty_like(X)

        if split_baddbmm:
            BX_matmul = torch.bmm if X.ndim > 2 else torch.mm
        else:
            aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

        for a, b, c in POLAR_EXPRESS_COEFFS:
            XXT(X, out=A)
            ba_plus_cAA(A, alpha=c, beta=b, out=B)
            if split_baddbmm:
                BX_matmul(B, X, out=C)
                C.add_(X, alpha=a)
            else:
                aX_plus_BX(X, B, X, beta=a, out=C)
            X, C = C, X

    return X


# ---------------------------------------------------------------------------
# Per-parameter configuration
# ---------------------------------------------------------------------------


@dataclass
class ParamConfig:
    """Configuration for a single named parameter."""

    label: str
    optim: str  # "adam" or "normuon"
    comms: str  # "none", "replicated", or "sharded"
    adam_betas: tuple[float, float] | None
    lr_mul: float
    wd_mul: float
    lr: float
    initial_lr: float
    weight_decay: float
    # Adam-specific
    eps: float | None = None
    # NorMuon-specific
    reshape: tuple | None = None
    chunk_size: int | None = None
    momentum: float | None = None
    beta2: float | None = None
    per_matrix_lr_mul: list[float] | None = None


# ---------------------------------------------------------------------------
# NorMuon + Adam optimizer
# ---------------------------------------------------------------------------


class NorMuonAdam:
    """Combined NorMuon (projection matrices) and Adam (everything else).

    This is a *non-standard* optimizer interface: it does **not** subclass
    ``torch.optim.Optimizer``.  Instead it uses a ``param_table`` dict that
    maps parameter *labels* (stored as ``param.label``) to per-parameter
    hyperparameters.

    Communication is explicit (``scatter_order`` / ``work_order``) rather
    than hook-driven, giving fine-grained control over overlapping
    computation with all-reduce / reduce-scatter / all-gather.

    Parameters
    ----------
    named_params : iterable of (name, Parameter)
        Typically ``model.named_parameters()``.
    param_table : dict[str, dict]
        Per-label configuration.  Required keys: ``optim`` ("adam" | "normuon"),
        ``comms`` ("none" | "replicated" | "sharded").
        Optional keys: ``adam_betas``, ``lr_mul``, ``wd_mul``.
    scatter_order, work_order : list[str]
        Label orderings for communication launch and update math respectively.
    adam_defaults, normuon_defaults : dict
        Default hyper-parameters for each optimizer type.
    """

    def __init__(
        self,
        named_params,
        param_table: dict,
        scatter_order: list[str],
        work_order: list[str],
        adam_defaults: dict,
        normuon_defaults: dict,
    ):
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.adam_defaults = adam_defaults
        self.normuon_defaults = normuon_defaults
        self.param_table = param_table
        self.scatter_order = scatter_order
        self.work_order = work_order

        self.param_cfgs: dict[nn.Parameter, ParamConfig] = {}
        self.param_states: dict[nn.Parameter, dict] = {}
        self._param_by_label: dict[str, nn.Parameter] = {}

        for _name, param in named_params:
            label = getattr(param, "label", None)
            assert label is not None and label in param_table, f"param {_name!r} needs a valid .label"
            assert label not in self._param_by_label, f"duplicate label {label!r}"
            self._param_by_label[label] = param
            self._build_param_cfg(param, label)

        present = set(self._param_by_label.keys())
        assert set(scatter_order) == present and set(work_order) == present

        if self.world_size == 1:
            for p_cfg in self.param_cfgs.values():
                p_cfg.comms = "none"

        self._init_state()

        # 0-D CPU tensors avoid torch.compile recompilation
        self._step_size_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eff_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._eff_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

        self._reduce_futures: dict[nn.Parameter, tuple] = {}

    # -----------------------------------------------------------------------
    # Config + state init
    # -----------------------------------------------------------------------

    def _build_param_cfg(self, param: nn.Parameter, label: str):
        entry = self.param_table[label]
        optim = entry["optim"]
        comms = entry["comms"]
        adam_betas = entry.get("adam_betas")
        lr_mul = entry.get("lr_mul", 1.0)
        wd_mul = entry.get("wd_mul", 1.0)

        if optim == "adam":
            chunk_size = param.shape[0] // self.world_size if comms == "sharded" else None
            p_cfg = ParamConfig(
                label=label,
                optim=optim,
                comms=comms,
                adam_betas=tuple(adam_betas) if adam_betas else None,
                lr_mul=lr_mul,
                wd_mul=wd_mul,
                lr=self.adam_defaults["lr"],
                initial_lr=self.adam_defaults["lr"],
                weight_decay=self.adam_defaults["weight_decay"],
                eps=self.adam_defaults["eps"],
                chunk_size=chunk_size,
            )
        elif optim == "normuon":
            reshape = getattr(param, "reshape", None)
            if reshape is None:
                raise ValueError(f"NorMuon param {label} must have a .reshape attribute")
            if reshape[0] % self.world_size != 0:
                raise ValueError(f"reshape[0]={reshape[0]} must be divisible by world_size={self.world_size}")

            chunk_size = reshape[0] // self.world_size
            chunk_shape = (chunk_size, *reshape[1:])
            shape_mult = max(1.0, chunk_shape[-2] / chunk_shape[-1]) ** 0.5 if len(chunk_shape) >= 2 else 1.0
            lr_mul = shape_mult * lr_mul

            per_matrix_lr_mul = None
            # Allow caller to attach per_matrix_lr_mul via param_table
            if "per_matrix_lr_fn" in entry:
                rank = dist.get_rank() if dist.is_initialized() else 0
                start_idx = rank * chunk_size
                per_matrix_lr_mul = [entry["per_matrix_lr_fn"](start_idx + i) for i in range(chunk_size)]

            p_cfg = ParamConfig(
                label=label,
                optim=optim,
                comms=comms,
                adam_betas=tuple(adam_betas) if adam_betas else None,
                lr_mul=lr_mul,
                wd_mul=wd_mul,
                lr=self.normuon_defaults["lr"],
                initial_lr=self.normuon_defaults["lr"],
                weight_decay=self.normuon_defaults["weight_decay"],
                reshape=reshape,
                chunk_size=chunk_size,
                momentum=self.normuon_defaults["momentum"],
                beta2=self.normuon_defaults["beta2"],
                per_matrix_lr_mul=per_matrix_lr_mul,
            )
        else:
            raise ValueError(f"Unknown optim type: {optim}")

        self.param_cfgs[param] = p_cfg

    def _init_state(self):
        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "adam":
                chunk = param[: p_cfg.chunk_size] if p_cfg.comms == "sharded" else param
                exp_avg = torch.zeros_like(chunk, dtype=torch.float32, device=param.device)
                self.param_states[param] = dict(step=0, exp_avg=exp_avg, exp_avg_sq=torch.zeros_like(exp_avg))

            elif p_cfg.optim == "normuon":
                chunk_shape = (p_cfg.chunk_size, *p_cfg.reshape[1:])
                momentum_buffer = torch.zeros(chunk_shape, dtype=torch.float32, device=param.device)

                if chunk_shape[-2] >= chunk_shape[-1]:
                    second_mom_shape = (*chunk_shape[:-1], 1)
                else:
                    second_mom_shape = (*chunk_shape[:-2], 1, chunk_shape[-1])
                second_momentum_buffer = torch.zeros(second_mom_shape, dtype=torch.float32, device=param.device)
                mantissa = torch.zeros(chunk_shape, dtype=torch.uint16, device=param.device)

                self.param_states[param] = dict(
                    momentum_buffer=momentum_buffer,
                    second_momentum_buffer=second_momentum_buffer,
                    mantissa=mantissa,
                )

    # -----------------------------------------------------------------------
    # Communication helpers
    # -----------------------------------------------------------------------

    def _launch_reduce(self, param: nn.Parameter, grad: Tensor):
        p_cfg = self.param_cfgs[param]

        if p_cfg.comms == "none":
            if p_cfg.optim == "normuon":
                grad = grad.view(p_cfg.reshape)
            self._reduce_futures[param] = (None, grad)
        elif p_cfg.comms == "replicated":
            future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
            self._reduce_futures[param] = (future, grad)
        elif p_cfg.comms == "sharded":
            if p_cfg.optim == "normuon":
                grad_reshaped = grad.view(p_cfg.reshape)
                grad_chunk = torch.empty(
                    (p_cfg.chunk_size, *grad_reshaped.shape[1:]),
                    dtype=grad.dtype,
                    device=grad.device,
                )
                future = dist.reduce_scatter_tensor(
                    grad_chunk, grad_reshaped.contiguous(), op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                self._reduce_futures[param] = (future, grad_chunk)
            else:
                grad_chunk = torch.empty_like(grad[: p_cfg.chunk_size])
                future = dist.reduce_scatter_tensor(grad_chunk, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                self._reduce_futures[param] = (future, grad_chunk)

    def _launch_gather(self, param: nn.Parameter, p_slice: Tensor) -> torch.futures.Future:
        p_cfg = self.param_cfgs[param]
        if p_cfg.optim == "normuon":
            full_param = param.data.view(p_cfg.reshape)
            assert full_param.is_contiguous()
            return dist.all_gather_into_tensor(full_param, p_slice.contiguous(), async_op=True).get_future()
        else:
            return dist.all_gather_into_tensor(param, p_slice.contiguous(), async_op=True).get_future()

    # -----------------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------------

    def reset_normuon(self):
        """Zero NorMuon momentum buffers (call on training reset)."""
        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "normuon":
                s = self.param_states[param]
                s["momentum_buffer"].zero_()
                s["mantissa"].zero_()
                s["second_momentum_buffer"].zero_()

    def state_dict(self):
        return {
            "param_states": {id(p): s for p, s in self.param_states.items()},
            "param_cfgs": {id(p): s for p, s in self.param_cfgs.items()},
        }

    def load_state_dict(self, state_dict):
        id_to_param = {id(p): p for p in self.param_cfgs}
        for param_id, saved in state_dict["param_states"].items():
            if param_id in id_to_param:
                param = id_to_param[param_id]
                cur = self.param_states[param]
                for k, v in saved.items():
                    if isinstance(v, torch.Tensor) and k in cur:
                        cur[k] = v.to(dtype=cur[k].dtype, device=cur[k].device)
                    else:
                        cur[k] = v

    # -----------------------------------------------------------------------
    # Optimizer step
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def step(self, do_adam: bool = True):
        """Run one optimiser step.

        Parameters
        ----------
        do_adam : bool
            When ``False`` only NorMuon params are updated (useful for
            alternating NorMuon-every-step / Adam-every-other-step).
        """
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Phase 1 — launch all reduces
        for label in self.scatter_order:
            param = self._param_by_label[label]
            p_cfg = self.param_cfgs[param]
            if p_cfg.optim == "adam" and not do_adam:
                continue
            if param.grad is None:
                continue
            self._launch_reduce(param, param.grad)

        # Phase 2 — wait → compute update → launch gather
        gather_futures: list = []
        for label in self.work_order:
            param = self._param_by_label[label]
            if param not in self._reduce_futures:
                continue
            p_cfg = self.param_cfgs[param]
            if p_cfg.optim == "adam" and not do_adam:
                continue

            future, grad_chunk = self._reduce_futures[param]
            if future is not None:
                future.wait()

            if p_cfg.optim == "adam":
                p_slice = self._adam_update(param, grad_chunk, p_cfg, rank)
            else:
                p_slice = self._normuon_update(param, grad_chunk, p_cfg, rank)

            if p_cfg.comms == "sharded" and self.world_size > 1:
                gather_futures.append(self._launch_gather(param, p_slice))

        # Phase 3 — wait for gathers
        for fut in gather_futures:
            fut.wait()

        self._reduce_futures.clear()

        # Clear grads for updated params
        for param, p_cfg in self.param_cfgs.items():
            if p_cfg.optim == "adam" and not do_adam:
                continue
            param.grad = None

    # -----------------------------------------------------------------------
    # Adam update
    # -----------------------------------------------------------------------

    def _adam_update(self, param: nn.Parameter, grad_chunk: Tensor, p_cfg: ParamConfig, rank: int) -> Tensor:
        beta1, beta2 = p_cfg.adam_betas
        lr = p_cfg.lr * p_cfg.lr_mul

        if p_cfg.comms == "sharded":
            p_slice = param[rank * p_cfg.chunk_size : (rank + 1) * p_cfg.chunk_size]
        else:
            p_slice = param

        s = self.param_states[param]
        s["step"] += 1
        t = s["step"]

        bias1, bias2 = 1 - beta1**t, 1 - beta2**t
        self._step_size_t.fill_(lr * (bias2**0.5 / bias1))
        self._eff_wd_t.fill_(lr * lr * p_cfg.weight_decay * p_cfg.wd_mul)

        NorMuonAdam._adam_update_step(
            p_slice, grad_chunk, s["exp_avg"], s["exp_avg_sq"], beta1, beta2, p_cfg.eps, self._step_size_t, self._eff_wd_t
        )
        return p_slice

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _adam_update_step(p_slice, g_slice, exp_avg, exp_avg_sq, beta1, beta2, eps, step_size_t, eff_wd_t):
        exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
        update = exp_avg.div(exp_avg_sq.sqrt().add_(eps)).mul_(step_size_t)
        # Cautious weight decay — only decay weights that agree with gradient
        mask = (update * p_slice) > 0
        update.addcmul_(p_slice, mask, value=eff_wd_t)
        p_slice.add_(other=update, alpha=-1.0)

    # -----------------------------------------------------------------------
    # NorMuon update
    # -----------------------------------------------------------------------

    def _normuon_update(self, param: nn.Parameter, grad_chunk: Tensor, p_cfg: ParamConfig, rank: int) -> Tensor:
        chunk_shape = grad_chunk.shape
        s = self.param_states[param]
        grad_chunk = grad_chunk.float()

        self._momentum_t.fill_(p_cfg.momentum)
        self._eff_lr_t.fill_(p_cfg.lr_mul * p_cfg.lr)
        self._eff_wd_t.fill_(p_cfg.wd_mul * p_cfg.weight_decay * p_cfg.lr)

        is_large_matrix = chunk_shape[-2] > 1024
        v_chunk = polar_express(grad_chunk, s["momentum_buffer"], self._momentum_t, split_baddbmm=is_large_matrix)

        red_dim = -1 if chunk_shape[-2] >= chunk_shape[-1] else -2
        v_chunk = NorMuonAdam._apply_normuon_variance_reduction(v_chunk, s["second_momentum_buffer"], p_cfg.beta2, red_dim)

        param_view = param.data.view(p_cfg.reshape)
        p_slice = param_view[rank * p_cfg.chunk_size : (rank + 1) * p_cfg.chunk_size]

        if p_cfg.per_matrix_lr_mul is not None:
            for mat_idx in range(p_cfg.chunk_size):
                self._eff_lr_t.fill_(p_cfg.lr_mul * p_cfg.per_matrix_lr_mul[mat_idx] * p_cfg.lr)
                self._eff_wd_t.fill_(p_cfg.wd_mul * p_cfg.weight_decay * p_cfg.lr)
                NorMuonAdam._cautious_wd_and_update_inplace(
                    p_slice[mat_idx].view(torch.uint16),
                    s["mantissa"][mat_idx],
                    v_chunk[mat_idx],
                    self._eff_wd_t,
                    self._eff_lr_t,
                )
        else:
            NorMuonAdam._cautious_wd_and_update_inplace(
                p_slice.view(torch.uint16), s["mantissa"], v_chunk, self._eff_wd_t, self._eff_lr_t
            )

        return p_slice

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _cautious_wd_and_update_inplace(p, mantissa, grad, wd_tensor, lr_tensor):
        """Cautious WD + mantissa-tracked bf16 update.

        bf16 has only 7 mantissa bits; we store the lower 16 bits of the
        fp32 representation separately so tiny updates accumulate properly.
        """
        assert p.dtype == mantissa.dtype == torch.uint16
        grad = grad.float()
        wd_factor = wd_tensor.to(torch.float32)
        lr_factor = lr_tensor.to(torch.float32)
        p_precise_raw = (p.to(torch.uint32) << 16) | mantissa.to(torch.uint32)
        p_precise = p_precise_raw.view(torch.float32)
        mask = (grad * p_precise) >= 0
        p_precise.copy_(p_precise - (p_precise * mask * wd_factor * lr_factor) - (grad * lr_factor))
        p.copy_((p_precise_raw >> 16).to(torch.uint16))
        mantissa.copy_(p_precise_raw.to(torch.uint16))

    @staticmethod
    @torch.compile(dynamic=False, fullgraph=True)
    def _apply_normuon_variance_reduction(v_chunk, second_momentum_buffer, beta2, red_dim):
        """Low-rank variance normalisation (Adafactor-inspired)."""
        v_mean = v_chunk.float().square().mean(dim=red_dim, keepdim=True)
        red_dim_size = v_chunk.size(red_dim)
        v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
        v_norm = v_norm_sq.sqrt_()
        second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
        step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()
        scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
        v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()
        final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))
        return v_chunk.mul_(final_scale.type_as(v_chunk))
