"""
Looped (depth-recurrent) GPT for algorithmic reasoning.

A clean, A100-compatible implementation extracted from the modded-nanogpt
speedrun.  All speedrun-specific tricks (paired heads, two-lane residual,
bigram embeddings, value embeddings, smear gate, backout, MTP, FP8) are
removed.  What remains:

  - Standard transformer blocks: RMSNorm → Attention → Residual → RMSNorm → MLP(ReLU²) → Residual
  - Three-phase forward: prelude → recurrence × n_loop → coda
  - Input injection: cat(prelude_output, state) → linear projection
  - Post-loop RMSNorm to prevent activation blowup
  - Truncated BPTT for memory-efficient deep recurrence
  - Softcapping on output logits (from Gemma 2)
  - Parameter banks for efficient batched Muon optimisation

Architecture decisions are driven by the looped reasoning use-case:
small model (~25M params), deep effective depth via looping, tiny vocab
for character-level reasoning tasks.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


def next_multiple_of_n(v: float | int, *, n: int) -> int:
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


# ---------------------------------------------------------------------------
# Softcapped cross-entropy (no FP8 — uses standard PyTorch)
# ---------------------------------------------------------------------------


def softcapped_cross_entropy(
    x: Tensor,
    weight: Tensor,
    targets: Tensor,
    cap_a: float = 23.0,
    cap_b: float = 5.0,
    cap_c: float = 7.5,
) -> Tensor:
    """Compute cross-entropy with softcapped logits.

    ``logits = cap_a * sigmoid((x @ weight.T + cap_b) / cap_c)``

    This bounds logits to [0, cap_a], preventing logit explosion while
    preserving gradients throughout the range (from Gemma 2).
    """
    logits = F.linear(x, weight)
    logits = cap_a * torch.sigmoid((logits + cap_b) / cap_c)
    return F.cross_entropy(logits.float(), targets, reduction="mean")


# ---------------------------------------------------------------------------
# MLP with ReLU² activation (no fused kernel — simple PyTorch for A100)
# ---------------------------------------------------------------------------


class ReLUSquaredMLP(nn.Module):
    """MLP with ReLU² activation: out = relu(x @ W_up)² @ W_down.

    ReLU² is ~1-2% better than GELU (arXiv:2109.08668). Weights are
    *not* stored here — they live in the model's parameter bank for
    efficient batched Muon optimisation. Weights are passed in forward().
    """

    def forward(self, x: Tensor, w_up: Tensor, w_down: Tensor) -> Tensor:
        h = F.linear(x, w_up)
        h = F.relu(h)
        h = h * h  # ReLU²
        return F.linear(h, w_down)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention.

    Uses PyTorch's ``scaled_dot_product_attention`` which dispatches to
    FlashAttention 2 on A100 or the math backend as fallback.

    Weights are passed in via forward() (from the parameter bank).
    """

    def __init__(self, model_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.model_dim = model_dim
        self.scale = head_dim**-0.5

    def forward(self, x: Tensor, qkv_w: Tensor, o_w: Tensor) -> Tensor:
        B, T, _D = x.shape

        # Project to Q, K, V  (qkv_w: [3 * model_dim, model_dim])
        qkv = F.linear(x, qkv_w)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, T, H, D)

        # QK-norm (stabilises training, allows higher LR)
        q, k = norm(q), norm(k)

        # (B, T, H, D) → (B, H, T, D) for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # SDPA dispatches to FlashAttention 2 on A100
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)

        y = y.transpose(1, 2).contiguous().view(B, T, self.model_dim)
        y = F.linear(y, o_w)
        return y


# ---------------------------------------------------------------------------
# Single transformer block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """One transformer block: norm → attn → residual → norm → mlp → residual.

    All weight tensors are passed in from the parameter bank, so this
    module has *no* learned parameters itself — it's pure computation.
    """

    def __init__(self, model_dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(model_dim, num_heads, head_dim)
        self.mlp = ReLUSquaredMLP()

    def forward(
        self,
        x: Tensor,
        qkv_w: Tensor,
        o_w: Tensor,
        mlp_up_w: Tensor,
        mlp_down_w: Tensor,
    ) -> Tensor:
        # Attention
        x = x + self.attn(norm(x), qkv_w, o_w)
        # MLP
        x = x + self.mlp(norm(x), mlp_up_w, mlp_down_w)
        return x


# ---------------------------------------------------------------------------
# Looped GPT model
# ---------------------------------------------------------------------------


@dataclass
class LoopedGPTConfig:
    """Configuration for the looped transformer."""

    vocab_size: int = 80  # small for character-level reasoning tasks
    model_dim: int = 512
    num_heads: int = 8
    head_dim: int = 64  # model_dim // num_heads
    mlp_dim: int = 2048  # 4 * model_dim
    max_seq_len: int = 512
    # Layer counts
    n_prelude: int = 2
    n_recur: int = 4
    n_coda: int = 2
    # Looping
    n_loop: int = 1  # recurrence iterations (1 = no looping)
    bptt_k: int | None = None  # truncated BPTT depth (None = full backprop)
    input_injection: str = "inject"  # "inject" or "passthrough"
    # Output
    softcap_a: float = 23.0
    softcap_b: float = 5.0
    softcap_c: float = 7.5


class LoopedGPT(nn.Module):
    """Depth-recurrent GPT for algorithmic reasoning.

    The model has three phases:

    1. **Prelude** — ``n_prelude`` transformer blocks run once, building an
       initial representation.
    2. **Recurrence** — ``n_recur`` blocks run ``n_loop`` times.  Each
       iteration takes the previous iteration's output, optionally
       concatenated with the prelude output (input injection), and
       produces a refined representation.
    3. **Coda** — ``n_coda`` blocks run once, producing the final output.

    All transformer block weights are stored in parameter banks
    (``attn_bank``, ``mlp_bank``) for efficient batched Muon optimisation.

    Parameters are auto-labelled for the NorMuon+Adam optimizer.
    """

    def __init__(self, cfg: LoopedGPTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.model_dim
        total_layers = cfg.n_prelude + cfg.n_recur + cfg.n_coda

        # --- Embeddings ---
        effective_vocab = next_multiple_of_n(cfg.vocab_size, n=64)
        self.embed = nn.Embedding(effective_vocab, d)
        self.effective_vocab = effective_vocab

        # --- Output head ---
        self.lm_head = nn.Linear(d, effective_vocab, bias=False)
        # Tie weights at init (can be split later if needed)
        with torch.no_grad():
            self.lm_head.weight.copy_(self.embed.weight)

        # --- Parameter banks ---
        # Attention: stores [QKV_concat | O] for each layer
        # QKV: (3 * model_dim, model_dim),  O: (model_dim, model_dim)
        # Total per layer: (4 * model_dim, model_dim)
        self.attn_bank = nn.Parameter(torch.empty(total_layers, 4 * d, d))
        self.attn_bank.reshape = (total_layers * 4, d, d)

        # MLP: stores [W_up, W_down] for each layer
        # W_up: (mlp_dim, model_dim), W_down: (model_dim, mlp_dim)
        # We pad to even number of layers if needed for clean GPU sharding
        n_mlp_banks = next_multiple_of_n(total_layers, n=2)
        self.mlp_bank = nn.Parameter(torch.empty(n_mlp_banks, 2, cfg.mlp_dim, d))
        self.mlp_bank.reshape = (n_mlp_banks * 2, cfg.mlp_dim, d)
        self._n_mlp_banks = n_mlp_banks

        # Init weights
        std = 0.5 * d**-0.5
        bound = (3**0.5) * std
        with torch.no_grad():
            self.attn_bank.uniform_(-bound, bound)
            self.mlp_bank[:total_layers, 0, :, :].uniform_(-bound, bound)  # W_up
            self.mlp_bank[:total_layers, 1, :, :].zero_()  # W_down (zero init)
            if n_mlp_banks > total_layers:
                self.mlp_bank[total_layers:].zero_()  # padding

        # --- Shared transformer blocks ---
        # We use a single TransformerBlock instance for all layers within
        # each phase (they're stateless — weights come from the banks)
        self.block = TransformerBlock(d, cfg.num_heads, cfg.head_dim)

        # --- Looping modules ---
        if cfg.n_loop > 1:
            if cfg.input_injection == "inject":
                self.inject = nn.Linear(2 * d, d, bias=False)
                with torch.no_grad():
                    self.inject.weight.zero_()
                    self.inject.weight[:d, :d].copy_(torch.eye(d))
            self.norm_recur = nn.RMSNorm(d, elementwise_affine=True)

        # --- Auto-label parameters for optimizer ---
        for name, param in self.named_parameters():
            param.label = name.replace(".weight", "")

    def _get_layer_weights(self, layer_idx: int):
        """Unpack weights for a single transformer block from the banks."""
        d = self.cfg.model_dim
        attn_w = self.attn_bank[layer_idx]  # (4d, d)
        qkv_w = attn_w[: 3 * d]  # (3d, d)
        o_w = attn_w[3 * d :]  # (d, d)
        mlp_up_w = self.mlp_bank[layer_idx, 0]  # (mlp_dim, d)
        # W_down is stored as (mlp_dim, d) but we need (d, mlp_dim) for F.linear
        # Actually F.linear(x, W) computes x @ W.T, so W_down shape (d, mlp_dim)
        # But our bank stores (mlp_dim, d), so F.linear(h, W_down) = h @ W_down.T
        # which is (*, mlp_dim) @ (d, mlp_dim).T = (*, mlp_dim) @ (mlp_dim, d) = (*, d) ✓
        # Wait — our bank stores (mlp_dim, d). F.linear(h, self.mlp_bank[i,1]) would
        # compute h @ bank[i,1].T = (*, mlp_dim) @ (d, mlp_dim) — wrong shapes.
        # We need the down projection weight to be (d, mlp_dim) for F.linear.
        # Solution: store it transposed OR use a different convention.
        # Let's keep it simple: bank[i,1] is (mlp_dim, d) = W_down in (out, in) format
        # But MLP down-proj goes from mlp_dim → d, so weight should be (d, mlp_dim).
        # So bank[i,1].T gives us (d, mlp_dim) — but .T creates a non-contiguous view.
        # For simplicity, let's just store both as (larger_dim, smaller_dim) and handle it:
        mlp_down_w = self.mlp_bank[layer_idx, 1].T  # (d, mlp_dim) for F.linear — wrong
        # Actually: F.linear(x, w) = x @ w.T. If x is (B,T,mlp_dim) and we want output (B,T,d),
        # then w must be (d, mlp_dim). So mlp_bank[i,1] must be (d, mlp_dim).
        # But we stored it as (mlp_dim, d). So we need .T → (d, mlp_dim). Non-contiguous.
        # Let's just make it contiguous:
        mlp_down_w = self.mlp_bank[layer_idx, 1].T.contiguous()
        return qkv_w, o_w, mlp_up_w, mlp_down_w

    def forward(self, input_ids: Tensor, targets: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        input_ids : (B, T) int tensor
        targets : (B, T) int tensor, optional
            If provided, returns (loss, logits). Otherwise just logits.

        Returns
        -------
        loss, logits  or  logits
        """
        cfg = self.cfg
        B, T = input_ids.shape

        # --- Embed ---
        x = self.embed(input_ids)
        x = norm(x)

        # --- Layer indices ---
        prelude_end = cfg.n_prelude
        recur_end = prelude_end + cfg.n_recur
        # coda uses layers [recur_end, recur_end + n_coda)

        # --- 1. Prelude ---
        for i in range(prelude_end):
            qkv_w, o_w, mlp_up_w, mlp_down_w = self._get_layer_weights(i)
            x = self.block(x, qkv_w, o_w, mlp_up_w, mlp_down_w)

        # --- 2. Recurrence ---
        has_recurrence = cfg.n_loop > 1
        if has_recurrence:
            e = x  # prelude output (for input injection)

        for loop_iter in range(cfg.n_loop):
            if has_recurrence:
                if loop_iter == 0:
                    s = x  # initial state = prelude output
                # Input injection
                if cfg.input_injection == "inject":
                    x = self.inject(torch.cat([e, s], dim=-1))
                else:  # passthrough
                    x = s

            for i in range(prelude_end, recur_end):
                qkv_w, o_w, mlp_up_w, mlp_down_w = self._get_layer_weights(i)
                x = self.block(x, qkv_w, o_w, mlp_up_w, mlp_down_w)

            if has_recurrence:
                s = self.norm_recur(x)
                # Truncated BPTT: detach early iterations
                if cfg.bptt_k is not None and loop_iter < cfg.n_loop - cfg.bptt_k:
                    s = s.detach()

        if has_recurrence:
            x = s

        # --- 3. Coda ---
        for i in range(recur_end, recur_end + cfg.n_coda):
            qkv_w, o_w, mlp_up_w, mlp_down_w = self._get_layer_weights(i)
            x = self.block(x, qkv_w, o_w, mlp_up_w, mlp_down_w)

        # --- Output ---
        x = norm(x)

        if targets is not None:
            loss = softcapped_cross_entropy(
                x.view(-1, x.size(-1)),
                self.lm_head.weight,
                targets.view(-1),
                cap_a=cfg.softcap_a,
                cap_b=cfg.softcap_b,
                cap_c=cfg.softcap_c,
            )
            logits = F.linear(x, self.lm_head.weight)
            return loss, logits
        else:
            logits = F.linear(x, self.lm_head.weight)
            logits = cfg.softcap_a * torch.sigmoid((logits + cfg.softcap_b) / cfg.softcap_c)
            return logits
