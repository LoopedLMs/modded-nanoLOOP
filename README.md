# modded-nanoLOOP

Looped Transformer speedrun — depth-recurrent GPT-2 on FineWeb.

A research fork of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) that replaces the standard deep transformer with a **looped (depth-recurrent) architecture** for algorithmic reasoning tasks.

## Architecture

The model has three phases:

1. **Prelude** — a few transformer blocks run once, building an initial representation
2. **Recurrence** — a set of blocks run `n_loop` times with input injection (`cat(prelude_out, state) → projection`)
3. **Coda** — final blocks run once, producing the output

This gives deep effective depth from a small parameter budget (~25M params). Key design choices:

- RMSNorm → Attention → Residual → RMSNorm → MLP(ReLU²) → Residual
- QK-norm, FlashAttention 2, softcapped logits (from Gemma 2)
- Parameter banks for batched NorMuon + Adam optimization
- Truncated BPTT for memory-efficient deep recurrence
- Polar Express orthogonalization in the optimizer

## Tasks

Character-level reasoning tasks with a shared 57-token vocab:

| Task | Description |
|------|-------------|
| Arithmetic | Addition with reversed output for carry propagation |
| SAT | Random k-SAT — find a satisfying assignment |
| Grid | 2D grid transformations (flip, rotate) |
| Pathfinding | Grid maze shortest path via BFS |

## Quick start

```bash
# Install dependencies
uv sync

# Single GPU
uv run python train.py

# Multi-GPU
uv run torchrun --standalone --nproc_per_node=2 train.py
```

Configuration is done by editing the `TrainConfig` dataclass in [train.py](train.py).

## Files

| File | Purpose |
|------|---------|
| `model.py` | LoopedGPT architecture (prelude → recurrence → coda) |
| `optimizer.py` | NorMuon + Adam optimizer with Polar Express |
| `train.py` | Training loop for reasoning tasks |
| `data/reasoning.py` | Task generators and character-level tokenizer |
| `kernels_triton.py` | Fused Triton kernels for the optimizer |
| `train_gpt.py` | Upstream FineWeb speedrun trainer (reference) |

## Tests

```bash
uv run pytest
```

## References

- [Muon optimizer](https://kellerjordan.github.io/posts/muon/)
- [Polar Express](https://arxiv.org/pdf/2505.16932)
- [NorMuon](https://arxiv.org/pdf/2510.05491)
- [modded-nanogpt speedrun](https://github.com/KellerJordan/modded-nanogpt)
