"""
Training loop for the looped GPT on reasoning tasks.

Minimal, single-file training script that uses:
  - ``model.py`` — LoopedGPT architecture
  - ``optimizer.py`` — NorMuon + Adam
  - ``data/reasoning.py`` — task generators

Supports single-GPU and multi-GPU (via torchrun) training.

Usage
-----
Single GPU::

    uv run python train.py

Multi-GPU::

    uv run torchrun --standalone --nproc_per_node=2 train.py

Configuration is done by editing the ``TrainConfig`` dataclass below.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from data.reasoning import (
    TASK_REGISTRY,
    VOCAB_SIZE,
    TaskMix,
    generate_batch,
)
from model import LoopedGPT, LoopedGPTConfig
from optimizer import NorMuonAdam

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """All training hyperparameters in one place."""

    # Model
    model_dim: int = 512
    num_heads: int = 8
    head_dim: int = 64
    mlp_dim: int = 2048
    n_prelude: int = 2
    n_recur: int = 4
    n_coda: int = 2
    n_loop: int = 4
    bptt_k: int | None = None  # None = full backprop
    input_injection: str = "inject"
    max_seq_len: int = 256

    # Data
    tasks: list[str] = field(default_factory=lambda: ["arithmetic", "sat", "grid", "maze"])
    task_weights: list[float] = field(default_factory=lambda: [0.4, 0.3, 0.15, 0.15])
    difficulty: int = 4
    batch_size: int = 64  # per GPU

    # Optimiser
    adam_lr: float = 0.008
    adam_eps: float = 1e-10
    adam_wd: float = 0.005
    normuon_lr: float = 0.023
    normuon_momentum: float = 0.95
    normuon_beta2: float = 0.95
    normuon_wd: float = 1.2

    # Schedule
    num_steps: int = 5000
    warmup_steps: int = 250
    log_every: int = 50
    eval_every: int = 500
    eval_batches: int = 20
    seed: int = 42


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------


def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialize distributed training if launched via torchrun."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def is_master(rank: int) -> bool:
    return rank == 0


# ---------------------------------------------------------------------------
# LR schedule (warmup + cosine decay)
# ---------------------------------------------------------------------------


def get_lr_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Build optimizer
# ---------------------------------------------------------------------------


def build_optimizer(model: LoopedGPT, cfg: TrainConfig) -> NorMuonAdam:
    """Wire up the NorMuon+Adam optimizer for the LoopedGPT model."""
    param_table = {
        "attn_bank": {"optim": "normuon", "comms": "sharded"},
        "mlp_up_bank": {"optim": "normuon", "comms": "sharded"},
        "mlp_down_bank": {"optim": "normuon", "comms": "sharded"},
        "embed": {"optim": "adam", "comms": "sharded", "adam_betas": [0.9, 0.95]},
    }
    # Looping modules (conditional)
    if hasattr(model, "inject"):
        param_table["inject"] = {
            "optim": "adam",
            "comms": "replicated",
            "adam_betas": [0.9, 0.95],
            "lr_mul": 1.0,
            "wd_mul": 0.0,
        }
    if hasattr(model, "norm_recur"):
        param_table["norm_recur"] = {
            "optim": "adam",
            "comms": "replicated",
            "adam_betas": [0.9, 0.95],
            "lr_mul": 5.0,
            "wd_mul": 0.0,
        }

    labels = list(param_table.keys())
    return NorMuonAdam(
        model.named_parameters(),
        param_table=param_table,
        scatter_order=labels,
        work_order=labels,
        adam_defaults=dict(lr=cfg.adam_lr, eps=cfg.adam_eps, weight_decay=cfg.adam_wd),
        normuon_defaults=dict(
            lr=cfg.normuon_lr, momentum=cfg.normuon_momentum, beta2=cfg.normuon_beta2, weight_decay=cfg.normuon_wd
        ),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model: LoopedGPT, mix: TaskMix, cfg: TrainConfig, device: torch.device) -> float:
    """Compute average loss over eval_batches."""
    model.eval()
    total_loss = 0.0
    rng = random.Random(cfg.seed + 999)
    for _ in range(cfg.eval_batches):
        input_ids, target_ids = generate_batch(mix, cfg.batch_size, rng)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        loss, _logits = model(input_ids, target_ids)
        total_loss += loss.item()
    model.train()
    return total_loss / cfg.eval_batches


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig | None = None):
    if cfg is None:
        cfg = TrainConfig()

    rank, world_size, device = setup_distributed()
    master = is_master(rank)

    if master:
        print(f"Training LoopedGPT — {world_size} GPU(s)")
        print(f"  model: {cfg.model_dim}d, {cfg.n_prelude}+{cfg.n_recur}+{cfg.n_coda} layers, n_loop={cfg.n_loop}")
        print(f"  tasks: {cfg.tasks} (weights: {cfg.task_weights})")
        print(f"  steps: {cfg.num_steps}, batch_size: {cfg.batch_size}/GPU, seq_len: {cfg.max_seq_len}")

    # --- Model ---
    model_cfg = LoopedGPTConfig(
        vocab_size=VOCAB_SIZE,
        model_dim=cfg.model_dim,
        num_heads=cfg.num_heads,
        head_dim=cfg.head_dim,
        mlp_dim=cfg.mlp_dim,
        max_seq_len=cfg.max_seq_len,
        n_prelude=cfg.n_prelude,
        n_recur=cfg.n_recur,
        n_coda=cfg.n_coda,
        n_loop=cfg.n_loop,
        bptt_k=cfg.bptt_k,
        input_injection=cfg.input_injection,
    )
    model = LoopedGPT(model_cfg).to(device)

    # Cast to bf16
    for p in model.parameters():
        if p.dtype == torch.float32 and p.ndim >= 2:
            p.data = p.data.bfloat16()

    # Broadcast params in multi-GPU
    if world_size > 1:
        for p in model.parameters():
            dist.broadcast(p.detach(), 0)

    n_params = sum(p.numel() for p in model.parameters())
    if master:
        print(f"  parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    # Compile
    compiled_model = torch.compile(model, dynamic=False)

    # --- Optimizer ---
    optimizer = build_optimizer(model, cfg)

    # --- Data ---
    task_generators = [TASK_REGISTRY[t]() for t in cfg.tasks]
    mix = TaskMix(
        tasks=task_generators,
        weights=cfg.task_weights,
        difficulty=cfg.difficulty,
        max_seq_len=cfg.max_seq_len,
    )
    rng = random.Random(cfg.seed + rank)

    # --- Training ---
    if master:
        print("\nStarting training...")

    model.train()
    t0 = time.perf_counter()

    for step in range(cfg.num_steps):
        # LR schedule
        lr_mul = get_lr_multiplier(step, cfg.warmup_steps, cfg.num_steps)
        for p_cfg in optimizer.param_cfgs.values():
            p_cfg.lr = p_cfg.initial_lr * lr_mul

        # Generate batch
        input_ids, target_ids = generate_batch(mix, cfg.batch_size, rng)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # Forward + backward
        loss, _logits = compiled_model(input_ids, target_ids)
        loss.backward()

        # Optimizer step (NorMuon every step, Adam every other step)
        do_adam = step % 2 == 1
        optimizer.step(do_adam=do_adam)

        # Logging
        if master and (step % cfg.log_every == 0 or step == cfg.num_steps - 1):
            elapsed = time.perf_counter() - t0
            ms_per_step = 1000 * elapsed / (step + 1)
            print(f"  step {step:5d}/{cfg.num_steps}  loss={loss.item():.4f}  lr_mul={lr_mul:.3f}  {ms_per_step:.1f}ms/step")

        # Evaluation
        if master and (step % cfg.eval_every == 0 or step == cfg.num_steps - 1):
            val_loss = evaluate(model, mix, cfg, device)
            print(f"  [eval] step {step:5d}  val_loss={val_loss:.4f}")
            model.train()

    # --- Done ---
    elapsed = time.perf_counter() - t0
    if master:
        print(f"\nTraining complete: {elapsed:.1f}s total, {1000 * elapsed / cfg.num_steps:.1f}ms/step avg")
        print(f"Peak memory: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    train()
