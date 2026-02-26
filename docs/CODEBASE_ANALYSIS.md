# modded-nanoLOOP: Codebase Deep Dive & Roadmap

> *From a speedrun-optimized GPT-2 to a looped reasoning machine — what to keep, what to cut, and how to get there.*

---

## Table of Contents

1. [What Is This Codebase?](#1-what-is-this-codebase)
2. [Architecture Overview: The 11-Layer GPT](#2-architecture-overview-the-11-layer-gpt)
3. [The Optimization Stack (Deep Dive)](#3-the-optimization-stack-deep-dive)
4. [A100 Compatibility Audit](#4-a100-compatibility-audit)
5. [The Looped Architecture (Current State)](#5-the-looped-architecture-current-state)
6. [Keep / Rework / Remove Decision Matrix](#6-keep--rework--remove-decision-matrix)
7. [Reasoning Data Integration Strategy](#7-reasoning-data-integration-strategy)
8. [Recommended Path Forward](#8-recommended-path-forward)

---

## 1. What Is This Codebase?

This repository is a **heavily optimized fork of modded-nanogpt** — a competitive "speedrun" project where contributors race to train GPT-2 (124M parameters) to a target validation loss on FineWeb data in the shortest wall-clock time possible. The current record sits around ~24 seconds on 8×H100 GPUs.

**Why does this matter for us?** The speedrun pressure produced genuinely useful engineering:
- A custom optimizer (NorMuon) that converges faster than Adam alone
- Fused Triton kernels that avoid memory bottlenecks
- Efficient distributed training with carefully overlapped communication

But it also produced **extreme overfitting to a specific workload**: 11 layers of GPT-2 on English web text, targeting a specific val-loss in minimal time. Many design choices (paired heads on specific layers, hardcoded skip connections between layers 3→6, the exact training schedule) are "winning lottery tickets" for *this particular race* and won't transfer to our looped reasoning architecture.

**Our goal** is to extract the efficient infrastructure while building something fundamentally different: a small, **depth-recurrent** (looped) transformer that can learn algorithmic reasoning tasks.

### File Map

| File | Lines | What It Does |
|------|-------|-------------|
| `train_gpt.py` | ~2230 | **Everything**: model, optimizer, data loader, training loop |
| `triton_kernels.py` | ~850 | Custom Triton kernels for Muon optimizer, MLP, cross-entropy, transpose ops |
| `train_gpt_medium.py` | ~2500 | Separate 350M-param variant (not relevant for us) |
| `data/*.py` | ~20 each | HuggingFace download scripts for FineWeb tokenized shards |

The monolithic `train_gpt.py` is both a strength (everything in one file, easy to understand) and a weakness (2200 lines with no separation of concerns).

---

## 2. Architecture Overview: The 11-Layer GPT

Let's walk through the model, piece by piece. Understanding *why* each component exists helps us decide what stays.

### 2.1 Embeddings (3 kinds!)

```
Input tokens → [Token Embed] + [Bigram Hash Embed] + [Value Embeds]
```

**Token embedding** (`self.embed`): Standard learned embedding, 50304×768. Tied to `lm_head` for the first 2/3 of training, then split.

**Bigram hash embedding** (`self.bigram_embed`): A hash-based embedding that encodes *pairs* of consecutive tokens. The idea: "the" followed by "cat" gets a different representation than "the" followed by "dog", giving the model bigram context for free without any attention. Uses ~250k entries (5× vocab) with a hash function to map token pairs to indices.

> **Why it exists**: In the speedrun, every bit of "free" context helps reach the loss target faster. Bigrams give the model a cheap shortcut for next-token prediction.
>
> **For us**: Reasoning tasks are about learning algorithms, not memorizing bigram statistics. **Remove**.

**Value embeddings** (`self.value_embeds`): 5 separate embedding tables, each mapping tokens to vectors. These are *added to the values* in attention at 5 specific layers (layers 1, 2, 8, 9, 10). Think of them as giving the attention mechanism pre-computed "what to write" signals based on the token identity, independent of context.

> **Why it exists**: Inspired by the "value residual" paper (arXiv:2410.17897). Helps early layers build useful representations faster.
>
> **For us**: These are heavily tuned to English text patterns. They add significant memory (5×50304×768 in bf16 ≈ 290MB). **Remove for reasoning tasks** — the model should learn representations from the task structure, not pre-baked token features.

### 2.2 The Residual Stream: Two Lanes

Starting at layer 7, the model splits into a **two-lane residual stream**:

```
Layers 0-6:  [lane0] ← single stream (standard residual)
Layers 7-10: [lane0] + [lane1] ← two parallel streams
                                   attn reads lane0, MLP reads lane1
Output: (lane0 + lane1) / 2
```

Each layer's attention output is added to both lanes (with different scaling), and MLP output is added to both lanes. This is the "parallel residual" pattern — it lets attention and MLP operate on slightly different views of the data.

> **Why it exists**: Empirically found to be ~45 steps faster in the speedrun. Parallel residuals reduce sequential dependency within a layer.
>
> **For us**: This adds complexity to the looping mechanism (which lane does the recurrence operate on?). **Simplify to a single residual stream** initially. Can re-add later if experiments show benefit.

### 2.3 Attention: Four Flavors

The codebase implements an unusually complex attention system:

1. **Standard causal attention** with RoPE (half-truncated by @YouJiacheng)
2. **Paired-head attention** on layers 0, 2, 5, 9 — adjacent heads attend to each other's keys (doubles sequence length, halves window)
3. **Sliding window attention** with sizes that change during training (128→384→640→768 for short windows, 384→896→1408→1664 for long)
4. **Key offset** on long-window layers — shifts key representations forward by one position to enable 1-layer induction heads
5. **No attention at all** on layer 6 (MLP-only layer to save compute)

All attention uses **Flash Attention 3** (via `varunneal/flash-attention-3` kernel package) with variable-length sequences.

> **Why it exists**: Every speedrun trick here wins fractions of a second. Paired heads improve information routing. Sliding windows save FLOPs. Key offset helps induction.
>
> **For us**: Flash Attention 3 is **H100-only** (requires SM90 TMA features). On A100, we need Flash Attention 2 or PyTorch's `scaled_dot_product_attention`. Paired heads, the MLP-only layer 6, and the specific skip patterns are overfitted to this 11-layer layout. **Replace with standard causal attention + FlashAttention 2**. Consider sliding windows only if sequence lengths warrant it.

### 2.4 Skip Connections (Hardcoded)

Layer 3's post-attention output is saved and injected (gated) into layer 6. This is a specific long-range skip connection tuned for this architecture.

> **For us**: **Remove**. In a looped architecture, skip connections across loop iterations are the interesting design choice — and they need to be systematic, not hardcoded.

### 2.5 MLP: Fused ReLU²

```python
ReLUSqrdMLP(norm(x), c_fc, c_proj)  # = (relu(x @ W1.T))² @ W2.T
```

The activation function is ReLU², which is ~1-2% better than GELU (from the SwiGLU lineage, arXiv:2109.08668). It's fused into a single Triton kernel that does the matmul and activation in one pass.

> **For us**: ReLU² is a solid choice regardless of architecture. The Triton kernel, however, **uses TensorDescriptors** (Triton's TMA-based API, `triton.tools.tensor_descriptor`), which is an **SM90+ (H100+) feature**. On A100, we need a fallback. **Keep the mathematical idea (ReLU²), rewrite the kernel** or use a simple PyTorch implementation.

### 2.6 Output Head: Fused Softcapped Cross-Entropy

The loss computation fuses the linear projection, softcapping, and cross-entropy into a single Triton kernel:

```
logits = softcap(x @ lm_head.W)  →  23 * sigmoid((logits + 5) / 7.5)
loss = cross_entropy(logits, targets)
```

This also includes **Multi-Token Prediction (MTP)**: the model predicts not just the next token, but the next 2-3 tokens with decreasing weights. MTP is phased out during training (3→2→1 targets).

> **For us**: The softcapping is a nice trick (from Gemma 2) that prevents logit explosion. The fused kernel uses **FP8 matmuls internally** (H100-only). MTP is designed for language modeling, not reasoning tasks where we typically want exact answers. **Keep softcapping as a concept, rewrite without FP8**. **Remove MTP** for reasoning tasks.

### 2.7 Smear Gate & Backout

**Smear gate**: Mixes each token's embedding with the *previous* token's embedding (gated). A positional shortcut.

**Backout**: Subtracts a scaled version of the hidden state at layer 7 from the final output. The idea is that early layers build representations needed for later computation but not for final prediction — backing them out improves the prediction head.

> **For us**: Both are speedrun-specific tricks. **Remove both**. Backout in particular makes no sense for a looped model where the "early" layers are run multiple times.

---

## 3. The Optimization Stack (Deep Dive)

### 3.1 NorMuon Optimizer

This is arguably the **most valuable piece** of the codebase. NorMuon is a variant of Muon (Momentum Orthogonalized by Newton-Schulz) with several enhancements:

**How Muon works (intuitively):**
1. Compute gradient with Nesterov momentum (standard SGD)
2. Find the nearest orthogonal matrix to that gradient update
3. Apply that orthogonal update to the weights

*Why orthogonal?* Orthogonal updates preserve the spectral structure of weight matrices — they rotate the weight space without changing the scale. This leads to more stable training with larger learning rates.

**How the orthogonalization works (Polar Express):**

The Newton-Schulz iteration (now replaced by "Polar Express") computes the polar decomposition of a matrix — finding the nearest orthogonal matrix. It runs 5 iterations of a polynomial recurrence in bf16, using custom Triton kernels for the matrix operations (`XXT`, `XTX`, `ba_plus_cAA`).

```
For each iteration:
  A = X.T @ X  (or X @ X.T for wide matrices)
  B = b*A + c*(A@A)
  X = a*X + X @ B
```

**NorMuon additions over standard Muon:**
- **Low-rank variance estimator** (from Adafactor) — normalizes the update to reduce variance across matrix dimensions
- **Cautious weight decay** — only decays weights that agree in sign with the gradient (prevents decay from fighting the gradient)
- **Mantissa tracking** — stores the lower 16 bits of the bf16→fp32 parameter representation to accumulate small updates that would otherwise be lost to bf16 rounding

> **For us**: **Keep NorMuon**. It's architecture-agnostic and works for any model with 2D weight matrices (which is all of them). The Triton kernels for the polar decomposition (`XXT`, `XTX`, `ba_plus_cAA`) use standard Triton operations — **no TMA descriptors, A100-compatible**. The only issue is that block sizes are hardcoded for H100 autotuning — they should be **re-tuned for A100** but will work as-is.

### 3.2 Adam (for Non-Matrix Parameters)

Embeddings, scalars, gate weights, and biases use standard Adam with bias correction. The implementation is clean and compiled with `torch.compile`.

> **For us**: **Keep**. Nothing H100-specific here.

### 3.3 FP8 Matmuls

The `CastedLinearT` layer uses FP8 (float8_e4m3fn) for forward pass matmuls and FP8 (float8_e5m2) for backward pass gradients via `torch._scaled_mm`. This is done through a custom autograd function that:

1. Quantizes activations and weights to FP8
2. Uses `torch._scaled_mm` (which calls cuBLAS FP8 tensor cores)
3. Stores the FP8 tensors for the backward pass (saving memory vs storing bf16)

The `FusedSoftcappedCrossEntropy` kernel also uses FP8 internally.

> **For us**: **FP8 matmuls require SM89+ (H100/L40S)**. A100 has no FP8 tensor core support. There's already an `DISABLE_FP8` environment variable (line 1241) but the fused cross-entropy kernel always uses FP8 internally.
>
> **Action**: **Remove FP8 entirely**. Use bf16 matmuls throughout. This is a significant code simplification. The `CastedLinearT` class with transposed weight storage is still useful for gradient accumulation — just remove the FP8 path.

### 3.4 Transposed Weight Storage

`CastedLinearT` stores weights as `(in_features, out_features)` instead of the standard `(out_features, in_features)`. This means the gradient accumulation `x.T @ grad` produces a tensor already in the right layout, avoiding a slow transpose kernel.

> **For us**: **Keep**. This is a pure memory-layout optimization that works on any GPU.

### 3.5 Parameter Banks

Instead of having separate `nn.Linear` modules for each layer, all attention weights are stored in a single tensor `attn_bank` of shape `(10, 4*768, 768)` and all MLP weights in `mlp_bank` of shape `(12, 2, 3072, 768)`. This enables:

1. **Efficient sharded optimization**: The entire bank is treated as one parameter for reduce-scatter/all-gather
2. **Even distribution across GPUs**: 40 attention matrices / 8 GPUs = 5 per GPU
3. **The Muon optimizer operates on the reshaped bank as a batch**, running polar decomposition on all matrices simultaneously

> **For us**: **Keep the concept, but simplify**. For a looped model, all iterations share the same weights anyway — so the bank is naturally smaller (just the unique layers). The sharding is only needed for multi-GPU distributed training.

### 3.6 Distributed Communication

The codebase implements a sophisticated communication strategy:
- **Explicit scatter/work ordering** instead of hook-based gradient communication
- **Sparse communication** for the bigram embedding (only sends gradient rows that were actually used)
- **Overlapped computation and communication**: small parameter updates run while large reduce-scatters are in flight

> **For us**: **Keep the infrastructure but simplify**. Remove sparse bigram communication (no bigram embed). The explicit ordering pattern is good practice for any distributed training.

### 3.7 Training Schedule

An elaborate multi-stage schedule manages:
- Batch size ramp: 8→16→24 sequences × 2048 tokens
- Window size ramp: short and long windows grow over training
- MTP weight decay: 3-token → 2-token → 1-token prediction
- LR warmup and cooldown with per-batch-size scaling
- Embed/lm_head tying → splitting at 2/3 of training
- YaRN RoPE extension when window sizes change

> **For us**: **Remove most of this**. Our reasoning tasks have fixed sequence lengths and don't need progressive window scaling. Keep the basic LR schedule (warmup + cosine/linear decay). The embed/lm_head split trick is language-modeling-specific.

---

## 4. A100 Compatibility Audit

Here's every component that won't work on A100 (SM80) out of the box:

| Component | Why It Fails on A100 | Severity | Fix |
|-----------|---------------------|----------|-----|
| **Flash Attention 3** (`varunneal/flash-attention-3`) | Requires SM90 TMA hardware | **Blocker** | Use FA2 or `torch.nn.functional.scaled_dot_product_attention` |
| **FP8 matmuls** (`torch._scaled_mm` with float8) | No FP8 tensor cores on A100 | **Blocker** | Use bf16 matmuls |
| **Fused cross-entropy kernel** (uses FP8 internally) | FP8 quantization + `_scaled_mm` | **Blocker** | Rewrite with bf16 or use standard `F.cross_entropy` |
| **TensorDescriptor in MLP kernel** | `triton.tools.tensor_descriptor` is SM90+ TMA | **Blocker** | Rewrite MLP kernel without TMA descriptors |
| **Triton kernel block sizes** (XXT, XTX, ba_plus_cAA) | Hardcoded for H100 autotuning | **Performance** | Re-tune for A100 (will work but suboptimally) |
| **`kernels` package** (flash-attention-3) | Pulls H100-specific kernel builds | **Blocker** | Remove dependency, use standard packages |

### What Works Fine on A100

- All Triton kernels in `triton_kernels.py` *except* the MLP kernel (the XXT/XTX/ba_plus_cAA kernels use standard Triton ops)
- The NorMuon optimizer (Polar Express)
- All PyTorch operations (RMS norm, embeddings, etc.)
- The distributed communication infrastructure
- `torch.compile` with bf16

---

## 5. The Looped Architecture (Current State)

The looping support was added in commit `f550f22` ("Add looped (depth-recurrent) transformer support"). Here's what exists:

### 5.1 Configuration

```python
@dataclass
class Hyperparameters:
    n_loop: int = 1          # iterations (1 = no looping)
    recur_start: int = 0     # first recurrence layer (inclusive)
    recur_end: int = 0       # last recurrence layer (exclusive)
    bptt_k: int | None = None  # truncated BPTT depth
    input_injection: str = "passthrough"  # "inject", "inject_random", "passthrough"
```

### 5.2 Forward Pass Structure

```
Prelude:    layers [0, recur_start)           — run once
Recurrence: layers [recur_start, recur_end)   — run n_loop times
Coda:       layers [recur_end, num_layers)    — run once
```

### 5.3 Input Injection Modes

1. **"passthrough"**: Recurrence just feeds the output of iteration *k* as input to iteration *k+1*
2. **"inject"**: Concatenates prelude output with recurrent state, projects through a learned linear layer (`[I|0]` init so it starts as identity)
3. **"inject_random"**: Same as inject but initializes the state randomly instead of from the prelude output

### 5.4 Post-Recurrence Normalization

After each loop iteration, the output is passed through `self.norm_recur` (RMSNorm) to prevent activation blowup — a known issue with recurrent depth.

### 5.5 Truncated BPTT

If `bptt_k` is set, gradients are detached for all iterations except the last `k`, saving memory during backprop.

### 5.6 What's Missing / Problematic

1. **Massive code duplication**: The layer-processing logic is copy-pasted three times (prelude, recurrence, coda) — ~100 lines each. Any change requires updating all three.

2. **All 11-layer complexity is preserved in the loop**: The recurrence block inherits paired heads, skip connections, parallel lanes, value embeddings, etc. This makes the loop extremely heavy and hard to reason about.

3. **No adaptive iteration count**: The model runs a fixed `n_loop` iterations — there's no mechanism to decide "I need more thinking for this input".

4. **No intermediate supervision**: Loss is only computed on the final output. Intermediate iterations could benefit from auxiliary losses.

5. **The skip connections (3→6) must be within the recur block**: This is explicitly noted in a comment but creates a fragile dependency.

6. **No proper positional encoding for loop iterations**: The model uses the same RoPE across all iterations, which means it can't distinguish "pass 1 through layer 3" from "pass 2 through layer 3".

---

## 6. Keep / Rework / Remove Decision Matrix

### KEEP (High-Value, Architecture-Agnostic)

| Component | Location | Why Keep |
|-----------|----------|----------|
| **NorMuon optimizer** | `train_gpt.py:367-940` | State-of-the-art optimizer for transformer weight matrices. Works on any 2D parameter. |
| **Polar Express** (orthogonalization) | `train_gpt.py:158-248` | Fast bf16 Newton-Schulz iteration. A100-compatible. |
| **Triton kernels for Muon** (XXT, XTX, ba_plus_cAA) | `triton_kernels.py:1-397` | Fused symmetric matmul kernels. A100-compatible (re-tune block sizes). |
| **Transpose copy/add kernels** | `triton_kernels.py:636-757` | Useful for any tied-weight scenario. A100-compatible. |
| **Transposed weight storage** | `train_gpt.py:949-976` | Memory layout optimization. GPU-agnostic. |
| **Distributed training infra** | `train_gpt.py:46-57` + optimizer comms | Multi-GPU reduce-scatter/all-gather. Architecture-agnostic. |
| **Parameter bank pattern** | `train_gpt.py:1217-1223` | Enables efficient batched Muon optimization. Simplify for looped model. |
| **Cautious weight decay** | `train_gpt.py:909-925` | Better than standard decoupled WD. Architecture-agnostic. |
| **Mantissa tracking** | `train_gpt.py:909-925` | Enables higher precision bf16 updates. Architecture-agnostic. |
| **ReLU² activation** | Mathematical idea | Simple, effective. Rewrite kernel for A100. |
| **RMSNorm** | `train_gpt.py:945-946` | Standard, fast. Keep. |
| **Softcapping** (concept) | `train_gpt.py:1561` | Prevents logit explosion. Rewrite without FP8. |
| **Truncated BPTT** | `train_gpt.py:1496-1498` | Essential for memory-efficient deep recurrence. |
| **Input injection** | `train_gpt.py:1287-1293` | Good mechanism for looped models. Simplify. |
| **Post-recurrence norm** | `train_gpt.py:1293` | Prevents activation blowup in recurrence. Essential. |

### REWORK (Good Ideas, Wrong Implementation for Us)

| Component | What to Change | Why |
|-----------|---------------|-----|
| **Attention** | Replace FA3 → FA2 or SDPA. Remove paired heads, key offset. Keep sliding window optionally. | A100 compat + simplicity |
| **MLP kernel** | Rewrite without TMA TensorDescriptors. Use standard Triton matmul + ReLU². | A100 compat |
| **Cross-entropy** | Remove FP8, remove MTP. Keep softcapping as a simple PyTorch op. | A100 compat + reasoning tasks don't need MTP |
| **CastedLinearT** | Remove FP8 path. Keep transposed storage for gradient efficiency. | A100 compat |
| **Forward pass** | Extract layer logic into a single `transformer_block()` function. Call it from prelude/recurrence/coda without duplication. | Maintainability |
| **Training schedule** | Replace with simple warmup + cosine LR decay. Fixed batch size. | Reasoning tasks have fixed structure |
| **Data loader** | Replace FineWeb pipeline with reasoning task generators. Keep the distributed infrastructure. | New task domain |
| **Parameter banks** | Simplify: for looped model, there's naturally only ~3-4 unique layers. | Less complexity |

### REMOVE (Speedrun-Specific, No Value for Us)

| Component | Location | Why Remove |
|-----------|----------|-----------|
| **Bigram hash embedding** | `train_gpt.py:1251-1252, 1655-1670` | Language-modeling shortcut. Irrelevant for reasoning. |
| **Value embeddings** | `train_gpt.py:1195` | Token-specific attention biases tuned for English text. |
| **Two-lane residual stream** | `train_gpt.py:1254-1261` | Adds looping complexity for marginal gain on reasoning. |
| **Skip connections (3→6)** | `train_gpt.py:1306-1307, 1390-1395` | Hardcoded for 11-layer layout. |
| **Smear gate** | `train_gpt.py:1186-1187, 1351-1352` | Positional shortcut for language modeling. |
| **Backout** | `train_gpt.py:1309, 1557-1558` | Makes no sense for looped models. |
| **MLP-only layer 6** | `train_gpt.py:1398-1400` | Speedrun trick, not principled. |
| **Paired head attention** | `train_gpt.py:1105-1124, 1234-1236` | Complex, tuned for specific layers. |
| **YaRN progressive scaling** | `train_gpt.py:981-1048` | Window schedule is being removed. |
| **Multi-Token Prediction** | `train_gpt.py:1563` | Language modeling technique. |
| **Embed/lm_head tying→splitting** | `train_gpt.py:1248-1249, split logic` | Language modeling trick. |
| **Sparse bigram communication** | `train_gpt.py:252-339` | Only for bigram embed (being removed). |
| **train_gpt_medium.py** | Entire file | 350M variant not relevant. |
| **Attn gate / VE gate banks** | `train_gpt.py:1198-1199` | Tied to value embeddings and specific layer patterns. |
| **x0_lambdas, bigram_lambdas** | `train_gpt.py:1264-1265` | Tied to input injection patterns being removed. |
| **Post-lambdas** (2-lane scaling) | `train_gpt.py:1258-1261` | Tied to 2-lane architecture being removed. |

---

## 7. Reasoning Data Integration Strategy

Our target tasks require a fundamentally different data pipeline than FineWeb. Here's how to think about each task and how to combine them.

### 7.1 Task Overview

#### Arithmetic (Fan et al. — Length Generalization)

**What it is**: Addition, multiplication, and other arithmetic on numbers of varying digit length. The key challenge is *length generalization* — training on 5-digit addition and testing on 20-digit addition.

**Why it's great for looped models**: Arithmetic is inherently iterative. Humans add digit-by-digit with carries. A looped model can learn "one iteration = process one more digit" — and generalize to longer numbers by running more iterations.

**Data format (autoregressive)**:
```
Input:  "3 4 5 + 6 7 8 ="
Output: "1 0 2 3"
```

Tokens are individual digits + operators. Reverse output order (least significant first) is critical for carry propagation in autoregressive generation.

#### SAT (from Recurrent Transformer Paper)

**What it is**: Boolean satisfiability problems — given a formula like `(x1 OR NOT x2) AND (x2 OR x3)`, find an assignment of True/False to variables that makes it all true.

**Why it's great for looped models**: SAT solving is NP-complete, and practical solvers work iteratively (unit propagation, backtracking). A looped model can use each iteration to propagate constraints or try different assignments.

**Data format**:
```
Input:  "x1 x2 x3 | x1 -x2 | x2 x3 | -x1 x3 ="
Output: "1 0 1"  (x1=T, x2=F, x3=T)
```

#### Grid/ARC-like Tasks (from TRM / ARC-AGI)

**What it is**: Small 2D grid transformations — things like "rotate the colored shape 90 degrees" or "fill in the pattern". ARC (Abstraction and Reasoning Corpus) tests are considered a benchmark for general intelligence.

**Why it's great for looped models**: Grid tasks often require iterating over the grid, applying a rule, checking the result, and refining. Each loop iteration can refine the output grid.

**Data format**:
```
Input:  "3x3 | 1 0 1 | 0 1 0 | 1 0 1 -> 3x3 |"
Output: "0 1 0 | 1 0 1 | 0 1 0"
```

#### Graph Traversal / Maze Pathfinding

**What it is**: Given a graph (as an adjacency list or grid), find a path from start to goal.

**Why it's great for looped models**: BFS/DFS are iterative algorithms. Each loop iteration = explore one more step of the frontier.

**Data format (adjacency list)**:
```
Input:  "nodes: A B C D | edges: A-B B-C C-D A-D | path: A -> D ="
Output: "A D"  (shortest path)
```

**Data format (maze)**:
```
Input:  "5x5 | # # # # # | # S . . # | # # # . # | # . . . # | # # # G # ="
Output: "S D D R R R D D R"  (directions)
```

### 7.2 Unified Data Pipeline Design

The key insight: all these tasks share a common structure:

```
[Task Prefix] [Input Encoding] [Separator] [Output Encoding]
```

We recommend building a **task-composable data generator** with these properties:

**1. Shared tokenizer**: Use a small, custom character-level or byte-level tokenizer. Our vocab needs:
- Digits: `0-9`
- Letters: `a-z` (for variables, node names)
- Operators: `+ - * / = | & ~ -> .`
- Structural: `( ) [ ] { } | : ,`
- Grid: `# . S G` (walls, empty, start, goal)
- Special: `[PAD] [BOS] [EOS] [SEP]`

This gives us a vocab of ~60-80 tokens — tiny compared to GPT-2's 50k. This is deliberate: **small vocab = less embedding overhead, more capacity for reasoning**.

**2. Task mixing**: Each training batch can contain examples from multiple tasks. This is simply achieved by:

```python
class ReasoningDataset:
    def __init__(self, tasks: list[TaskGenerator], mix_weights: list[float]):
        self.tasks = tasks
        self.weights = mix_weights

    def generate_batch(self, batch_size, max_seq_len):
        examples = []
        for _ in range(batch_size):
            task = random.choices(self.tasks, weights=self.weights)[0]
            example = task.generate(max_seq_len)
            examples.append(example)
        return collate(examples)
```

**3. Difficulty curriculum**: For length generalization, start with easy examples (short numbers, small grids, few variables) and progressively increase difficulty. This maps naturally to the existing training schedule infrastructure.

**4. Variable-length sequences**: The existing `flash_attn_varlen` infrastructure handles variable-length sequences efficiently. Keep this for batching examples of different lengths.

### 7.3 Integration with Looped Architecture

The loop count should ideally scale with problem difficulty:

| Task | Easy | Hard | Loop Scaling |
|------|------|------|-------------|
| Arithmetic | 3-digit add | 20-digit add | ~1 iter per digit |
| SAT | 3 variables | 20 variables | ~1-3 iters per variable |
| Grid | 3×3 | 10×10 | ~1 iter per row/col |
| Graph | 4 nodes | 20 nodes | ~1 iter per BFS layer |

This motivates **adaptive computation** — letting the model decide how many iterations to run. Two approaches:

1. **Fixed schedule during training, variable during eval**: Train with `n_loop=k` for difficulty level *d*, evaluate with more iterations
2. **Halting mechanism** (like Universal Transformers / PonderNet): The model outputs a halting probability at each iteration; training uses the expected loss under the halting distribution

For initial experiments, fixed `n_loop` per difficulty level is simpler and more debuggable.

---

## 8. Recommended Path Forward

### Phase 1: Clean Slate — Minimal Looped Transformer

**Goal**: A clean, maintainable, A100-compatible looped transformer that trains.

1. **Extract and refactor into modules**:
   - `model.py`: Clean GPT with a single `TransformerBlock` class, called in a loop
   - `optimizer.py`: NorMuon + Adam (extracted from train_gpt.py)
   - `kernels.py`: A100-compatible Triton kernels (XXT, XTX, ba_plus_cAA, transpose ops)
   - `data.py`: Reasoning task generators
   - `train.py`: Training loop

2. **Model architecture** (target ~20-50M params for fast iteration):
   ```
   Embed → [Block × L_prelude] → [Block × L_recur] × n_loop → [Block × L_coda] → Head

   Block = RMSNorm → Attention → Residual → RMSNorm → MLP(ReLU²) → Residual
   ```
   - `model_dim=512`, `num_heads=8`, `head_dim=64`
   - `L_prelude=2`, `L_recur=4`, `L_coda=2` (8 unique layers, effectively deeper with looping)
   - Standard causal attention with Flash Attention 2 or PyTorch SDPA
   - Input injection: `inject` mode (concat + project)
   - Post-loop RMSNorm
   - Truncated BPTT

3. **Remove everything from the "Remove" list**. No bigrams, no value embeddings, no paired heads, no two-lane residual, no smear gate, no backout, no MTP, no YaRN.

4. **Simplify training schedule**: Warmup (linear, ~5% of steps) → Cosine decay → 0. Single batch size. Fixed sequence length per task difficulty.

### Phase 2: Reasoning Tasks

5. **Implement task generators** (start with arithmetic, then add SAT):
   - Character-level tokenizer (~80 tokens)
   - Curriculum: train on easy→hard difficulty
   - Evaluation: test on harder-than-training instances (length generalization)

6. **Experiment with loop counts**: Start with n_loop=4, sweep {2, 4, 8, 16}. Measure both training loss and out-of-distribution generalization.

### Phase 3: Advanced Looping

7. **Add intermediate supervision**: Compute loss at every loop iteration (weighted), not just the final one. This gives the model gradient signal at every depth.

8. **Explore adaptive computation**: PonderNet-style halting or a simpler "fixed budget, test with more" approach.

9. **Explore cross-iteration state passing**: Beyond simple hidden state — consider adding a small "scratch pad" memory that persists across iterations.

### What This Preserves from the Original

- **NorMuon optimizer** (the biggest win — faster convergence than Adam alone)
- **Triton kernels for Muon** (efficient orthogonalization)
- **Transposed weight storage** (better gradient accumulation)
- **Distributed training infrastructure** (multi-GPU support)
- **`torch.compile`** (significant speedup for free)
- **Truncated BPTT** (memory-efficient deep recurrence)
- **The monolithic single-file spirit** (just organized into a few focused modules)

### Estimated Parameter Budget

| Component | Parameters | Notes |
|-----------|-----------|-------|
| Token embed | 512 × 80 = 40K | Tiny vocab |
| Prelude (2 blocks) | 2 × (4 × 512² + 2 × 512 × 2048) = 6.3M | Attn + MLP |
| Recur block (4 blocks) | 4 × (4 × 512² + 2 × 512 × 2048) = 12.6M | Shared across loops |
| Coda (2 blocks) | 2 × (4 × 512² + 2 × 512 × 2048) = 6.3M | Attn + MLP |
| LM head | 512 × 80 = 40K | Tiny vocab |
| Inject projection | 1024 × 512 = 524K | Input injection |
| Norms, misc | ~50K | RMSNorm weights |
| **Total** | **~25.8M** | **With n_loop=8: effective depth = 2+32+2 = 36 layers** |

This is a model that's small enough to train in minutes on 2×A100 for each experiment, but deep enough (via looping) to learn multi-step reasoning.

---

*This document should be treated as a living reference. Update it as experiments reveal which components matter and which don't.*
