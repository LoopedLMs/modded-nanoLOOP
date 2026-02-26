"""
Reasoning task generators for looped transformer training.

Each generator produces (input_ids, target_ids) pairs using a shared
character-level tokenizer.  Tasks can be mixed in arbitrary proportions
and difficulty is controllable for curriculum learning.

Supported tasks
---------------
- **Arithmetic** (addition, reverse output for carry propagation)
- **SAT** (random k-SAT instances, find satisfying assignment)
- **Grid** (simple 2D grid transformations — flip, rotate)
- **Pathfinding** (grid maze, find shortest path via BFS)

All tasks output sequences in the format::

    [BOS] <task_prefix> <input_encoding> [SEP] <output_encoding> [EOS] [PAD]*
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Tokenizer — tiny character-level vocab for reasoning tasks
# ---------------------------------------------------------------------------

# Special tokens
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SEP_TOKEN = "="

# Build vocab: specials + digits + letters + operators + structural
_SPECIAL = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
_DIGITS = list("0123456789")
_LETTERS = list("abcdefghijklmnopqrstuvwxyz")
_OPS = list("+-*/=|&~>.")
_STRUCT = list("()[]{}#SG,: ")
_BOOL = ["T", "F"]

VOCAB = _SPECIAL + _DIGITS + _LETTERS + _OPS + _STRUCT + _BOOL
VOCAB_SIZE = len(VOCAB)

_tok2id = {t: i for i, t in enumerate(VOCAB)}
_id2tok = dict(enumerate(VOCAB))

PAD_ID = _tok2id[PAD_TOKEN]
BOS_ID = _tok2id[BOS_TOKEN]
EOS_ID = _tok2id[EOS_TOKEN]
SEP_ID = _tok2id[SEP_TOKEN]


def encode(text: str) -> list[int]:
    """Encode a string to token ids (character-level)."""
    return [_tok2id[ch] for ch in text]


def decode(ids: list[int] | torch.Tensor) -> str:
    """Decode token ids back to string."""
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return "".join(_id2tok[i] for i in ids)


# ---------------------------------------------------------------------------
# Base task interface
# ---------------------------------------------------------------------------


@dataclass
class Example:
    """A single reasoning example."""

    input_text: str  # human-readable input
    output_text: str  # human-readable output
    task_prefix: str  # e.g. "add", "sat", "grid", "maze"


class TaskGenerator(ABC):
    """Base class for reasoning task generators."""

    @abstractmethod
    def generate(self, difficulty: int, rng: random.Random) -> Example:
        """Generate one example at the given difficulty level.

        Higher difficulty = longer/harder instances.
        """
        ...

    @property
    @abstractmethod
    def prefix(self) -> str:
        """Short task identifier used as sequence prefix."""
        ...


# ---------------------------------------------------------------------------
# Arithmetic (addition with reversed output)
# ---------------------------------------------------------------------------


class ArithmeticTask(TaskGenerator):
    """Addition of two numbers with reversed (LSB-first) output.

    Reversing the output lets an autoregressive model process carries
    left-to-right, which maps naturally to iterative computation.

    Difficulty controls the number of digits in each operand.

    Example (difficulty=3):
        Input:  "add 3 4 5 + 6 7 8"
        Output: "3 2 0 1"  (345 + 678 = 1023, reversed: 3201)
    """

    @property
    def prefix(self) -> str:
        return "add"

    def generate(self, difficulty: int, rng: random.Random) -> Example:
        n_digits = max(1, difficulty)
        lo = 10 ** (n_digits - 1) if n_digits > 1 else 0
        hi = 10**n_digits - 1
        a = rng.randint(lo, hi)
        b = rng.randint(lo, hi)
        result = a + b
        # Space-separated digits
        a_str = " ".join(str(a))
        b_str = " ".join(str(b))
        r_str = " ".join(reversed(str(result)))  # LSB first
        input_text = f"{a_str} + {b_str}"
        return Example(input_text=input_text, output_text=r_str, task_prefix=self.prefix)


# ---------------------------------------------------------------------------
# SAT (random k-SAT)
# ---------------------------------------------------------------------------


class SATTask(TaskGenerator):
    """Random k-SAT: find a satisfying assignment.

    Generates a random 3-SAT formula that is guaranteed satisfiable
    (we first pick a satisfying assignment, then build clauses around it).

    Difficulty controls the number of variables (clauses = 4.2 × vars).

    Example (difficulty=3, vars=a,b,c):
        Input:  "sat a b ~c | ~a c | b c"
        Output: "T F T"
    """

    @property
    def prefix(self) -> str:
        return "sat"

    def generate(self, difficulty: int, rng: random.Random) -> Example:
        n_vars = max(2, difficulty)
        n_clauses = max(1, int(4.2 * n_vars))  # near phase transition
        var_names = [chr(ord("a") + i) for i in range(min(n_vars, 26))]

        # Pick a satisfying assignment
        assignment = {v: rng.choice([True, False]) for v in var_names}

        clauses = []
        for _ in range(n_clauses):
            clause_vars = rng.sample(var_names, min(3, n_vars))
            literals = []
            # Ensure at least one literal is satisfied
            satisfied = False
            for v in clause_vars:
                negated = rng.choice([True, False])
                lit_val = (not assignment[v]) if negated else assignment[v]
                if lit_val:
                    satisfied = True
                literals.append(f"~{v}" if negated else v)
            # If none satisfied, flip one literal to make it satisfied
            if not satisfied:
                idx = rng.randrange(len(literals))
                v = clause_vars[idx]
                # Make this literal match the assignment
                literals[idx] = v if assignment[v] else f"~{v}"
            clauses.append(" ".join(literals))

        input_text = " | ".join(clauses)
        output_text = " ".join("T" if assignment[v] else "F" for v in var_names)
        return Example(input_text=input_text, output_text=output_text, task_prefix=self.prefix)


# ---------------------------------------------------------------------------
# Grid transformation
# ---------------------------------------------------------------------------


class GridTask(TaskGenerator):
    """Simple 2D grid transformations (horizontal flip).

    Difficulty controls grid size (difficulty × difficulty).

    Example (difficulty=3):
        Input:  "flip 1 0 1 , 0 1 0 , 1 1 0"
        Output: "1 0 1 , 0 1 0 , 0 1 1"
    """

    @property
    def prefix(self) -> str:
        return "grid"

    def generate(self, difficulty: int, rng: random.Random) -> Example:
        size = max(2, difficulty)
        grid = [[rng.randint(0, 1) for _ in range(size)] for _ in range(size)]

        # Horizontal flip
        flipped = [row[::-1] for row in grid]

        def grid_str(g):
            return " , ".join(" ".join(str(c) for c in row) for row in g)

        input_text = f"flip {grid_str(grid)}"
        output_text = grid_str(flipped)
        return Example(input_text=input_text, output_text=output_text, task_prefix=self.prefix)


# ---------------------------------------------------------------------------
# Maze pathfinding (BFS on grid)
# ---------------------------------------------------------------------------


class MazeTask(TaskGenerator):
    """Grid maze pathfinding — find shortest path from S to G.

    Generates a random maze with guaranteed solution (carved from
    a spanning tree). Output is a sequence of moves: u/d/l/r.

    Difficulty controls maze size.

    Example (difficulty=3 → 7×7 maze, i.e. 3×3 cells with walls):
        Input:  "maze # # # # # , # S . . # , # # # . # , # . . . # , # # # G #"
        Output: "d d r r r d d"  (but with actual shortest path)
    """

    @property
    def prefix(self) -> str:
        return "maze"

    def generate(self, difficulty: int, rng: random.Random) -> Example:
        # Maze size: (2*difficulty+1) × (2*difficulty+1) for walls between cells
        n = max(2, difficulty)
        h, w = 2 * n + 1, 2 * n + 1

        # Initialize grid: all walls
        grid = [["#"] * w for _ in range(h)]

        # Carve cells at odd positions
        for r in range(1, h, 2):
            for c in range(1, w, 2):
                grid[r][c] = "."

        # Randomised DFS to carve passages
        visited = set()
        start_cell = (1, 1)
        stack = [start_cell]
        visited.add(start_cell)

        while stack:
            r, c = stack[-1]
            neighbours = []
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                nr, nc = r + dr, c + dc
                if 0 < nr < h and 0 < nc < w and (nr, nc) not in visited:
                    neighbours.append((nr, nc, r + dr // 2, c + dc // 2))
            if neighbours:
                nr, nc, wr, wc = rng.choice(neighbours)
                grid[wr][wc] = "."  # carve wall between cells
                visited.add((nr, nc))
                stack.append((nr, nc))
            else:
                stack.pop()

        # Place S and G
        grid[1][1] = "S"
        grid[h - 2][w - 2] = "G"

        # BFS for shortest path
        sr, sc = 1, 1
        gr, gc = h - 2, w - 2
        queue = deque([(sr, sc, [])])
        bfs_visited = {(sr, sc)}
        path = []

        while queue:
            r, c, moves = queue.popleft()
            if r == gr and c == gc:
                path = moves
                break
            for dr, dc, move in [(-1, 0, "u"), (1, 0, "d"), (0, -1, "l"), (0, 1, "r")]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in bfs_visited and grid[nr][nc] != "#":
                    bfs_visited.add((nr, nc))
                    queue.append((nr, nc, moves + [move]))

        def grid_str(g):
            return " , ".join(" ".join(cell for cell in row) for row in g)

        input_text = grid_str(grid)
        output_text = " ".join(path)
        return Example(input_text=input_text, output_text=output_text, task_prefix=self.prefix)


# ---------------------------------------------------------------------------
# Dataset: collation and batching
# ---------------------------------------------------------------------------

# Registry of all tasks
TASK_REGISTRY: dict[str, type[TaskGenerator]] = {
    "arithmetic": ArithmeticTask,
    "sat": SATTask,
    "grid": GridTask,
    "maze": MazeTask,
}


def example_to_ids(example: Example, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert an Example to padded (input_ids, target_ids) tensors.

    Format: [BOS] <prefix> [SPACE] <input> [SEP] <output> [EOS] [PAD]*

    The model is trained to predict everything after [SEP] — tokens
    before [SEP] have target = -100 (ignored by cross-entropy).
    """
    prefix_str = example.task_prefix + " "
    full_str = prefix_str + example.input_text + "=" + example.output_text

    ids = [BOS_ID] + encode(full_str) + [EOS_ID]

    # Find separator position to mask input from loss
    sep_pos = len([BOS_ID] + encode(prefix_str + example.input_text))

    # Truncate if too long
    if len(ids) > max_seq_len:
        ids = ids[:max_seq_len]

    # Pad
    n_pad = max_seq_len - len(ids)
    ids = ids + [PAD_ID] * n_pad

    input_ids = torch.tensor(ids[:-1], dtype=torch.long)
    target_ids = torch.tensor(ids[1:], dtype=torch.long)

    # Mask: don't compute loss on input portion or padding
    # Everything before sep_pos-1 in target (shifted) should be masked
    target_ids[: sep_pos - 1] = -100
    target_ids[target_ids == PAD_ID] = -100

    return input_ids, target_ids


@dataclass
class TaskMix:
    """Weighted mixture of reasoning tasks."""

    tasks: list[TaskGenerator]
    weights: list[float]
    difficulty: int = 3
    max_seq_len: int = 256


def generate_batch(
    mix: TaskMix,
    batch_size: int,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of (input_ids, target_ids) from the task mixture.

    Returns
    -------
    input_ids : (B, T) int64
    target_ids : (B, T) int64, with -100 for masked positions
    """
    if rng is None:
        rng = random.Random()

    all_inputs, all_targets = [], []
    for _ in range(batch_size):
        task = rng.choices(mix.tasks, weights=mix.weights, k=1)[0]
        example = task.generate(mix.difficulty, rng)
        inp, tgt = example_to_ids(example, mix.max_seq_len)
        all_inputs.append(inp)
        all_targets.append(tgt)

    return torch.stack(all_inputs), torch.stack(all_targets)
