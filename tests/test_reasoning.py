"""Tests for data/reasoning.py — tokenizer, task generators, and batching."""

from __future__ import annotations

import random

import torch

from data.reasoning import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SEP_ID,
    VOCAB,
    VOCAB_SIZE,
    ArithmeticTask,
    GridTask,
    MazeTask,
    SATTask,
    TaskMix,
    decode,
    encode,
    example_to_ids,
    generate_batch,
)


class TestTokenizer:
    def test_vocab_size_matches(self):
        assert len(VOCAB) == VOCAB_SIZE

    def test_encode_decode_roundtrip(self):
        text = "123+456=579"
        ids = encode(text)
        assert decode(ids) == text

    def test_encode_all_vocab_chars(self):
        """Every character in the vocab should encode to a unique id."""
        for i, tok in enumerate(VOCAB):
            if len(tok) == 1:
                assert encode(tok) == [i]

    def test_decode_tensor(self):
        ids = torch.tensor([BOS_ID, SEP_ID, EOS_ID])
        result = decode(ids)
        assert "<bos>" in result
        assert "=" in result
        assert "<eos>" in result

    def test_encode_unknown_char_raises(self):
        """Characters not in vocab should raise KeyError."""
        import pytest

        with pytest.raises(KeyError):
            encode("🔥")


class TestArithmeticTask:
    def test_generates_valid_example(self):
        task = ArithmeticTask()
        rng = random.Random(42)
        ex = task.generate(difficulty=3, rng=rng)
        assert ex.task_prefix == "add"
        assert "+" in ex.input_text
        # Output should be space-separated reversed digits of the sum
        assert len(ex.output_text) > 0

    def test_result_is_correct(self):
        task = ArithmeticTask()
        rng = random.Random(42)
        for _ in range(20):
            ex = task.generate(difficulty=3, rng=rng)
            # Parse input: "d d d + d d d"
            parts = ex.input_text.split(" + ")
            a = int(parts[0].replace(" ", ""))
            b = int(parts[1].replace(" ", ""))
            expected = str(a + b)[::-1]
            actual = ex.output_text.replace(" ", "")
            assert actual == expected, f"{a}+{b}: expected {expected!r}, got {actual!r}"

    def test_difficulty_controls_digits(self):
        task = ArithmeticTask()
        rng = random.Random(42)
        ex1 = task.generate(difficulty=1, rng=rng)
        ex5 = task.generate(difficulty=5, rng=rng)
        # difficulty=1 produces 1-digit numbers, difficulty=5 produces 5-digit
        digits_1 = ex1.input_text.split(" + ")[0].replace(" ", "")
        digits_5 = ex5.input_text.split(" + ")[0].replace(" ", "")
        assert len(digits_1) == 1
        assert len(digits_5) == 5


class TestSATTask:
    def test_generates_valid_example(self):
        task = SATTask()
        rng = random.Random(42)
        ex = task.generate(difficulty=3, rng=rng)
        assert ex.task_prefix == "sat"
        assert "|" in ex.input_text
        assert all(c in ("T", "F", " ") for c in ex.output_text)

    def test_assignment_satisfies_formula(self):
        task = SATTask()
        rng = random.Random(42)
        for _ in range(20):
            ex = task.generate(difficulty=4, rng=rng)
            # Parse assignment
            n_vars = 4
            var_names = [chr(ord("a") + i) for i in range(n_vars)]
            values = ex.output_text.split()
            assignment = {v: val == "T" for v, val in zip(var_names, values)}
            # Parse and check clauses
            clauses = ex.input_text.split(" | ")
            for clause in clauses:
                literals = clause.strip().split()
                satisfied = False
                for lit in literals:
                    if lit.startswith("~"):
                        satisfied = satisfied or not assignment[lit[1:]]
                    else:
                        satisfied = satisfied or assignment[lit]
                assert satisfied, f"Clause {clause!r} not satisfied by {assignment}"


class TestGridTask:
    def test_generates_valid_example(self):
        task = GridTask()
        rng = random.Random(42)
        ex = task.generate(difficulty=3, rng=rng)
        assert ex.task_prefix == "grid"
        assert "flip" in ex.input_text

    def test_flip_is_correct(self):
        task = GridTask()
        rng = random.Random(42)
        for _ in range(10):
            ex = task.generate(difficulty=3, rng=rng)
            # Parse input grid (after "flip ")
            input_grid_str = ex.input_text[len("flip ") :]
            input_rows = [row.strip().split() for row in input_grid_str.split(",")]
            output_rows = [row.strip().split() for row in ex.output_text.split(",")]
            for in_row, out_row in zip(input_rows, output_rows):
                assert in_row[::-1] == out_row


class TestMazeTask:
    def test_generates_valid_example(self):
        task = MazeTask()
        rng = random.Random(42)
        ex = task.generate(difficulty=3, rng=rng)
        assert ex.task_prefix == "maze"
        assert all(m in ("u", "d", "l", "r", " ") for m in ex.output_text)

    def test_path_reaches_goal(self):
        task = MazeTask()
        rng = random.Random(42)
        for _ in range(10):
            ex = task.generate(difficulty=3, rng=rng)
            # Parse grid
            rows = [row.strip().split() for row in ex.input_text.split(",")]
            # Find S
            sr, sc = None, None
            gr, gc = None, None
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    if cell == "S":
                        sr, sc = r, c
                    elif cell == "G":
                        gr, gc = r, c
            assert sr is not None and gr is not None
            # Follow path
            r, c = sr, sc
            moves = ex.output_text.split()
            for move in moves:
                if move == "u":
                    r -= 1
                elif move == "d":
                    r += 1
                elif move == "l":
                    c -= 1
                elif move == "r":
                    c += 1
                assert rows[r][c] != "#", f"Path walks into wall at ({r},{c})"
            assert (r, c) == (gr, gc), f"Path ends at ({r},{c}), expected ({gr},{gc})"


class TestExampleToIds:
    def test_output_shapes(self):
        task = ArithmeticTask()
        rng = random.Random(42)
        ex = task.generate(difficulty=2, rng=rng)
        inp, tgt = example_to_ids(ex, max_seq_len=64)
        assert inp.shape == (63,)
        assert tgt.shape == (63,)

    def test_input_portion_is_masked(self):
        task = ArithmeticTask()
        rng = random.Random(42)
        ex = task.generate(difficulty=2, rng=rng)
        inp, tgt = example_to_ids(ex, max_seq_len=64)
        # There should be some masked (-100) and some unmasked positions
        non_masked = (tgt != -100).nonzero(as_tuple=True)[0]
        masked = (tgt == -100).nonzero(as_tuple=True)[0]
        assert len(non_masked) > 0, "Should have unmasked positions (output tokens)"
        assert len(masked) > 0, "Should have masked positions (input tokens)"
        # The first few positions should all be masked (the input portion)
        assert tgt[0].item() == -100, "First target position should be masked"

    def test_padding_is_masked(self):
        task = ArithmeticTask()
        rng = random.Random(42)
        ex = task.generate(difficulty=1, rng=rng)
        inp, tgt = example_to_ids(ex, max_seq_len=128)
        # Padding positions in target should be -100
        pad_mask = inp == PAD_ID
        assert (tgt[pad_mask] == -100).all()

    def test_starts_with_bos(self):
        task = ArithmeticTask()
        rng = random.Random(42)
        ex = task.generate(difficulty=2, rng=rng)
        inp, tgt = example_to_ids(ex, max_seq_len=64)
        assert inp[0].item() == BOS_ID


class TestGenerateBatch:
    def test_batch_shapes(self):
        tasks = [ArithmeticTask(), SATTask()]
        mix = TaskMix(tasks=tasks, weights=[0.5, 0.5], difficulty=2, max_seq_len=64)
        rng = random.Random(42)
        inp, tgt = generate_batch(mix, batch_size=8, rng=rng)
        assert inp.shape == (8, 63)
        assert tgt.shape == (8, 63)
        assert inp.dtype == torch.long
        assert tgt.dtype == torch.long
