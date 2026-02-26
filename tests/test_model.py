"""Tests for model.py — LoopedGPT architecture."""

from __future__ import annotations

import pytest
import torch

from model import LoopedGPTConfig, next_multiple_of_n


class TestNextMultipleOfN:
    def test_exact_multiple(self):
        assert next_multiple_of_n(64, n=64) == 64

    def test_rounds_up(self):
        assert next_multiple_of_n(65, n=64) == 128

    def test_small_value(self):
        assert next_multiple_of_n(1, n=64) == 64

    def test_float_input(self):
        assert next_multiple_of_n(57.0, n=64) == 64

    def test_zero(self):
        assert next_multiple_of_n(0, n=64) == 0


class TestLoopedGPTConfig:
    def test_defaults(self):
        cfg = LoopedGPTConfig()
        assert cfg.vocab_size == 57
        assert cfg.model_dim == 512
        assert cfg.n_prelude == 2
        assert cfg.n_recur == 4
        assert cfg.n_coda == 2


# GPU-dependent tests are skipped if CUDA is unavailable
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@requires_cuda
class TestLoopedGPTForward:
    def _make_model(self, **kwargs):
        from model import LoopedGPT

        cfg = LoopedGPTConfig(
            vocab_size=57,
            model_dim=64,
            num_heads=4,
            head_dim=16,
            mlp_dim=128,
            max_seq_len=32,
            n_prelude=1,
            n_recur=2,
            n_coda=1,
            **kwargs,
        )
        return LoopedGPT(cfg).cuda()

    def test_forward_shapes_no_targets(self):
        model = self._make_model(n_loop=1)
        x = torch.randint(0, 57, (2, 16)).cuda()
        logits = model(x)
        effective_vocab = next_multiple_of_n(57, n=64)
        assert logits.shape == (2, 16, effective_vocab)

    def test_forward_shapes_with_targets(self):
        model = self._make_model(n_loop=1)
        x = torch.randint(0, 57, (2, 16)).cuda()
        targets = torch.randint(0, 57, (2, 16)).cuda()
        loss, logits = model(x, targets)
        assert loss.shape == ()
        assert loss.item() > 0
        effective_vocab = next_multiple_of_n(57, n=64)
        assert logits.shape == (2, 16, effective_vocab)

    def test_looped_forward(self):
        model = self._make_model(n_loop=3, input_injection="inject")
        x = torch.randint(0, 57, (2, 16)).cuda()
        targets = torch.randint(0, 57, (2, 16)).cuda()
        loss, logits = model(x, targets)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_passthrough_mode(self):
        model = self._make_model(n_loop=3, input_injection="passthrough")
        x = torch.randint(0, 57, (2, 16)).cuda()
        targets = torch.randint(0, 57, (2, 16)).cuda()
        loss, logits = model(x, targets)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_bptt_truncation(self):
        model = self._make_model(n_loop=4, bptt_k=2)
        x = torch.randint(0, 57, (2, 16)).cuda()
        targets = torch.randint(0, 57, (2, 16)).cuda()
        loss, _ = model(x, targets)
        loss.backward()
        # Verify gradients exist
        assert model.attn_bank.grad is not None

    def test_weight_tying(self):
        model = self._make_model(n_loop=1)
        assert model.lm_head.weight is model.embed.weight

    def test_param_labels(self):
        model = self._make_model(n_loop=2, input_injection="inject")
        labels = {getattr(p, "label", None) for _, p in model.named_parameters()}
        assert "embed" in labels
        assert "attn_bank" in labels
        assert "mlp_up_bank" in labels
        assert "mlp_down_bank" in labels
        assert "inject" in labels
        assert "norm_recur" in labels
