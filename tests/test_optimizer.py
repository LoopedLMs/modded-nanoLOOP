"""Tests for optimizer.py — NorMuonAdam state_dict serialization."""

from __future__ import annotations

from optimizer import NorMuonAdam


class TestStateDict:
    def test_state_dict_uses_labels(self):
        """state_dict keys must be parameter labels, not object ids."""
        from torch import nn

        # Create a minimal model with labelled params
        embed = nn.Embedding(64, 16)
        embed.weight.label = "embed"

        param_table = {
            "embed": {"optim": "adam", "comms": "none", "adam_betas": [0.9, 0.95]},
        }
        opt = NorMuonAdam(
            [("embed.weight", embed.weight)],
            param_table=param_table,
            scatter_order=["embed"],
            work_order=["embed"],
            adam_defaults=dict(lr=0.001, eps=1e-8, weight_decay=0.01),
            normuon_defaults=dict(lr=0.01, momentum=0.95, beta2=0.95, weight_decay=0.1),
        )

        sd = opt.state_dict()
        assert "param_states" in sd
        assert "embed" in sd["param_states"]
        # No int keys (from id())
        for key in sd["param_states"]:
            assert isinstance(key, str), f"Expected string key, got {type(key)}: {key}"

    def test_load_state_dict_roundtrip(self):
        """state_dict should survive save/load cycle."""
        from torch import nn

        embed = nn.Embedding(64, 16)
        embed.weight.label = "embed"

        param_table = {
            "embed": {"optim": "adam", "comms": "none", "adam_betas": [0.9, 0.95]},
        }

        def make_opt():
            return NorMuonAdam(
                [("embed.weight", embed.weight)],
                param_table=param_table,
                scatter_order=["embed"],
                work_order=["embed"],
                adam_defaults=dict(lr=0.001, eps=1e-8, weight_decay=0.01),
                normuon_defaults=dict(lr=0.01, momentum=0.95, beta2=0.95, weight_decay=0.1),
            )

        opt1 = make_opt()
        # Modify state to ensure load actually changes something
        for state in opt1.param_states.values():
            state["step"] = 42
            state["exp_avg"].fill_(1.0)

        sd = opt1.state_dict()

        opt2 = make_opt()
        opt2.load_state_dict(sd)

        for state in opt2.param_states.values():
            assert state["step"] == 42
            assert (state["exp_avg"] == 1.0).all()
