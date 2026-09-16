from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from minimax_h3_keyless.activation_capture import CapturedPilotCase
from minimax_h3_keyless.pilot import PilotCase
from minimax_h3_keyless.pilot_replay import pilot_case_to_device, verify_captured_pilot_replay


class TinyAttention(nn.Module):
    def forward(self, x, rope_freqs=None, transformer_options=None):
        return x * 0.25


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttention()

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
        h = x + t_emb[0, 0]
        x.add_(self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options))
        return x


def _case() -> PilotCase:
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    return PilotCase(
        x=x,
        t_emb=torch.tensor([[2.0, 3.0]], dtype=torch.float32),
        mod_segments=((0, 3, torch.tensor([0, 0, 0], dtype=torch.long)),),
        rope_freqs=torch.ones(1, 3, 1, 1, 2, 2, dtype=torch.float32),
        transformer_options={"bounded_mask": torch.ones(3, dtype=torch.bool)},
        case_id="replay-case",
        sigma=0.5,
        modality_label="video",
        position_ids=torch.arange(9, dtype=torch.float64).reshape(3, 3),
        context={"artifact": "capture-receipt"},
    )


def test_pilot_case_device_boundary_moves_all_execution_tensors_without_changing_metadata() -> None:
    case = _case()
    moved = pilot_case_to_device(case, "cpu")
    assert moved.context is case.context
    assert moved.case_id == case.case_id
    assert moved.x.dtype == case.x.dtype
    assert moved.t_emb.dtype == case.t_emb.dtype
    assert moved.position_ids.dtype == torch.float64
    assert moved.mod_segments[0][2].device.type == "cpu"
    assert moved.transformer_options["bounded_mask"].device.type == "cpu"
    torch.testing.assert_close(moved.x, case.x)
    torch.testing.assert_close(moved.rope_freqs, case.rope_freqs)


def test_captured_replay_reproduces_actual_post_adaln_attention_input_exactly() -> None:
    case = _case()
    captured = CapturedPilotCase(
        block_index=25,
        case=case,
        attention_input=case.x + case.t_emb[0, 0],
        captured_bytes=1234,
    )
    report = verify_captured_pilot_replay(TinyBlock(), captured, device="cpu")
    assert report.block_index == 25
    assert report.rows == 3
    assert report.attention_input_max_abs_error == 0.0
    assert report.attention_input_mean_abs_error == 0.0
    assert report.block_output_finite is True
    assert report.attention_output_finite is True


def test_captured_replay_fails_closed_when_execution_point_does_not_match_live_capture() -> None:
    case = _case()
    captured = CapturedPilotCase(
        block_index=49,
        case=case,
        attention_input=case.x + case.t_emb[0, 0] + 0.25,
        captured_bytes=1234,
    )
    with pytest.raises(AssertionError, match="post-AdaLN attention input diverged"):
        verify_captured_pilot_replay(TinyBlock(), captured, device="cpu")
