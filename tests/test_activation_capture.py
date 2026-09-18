from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from minimax_h3_keyless.activation_capture import PilotActivationCapture
from minimax_h3_keyless.contracts import PROVIDER_KEY


class TinyAttention(nn.Module):
    def forward(self, x, rope_freqs=None, transformer_options=None):
        return x * 0.25


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttention()

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None, attention=None):
        attention = self.attn if attention is None else attention
        h = x + t_emb[0, 0]
        x.add_(attention(h, rope_freqs=rope_freqs, transformer_options=transformer_options))
        return x


class TinyModel(nn.Module):
    def __init__(self, depth=3):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock() for _ in range(depth)])

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
        for block in self.blocks:
            x = block(
                x,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options=transformer_options,
            )
        return x


def _inputs():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    t_emb = torch.tensor([[2.0, 3.0]])
    rope = torch.ones(1, 3, 1, 1, 2, 2)
    segments = ((0, 3, 0),)
    layout = SimpleNamespace(
        position_ids=torch.arange(9, dtype=torch.float64).reshape(3, 3),
        signature=(1, 1, 1, 1, 1),
        seq_len=3,
        segments=(("video", 3),),
    )
    options = {"minimax_h3_layout": layout, "minimax_h3_sigma_shift_video": 12.0}
    return x, t_emb, segments, rope, options


def test_live_capture_clones_each_mutable_activation_state_and_post_adaln_input() -> None:
    model = TinyModel()
    x, t_emb, segments, rope, options = _inputs()
    original = x.clone()
    with PilotActivationCapture(
        model,
        case_id="case-1",
        sigma=0.5,
        modality_label="video",
        max_capture_bytes=1 << 20,
        block_indices=(0, 2),
        context={"split": "holdout"},
    ) as capture:
        model(x, t_emb, segments, rope, transformer_options=options)
    records = capture.records()
    assert [record.block_index for record in records] == [0, 2]

    first = records[0]
    torch.testing.assert_close(first.case.x, original)
    torch.testing.assert_close(first.attention_input, original + 2.0)
    torch.testing.assert_close(first.case.position_ids, options["minimax_h3_layout"].position_ids)
    assert first.case.transformer_options == {}
    assert first.case.context["split"] == "holdout"
    capture_context = first.case.context["minimax_h3_keyless_live_capture_v1"]
    assert capture_context["layout"]["seq_len"] == 3
    assert "minimax_h3_layout" in capture_context["transformer_options"]
    assert first.captured_bytes > 0

    # The same residual tensor object is mutated in place through every block. A correct
    # capture must snapshot its numerical state per observation rather than deduplicating
    # by object identity. Block 2 sees the result of blocks 0 and 1.
    after_block0 = original * 1.25 + 0.5
    expected_block2_input = after_block0 * 1.25 + 0.5
    second = records[1]
    torch.testing.assert_close(second.case.x, expected_block2_input)
    torch.testing.assert_close(second.attention_input, expected_block2_input + 2.0)
    assert not torch.equal(first.case.x, second.case.x)

    # The original stream is mutated by the model, but captured tensors are private CPU copies.
    assert not torch.equal(x, original)
    torch.testing.assert_close(first.case.x, original)


def test_reused_immutable_timestep_rope_and_position_tensors_are_interned() -> None:
    model = TinyModel()
    x, t_emb, segments, rope, options = _inputs()
    # x and post-AdaLN h are mutable snapshots per block. t_emb/rope/position_ids are
    # immutable across the H3 block loop and therefore count once.
    unique_bytes = (
        3 * x.numel() * x.element_size()
        + t_emb.numel() * t_emb.element_size()
        + rope.numel() * rope.element_size()
        + options["minimax_h3_layout"].position_ids.numel() * options["minimax_h3_layout"].position_ids.element_size()
        + 3 * x.numel() * x.element_size()
    )
    with PilotActivationCapture(
        model,
        case_id="case-intern",
        sigma=0.25,
        modality_label="video",
        max_capture_bytes=unique_bytes,
        block_indices=(0, 1, 2),
    ) as capture:
        model(x, t_emb, segments, rope, transformer_options=options)
    records = capture.records()
    assert records[-1].captured_bytes == unique_bytes
    assert records[0].case.t_emb is records[1].case.t_emb is records[2].case.t_emb
    assert records[0].case.rope_freqs is records[1].case.rope_freqs is records[2].case.rope_freqs
    assert records[0].case.position_ids is records[1].case.position_ids is records[2].case.position_ids


def test_capture_fails_closed_when_explicit_budget_is_exceeded_and_removes_hooks() -> None:
    model = TinyModel(depth=1)
    x, t_emb, segments, rope, options = _inputs()
    with pytest.raises(RuntimeError, match="byte budget"):
        with PilotActivationCapture(
            model,
            case_id="too-large",
            sigma=0.5,
            modality_label="video",
            max_capture_bytes=1,
            block_indices=(0,),
        ):
            model(x, t_emb, segments, rope, transformer_options=options)

    # Hook cleanup is unconditional; the same model runs normally after the failed capture.
    x2, t2, segments2, rope2, options2 = _inputs()
    model(x2, t2, segments2, rope2, transformer_options=options2)


def test_capture_rejects_multiple_selected_block_executions_in_one_session() -> None:
    model = TinyModel(depth=1)
    x, t_emb, segments, rope, options = _inputs()
    with pytest.raises(RuntimeError, match="more than once"):
        with PilotActivationCapture(
            model,
            case_id="duplicate-eval",
            sigma=0.5,
            modality_label="video",
            max_capture_bytes=1 << 20,
            block_indices=(0,),
        ):
            model(x.clone(), t_emb, segments, rope, transformer_options=options)
            model(x.clone(), t_emb, segments, rope, transformer_options=options)


def test_plain_native_capture_rejects_keyless_provider_and_attention_override() -> None:
    model = TinyModel(depth=1)
    x, t_emb, segments, rope, options = _inputs()
    options = dict(options)
    options[PROVIDER_KEY] = object()
    with pytest.raises(RuntimeError, match="Keyless provider"):
        with PilotActivationCapture(
            model,
            case_id="provider",
            sigma=0.5,
            modality_label="video",
            max_capture_bytes=1 << 20,
            block_indices=(0,),
        ):
            model(x.clone(), t_emb, segments, rope, transformer_options=options)

    class Override(nn.Module):
        def forward(self, x, rope_freqs=None, transformer_options=None):
            return torch.zeros_like(x)

    capture = PilotActivationCapture(
        model,
        case_id="override",
        sigma=0.5,
        modality_label="video",
        max_capture_bytes=1 << 20,
        block_indices=(0,),
    )
    with pytest.raises(RuntimeError, match="attention override"):
        with capture:
            model.blocks[0](
                x.clone(),
                t_emb,
                segments,
                rope,
                transformer_options=_inputs()[4],
                attention=Override(),
            )


def test_capture_requires_every_selected_block_to_be_observed() -> None:
    model = TinyModel(depth=2)
    x, t_emb, segments, rope, options = _inputs()
    with PilotActivationCapture(
        model,
        case_id="missing",
        sigma=0.5,
        modality_label="video",
        max_capture_bytes=1 << 20,
        block_indices=(0, 1),
    ) as capture:
        model.blocks[0](x, t_emb, segments, rope, transformer_options=options)
    with pytest.raises(RuntimeError, match="did not observe"):
        capture.records()
