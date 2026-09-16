from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from minimax_h3_keyless.attention import KeylessAttentionTrain
from minimax_h3_keyless.initialization import initialize_training_attention_from_native
from minimax_h3_keyless.ops import normalized_positioned, torch_sdpa_attention
from minimax_h3_keyless.progressive import (
    PROGRESSIVE_PREFIX_CONTEXT_KEY,
    ProgressiveAcceptedBlock,
    ProgressivePrefix,
    progressive_capture_session,
    validate_progressive_model_prefix,
)


class TinyNativeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = 2
        self.head_dim = 2
        self.qkv_proj = nn.Linear(4, 12, bias=False)
        self.q_norm = nn.RMSNorm(2, eps=1e-5)
        self.k_norm = nn.RMSNorm(2, eps=1e-5)
        self.out_proj = nn.Linear(4, 4, bias=False)
        self.to_gate_compress = None

    def forward(self, x, rope_freqs=None, transformer_options=None):
        q, k, v = self.qkv_proj(x).split(4, dim=-1)
        q = normalized_positioned(q.view(x.shape[0], 2, 2), self.q_norm.weight, 1e-5, rope_freqs)
        k = normalized_positioned(k.view(x.shape[0], 2, 2), self.k_norm.weight, 1e-5, rope_freqs)
        v = v.view(x.shape[0], 2, 2)
        out = torch_sdpa_attention(q, k, v, scale=2 ** -0.5).reshape(x.shape[0], 4)
        return self.out_proj(out)


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(4, eps=1e-5)
        self.attn = TinyNativeAttention()
        self.mlp = nn.Linear(4, 4, bias=False)

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
        h = self.norm1(x)
        x = x + self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
        return x + 0.01 * self.mlp(x)


class TinyCore50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(TinyBlock() for _ in range(50))

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
        options = {} if transformer_options is None else transformer_options
        for block in self.blocks:
            x = block(x, t_emb, mod_segments, rope_freqs, transformer_options=options)
        return x


def _accepted(index: int) -> ProgressiveAcceptedBlock:
    return ProgressiveAcceptedBlock(
        block_index=index,
        final_stage="route",
        checkpoint_sha256=(f"{index + 1:02x}" * 32),
        result_sha256=(f"{index + 51:02x}" * 32),
    )


def _prefix(count: int) -> ProgressivePrefix:
    return ProgressivePrefix(
        sweep_id="sweep-001",
        code_commit="deadbeef",
        stage_a_campaign_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        gate_manifest_sha256="c" * 64,
        accepted=tuple(_accepted(index) for index in range(count)),
    )


def _convert_prefix(model: TinyCore50, count: int) -> None:
    for index in range(count):
        native = model.blocks[index].attn
        attention = KeylessAttentionTrain(
            4,
            2,
            2,
            1e-5,
            block_index=index,
            dtype=native.qkv_proj.weight.dtype,
        )
        initialize_training_attention_from_native(
            attention,
            qkv_weight=native.qkv_proj.weight,
            q_norm_weight=native.q_norm.weight,
            k_norm_weight=native.k_norm.weight,
            out_proj_weight=native.out_proj.weight,
            route_mode="identity",
            lambda_relative=0.0,
        )
        model.blocks[index].attn = attention


def test_progressive_prefix_requires_exact_contiguous_early_to_late_acceptance() -> None:
    with pytest.raises(ValueError, match="contiguous early->late"):
        ProgressivePrefix(
            sweep_id="bad",
            code_commit="deadbeef",
            stage_a_campaign_sha256="a" * 64,
            dataset_manifest_sha256="b" * 64,
            gate_manifest_sha256="c" * 64,
            accepted=(_accepted(0), _accepted(2)),
        )

    prefix = _prefix(0)
    advanced = prefix.advance(
        block_index=0,
        final_stage="value",
        checkpoint_sha256="d" * 64,
        result_sha256="e" * 64,
    )
    assert advanced.accepted_blocks == (0,)
    assert advanced.next_block == 1
    assert advanced.identity_sha256 != prefix.identity_sha256
    with pytest.raises(ValueError, match="expected block 1"):
        advanced.advance(
            block_index=2,
            final_stage="value",
            checkpoint_sha256="f" * 64,
            result_sha256="1" * 64,
        )


def test_progressive_model_prefix_rejects_missing_or_premature_conversion() -> None:
    torch.manual_seed(601)
    model = TinyCore50()
    _convert_prefix(model, 1)
    with pytest.raises(RuntimeError, match="accepted progressive block 1"):
        validate_progressive_model_prefix(model, _prefix(2))

    model = TinyCore50()
    _convert_prefix(model, 3)
    with pytest.raises(RuntimeError, match="unaccepted progressive block 2"):
        validate_progressive_model_prefix(model, _prefix(2))


def test_progressive_capture_observes_only_next_block_and_binds_prefix_digest() -> None:
    torch.manual_seed(602)
    model = TinyCore50()
    _convert_prefix(model, 3)
    prefix = _prefix(3)
    validate_progressive_model_prefix(model, prefix)

    x = torch.randn(5, 4)
    with progressive_capture_session(
        model,
        prefix,
        case_id="case-a::sigma=0.5",
        sigma=0.5,
        modality_label="video",
        max_capture_bytes=1024 * 1024,
        context={"split": "train"},
    ) as capture:
        output = model(
            x,
            torch.zeros(1, 1),
            ((0, 5, 0),),
            None,
            transformer_options={},
        )
    assert torch.isfinite(output).all()
    records = capture.records()
    assert len(records) == 1
    record = records[0]
    assert record.block_index == 3
    assert record.case.context["split"] == "train"
    bound = record.case.context[PROGRESSIVE_PREFIX_CONTEXT_KEY]
    assert bound["prefix_identity_sha256"] == prefix.identity_sha256
    assert bound["accepted_blocks"] == [0, 1, 2]
    assert bound["next_block"] == 3


def test_progressive_capture_rejects_complete_prefix() -> None:
    complete = _prefix(50)
    model = TinyCore50()
    _convert_prefix(model, 50)
    assert complete.complete is True
    with pytest.raises(RuntimeError, match="no next block"):
        progressive_capture_session(
            model,
            complete,
            case_id="done",
            sigma=0.5,
            modality_label="video",
            max_capture_bytes=1024,
        )
