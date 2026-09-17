from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from minimax_h3_keyless.activation_capture import CapturedPilotCase
from minimax_h3_keyless.attention import KeylessAttentionDeploy, KeylessAttentionTrain
from minimax_h3_keyless.initialization import initialize_training_attention_from_native
from minimax_h3_keyless.ops import normalized_positioned, torch_sdpa_attention
from minimax_h3_keyless.pilot import PilotCase, set_pilot_block_stage
from minimax_h3_keyless.pilot_campaign import DATASET_SCHEMA, GATE_SCHEMA, validate_pilot_gate_manifest
from minimax_h3_keyless.pilot_runner import StageATrainStage
from minimax_h3_keyless.progressive import (
    PROGRESSIVE_PREFIX_CONTEXT_KEY,
    ProgressivePrefix,
    validate_progressive_model_prefix,
)
from minimax_h3_keyless.progressive_acceptance import accept_progressive_block
from minimax_h3_keyless.progressive_artifacts import persist_progressive_block_artifacts
from minimax_h3_keyless.progressive_capture_io import (
    ProgressiveCaptureProvenance,
    write_progressive_capture_bundle,
)
from minimax_h3_keyless.progressive_capture_set import (
    ProgressiveCaptureArtifactRef,
    load_progressive_block_capture_set,
)
from minimax_h3_keyless.progressive_runner import run_progressive_block_training


LIVE_CAPTURE_KEY = "minimax_h3_keyless_live_capture_v1"


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
        q = normalized_positioned(q.view(-1, 2, 2), self.q_norm.weight, 1e-5, rope_freqs)
        k = normalized_positioned(k.view(-1, 2, 2), self.k_norm.weight, 1e-5, rope_freqs)
        v = v.view(-1, 2, 2)
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


def _builder(teacher, block_index, route_mode, lambda_relative):
    assert route_mode == "identity" and lambda_relative == 0.0
    student = copy.deepcopy(teacher)
    native = teacher.attn
    attention = KeylessAttentionTrain(4, 2, 2, 1e-5, block_index=block_index, dtype=torch.float32)
    report = initialize_training_attention_from_native(
        attention,
        qkv_weight=native.qkv_proj.weight,
        q_norm_weight=native.q_norm.weight,
        k_norm_weight=native.k_norm.weight,
        out_proj_weight=native.out_proj.weight,
        route_mode="identity",
        lambda_relative=0.0,
    )
    student.attn = attention
    set_pilot_block_stage(student, "route")
    return student, report


def _gate_manifest():
    return {
        "schema": GATE_SCHEMA,
        "thresholds": {
            "stage_a_min_case_total_improvement_fraction": 0.0,
            "stage_a_min_mean_total_relative_improvement": 0.0,
            "stage_a_max_mean_attention_normalized_mse": 1e9,
            "stage_a_max_mean_block_normalized_mse": 1e9,
            "stage_a_max_worst_attention_normalized_mse": 1e9,
            "stage_a_max_worst_block_normalized_mse": 1e9,
            "stage_a_min_mean_attention_cosine": -1.0,
            "stage_a_min_mean_block_cosine": -1.0,
            "stage_a_minimum_attention_cosine": -1.0,
            "stage_a_minimum_block_cosine": -1.0,
            "stage_a_max_modality_nmse_ratio_to_best_baseline": 1e9,
            "stage_a_max_gradient_l2_norm": 1e9,
        },
        "calibration_evidence": {"fixture": "acceptance"},
    }


def _manifest():
    def row(case_id: str, split: str):
        return {
            "case_id": case_id,
            "split": split,
            "prompt": case_id,
            "seed": 3,
            "schedule": {"name": "fixture"},
            "modality_label": "video",
            "resolution": [64, 64],
            "duration_seconds": 1.0,
            "sigmas": [0.2, 0.8],
            "coverage_tags": [],
            "assets": [],
        }
    return {"schema": DATASET_SCHEMA, "cases": [row("train", "train"), row("holdout", "holdout")]}


def _prefix(manifest, gate_manifest):
    from minimax_h3_keyless.pilot_campaign import canonical_json_sha256

    return ProgressivePrefix(
        sweep_id="sweep-acceptance",
        code_commit="deadbeef",
        stage_a_campaign_sha256="a" * 64,
        dataset_manifest_sha256=canonical_json_sha256(manifest),
        gate_manifest_sha256=validate_pilot_gate_manifest(gate_manifest),
    )


def _record(teacher: TinyBlock, prefix: ProgressivePrefix, case_id: str, sigma: float):
    torch.manual_seed(900 + int(sigma * 100))
    x = torch.randn(5, 4)
    case = PilotCase(
        x=x,
        t_emb=torch.zeros(1, 1),
        mod_segments=((0, 5, 0),),
        rope_freqs=None,
        transformer_options={},
        case_id=case_id,
        sigma=sigma,
        modality_label="video",
        context={
            PROGRESSIVE_PREFIX_CONTEXT_KEY: prefix.capture_context(),
            LIVE_CAPTURE_KEY: {
                "block_index": 0,
                "layout": {"seq_len": 5, "segments": [[0, 5, "video"]]},
            },
        },
    )
    seen = {}

    def hook(module, args, kwargs):
        seen["h"] = args[0].detach().clone()

    handle = teacher.attn.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        teacher(x.detach().clone(), case.t_emb, case.mod_segments, None, transformer_options={})
    finally:
        handle.remove()
    return CapturedPilotCase(
        block_index=0,
        case=case,
        attention_input=seen["h"],
        captured_bytes=512,
    )


def _captures(tmp_path, teacher, prefix, manifest):
    refs = []
    for case_id in ("train", "holdout"):
        for sigma in (0.2, 0.8):
            record = _record(teacher, prefix, case_id, sigma)
            provenance = ProgressiveCaptureProvenance(
                code_commit=prefix.code_commit,
                comfy_commit="comfy-fixture",
                dataset_manifest_sha256=prefix.dataset_manifest_sha256,
                gate_manifest_sha256=prefix.gate_manifest_sha256,
                stage_a_campaign_sha256=prefix.stage_a_campaign_sha256,
                prefix_identity_sha256=prefix.identity_sha256,
                target_block=0,
                execution_descriptor="progressive-acceptance-fixture",
            )
            written = write_progressive_capture_bundle(
                tmp_path / f"{case_id}-{sigma}.capture.pt",
                record,
                provenance=provenance,
            )
            refs.append(
                ProgressiveCaptureArtifactRef(
                    bundle_path=written.bundle_path,
                    receipt_path=written.receipt_path,
                    receipt_sha256=written.receipt_sha256,
                )
            )
    return load_progressive_block_capture_set(
        refs,
        manifest,
        prefix,
        minimum_cases=2,
        minimum_sigma_strata=2,
    )


def _no_update_optimizer(parameters, learning_rate, weight_decay):
    return torch.optim.SGD(parameters, lr=0.0, weight_decay=0.0)


def _candidate(tmp_path):
    torch.manual_seed(901)
    model = TinyCore50()
    gate_manifest = _gate_manifest()
    manifest = _manifest()
    prefix = _prefix(manifest, gate_manifest)
    validate_progressive_model_prefix(model, prefix)
    captures = _captures(tmp_path / "captures", model.blocks[0], prefix, manifest)
    result = run_progressive_block_training(
        model.blocks[0],
        captures,
        prefix,
        device="cpu",
        gate_manifest=gate_manifest,
        train_plan=(StageATrainStage("route", 1, 1e-3),),
        student_builder=_builder,
        optimizer_factory=_no_update_optimizer,
        require_bf16_teacher=False,
    )
    assert result.gate.passed is True
    receipt = persist_progressive_block_artifacts(
        tmp_path / "results",
        prefix=prefix,
        captures=captures,
        result=result,
    )
    return model, gate_manifest, prefix, captures, result, receipt


def test_progressive_acceptance_persists_folds_installs_and_advances_atomically(tmp_path) -> None:
    model, gate_manifest, prefix, captures, result, receipt = _candidate(tmp_path)
    assert isinstance(model.blocks[0].attn, TinyNativeAttention)

    advanced = accept_progressive_block(
        model,
        prefix,
        captures,
        result,
        receipt,
        gate_manifest=gate_manifest,
        fold_atol=1e-5,
        fold_rtol=1e-5,
    )

    assert prefix.accepted_blocks == ()
    assert advanced.accepted_blocks == (0,)
    assert advanced.accepted[0].checkpoint_sha256 == receipt.checkpoint_sha256
    assert advanced.accepted[0].result_sha256 == receipt.result_sha256
    assert isinstance(model.blocks[0].attn, KeylessAttentionDeploy)
    assert isinstance(model.blocks[1].attn, TinyNativeAttention)
    assert not hasattr(model.blocks[0].attn, "query_route")
    validate_progressive_model_prefix(model, advanced)


def test_progressive_acceptance_rejects_forged_holdout_selection_marker_before_mutation(tmp_path) -> None:
    model, gate_manifest, prefix, captures, result, receipt = _candidate(tmp_path)
    original_attention = model.blocks[0].attn
    forged = replace(result, selection_split="holdout")

    with pytest.raises(RuntimeError, match="train-only initialization selection"):
        accept_progressive_block(
            model,
            prefix,
            captures,
            forged,
            receipt,
            gate_manifest=gate_manifest,
            fold_atol=1e-5,
            fold_rtol=1e-5,
        )

    assert model.blocks[0].attn is original_attention
    validate_progressive_model_prefix(model, prefix)
