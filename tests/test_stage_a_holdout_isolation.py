from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn as nn

from minimax_h3_keyless.activation_capture import CapturedPilotCase
from minimax_h3_keyless.attention import KeylessAttentionTrain
from minimax_h3_keyless.initialization import initialize_training_attention_from_native
from minimax_h3_keyless.ops import normalized_positioned, torch_sdpa_attention
from minimax_h3_keyless.pilot import PilotCase, PilotLossWeights, set_pilot_block_stage
from minimax_h3_keyless.pilot_artifacts import StageAArtifactRequest
from minimax_h3_keyless.pilot_campaign import GATE_SCHEMA, validate_pilot_gate_manifest
from minimax_h3_keyless.pilot_capture_set import StageACaptureSet
from minimax_h3_keyless.pilot_completed_v3 import load_completed_stage_a_block_evidence
from minimax_h3_keyless.pilot_runner import StageATrainStage, run_stage_a_block_pilot


class TinyNativeAttention(nn.Module):
    def __init__(self):
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
        q = q.view(x.shape[0], 2, 2)
        k = k.view(x.shape[0], 2, 2)
        v = v.view(x.shape[0], 2, 2)
        q = normalized_positioned(q, self.q_norm.weight, self.q_norm.eps, rope_freqs)
        k = normalized_positioned(k, self.k_norm.weight, self.k_norm.eps, rope_freqs)
        out = torch_sdpa_attention(q, k, v, scale=2 ** -0.5).reshape(x.shape[0], 4)
        return self.out_proj(out)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.RMSNorm(4, eps=1e-5)
        self.attn = TinyNativeAttention()
        self.mlp = nn.Linear(4, 4, bias=False)

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options=None):
        h = self.norm1(x)
        x = x + self.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
        return x + 0.01 * self.mlp(x)


def _builder(teacher, block_index, route_mode, lambda_relative):
    student = copy.deepcopy(teacher)
    native = teacher.attn
    attention = KeylessAttentionTrain(4, 2, 2, 1e-5, block_index=block_index, dtype=torch.float32)
    report = initialize_training_attention_from_native(
        attention,
        qkv_weight=native.qkv_proj.weight,
        q_norm_weight=native.q_norm.weight,
        k_norm_weight=native.k_norm.weight,
        out_proj_weight=native.out_proj.weight,
        route_mode=route_mode,
        lambda_relative=lambda_relative,
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
        "calibration_evidence": {"fixture": "holdout-isolation"},
    }


def _capture_set(teacher: TinyBlock) -> StageACaptureSet:
    torch.manual_seed(701)
    train = []
    holdout = []
    for split, target in (("train", train), ("holdout", holdout)):
        for index, sigma in enumerate((0.2, 0.8)):
            x = torch.randn(5, 4)
            case = PilotCase(
                x=x,
                t_emb=torch.zeros(1, 1),
                mod_segments=((0, 5, 0),),
                rope_freqs=None,
                transformer_options={},
                case_id=f"{split}-{index}::sigma={sigma}",
                sigma=sigma,
                modality_label="video",
                context={
                    "stage_a_split": split,
                    "minimax_h3_keyless_live_capture_v1": {
                        "layout": {"seq_len": 5, "segments": [[0, 5, "video"]]}
                    },
                },
            )
            captured = {}

            def capture_hidden(module, args, kwargs):
                captured["h"] = args[0].detach().clone()

            handle = teacher.attn.register_forward_pre_hook(capture_hidden, with_kwargs=True)
            try:
                teacher(
                    case.x.detach().clone(),
                    case.t_emb,
                    case.mod_segments,
                    case.rope_freqs,
                    transformer_options={},
                )
            finally:
                handle.remove()
            target.append(
                CapturedPilotCase(
                    block_index=0,
                    case=case,
                    attention_input=captured["h"],
                    captured_bytes=case.x.numel() * case.x.element_size(),
                )
            )
    return StageACaptureSet(
        dataset_manifest_sha256="a" * 64,
        code_commit="test",
        comfy_commit="test",
        execution_descriptor="test",
        train_by_block={0: tuple(train), 25: (), 49: ()},
        holdout_by_block={0: tuple(holdout), 25: (), 49: ()},
        artifact_refs=(),
    )


def test_stage_a_selection_never_consumes_holdout_cases_and_v3_resume_revalidates(tmp_path: Path):
    torch.manual_seed(702)
    teacher = TinyBlock()
    corpus = _capture_set(teacher)
    gate_manifest = _gate_manifest()
    gate_sha = validate_pilot_gate_manifest(gate_manifest)
    request = StageAArtifactRequest(
        str(tmp_path),
        "isolated-run",
        "deadbeef",
        experiment_context_sha256="c" * 64,
    )
    result = run_stage_a_block_pilot(
        teacher,
        corpus,
        block_index=0,
        device="cpu",
        gate_manifest=gate_manifest,
        train_plan=(StageATrainStage("route", 1, 2e-3, max_grad_norm=100.0),),
        loss_weights=PilotLossWeights(attention_output=1.0, block_output=1.0),
        student_builder=_builder,
        artifact_request=request,
    )

    selection_ids = {
        case.case_id
        for evaluation in result.initialization_evaluations
        for case in evaluation.metrics.cases
    }
    holdout_ids = {case.case_id for case in result.candidate.cases}
    assert selection_ids
    assert all(case_id.startswith("train-") for case_id in selection_ids)
    assert holdout_ids
    assert all(case_id.startswith("holdout-") for case_id in holdout_ids)
    assert selection_ids.isdisjoint(holdout_ids)
    assert {case.case_id for case in result.identity_baseline.cases} == holdout_ids
    assert {case.case_id for case in result.least_squares_baseline.cases} == holdout_ids

    best_ls = min(
        (
            evaluation
            for evaluation in result.initialization_evaluations
            if evaluation.route_mode == "least_squares"
        ),
        key=lambda row: (
            row.metrics.mean_attention_normalized_mse,
            row.metrics.mean_block_normalized_mse,
            row.lambda_relative,
        ),
    )
    assert result.least_squares_baseline_lambda_relative == best_ls.lambda_relative
    assert result.artifact is not None
    payload = json.loads(Path(result.artifact.result_path).read_text(encoding="utf-8"))
    assert payload["result"]["selection_split"] == "train_complete_cases"

    completed = load_completed_stage_a_block_evidence(
        request,
        block_index=0,
        final_stage="route",
        expected_dataset_manifest_sha256="a" * 64,
        expected_gate_manifest_sha256=gate_sha,
        gate_manifest=gate_manifest,
    )
    assert completed is not None
    assert completed.block_index == 0
