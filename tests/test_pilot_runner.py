from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from minimax_h3_keyless.activation_capture import CapturedPilotCase
from minimax_h3_keyless.attention import KeylessAttentionTrain
from minimax_h3_keyless.initialization import initialize_training_attention_from_native
from minimax_h3_keyless.ops import normalized_positioned, torch_sdpa_attention
from minimax_h3_keyless.pilot import PilotCase, PilotLossWeights, set_pilot_block_stage
from minimax_h3_keyless.pilot_artifacts import StageAArtifactRequest
from minimax_h3_keyless.pilot_campaign import GATE_SCHEMA, PilotAggregateMetrics, PilotCaseMetrics
from minimax_h3_keyless.pilot_capture_set import StageACaptureSet
from minimax_h3_keyless.pilot_runner import (
    StageAInitializationEvaluation,
    StageATrainStage,
    run_stage_a_block_pilot,
    select_stage_a_initialization,
    validate_stage_a_train_plan,
)
from minimax_h3_keyless.route_fit import RouteActivationFitDiagnostics


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
        "calibration_evidence": {"fixed_suite_sha256": "a" * 64},
    }


def _capture_set(teacher: TinyBlock):
    torch.manual_seed(501)
    train = []
    holdout = []
    for split, target in (("train", train), ("holdout", holdout)):
        for n, sigma in enumerate((0.2, 0.8)):
            x = torch.randn(5, 4)
            segments = [[0, 5, "video"]] if n == 0 else [[0, 2, "audio"], [2, 5, "video"]]
            case = PilotCase(
                x=x,
                t_emb=torch.zeros(1, 1),
                mod_segments=((0, 5, 0),),
                rope_freqs=None,
                transformer_options={},
                case_id=f"{split}-{n}::sigma={sigma}",
                sigma=sigma,
                modality_label="video" if n == 0 else "audio-video",
                context={
                    "stage_a_split": split,
                    "minimax_h3_keyless_live_capture_v1": {
                        "layout": {"seq_len": 5, "segments": segments}
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


def _metric(attn, block):
    case = PilotCaseMetrics(
        case_id="x",
        modality_label="video",
        sigma=0.5,
        rows=1,
        total=attn + block,
        attention_normalized_mse=attn,
        block_normalized_mse=block,
        attention_cosine=0.9,
        block_cosine=0.9,
    )
    return PilotAggregateMetrics(
        case_count=1,
        mean_total=case.total,
        mean_attention_normalized_mse=attn,
        mean_block_normalized_mse=block,
        mean_attention_cosine=0.9,
        mean_block_cosine=0.9,
        worst_attention_normalized_mse=attn,
        worst_block_normalized_mse=block,
        minimum_attention_cosine=0.9,
        minimum_block_cosine=0.9,
        by_modality={
            "video": {
                "case_count": 1,
                "mean_attention_normalized_mse": attn,
                "mean_block_normalized_mse": block,
                "mean_attention_cosine": 0.9,
                "mean_block_cosine": 0.9,
            }
        },
        cases=(case,),
    )


def _diagnostics(lambda_relative: float) -> RouteActivationFitDiagnostics:
    return RouteActivationFitDiagnostics(
        rows=10,
        lambda_relative=lambda_relative,
        lambda_actual=(lambda_relative, lambda_relative),
        smallest_singular_value=(1.0, 1.5),
        largest_singular_value=(2.0, 3.0),
        numerical_rank=(2, 2),
        full_rank_condition_number=(2.0, 2.0),
    )


def test_stage_plan_enforces_monotonic_escalation_and_value_before_norm_out() -> None:
    plan = validate_stage_a_train_plan(
        (
            StageATrainStage("route", 1, 1e-3),
            StageATrainStage("value", 1, 1e-4),
            StageATrainStage("norm_out", 1, 1e-5),
        )
    )
    assert [row.stage for row in plan] == ["route", "value", "norm_out"]
    with pytest.raises(ValueError, match="start with route"):
        validate_stage_a_train_plan((StageATrainStage("query", 1, 1e-3),))
    with pytest.raises(ValueError, match="repeats"):
        validate_stage_a_train_plan(
            (StageATrainStage("route", 1, 1e-3), StageATrainStage("route", 1, 1e-3))
        )
    with pytest.raises(ValueError, match="requires value"):
        validate_stage_a_train_plan(
            (StageATrainStage("route", 1, 1e-3), StageATrainStage("norm_out", 1, 1e-4))
        )


def test_initialization_selection_prefers_holdout_attention_error_then_block_error() -> None:
    evaluations = (
        StageAInitializationEvaluation("identity", 0.0, _metric(0.3, 0.1)),
        StageAInitializationEvaluation("least_squares", 0.0, _metric(0.2, 0.2), _diagnostics(0.0)),
        StageAInitializationEvaluation(
            "least_squares", 1e-4, _metric(0.2, 0.15), _diagnostics(1e-4)
        ),
        StageAInitializationEvaluation(
            "least_squares", 1e-2, _metric(0.4, 0.01), _diagnostics(1e-2)
        ),
    )
    selected = select_stage_a_initialization(evaluations)
    assert selected.route_mode == "least_squares"
    assert selected.lambda_relative == 1e-4


def test_initialization_selection_rejects_duplicate_ls_lambda() -> None:
    evaluations = (
        StageAInitializationEvaluation("identity", 0.0, _metric(0.3, 0.1)),
        StageAInitializationEvaluation("least_squares", 0.0, _metric(0.2, 0.2), _diagnostics(0.0)),
        StageAInitializationEvaluation(
            "least_squares", 1e-4, _metric(0.2, 0.15), _diagnostics(1e-4)
        ),
        StageAInitializationEvaluation(
            "least_squares", 1e-4, _metric(0.1, 0.1), _diagnostics(1e-4)
        ),
        StageAInitializationEvaluation(
            "least_squares", 1e-2, _metric(0.4, 0.01), _diagnostics(1e-2)
        ),
    )
    with pytest.raises(ValueError, match="LS lambdas"):
        select_stage_a_initialization(evaluations)


def test_initialization_selection_rejects_ls_without_activation_diagnostics() -> None:
    evaluations = (
        StageAInitializationEvaluation("identity", 0.0, _metric(0.3, 0.1)),
        StageAInitializationEvaluation("least_squares", 0.0, _metric(0.2, 0.2)),
        StageAInitializationEvaluation(
            "least_squares", 1e-4, _metric(0.2, 0.15), _diagnostics(1e-4)
        ),
        StageAInitializationEvaluation(
            "least_squares", 1e-2, _metric(0.4, 0.01), _diagnostics(1e-2)
        ),
    )
    with pytest.raises(ValueError, match="captured-activation diagnostics"):
        select_stage_a_initialization(evaluations)


def test_block_runner_replays_grid_trains_and_persists_immutable_evidence(tmp_path: Path) -> None:
    torch.manual_seed(502)
    teacher = TinyBlock()
    corpus = _capture_set(teacher)
    request = StageAArtifactRequest(str(tmp_path), "tiny-run", "deadbeef")
    result = run_stage_a_block_pilot(
        teacher,
        corpus,
        block_index=0,
        device="cpu",
        gate_manifest=_gate_manifest(),
        train_plan=(StageATrainStage("route", 1, 2e-3, max_grad_norm=100.0),),
        loss_weights=PilotLossWeights(attention_output=1.0, block_output=1.0),
        student_builder=_builder,
        artifact_request=request,
    )
    assert len(result.replay_reports) == 4
    assert all(report.attention_input_max_abs_error == 0.0 for report in result.replay_reports)
    assert len(result.initialization_evaluations) == 4
    assert all(len(row.attention_diagnostics) == 2 for row in result.initialization_evaluations)
    ls_rows = [
        row for row in result.initialization_evaluations if row.route_mode == "least_squares"
    ]
    assert {row.lambda_relative for row in ls_rows} == {0.0, 1e-4, 1e-2}
    assert all(row.route_fit_diagnostics is not None for row in ls_rows)
    assert all(row.route_fit_diagnostics.rows == 10 for row in ls_rows)
    assert len(result.training_events) == 2
    assert result.candidate.case_count == 2
    assert len(result.candidate_attention_diagnostics) == 2
    assert result.candidate_attention_diagnostics[0].modality_kinds == ("video",)
    assert result.candidate_attention_diagnostics[1].modality_kinds == ("audio", "video")
    assert result.gate.block_index == 0
    assert all(torch.isfinite(torch.tensor(event.report.total)) for event in result.training_events)
    assert result.artifact is not None
    assert Path(result.artifact.checkpoint_path).exists()
    assert Path(result.artifact.result_path).exists()
    with pytest.raises(FileExistsError, match="immutable"):
        run_stage_a_block_pilot(
            teacher,
            corpus,
            block_index=0,
            device="cpu",
            gate_manifest=_gate_manifest(),
            train_plan=(StageATrainStage("route", 1, 2e-3),),
            student_builder=_builder,
            artifact_request=request,
        )
