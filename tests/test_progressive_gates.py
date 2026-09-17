from __future__ import annotations

import pytest

from minimax_h3_keyless.pilot import PilotStepReport
from minimax_h3_keyless.pilot_campaign import (
    GATE_SCHEMA,
    PilotAggregateMetrics,
    PilotCaseMetrics,
    PilotTrainingEvent,
)
from minimax_h3_keyless.pilot_gates import stage_a_policy_from_gate_manifest
from minimax_h3_keyless.progressive_gates import (
    evaluate_progressive_block_gate,
    progressive_execution_policy_from_gate_manifest,
)


def _manifest():
    return {
        "schema": GATE_SCHEMA,
        "thresholds": {
            "stage_a_min_case_total_improvement_fraction": 1.0,
            "stage_a_min_mean_total_relative_improvement": 0.1,
            "stage_a_max_mean_attention_normalized_mse": 0.3,
            "stage_a_max_mean_block_normalized_mse": 0.2,
            "stage_a_max_worst_attention_normalized_mse": 0.4,
            "stage_a_max_worst_block_normalized_mse": 0.3,
            "stage_a_min_mean_attention_cosine": 0.9,
            "stage_a_min_mean_block_cosine": 0.92,
            "stage_a_minimum_attention_cosine": 0.85,
            "stage_a_minimum_block_cosine": 0.88,
            "stage_a_max_modality_nmse_ratio_to_best_baseline": 0.9,
            "stage_a_max_gradient_l2_norm": 50.0,
            "progressive_fold_atol": 0.002,
            "progressive_fold_rtol": 0.003,
        },
        "calibration_evidence": {"fixed_suite_sha256": "a" * 64},
    }


def _report(total: float, attn: float, block: float, cosine: float) -> PilotAggregateMetrics:
    case = PilotCaseMetrics(
        case_id="holdout::sigma=0.5",
        modality_label="video",
        sigma=0.5,
        rows=8,
        total=total,
        attention_normalized_mse=attn,
        block_normalized_mse=block,
        attention_cosine=cosine,
        block_cosine=cosine,
    )
    return PilotAggregateMetrics(
        case_count=1,
        mean_total=total,
        mean_attention_normalized_mse=attn,
        mean_block_normalized_mse=block,
        mean_attention_cosine=cosine,
        mean_block_cosine=cosine,
        worst_attention_normalized_mse=attn,
        worst_block_normalized_mse=block,
        minimum_attention_cosine=cosine,
        minimum_block_cosine=cosine,
        by_modality={
            "video": {
                "case_count": 1,
                "mean_attention_normalized_mse": attn,
                "mean_block_normalized_mse": block,
                "mean_attention_cosine": cosine,
                "mean_block_cosine": cosine,
            }
        },
        cases=(case,),
    )


def _events():
    return (
        PilotTrainingEvent(
            stage="route",
            epoch=0,
            case_id="train::sigma=0.5",
            report=PilotStepReport(
                total=0.2,
                attention_normalized_mse=0.1,
                block_normalized_mse=0.1,
                attention_cosine=0.95,
                block_cosine=0.95,
                trainable_parameters=10,
                gradient_l2_norm=2.0,
            ),
        ),
    )


def test_progressive_execution_policy_is_loaded_from_hash_bound_gate_manifest() -> None:
    policy = progressive_execution_policy_from_gate_manifest(_manifest())
    assert policy.fold_atol == 0.002
    assert policy.fold_rtol == 0.003


def test_progressive_execution_policy_requires_predeclared_nonnegative_tolerances() -> None:
    manifest = _manifest()
    del manifest["thresholds"]["progressive_fold_rtol"]
    with pytest.raises(ValueError, match="missing progressive execution thresholds"):
        progressive_execution_policy_from_gate_manifest(manifest)

    manifest = _manifest()
    manifest["thresholds"]["progressive_fold_atol"] = -1.0
    with pytest.raises(ValueError, match="finite non-negative"):
        progressive_execution_policy_from_gate_manifest(manifest)


def test_progressive_gate_reuses_pilot_policy_for_nonpilot_depth() -> None:
    result = evaluate_progressive_block_gate(
        block_index=37,
        candidate=_report(0.18, 0.10, 0.08, 0.96),
        identity_baseline=_report(0.5, 0.30, 0.20, 0.8),
        least_squares_baseline=_report(0.4, 0.22, 0.16, 0.85),
        training_events=_events(),
        policy=stage_a_policy_from_gate_manifest(_manifest()),
    )
    assert result.passed is True
    assert result.block_index == 37
    assert result.failures == ()


def test_progressive_gate_rejects_non_core_block() -> None:
    common = dict(
        candidate=_report(0.18, 0.10, 0.08, 0.96),
        identity_baseline=_report(0.5, 0.30, 0.20, 0.8),
        least_squares_baseline=_report(0.4, 0.22, 0.16, 0.85),
        training_events=_events(),
        policy=stage_a_policy_from_gate_manifest(_manifest()),
    )
    with pytest.raises(ValueError, match="within"):
        evaluate_progressive_block_gate(block_index=50, **common)
    with pytest.raises(ValueError, match="integer"):
        evaluate_progressive_block_gate(block_index=True, **common)
