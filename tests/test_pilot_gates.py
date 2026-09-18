from __future__ import annotations

import pytest

from minimax_h3_keyless.pilot import PilotStepReport
from minimax_h3_keyless.pilot_campaign import (
    GATE_SCHEMA,
    PilotAggregateMetrics,
    PilotCaseMetrics,
    PilotTrainingEvent,
)
from minimax_h3_keyless.pilot_gates import (
    StageABlockGateResult,
    evaluate_stage_a_block_gate,
    evaluate_stage_a_campaign_gate,
    stage_a_policy_from_gate_manifest,
)


def _gate_manifest(**updates):
    thresholds = {
        "stage_a_min_case_total_improvement_fraction": 1.0,
        "stage_a_min_mean_total_relative_improvement": 0.1,
        "stage_a_max_mean_attention_normalized_mse": 0.30,
        "stage_a_max_mean_block_normalized_mse": 0.20,
        "stage_a_max_worst_attention_normalized_mse": 0.40,
        "stage_a_max_worst_block_normalized_mse": 0.30,
        "stage_a_min_mean_attention_cosine": 0.90,
        "stage_a_min_mean_block_cosine": 0.92,
        "stage_a_minimum_attention_cosine": 0.85,
        "stage_a_minimum_block_cosine": 0.88,
        "stage_a_max_modality_nmse_ratio_to_best_baseline": 0.9,
        "stage_a_max_gradient_l2_norm": 50.0,
    }
    thresholds.update(updates)
    return {
        "schema": GATE_SCHEMA,
        "thresholds": thresholds,
        "calibration_evidence": {"fixed_suite_sha256": "a" * 64},
    }


def _report(
    *,
    totals=(0.20, 0.24),
    attention=(0.12, 0.16),
    block=(0.08, 0.10),
    attention_cos=(0.96, 0.94),
    block_cos=(0.97, 0.95),
) -> PilotAggregateMetrics:
    modalities = ("video", "audio-video")
    cases = tuple(
        PilotCaseMetrics(
            case_id=f"case-{i}",
            modality_label=modalities[i],
            sigma=(0.2, 0.8)[i],
            rows=8 + i,
            total=float(totals[i]),
            attention_normalized_mse=float(attention[i]),
            block_normalized_mse=float(block[i]),
            attention_cosine=float(attention_cos[i]),
            block_cosine=float(block_cos[i]),
        )
        for i in range(2)
    )
    by_modality = {
        row.modality_label: {
            "case_count": 1,
            "mean_attention_normalized_mse": row.attention_normalized_mse,
            "mean_block_normalized_mse": row.block_normalized_mse,
            "mean_attention_cosine": row.attention_cosine,
            "mean_block_cosine": row.block_cosine,
        }
        for row in cases
    }
    return PilotAggregateMetrics(
        case_count=2,
        mean_total=sum(totals) / 2,
        mean_attention_normalized_mse=sum(attention) / 2,
        mean_block_normalized_mse=sum(block) / 2,
        mean_attention_cosine=sum(attention_cos) / 2,
        mean_block_cosine=sum(block_cos) / 2,
        worst_attention_normalized_mse=max(attention),
        worst_block_normalized_mse=max(block),
        minimum_attention_cosine=min(attention_cos),
        minimum_block_cosine=min(block_cos),
        by_modality=by_modality,
        cases=cases,
    )


def _events(gradient=4.0):
    return (
        PilotTrainingEvent(
            stage="route",
            epoch=0,
            case_id="case-train",
            report=PilotStepReport(
                total=0.5,
                attention_normalized_mse=0.3,
                block_normalized_mse=0.2,
                attention_cosine=0.9,
                block_cosine=0.91,
                trainable_parameters=10,
                gradient_l2_norm=gradient,
            ),
        ),
    )


def _passing_inputs():
    candidate = _report()
    identity = _report(
        totals=(0.45, 0.50),
        attention=(0.30, 0.34),
        block=(0.20, 0.23),
        attention_cos=(0.78, 0.76),
        block_cos=(0.82, 0.80),
    )
    least_squares = _report(
        totals=(0.35, 0.39),
        attention=(0.22, 0.25),
        block=(0.15, 0.17),
        attention_cos=(0.84, 0.82),
        block_cos=(0.87, 0.85),
    )
    return candidate, identity, least_squares


def test_stage_a_policy_has_no_code_defaults_and_requires_all_manifest_thresholds() -> None:
    policy = stage_a_policy_from_gate_manifest(_gate_manifest())
    assert policy.minimum_case_total_improvement_fraction == 1.0
    assert policy.maximum_gradient_l2_norm == 50.0

    manifest = _gate_manifest()
    manifest["thresholds"].pop("stage_a_max_gradient_l2_norm")
    with pytest.raises(ValueError, match="missing Stage-A thresholds"):
        stage_a_policy_from_gate_manifest(manifest)

    with pytest.raises(ValueError, match="within"):
        stage_a_policy_from_gate_manifest(
            _gate_manifest(stage_a_min_case_total_improvement_fraction=1.1)
        )


def test_stage_a_block_gate_passes_only_against_same_holdout_and_both_initializations() -> None:
    candidate, identity, least_squares = _passing_inputs()
    result = evaluate_stage_a_block_gate(
        block_index=25,
        candidate=candidate,
        identity_baseline=identity,
        least_squares_baseline=least_squares,
        training_events=_events(),
        policy=stage_a_policy_from_gate_manifest(_gate_manifest()),
    )
    assert result.passed is True
    assert result.failures == ()
    assert result.case_fraction_improved_over_both == 1.0
    assert result.mean_total_relative_improvement_vs_identity > 0.1
    assert result.mean_total_relative_improvement_vs_least_squares > 0.1
    assert result.maximum_gradient_l2_norm == 4.0


def test_stage_a_gate_reports_baseline_improvement_modality_and_gradient_failures() -> None:
    candidate, identity, least_squares = _passing_inputs()
    # Degrade one held-out modality and one case enough to violate several independent gates.
    candidate = _report(
        totals=(0.36, 0.50),
        attention=(0.20, 0.50),
        block=(0.12, 0.35),
        attention_cos=(0.92, 0.70),
        block_cos=(0.94, 0.72),
    )
    result = evaluate_stage_a_block_gate(
        block_index=0,
        candidate=candidate,
        identity_baseline=identity,
        least_squares_baseline=least_squares,
        training_events=_events(gradient=60.0),
        policy=stage_a_policy_from_gate_manifest(_gate_manifest()),
    )
    assert result.passed is False
    joined = "\n".join(result.failures)
    assert "case improvement fraction" in joined
    assert "mean total relative improvement vs least-squares" in joined
    assert "modality 'audio-video'" in joined
    assert "gradient L2 norm" in joined


def test_stage_a_gate_rejects_mismatched_holdout_identity_instead_of_comparing_it() -> None:
    candidate, identity, least_squares = _passing_inputs()
    bad_cases = list(least_squares.cases)
    bad_cases[1] = PilotCaseMetrics(
        **{**bad_cases[1].__dict__, "case_id": "different-case"}
    )
    bad_ls = PilotAggregateMetrics(
        **{**least_squares.__dict__, "cases": tuple(bad_cases)}
    )
    with pytest.raises(ValueError, match="exact same held-out case IDs"):
        evaluate_stage_a_block_gate(
            block_index=49,
            candidate=candidate,
            identity_baseline=identity,
            least_squares_baseline=bad_ls,
            training_events=_events(),
            policy=stage_a_policy_from_gate_manifest(_gate_manifest()),
        )


def test_stage_a_gate_rejects_missing_or_nonfinite_training_evidence() -> None:
    candidate, identity, least_squares = _passing_inputs()
    policy = stage_a_policy_from_gate_manifest(_gate_manifest())
    with pytest.raises(ValueError, match="recorded training events"):
        evaluate_stage_a_block_gate(
            block_index=0,
            candidate=candidate,
            identity_baseline=identity,
            least_squares_baseline=least_squares,
            training_events=(),
            policy=policy,
        )
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_stage_a_block_gate(
            block_index=0,
            candidate=candidate,
            identity_baseline=identity,
            least_squares_baseline=least_squares,
            training_events=_events(gradient=float("nan")),
            policy=policy,
        )


def _block_result(block_index: int, passed: bool) -> StageABlockGateResult:
    return StageABlockGateResult(
        block_index=block_index,
        passed=passed,
        failures=() if passed else ("synthetic failure",),
        case_fraction_improved_over_both=1.0,
        mean_total_relative_improvement_vs_identity=0.5,
        mean_total_relative_improvement_vs_least_squares=0.4,
        maximum_modality_attention_nmse_ratio=0.7,
        maximum_modality_block_nmse_ratio=0.7,
        maximum_gradient_l2_norm=2.0,
    )


def test_stage_a_campaign_requires_all_three_prescribed_depth_pilots_to_pass() -> None:
    results = {i: _block_result(i, True) for i in (0, 25, 49)}
    assert evaluate_stage_a_campaign_gate(results).passed is True

    failed = dict(results)
    failed[25] = _block_result(25, False)
    report = evaluate_stage_a_campaign_gate(failed)
    assert report.passed is False
    assert report.failures == ("pilot block 25 failed 1 predeclared gate(s)",)

    with pytest.raises(ValueError, match="requires exactly"):
        evaluate_stage_a_campaign_gate({0: results[0], 25: results[25]})
