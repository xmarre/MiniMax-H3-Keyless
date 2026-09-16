from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .pilot import PilotStepReport
from .pilot_campaign import (
    PILOT_BLOCKS,
    PilotAggregateMetrics,
    PilotCaseMetrics,
    PilotTrainingEvent,
    validate_pilot_gate_manifest,
)


_STAGE_A_THRESHOLD_KEYS = (
    "stage_a_min_case_total_improvement_fraction",
    "stage_a_min_mean_total_relative_improvement",
    "stage_a_max_mean_attention_normalized_mse",
    "stage_a_max_mean_block_normalized_mse",
    "stage_a_max_worst_attention_normalized_mse",
    "stage_a_max_worst_block_normalized_mse",
    "stage_a_min_mean_attention_cosine",
    "stage_a_min_mean_block_cosine",
    "stage_a_minimum_attention_cosine",
    "stage_a_minimum_block_cosine",
    "stage_a_max_modality_nmse_ratio_to_best_baseline",
    "stage_a_max_gradient_l2_norm",
)


@dataclass(frozen=True)
class StageAGatePolicy:
    """Predeclared numerical policy for the Stage-A block-pilot exit.

    No thresholds are supplied by code. The policy must be parsed from the fixed gate
    manifest produced from calibration evidence before the pilot campaign starts.
    """

    minimum_case_total_improvement_fraction: float
    minimum_mean_total_relative_improvement: float
    maximum_mean_attention_normalized_mse: float
    maximum_mean_block_normalized_mse: float
    maximum_worst_attention_normalized_mse: float
    maximum_worst_block_normalized_mse: float
    minimum_mean_attention_cosine: float
    minimum_mean_block_cosine: float
    minimum_attention_cosine: float
    minimum_block_cosine: float
    maximum_modality_nmse_ratio_to_best_baseline: float
    maximum_gradient_l2_norm: float


@dataclass(frozen=True)
class StageABlockGateResult:
    block_index: int
    passed: bool
    failures: tuple[str, ...]
    case_fraction_improved_over_both: float
    mean_total_relative_improvement_vs_identity: float
    mean_total_relative_improvement_vs_least_squares: float
    maximum_modality_attention_nmse_ratio: float
    maximum_modality_block_nmse_ratio: float
    maximum_gradient_l2_norm: float


@dataclass(frozen=True)
class StageACampaignGateResult:
    passed: bool
    failures: tuple[str, ...]
    block_results: Mapping[int, StageABlockGateResult]


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _bounded(name: str, value: float, low: float, high: float) -> float:
    value = _finite(name, value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be within [{low},{high}]")
    return value


def stage_a_policy_from_gate_manifest(manifest: Mapping[str, object]) -> StageAGatePolicy:
    """Parse the required Stage-A thresholds from a validated fixed gate manifest."""
    validate_pilot_gate_manifest(manifest)
    thresholds = manifest["thresholds"]
    assert isinstance(thresholds, dict)
    missing = [key for key in _STAGE_A_THRESHOLD_KEYS if key not in thresholds]
    if missing:
        raise ValueError(f"pilot gate manifest is missing Stage-A thresholds: {missing}")

    case_fraction = _bounded(
        "stage_a_min_case_total_improvement_fraction",
        thresholds["stage_a_min_case_total_improvement_fraction"],
        0.0,
        1.0,
    )
    mean_improvement = _bounded(
        "stage_a_min_mean_total_relative_improvement",
        thresholds["stage_a_min_mean_total_relative_improvement"],
        0.0,
        1.0,
    )
    max_mean_attn = _finite(
        "stage_a_max_mean_attention_normalized_mse",
        thresholds["stage_a_max_mean_attention_normalized_mse"],
    )
    max_mean_block = _finite(
        "stage_a_max_mean_block_normalized_mse",
        thresholds["stage_a_max_mean_block_normalized_mse"],
    )
    max_worst_attn = _finite(
        "stage_a_max_worst_attention_normalized_mse",
        thresholds["stage_a_max_worst_attention_normalized_mse"],
    )
    max_worst_block = _finite(
        "stage_a_max_worst_block_normalized_mse",
        thresholds["stage_a_max_worst_block_normalized_mse"],
    )
    for name, value in (
        ("stage_a_max_mean_attention_normalized_mse", max_mean_attn),
        ("stage_a_max_mean_block_normalized_mse", max_mean_block),
        ("stage_a_max_worst_attention_normalized_mse", max_worst_attn),
        ("stage_a_max_worst_block_normalized_mse", max_worst_block),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    min_mean_attn_cos = _bounded(
        "stage_a_min_mean_attention_cosine",
        thresholds["stage_a_min_mean_attention_cosine"],
        -1.0,
        1.0,
    )
    min_mean_block_cos = _bounded(
        "stage_a_min_mean_block_cosine",
        thresholds["stage_a_min_mean_block_cosine"],
        -1.0,
        1.0,
    )
    min_attn_cos = _bounded(
        "stage_a_minimum_attention_cosine",
        thresholds["stage_a_minimum_attention_cosine"],
        -1.0,
        1.0,
    )
    min_block_cos = _bounded(
        "stage_a_minimum_block_cosine",
        thresholds["stage_a_minimum_block_cosine"],
        -1.0,
        1.0,
    )
    modality_ratio = _finite(
        "stage_a_max_modality_nmse_ratio_to_best_baseline",
        thresholds["stage_a_max_modality_nmse_ratio_to_best_baseline"],
    )
    max_grad = _finite(
        "stage_a_max_gradient_l2_norm",
        thresholds["stage_a_max_gradient_l2_norm"],
    )
    if modality_ratio <= 0:
        raise ValueError("stage_a_max_modality_nmse_ratio_to_best_baseline must be positive")
    if max_grad <= 0:
        raise ValueError("stage_a_max_gradient_l2_norm must be positive")

    return StageAGatePolicy(
        minimum_case_total_improvement_fraction=case_fraction,
        minimum_mean_total_relative_improvement=mean_improvement,
        maximum_mean_attention_normalized_mse=max_mean_attn,
        maximum_mean_block_normalized_mse=max_mean_block,
        maximum_worst_attention_normalized_mse=max_worst_attn,
        maximum_worst_block_normalized_mse=max_worst_block,
        minimum_mean_attention_cosine=min_mean_attn_cos,
        minimum_mean_block_cosine=min_mean_block_cos,
        minimum_attention_cosine=min_attn_cos,
        minimum_block_cosine=min_block_cos,
        maximum_modality_nmse_ratio_to_best_baseline=modality_ratio,
        maximum_gradient_l2_norm=max_grad,
    )


def _case_map(report: PilotAggregateMetrics, label: str) -> dict[str, PilotCaseMetrics]:
    if report.case_count != len(report.cases) or report.case_count <= 0:
        raise ValueError(f"{label} report has inconsistent/empty case accounting")
    out: dict[str, PilotCaseMetrics] = {}
    for case in report.cases:
        if not case.case_id or case.case_id in out:
            raise ValueError(f"{label} report has missing/duplicate case_id {case.case_id!r}")
        values = (
            case.total,
            case.attention_normalized_mse,
            case.block_normalized_mse,
            case.attention_cosine,
            case.block_cosine,
        )
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError(f"{label} report has non-finite metrics for {case.case_id!r}")
        if case.total < 0 or case.attention_normalized_mse < 0 or case.block_normalized_mse < 0:
            raise ValueError(f"{label} report has negative error metric for {case.case_id!r}")
        out[case.case_id] = case
    return out


def _validate_same_holdout(
    candidate: PilotAggregateMetrics,
    identity: PilotAggregateMetrics,
    least_squares: PilotAggregateMetrics,
) -> tuple[dict[str, PilotCaseMetrics], dict[str, PilotCaseMetrics], dict[str, PilotCaseMetrics]]:
    maps = (
        _case_map(candidate, "candidate"),
        _case_map(identity, "identity baseline"),
        _case_map(least_squares, "least-squares baseline"),
    )
    ids = [set(m) for m in maps]
    if ids[0] != ids[1] or ids[0] != ids[2]:
        raise ValueError("Stage-A candidate and baselines must use the exact same held-out case IDs")
    for case_id in ids[0]:
        modalities = {maps[i][case_id].modality_label for i in range(3)}
        if len(modalities) != 1:
            raise ValueError(f"held-out modality label differs for case {case_id!r}")
    return maps


def _relative_improvement(candidate: float, baseline: float) -> float:
    candidate = float(candidate)
    baseline = float(baseline)
    if baseline < 0 or candidate < 0:
        raise ValueError("relative improvement requires non-negative metrics")
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else float("-inf")
    return (baseline - candidate) / baseline


def _ratio(candidate: float, baseline: float) -> float:
    candidate = float(candidate)
    baseline = float(baseline)
    if baseline < 0 or candidate < 0:
        raise ValueError("NMSE ratios require non-negative values")
    if baseline == 0.0:
        return 1.0 if candidate == 0.0 else float("inf")
    return candidate / baseline


def _modality_metric(report: PilotAggregateMetrics, modality: str, key: str, label: str) -> float:
    row = report.by_modality.get(modality)
    if row is None or key not in row:
        raise ValueError(f"{label} report is missing modality metric {modality!r}/{key!r}")
    return _finite(f"{label} {modality} {key}", float(row[key]))


def _gradient_max(events: Sequence[PilotTrainingEvent]) -> float:
    if not events:
        raise ValueError("Stage-A exit requires recorded training events")
    maximum = 0.0
    for event in events:
        report: PilotStepReport = event.report
        values = (
            report.total,
            report.attention_normalized_mse,
            report.block_normalized_mse,
            report.attention_cosine,
            report.block_cosine,
            report.gradient_l2_norm,
        )
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError(
                f"Stage-A training event has non-finite metric at stage={event.stage!r}, "
                f"epoch={event.epoch}, case={event.case_id!r}"
            )
        if report.gradient_l2_norm < 0:
            raise ValueError("Stage-A gradient L2 norm cannot be negative")
        maximum = max(maximum, float(report.gradient_l2_norm))
    return maximum


def evaluate_stage_a_block_gate(
    *,
    block_index: int,
    candidate: PilotAggregateMetrics,
    identity_baseline: PilotAggregateMetrics,
    least_squares_baseline: PilotAggregateMetrics,
    training_events: Sequence[PilotTrainingEvent],
    policy: StageAGatePolicy,
) -> StageABlockGateResult:
    """Apply predeclared Stage-A gates without changing thresholds after results exist."""
    if block_index not in PILOT_BLOCKS:
        raise ValueError(f"Stage-A pilot block must be one of {PILOT_BLOCKS}")
    candidate_cases, identity_cases, ls_cases = _validate_same_holdout(
        candidate, identity_baseline, least_squares_baseline
    )

    failures: list[str] = []
    improved = 0
    for case_id, row in candidate_cases.items():
        if row.total < identity_cases[case_id].total and row.total < ls_cases[case_id].total:
            improved += 1
    case_fraction = improved / len(candidate_cases)
    if case_fraction < policy.minimum_case_total_improvement_fraction:
        failures.append(
            "held-out case improvement fraction below predeclared minimum: "
            f"{case_fraction:.6g} < {policy.minimum_case_total_improvement_fraction:.6g}"
        )

    identity_improvement = _relative_improvement(candidate.mean_total, identity_baseline.mean_total)
    ls_improvement = _relative_improvement(candidate.mean_total, least_squares_baseline.mean_total)
    if identity_improvement < policy.minimum_mean_total_relative_improvement:
        failures.append(
            "mean total relative improvement vs identity below predeclared minimum: "
            f"{identity_improvement:.6g} < {policy.minimum_mean_total_relative_improvement:.6g}"
        )
    if ls_improvement < policy.minimum_mean_total_relative_improvement:
        failures.append(
            "mean total relative improvement vs least-squares below predeclared minimum: "
            f"{ls_improvement:.6g} < {policy.minimum_mean_total_relative_improvement:.6g}"
        )

    absolute_checks = (
        (
            "mean attention normalized MSE",
            candidate.mean_attention_normalized_mse,
            policy.maximum_mean_attention_normalized_mse,
            "max",
        ),
        (
            "mean block normalized MSE",
            candidate.mean_block_normalized_mse,
            policy.maximum_mean_block_normalized_mse,
            "max",
        ),
        (
            "worst attention normalized MSE",
            candidate.worst_attention_normalized_mse,
            policy.maximum_worst_attention_normalized_mse,
            "max",
        ),
        (
            "worst block normalized MSE",
            candidate.worst_block_normalized_mse,
            policy.maximum_worst_block_normalized_mse,
            "max",
        ),
        (
            "mean attention cosine",
            candidate.mean_attention_cosine,
            policy.minimum_mean_attention_cosine,
            "min",
        ),
        (
            "mean block cosine",
            candidate.mean_block_cosine,
            policy.minimum_mean_block_cosine,
            "min",
        ),
        (
            "minimum attention cosine",
            candidate.minimum_attention_cosine,
            policy.minimum_attention_cosine,
            "min",
        ),
        (
            "minimum block cosine",
            candidate.minimum_block_cosine,
            policy.minimum_block_cosine,
            "min",
        ),
    )
    for name, value, threshold, direction in absolute_checks:
        value = _finite(name, value)
        if direction == "max" and value > threshold:
            failures.append(f"{name} exceeds predeclared maximum: {value:.6g} > {threshold:.6g}")
        if direction == "min" and value < threshold:
            failures.append(f"{name} below predeclared minimum: {value:.6g} < {threshold:.6g}")

    candidate_modalities = set(candidate.by_modality)
    if candidate_modalities != set(identity_baseline.by_modality) or candidate_modalities != set(
        least_squares_baseline.by_modality
    ):
        raise ValueError("Stage-A candidate and baselines must expose the same modality groups")
    max_attn_ratio = 0.0
    max_block_ratio = 0.0
    for modality in sorted(candidate_modalities):
        candidate_attn = _modality_metric(
            candidate, modality, "mean_attention_normalized_mse", "candidate"
        )
        candidate_block = _modality_metric(
            candidate, modality, "mean_block_normalized_mse", "candidate"
        )
        best_attn = min(
            _modality_metric(
                identity_baseline, modality, "mean_attention_normalized_mse", "identity baseline"
            ),
            _modality_metric(
                least_squares_baseline,
                modality,
                "mean_attention_normalized_mse",
                "least-squares baseline",
            ),
        )
        best_block = min(
            _modality_metric(
                identity_baseline, modality, "mean_block_normalized_mse", "identity baseline"
            ),
            _modality_metric(
                least_squares_baseline,
                modality,
                "mean_block_normalized_mse",
                "least-squares baseline",
            ),
        )
        attn_ratio = _ratio(candidate_attn, best_attn)
        block_ratio = _ratio(candidate_block, best_block)
        max_attn_ratio = max(max_attn_ratio, attn_ratio)
        max_block_ratio = max(max_block_ratio, block_ratio)
        if attn_ratio > policy.maximum_modality_nmse_ratio_to_best_baseline:
            failures.append(
                f"modality {modality!r} attention NMSE ratio indicates collapse: "
                f"{attn_ratio:.6g} > {policy.maximum_modality_nmse_ratio_to_best_baseline:.6g}"
            )
        if block_ratio > policy.maximum_modality_nmse_ratio_to_best_baseline:
            failures.append(
                f"modality {modality!r} block NMSE ratio indicates collapse: "
                f"{block_ratio:.6g} > {policy.maximum_modality_nmse_ratio_to_best_baseline:.6g}"
            )

    max_gradient = _gradient_max(training_events)
    if max_gradient > policy.maximum_gradient_l2_norm:
        failures.append(
            "gradient L2 norm exceeds predeclared stability maximum: "
            f"{max_gradient:.6g} > {policy.maximum_gradient_l2_norm:.6g}"
        )

    return StageABlockGateResult(
        block_index=block_index,
        passed=not failures,
        failures=tuple(failures),
        case_fraction_improved_over_both=case_fraction,
        mean_total_relative_improvement_vs_identity=identity_improvement,
        mean_total_relative_improvement_vs_least_squares=ls_improvement,
        maximum_modality_attention_nmse_ratio=max_attn_ratio,
        maximum_modality_block_nmse_ratio=max_block_ratio,
        maximum_gradient_l2_norm=max_gradient,
    )


def evaluate_stage_a_campaign_gate(
    block_results: Mapping[int, StageABlockGateResult],
) -> StageACampaignGateResult:
    """Require all three prescribed depth pilots to pass before a core50 sweep."""
    required = set(PILOT_BLOCKS)
    actual = set(block_results)
    if actual != required:
        raise ValueError(
            f"Stage-A campaign requires exactly pilot blocks {PILOT_BLOCKS}; "
            f"missing={sorted(required - actual)}, extra={sorted(actual - required)}"
        )
    failures: list[str] = []
    for block_index in PILOT_BLOCKS:
        result = block_results[block_index]
        if result.block_index != block_index:
            raise ValueError(
                f"Stage-A result map key {block_index} disagrees with result block {result.block_index}"
            )
        if not result.passed:
            failures.append(
                f"pilot block {block_index} failed {len(result.failures)} predeclared gate(s)"
            )
    return StageACampaignGateResult(
        passed=not failures,
        failures=tuple(failures),
        block_results=dict(block_results),
    )
