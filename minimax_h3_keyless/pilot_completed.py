from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .checkpoint import sha256_file
from .pilot import PilotStepReport
from .pilot_artifacts import (
    STAGE_A_RESULT_SCHEMA,
    StageAArtifactReceipt,
    StageAArtifactRequest,
)
from .pilot_attention_diagnostics import PilotAttentionDiagnostic
from .pilot_campaign import (
    RESUME_SCHEMA,
    PilotAggregateMetrics,
    PilotCaseMetrics,
    PilotRunIdentity,
    PilotTrainingEvent,
    _require_sha256,
    canonical_json_sha256,
)
from .pilot_gates import (
    StageABlockGateResult,
    evaluate_stage_a_block_gate,
    stage_a_policy_from_gate_manifest,
)
from .pilot_replay import CapturedReplayReport
from .pilot_runner import StageAInitializationEvaluation, select_stage_a_initialization
from .route_fit import RouteActivationFitDiagnostics


@dataclass(frozen=True)
class CompletedStageABlockEvidence:
    block_index: int
    identity: PilotRunIdentity
    stage: str
    step: int
    gate: StageABlockGateResult
    artifact: StageAArtifactReceipt


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys.difference(value))
    unknown = sorted(set(value).difference(keys))
    if missing or unknown:
        raise RuntimeError(f"{label} key mismatch: missing={missing}, unknown={unknown}")


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} must be numeric")
    out = float(value)
    if not math.isfinite(out):
        raise RuntimeError(f"{name} must be finite")
    return out


def _attention_diagnostic(row: Any) -> PilotAttentionDiagnostic:
    row = _object(row, "Stage-A attention diagnostic")
    keys = {
        "case_id",
        "sigma",
        "modality_label",
        "sampled_query_rows",
        "head_chunk_size",
        "key_chunk_size",
        "modality_kinds",
        "mean_centered_logit_nrmse",
        "mean_teacher_to_student_softmax_kl",
        "pre_out_normalized_rmse",
        "pre_out_cosine",
        "post_out_normalized_rmse",
        "post_out_cosine",
        "teacher_modality_mass",
        "student_modality_mass",
    }
    _exact_keys(row, keys, "Stage-A attention diagnostic")
    case_id = row["case_id"]
    if not isinstance(case_id, str) or not case_id:
        raise RuntimeError("Stage-A attention diagnostic case_id must be non-empty")
    sigma = row["sigma"]
    if sigma is not None:
        sigma = _finite("Stage-A attention diagnostic sigma", sigma)
        if not 0.0 <= sigma <= 1.0:
            raise RuntimeError("Stage-A attention diagnostic sigma must be within [0,1]")
    modality_label = row["modality_label"]
    if modality_label is not None and (
        not isinstance(modality_label, str) or not modality_label
    ):
        raise RuntimeError(
            "Stage-A attention diagnostic modality_label must be null or a non-empty string"
        )

    sampled = row["sampled_query_rows"]
    if not isinstance(sampled, list) or not sampled:
        raise RuntimeError("Stage-A attention diagnostic sampled_query_rows must be non-empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in sampled):
        raise RuntimeError("Stage-A attention diagnostic query rows must be non-negative integers")
    if sampled != sorted(set(sampled)):
        raise RuntimeError("Stage-A attention diagnostic query rows must be sorted and unique")

    chunks: dict[str, int] = {}
    for name in ("head_chunk_size", "key_chunk_size"):
        value = row[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"Stage-A attention diagnostic {name} must be a positive integer")
        chunks[name] = value

    modality_kinds = row["modality_kinds"]
    if (
        not isinstance(modality_kinds, list)
        or not modality_kinds
        or not all(isinstance(value, str) and value for value in modality_kinds)
        or len(set(modality_kinds)) != len(modality_kinds)
    ):
        raise RuntimeError(
            "Stage-A attention diagnostic modality_kinds must be unique non-empty strings"
        )

    centered = _finite(
        "Stage-A attention diagnostic centered-logit NRMSE",
        row["mean_centered_logit_nrmse"],
    )
    kl = _finite(
        "Stage-A attention diagnostic teacher-to-student KL",
        row["mean_teacher_to_student_softmax_kl"],
    )
    pre_nrmse = _finite(
        "Stage-A attention diagnostic pre-out NRMSE", row["pre_out_normalized_rmse"]
    )
    pre_cos = _finite("Stage-A attention diagnostic pre-out cosine", row["pre_out_cosine"])
    post_nrmse = _finite(
        "Stage-A attention diagnostic post-out NRMSE", row["post_out_normalized_rmse"]
    )
    post_cos = _finite("Stage-A attention diagnostic post-out cosine", row["post_out_cosine"])
    if centered < 0.0 or pre_nrmse < 0.0 or post_nrmse < 0.0:
        raise RuntimeError("Stage-A attention diagnostic NRMSE values must be non-negative")
    if kl < -1e-5:
        raise RuntimeError("Stage-A attention diagnostic KL is materially negative")
    if not -1.000001 <= pre_cos <= 1.000001 or not -1.000001 <= post_cos <= 1.000001:
        raise RuntimeError("Stage-A attention diagnostic cosine values are outside [-1,1]")

    masses: dict[str, tuple[float, ...]] = {}
    for name in ("teacher_modality_mass", "student_modality_mass"):
        values = row[name]
        if not isinstance(values, list) or len(values) != len(modality_kinds):
            raise RuntimeError(
                f"Stage-A attention diagnostic {name} must align with modality_kinds"
            )
        parsed = tuple(
            _finite(f"Stage-A attention diagnostic {name}", value) for value in values
        )
        if any(value < -1e-6 or value > 1.000001 for value in parsed):
            raise RuntimeError(f"Stage-A attention diagnostic {name} is outside [0,1]")
        if abs(sum(parsed) - 1.0) > 1e-4:
            raise RuntimeError(f"Stage-A attention diagnostic {name} does not sum to one")
        masses[name] = parsed

    return PilotAttentionDiagnostic(
        case_id=case_id,
        sigma=sigma,
        modality_label=modality_label,
        sampled_query_rows=tuple(sampled),
        head_chunk_size=chunks["head_chunk_size"],
        key_chunk_size=chunks["key_chunk_size"],
        modality_kinds=tuple(modality_kinds),
        mean_centered_logit_nrmse=centered,
        mean_teacher_to_student_softmax_kl=kl,
        pre_out_normalized_rmse=pre_nrmse,
        pre_out_cosine=pre_cos,
        post_out_normalized_rmse=post_nrmse,
        post_out_cosine=post_cos,
        teacher_modality_mass=masses["teacher_modality_mass"],
        student_modality_mass=masses["student_modality_mass"],
    )


def _attention_diagnostics(
    value: Any,
    cases: tuple[PilotCaseMetrics, ...],
) -> tuple[PilotAttentionDiagnostic, ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeError("Stage-A attention diagnostics must be a non-empty list")
    diagnostics = tuple(_attention_diagnostic(row) for row in value)
    expected_ids = tuple(case.case_id for case in cases)
    actual_ids = tuple(row.case_id for row in diagnostics)
    if actual_ids != expected_ids:
        raise RuntimeError("Stage-A attention diagnostic cases do not match aggregate case ordering")
    for diagnostic, case in zip(diagnostics, cases):
        if diagnostic.modality_label != case.modality_label:
            raise RuntimeError(
                "Stage-A attention diagnostic modality_label does not match aggregate case"
            )
        if diagnostic.sigma is None or case.sigma is None:
            if diagnostic.sigma is not case.sigma:
                raise RuntimeError("Stage-A attention diagnostic sigma does not match aggregate case")
        elif not math.isclose(diagnostic.sigma, float(case.sigma), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("Stage-A attention diagnostic sigma does not match aggregate case")
    return diagnostics


def _route_fit_diagnostics(
    value: Any,
    *,
    route_mode: str,
    lambda_relative: float,
) -> RouteActivationFitDiagnostics | None:
    if route_mode == "identity":
        if value is not None:
            raise RuntimeError("Stage-A identity initialization must not carry route-fit diagnostics")
        return None
    row = _object(value, "Stage-A route-fit diagnostics")
    keys = {
        "rows",
        "lambda_relative",
        "lambda_actual",
        "smallest_singular_value",
        "largest_singular_value",
        "numerical_rank",
        "full_rank_condition_number",
    }
    _exact_keys(row, keys, "Stage-A route-fit diagnostics")
    rows = row["rows"]
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise RuntimeError("Stage-A route-fit rows must be a positive integer")
    stored_relative = _finite("Stage-A route-fit lambda_relative", row["lambda_relative"])
    if stored_relative < 0 or not math.isclose(stored_relative, lambda_relative, rel_tol=0.0, abs_tol=1e-15):
        raise RuntimeError("Stage-A route-fit lambda_relative does not match its initialization row")

    names = (
        "lambda_actual",
        "smallest_singular_value",
        "largest_singular_value",
        "numerical_rank",
        "full_rank_condition_number",
    )
    arrays = {name: row[name] for name in names}
    if any(not isinstance(arrays[name], list) for name in names):
        raise RuntimeError("Stage-A route-fit per-head diagnostics must be JSON arrays")
    lengths = {len(arrays[name]) for name in names}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
        raise RuntimeError("Stage-A route-fit per-head diagnostics have inconsistent lengths")

    lambda_actual = tuple(
        _finite("Stage-A route-fit lambda_actual", value) for value in arrays["lambda_actual"]
    )
    smallest = tuple(
        _finite("Stage-A route-fit smallest singular value", value)
        for value in arrays["smallest_singular_value"]
    )
    largest = tuple(
        _finite("Stage-A route-fit largest singular value", value)
        for value in arrays["largest_singular_value"]
    )
    ranks: list[int] = []
    for value in arrays["numerical_rank"]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError("Stage-A route-fit numerical ranks must be non-negative integers")
        ranks.append(value)
    conditions: list[float | None] = []
    for value in arrays["full_rank_condition_number"]:
        if value is None:
            conditions.append(None)
            continue
        condition = _finite("Stage-A route-fit condition number", value)
        if condition < 1.0:
            raise RuntimeError("Stage-A route-fit condition numbers must be at least one")
        conditions.append(condition)
    if any(value < 0.0 for value in lambda_actual):
        raise RuntimeError("Stage-A route-fit lambda_actual values must be non-negative")
    if any(value < 0.0 for value in (*smallest, *largest)):
        raise RuntimeError("Stage-A route-fit singular values must be non-negative")
    if any(lo > hi for lo, hi in zip(smallest, largest)):
        raise RuntimeError("Stage-A route-fit singular-value bounds are inconsistent")

    return RouteActivationFitDiagnostics(
        rows=rows,
        lambda_relative=stored_relative,
        lambda_actual=lambda_actual,
        smallest_singular_value=smallest,
        largest_singular_value=largest,
        numerical_rank=tuple(ranks),
        full_rank_condition_number=tuple(conditions),
    )


def _case_metrics(row: Any) -> PilotCaseMetrics:
    row = _object(row, "Stage-A case metrics")
    keys = {
        "case_id", "modality_label", "sigma", "rows", "total",
        "attention_normalized_mse", "block_normalized_mse",
        "attention_cosine", "block_cosine",
    }
    _exact_keys(row, keys, "Stage-A case metrics")
    if not isinstance(row["case_id"], str) or not row["case_id"]:
        raise RuntimeError("Stage-A case metric case_id must be non-empty")
    if not isinstance(row["modality_label"], str) or not row["modality_label"]:
        raise RuntimeError("Stage-A case metric modality_label must be non-empty")
    if isinstance(row["rows"], bool) or not isinstance(row["rows"], int) or row["rows"] <= 0:
        raise RuntimeError("Stage-A case metric rows must be a positive integer")
    sigma = row["sigma"]
    if sigma is not None:
        sigma = _finite("Stage-A case sigma", sigma)
    errors = (
        _finite("Stage-A case total", row["total"]),
        _finite("Stage-A case attention NMSE", row["attention_normalized_mse"]),
        _finite("Stage-A case block NMSE", row["block_normalized_mse"]),
    )
    if any(value < 0 for value in errors):
        raise RuntimeError("Stage-A case error metrics must be non-negative")
    attn_cos = _finite("Stage-A case attention cosine", row["attention_cosine"])
    block_cos = _finite("Stage-A case block cosine", row["block_cosine"])
    if not -1.0 <= attn_cos <= 1.0 or not -1.0 <= block_cos <= 1.0:
        raise RuntimeError("Stage-A case cosine metrics must be within [-1,1]")
    return PilotCaseMetrics(
        case_id=row["case_id"],
        modality_label=row["modality_label"],
        sigma=sigma,
        rows=row["rows"],
        total=errors[0],
        attention_normalized_mse=errors[1],
        block_normalized_mse=errors[2],
        attention_cosine=attn_cos,
        block_cosine=block_cos,
    )


def _aggregate(row: Any) -> PilotAggregateMetrics:
    row = _object(row, "Stage-A aggregate metrics")
    keys = {
        "case_count", "mean_total", "mean_attention_normalized_mse",
        "mean_block_normalized_mse", "mean_attention_cosine", "mean_block_cosine",
        "worst_attention_normalized_mse", "worst_block_normalized_mse",
        "minimum_attention_cosine", "minimum_block_cosine", "by_modality", "cases",
    }
    _exact_keys(row, keys, "Stage-A aggregate metrics")
    raw_cases = row["cases"]
    if not isinstance(raw_cases, list):
        raise RuntimeError("Stage-A aggregate cases must be a list")
    cases = tuple(_case_metrics(case) for case in raw_cases)
    if isinstance(row["case_count"], bool) or not isinstance(row["case_count"], int):
        raise RuntimeError("Stage-A aggregate case_count must be an integer")
    if row["case_count"] <= 0 or row["case_count"] != len(cases):
        raise RuntimeError("Stage-A aggregate case_count is inconsistent")
    by_modality = row["by_modality"]
    if not isinstance(by_modality, dict) or not by_modality:
        raise RuntimeError("Stage-A aggregate by_modality must be a non-empty object")
    numeric = {
        key: _finite(f"Stage-A aggregate {key}", row[key])
        for key in keys
        if key not in {"case_count", "by_modality", "cases"}
    }
    for key in (
        "mean_total", "mean_attention_normalized_mse", "mean_block_normalized_mse",
        "worst_attention_normalized_mse", "worst_block_normalized_mse",
    ):
        if numeric[key] < 0:
            raise RuntimeError(f"Stage-A aggregate {key} must be non-negative")
    for key in (
        "mean_attention_cosine", "mean_block_cosine",
        "minimum_attention_cosine", "minimum_block_cosine",
    ):
        if not -1.0 <= numeric[key] <= 1.0:
            raise RuntimeError(f"Stage-A aggregate {key} must be within [-1,1]")
    return PilotAggregateMetrics(
        case_count=row["case_count"],
        mean_total=numeric["mean_total"],
        mean_attention_normalized_mse=numeric["mean_attention_normalized_mse"],
        mean_block_normalized_mse=numeric["mean_block_normalized_mse"],
        mean_attention_cosine=numeric["mean_attention_cosine"],
        mean_block_cosine=numeric["mean_block_cosine"],
        worst_attention_normalized_mse=numeric["worst_attention_normalized_mse"],
        worst_block_normalized_mse=numeric["worst_block_normalized_mse"],
        minimum_attention_cosine=numeric["minimum_attention_cosine"],
        minimum_block_cosine=numeric["minimum_block_cosine"],
        by_modality=by_modality,
        cases=cases,
    )


def _training_event(row: Any) -> PilotTrainingEvent:
    row = _object(row, "Stage-A training event")
    _exact_keys(row, {"stage", "epoch", "case_id", "report"}, "Stage-A training event")
    if row["stage"] not in ("route", "query", "value", "norm_out"):
        raise RuntimeError(f"invalid Stage-A training event stage: {row['stage']!r}")
    if isinstance(row["epoch"], bool) or not isinstance(row["epoch"], int) or row["epoch"] < 0:
        raise RuntimeError("Stage-A training event epoch must be a non-negative integer")
    if not isinstance(row["case_id"], str) or not row["case_id"]:
        raise RuntimeError("Stage-A training event case_id must be non-empty")
    report = _object(row["report"], "Stage-A step report")
    report_keys = {
        "total", "attention_normalized_mse", "block_normalized_mse",
        "attention_cosine", "block_cosine", "trainable_parameters", "gradient_l2_norm",
    }
    _exact_keys(report, report_keys, "Stage-A step report")
    values = {key: _finite(f"Stage-A step {key}", report[key]) for key in report_keys if key != "trainable_parameters"}
    if isinstance(report["trainable_parameters"], bool) or not isinstance(report["trainable_parameters"], int) or report["trainable_parameters"] <= 0:
        raise RuntimeError("Stage-A step trainable_parameters must be a positive integer")
    if values["total"] < 0 or values["attention_normalized_mse"] < 0 or values["block_normalized_mse"] < 0 or values["gradient_l2_norm"] < 0:
        raise RuntimeError("Stage-A step errors/gradient norm must be non-negative")
    return PilotTrainingEvent(
        stage=row["stage"],
        epoch=row["epoch"],
        case_id=row["case_id"],
        report=PilotStepReport(
            total=values["total"],
            attention_normalized_mse=values["attention_normalized_mse"],
            block_normalized_mse=values["block_normalized_mse"],
            attention_cosine=values["attention_cosine"],
            block_cosine=values["block_cosine"],
            trainable_parameters=report["trainable_parameters"],
            gradient_l2_norm=values["gradient_l2_norm"],
        ),
    )


def _initialization(row: Any) -> StageAInitializationEvaluation:
    row = _object(row, "Stage-A initialization evaluation")
    _exact_keys(
        row,
        {
            "route_mode",
            "lambda_relative",
            "metrics",
            "route_fit_diagnostics",
            "attention_diagnostics",
        },
        "Stage-A initialization evaluation",
    )
    if row["route_mode"] not in ("identity", "least_squares"):
        raise RuntimeError(f"invalid Stage-A route mode: {row['route_mode']!r}")
    lam = _finite("Stage-A initialization lambda_relative", row["lambda_relative"])
    if lam < 0:
        raise RuntimeError("Stage-A initialization lambda_relative must be non-negative")
    metrics = _aggregate(row["metrics"])
    diagnostics = _route_fit_diagnostics(
        row["route_fit_diagnostics"],
        route_mode=row["route_mode"],
        lambda_relative=lam,
    )
    attention_diagnostics = _attention_diagnostics(row["attention_diagnostics"], metrics.cases)
    return StageAInitializationEvaluation(
        route_mode=row["route_mode"],
        lambda_relative=lam,
        metrics=metrics,
        route_fit_diagnostics=diagnostics,
        attention_diagnostics=attention_diagnostics,
    )


def _replay(row: Any, block_index: int) -> CapturedReplayReport:
    row = _object(row, "Stage-A replay report")
    keys = {
        "block_index", "rows", "attention_input_max_abs_error",
        "attention_input_mean_abs_error", "block_output_finite", "attention_output_finite",
    }
    _exact_keys(row, keys, "Stage-A replay report")
    if row["block_index"] != block_index:
        raise RuntimeError("Stage-A replay report block_index does not match artifact")
    if isinstance(row["rows"], bool) or not isinstance(row["rows"], int) or row["rows"] <= 0:
        raise RuntimeError("Stage-A replay rows must be a positive integer")
    maximum = _finite("Stage-A replay max abs error", row["attention_input_max_abs_error"])
    mean = _finite("Stage-A replay mean abs error", row["attention_input_mean_abs_error"])
    if maximum < 0 or mean < 0:
        raise RuntimeError("Stage-A replay input errors must be non-negative")
    if row["block_output_finite"] is not True or row["attention_output_finite"] is not True:
        raise RuntimeError("Stage-A replay artifact records non-finite teacher output")
    return CapturedReplayReport(block_index, row["rows"], maximum, mean, True, True)


def _gate(row: Any) -> StageABlockGateResult:
    row = _object(row, "Stage-A block gate")
    keys = {
        "block_index", "passed", "failures", "case_fraction_improved_over_both",
        "mean_total_relative_improvement_vs_identity",
        "mean_total_relative_improvement_vs_least_squares",
        "maximum_modality_attention_nmse_ratio", "maximum_modality_block_nmse_ratio",
        "maximum_gradient_l2_norm",
    }
    _exact_keys(row, keys, "Stage-A block gate")
    failures = row["failures"]
    if not isinstance(failures, list) or not all(isinstance(x, str) for x in failures):
        raise RuntimeError("Stage-A gate failures must be a list of strings")
    if not isinstance(row["passed"], bool):
        raise RuntimeError("Stage-A gate passed must be boolean")
    numeric = {key: float(row[key]) for key in keys if key not in {"block_index", "passed", "failures"}}
    return StageABlockGateResult(
        block_index=int(row["block_index"]),
        passed=row["passed"],
        failures=tuple(failures),
        **numeric,
    )


def load_completed_stage_a_block_evidence(
    request: StageAArtifactRequest,
    *,
    block_index: int,
    final_stage: str,
    expected_dataset_manifest_sha256: str,
    expected_gate_manifest_sha256: str,
    gate_manifest: Mapping[str, object],
) -> CompletedStageABlockEvidence | None:
    """Validate and reuse one completed local block pilot after an interrupted campaign.

    This reads a trusted-local ``torch.save`` resume checkpoint with
    ``weights_only=False``. Do not point ``--resume-completed`` at untrusted artifacts.
    Missing checkpoint+result pairs return ``None``; a partial or contradictory pair is
    an error. Numerical evidence is re-parsed, the initialization selection and gate are
    recomputed, and the deterministic result payload hash must match the hash embedded in
    the immutable checkpoint.
    """
    if request.experiment_context_sha256 is None:
        raise ValueError("resumable Stage-A artifacts require experiment_context_sha256")
    expected_dataset = _require_sha256(
        "expected Stage-A dataset manifest SHA-256", expected_dataset_manifest_sha256
    )
    expected_gate = _require_sha256(
        "expected Stage-A gate manifest SHA-256", expected_gate_manifest_sha256
    )
    checkpoint_path, result_path = request.paths(block_index, final_stage)
    exists = (checkpoint_path.exists(), result_path.exists())
    if not any(exists):
        return None
    if not all(exists):
        raise RuntimeError(
            f"partial Stage-A evidence for block {block_index}: checkpoint={exists[0]}, result={exists[1]}"
        )

    try:
        top = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid Stage-A result JSON: {result_path}") from exc
    top = _object(top, "Stage-A block result")
    top_keys = {
        "schema", "identity", "stage", "step", "experiment_context_sha256",
        "checkpoint_filename", "checkpoint_sha256", "result_payload_sha256", "result",
    }
    _exact_keys(top, top_keys, "Stage-A block result")
    if top["schema"] != STAGE_A_RESULT_SCHEMA:
        raise RuntimeError("not a current Stage-A block result artifact")
    if top["stage"] != final_stage:
        raise RuntimeError("completed Stage-A artifact final stage does not match the current plan")
    if isinstance(top["step"], bool) or not isinstance(top["step"], int) or top["step"] < 0:
        raise RuntimeError("completed Stage-A artifact step must be a non-negative integer")
    if top["experiment_context_sha256"] != request.experiment_context_sha256:
        raise RuntimeError("completed Stage-A artifact experiment context does not match this campaign")
    if top["checkpoint_filename"] != checkpoint_path.name:
        raise RuntimeError("completed Stage-A artifact checkpoint filename is inconsistent")
    checkpoint_sha = _require_sha256("Stage-A checkpoint SHA-256", top["checkpoint_sha256"])
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise RuntimeError("completed Stage-A checkpoint SHA-256 does not match its result artifact")

    result_payload = _object(top["result"], "Stage-A numerical result payload")
    result_payload_sha = _require_sha256(
        "Stage-A result payload SHA-256", top["result_payload_sha256"]
    )
    if canonical_json_sha256(result_payload) != result_payload_sha:
        raise RuntimeError("completed Stage-A numerical result payload hash is invalid")

    identity_row = _object(top["identity"], "Stage-A run identity")
    try:
        identity = PilotRunIdentity(**identity_row)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("completed Stage-A run identity is invalid") from exc
    if identity.run_id != request.run_id or identity.code_commit != request.code_commit:
        raise RuntimeError("completed Stage-A run/code identity does not match this campaign")
    if identity.block_index != block_index:
        raise RuntimeError("completed Stage-A run identity has the wrong block index")
    if identity.dataset_manifest_sha256.lower() != expected_dataset:
        raise RuntimeError("completed Stage-A dataset identity does not match this campaign")
    if identity.gate_manifest_sha256.lower() != expected_gate:
        raise RuntimeError("completed Stage-A gate identity does not match this campaign")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        if not isinstance(checkpoint, dict) or checkpoint.get("schema") != RESUME_SCHEMA:
            raise RuntimeError("completed Stage-A checkpoint has the wrong resume schema")
        if checkpoint.get("identity") != asdict(identity):
            raise RuntimeError("completed Stage-A checkpoint identity differs from result JSON")
        if checkpoint.get("stage") != final_stage or checkpoint.get("step") != top["step"]:
            raise RuntimeError("completed Stage-A checkpoint stage/step differs from result JSON")
        extra = checkpoint.get("extra")
        if not isinstance(extra, dict):
            raise RuntimeError("completed Stage-A checkpoint is missing evidence metadata")
        if extra.get("stage_a_result_schema") != STAGE_A_RESULT_SCHEMA:
            raise RuntimeError("completed Stage-A checkpoint result schema marker is invalid")
        if extra.get("stage_a_result_payload_sha256") != result_payload_sha:
            raise RuntimeError("completed Stage-A checkpoint is not bound to its numerical result")
        if extra.get("stage_a_experiment_context_sha256") != request.experiment_context_sha256:
            raise RuntimeError("completed Stage-A checkpoint experiment context is invalid")
    finally:
        del checkpoint

    payload_keys = {
        "block_index", "replay_reports", "initialization_evaluations",
        "selected_route_mode", "selected_lambda_relative", "identity_baseline",
        "least_squares_baseline", "candidate", "candidate_attention_diagnostics",
        "training_events", "gate",
    }
    _exact_keys(result_payload, payload_keys, "Stage-A numerical result payload")
    if result_payload["block_index"] != block_index:
        raise RuntimeError("completed Stage-A numerical result has the wrong block index")
    replays_raw = result_payload["replay_reports"]
    evaluations_raw = result_payload["initialization_evaluations"]
    events_raw = result_payload["training_events"]
    if not isinstance(replays_raw, list) or not replays_raw:
        raise RuntimeError("completed Stage-A result has no replay evidence")
    if not isinstance(evaluations_raw, list):
        raise RuntimeError("completed Stage-A result initialization_evaluations must be a list")
    if not isinstance(events_raw, list) or not events_raw:
        raise RuntimeError("completed Stage-A result has no training events")
    tuple(_replay(row, block_index) for row in replays_raw)
    evaluations = tuple(_initialization(row) for row in evaluations_raw)
    selected = select_stage_a_initialization(evaluations)
    selected_lambda = _finite(
        "completed Stage-A selected lambda", result_payload["selected_lambda_relative"]
    )
    if result_payload["selected_route_mode"] != selected.route_mode or selected_lambda != selected.lambda_relative:
        raise RuntimeError("completed Stage-A selected initialization does not match its evaluation grid")
    if identity.route_mode != selected.route_mode or identity.lambda_relative != selected.lambda_relative:
        raise RuntimeError("completed Stage-A run identity does not match selected initialization")

    identity_eval = next(row for row in evaluations if row.route_mode == "identity")
    best_ls = min(
        (row for row in evaluations if row.route_mode == "least_squares"),
        key=lambda row: (
            row.metrics.mean_attention_normalized_mse,
            row.metrics.mean_block_normalized_mse,
            row.lambda_relative,
        ),
    )
    identity_baseline = _aggregate(result_payload["identity_baseline"])
    ls_baseline = _aggregate(result_payload["least_squares_baseline"])
    if asdict(identity_baseline) != asdict(identity_eval.metrics):
        raise RuntimeError("completed Stage-A identity baseline is inconsistent with initialization grid")
    if asdict(ls_baseline) != asdict(best_ls.metrics):
        raise RuntimeError("completed Stage-A least-squares baseline is inconsistent with initialization grid")
    candidate = _aggregate(result_payload["candidate"])
    _attention_diagnostics(result_payload["candidate_attention_diagnostics"], candidate.cases)
    events = tuple(_training_event(row) for row in events_raw)
    stored_gate = _gate(result_payload["gate"])
    if stored_gate.block_index != block_index:
        raise RuntimeError("completed Stage-A stored gate has the wrong block index")
    recomputed_gate = evaluate_stage_a_block_gate(
        block_index=block_index,
        candidate=candidate,
        identity_baseline=identity_baseline,
        least_squares_baseline=ls_baseline,
        training_events=events,
        policy=stage_a_policy_from_gate_manifest(gate_manifest),
    )
    if asdict(stored_gate) != asdict(recomputed_gate):
        raise RuntimeError("completed Stage-A gate does not recompute from its bound evidence")

    return CompletedStageABlockEvidence(
        block_index=block_index,
        identity=identity,
        stage=final_stage,
        step=top["step"],
        gate=recomputed_gate,
        artifact=StageAArtifactReceipt(
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha,
            result_path=str(result_path),
            result_sha256=sha256_file(result_path),
        ),
    )
