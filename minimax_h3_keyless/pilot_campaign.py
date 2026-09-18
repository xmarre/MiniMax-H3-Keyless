from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from .checkpoint import sha256_file
from .contracts import TARGET_MODEL_REVISION, TEACHER_SHA256
from .initialization import PilotTrainStage, RouteInitMode
from .pilot import (
    PilotCase,
    PilotLossWeights,
    PilotStepReport,
    pilot_loss,
    pilot_train_step,
    set_pilot_block_stage,
)


PILOT_BLOCKS = (0, 25, 49)
PILOT_LS_LAMBDAS = (0.0, 1e-4, 1e-2)
DATASET_SCHEMA = "minimax_h3_keyless_pilot_dataset_v1"
GATE_SCHEMA = "minimax_h3_keyless_pilot_gates_v1"
RESUME_SCHEMA = "minimax_h3_keyless_pilot_resume_v1"
_PILOT_STAGES: tuple[PilotTrainStage, ...] = ("route", "query", "value", "norm_out")


@dataclass(frozen=True)
class PilotRunIdentity:
    run_id: str
    code_commit: str
    dataset_manifest_sha256: str
    gate_manifest_sha256: str
    block_index: int
    route_mode: RouteInitMode
    lambda_relative: float
    teacher_model_revision: str = TARGET_MODEL_REVISION
    teacher_model_sha256: str = TEACHER_SHA256

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("pilot run_id must be non-empty")
        if not self.code_commit.strip():
            raise ValueError("pilot code_commit must be non-empty")
        for name, value in (
            ("dataset_manifest_sha256", self.dataset_manifest_sha256),
            ("gate_manifest_sha256", self.gate_manifest_sha256),
            ("teacher_model_sha256", self.teacher_model_sha256),
        ):
            _require_sha256(name, value)
        if self.teacher_model_revision != TARGET_MODEL_REVISION:
            raise ValueError("pilot teacher revision must be the pinned BF16 parent")
        if self.teacher_model_sha256.lower() != TEACHER_SHA256:
            raise ValueError("pilot teacher SHA-256 must be the pinned BF16 parent")
        if self.block_index not in PILOT_BLOCKS:
            raise ValueError(f"Stage-A pilot block must be one of {PILOT_BLOCKS}")
        if self.route_mode == "identity":
            if self.lambda_relative != 0.0:
                raise ValueError("identity route initialization requires lambda_relative=0")
        elif self.route_mode == "least_squares":
            if self.lambda_relative not in PILOT_LS_LAMBDAS:
                raise ValueError(
                    f"least-squares pilot lambda_relative must be one of {PILOT_LS_LAMBDAS}"
                )
        else:
            raise ValueError(f"unsupported pilot route mode: {self.route_mode!r}")


@dataclass(frozen=True)
class PilotCaseMetrics:
    case_id: str
    modality_label: str
    sigma: float | None
    rows: int
    total: float
    attention_normalized_mse: float
    block_normalized_mse: float
    attention_cosine: float
    block_cosine: float


@dataclass(frozen=True)
class PilotAggregateMetrics:
    case_count: int
    mean_total: float
    mean_attention_normalized_mse: float
    mean_block_normalized_mse: float
    mean_attention_cosine: float
    mean_block_cosine: float
    worst_attention_normalized_mse: float
    worst_block_normalized_mse: float
    minimum_attention_cosine: float
    minimum_block_cosine: float
    by_modality: Mapping[str, Mapping[str, float | int]]
    cases: tuple[PilotCaseMetrics, ...]


@dataclass(frozen=True)
class PilotTrainingEvent:
    stage: PilotTrainStage
    epoch: int
    case_id: str
    report: PilotStepReport


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a 64-hex SHA-256 string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be a 64-hex SHA-256 string") from exc
    return value.lower()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def load_json_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"manifest must decode to an object: {path}")
    return value


def validate_pilot_dataset_manifest(
    manifest: Mapping[str, Any],
    *,
    minimum_cases: int = 16,
    minimum_sigma_strata: int = 8,
    required_coverage_tags: Iterable[str] = (),
) -> str:
    """Validate the fixed Stage-A data manifest and return its canonical identity.

    The manifest stores case-level provenance only; it intentionally does not archive
    full hidden activations. Live H3 execution must reproduce those inputs during the
    pilot so train/holdout remains split by complete prompt/asset case. The defaults
    encode the design's minimum fixed-case/sigma coverage and must only be overridden
    by an explicitly identified experiment.
    """
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError(f"pilot dataset schema must be {DATASET_SCHEMA!r}")
    if minimum_cases <= 0 or minimum_sigma_strata <= 0:
        raise ValueError("pilot dataset minimums must be positive")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) < minimum_cases:
        raise ValueError(f"pilot dataset requires at least {minimum_cases} complete cases")
    ids: set[str] = set()
    splits: set[str] = set()
    sigma_values: set[float] = set()
    coverage: set[str] = set()
    asset_owners: dict[str, tuple[str, str]] = {}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"pilot case {index} must be an object")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"pilot case {index} has no case_id")
        if case_id in ids:
            raise ValueError(f"duplicate pilot case_id: {case_id}")
        ids.add(case_id)
        split = case.get("split")
        if split not in ("train", "holdout"):
            raise ValueError(f"pilot case {case_id!r} split must be train or holdout")
        splits.add(split)
        if not isinstance(case.get("prompt"), str):
            raise ValueError(f"pilot case {case_id!r} must record its prompt")
        if not isinstance(case.get("seed"), int):
            raise ValueError(f"pilot case {case_id!r} must record an integer seed")
        if "schedule" not in case:
            raise ValueError(f"pilot case {case_id!r} must record its sampler schedule")
        modality = case.get("modality_label")
        if not isinstance(modality, str) or not modality.strip():
            raise ValueError(f"pilot case {case_id!r} must record modality_label")
        resolution = case.get("resolution")
        if (
            not isinstance(resolution, list)
            or len(resolution) != 2
            or not all(isinstance(v, int) and v > 0 for v in resolution)
        ):
            raise ValueError(f"pilot case {case_id!r} resolution must be [height,width]")
        duration = case.get("duration_seconds")
        if not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or duration <= 0:
            raise ValueError(f"pilot case {case_id!r} duration_seconds must be positive")
        sigmas = case.get("sigmas")
        if not isinstance(sigmas, list) or not sigmas:
            raise ValueError(f"pilot case {case_id!r} must record sampled sigma strata")
        for sigma in sigmas:
            if not isinstance(sigma, (int, float)) or not math.isfinite(float(sigma)):
                raise ValueError(f"pilot case {case_id!r} contains a non-finite sigma")
            sigma = float(sigma)
            if not 0.0 <= sigma <= 1.0:
                raise ValueError(f"pilot case {case_id!r} sigma {sigma} is outside [0,1]")
            sigma_values.add(sigma)
        tags = case.get("coverage_tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) and tag for tag in tags):
            raise ValueError(f"pilot case {case_id!r} coverage_tags must be strings")
        coverage.update(tags)
        assets = case.get("assets", [])
        if not isinstance(assets, list):
            raise ValueError(f"pilot case {case_id!r} assets must be a list")
        for asset in assets:
            if not isinstance(asset, dict) or not isinstance(asset.get("path_or_uri"), str):
                raise ValueError(f"pilot case {case_id!r} contains an invalid asset record")
            asset_sha = _require_sha256(
                f"asset SHA-256 for {case_id}", asset.get("sha256", "")
            )
            previous = asset_owners.get(asset_sha)
            if previous is None:
                asset_owners[asset_sha] = (split, case_id)
            elif previous[0] != split:
                raise ValueError(
                    "pilot dataset asset crosses train/holdout split: "
                    f"sha256={asset_sha}, first_case={previous[1]!r} ({previous[0]}), "
                    f"current_case={case_id!r} ({split})"
                )
    if splits != {"train", "holdout"}:
        raise ValueError("pilot dataset must contain complete-case train and holdout splits")
    if len(sigma_values) < minimum_sigma_strata:
        raise ValueError(
            f"pilot dataset requires at least {minimum_sigma_strata} distinct sigma strata"
        )
    missing_tags = sorted(set(required_coverage_tags).difference(coverage))
    if missing_tags:
        raise ValueError(f"pilot dataset is missing required coverage tags: {missing_tags}")
    return canonical_json_sha256(manifest)


def validate_pilot_gate_manifest(manifest: Mapping[str, Any]) -> str:
    """Freeze implementation-stage gates before training without inventing thresholds here."""
    if manifest.get("schema") != GATE_SCHEMA:
        raise ValueError(f"pilot gate schema must be {GATE_SCHEMA!r}")
    thresholds = manifest.get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        raise ValueError("pilot gate manifest must contain predeclared thresholds")
    for name, value in thresholds.items():
        if not isinstance(name, str) or not name:
            raise ValueError("pilot gate threshold names must be non-empty strings")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"pilot gate threshold {name!r} must be finite")
    calibration = manifest.get("calibration_evidence")
    if not isinstance(calibration, dict) or not calibration:
        raise ValueError("pilot gate manifest must identify its numerical calibration evidence")
    return canonical_json_sha256(manifest)


def _metrics_from_loss(case: PilotCase, losses) -> PilotCaseMetrics:
    values = (
        losses.total,
        losses.attention_normalized_mse,
        losses.block_normalized_mse,
        losses.attention_cosine,
        losses.block_cosine,
    )
    if not all(torch.isfinite(value).item() for value in values):
        raise RuntimeError(f"non-finite holdout metric for case {case.case_id!r}")
    return PilotCaseMetrics(
        case_id=case.case_id,
        modality_label=case.modality_label or "unspecified",
        sigma=case.sigma,
        rows=int(case.x.shape[0]),
        total=float(losses.total.item()),
        attention_normalized_mse=float(losses.attention_normalized_mse.item()),
        block_normalized_mse=float(losses.block_normalized_mse.item()),
        attention_cosine=float(losses.attention_cosine.item()),
        block_cosine=float(losses.block_cosine.item()),
    )


def _mean(rows: Sequence[PilotCaseMetrics], field: str) -> float:
    return sum(float(getattr(row, field)) for row in rows) / len(rows)


def evaluate_pilot_cases(
    teacher_block: nn.Module,
    student_block: nn.Module,
    cases: Sequence[PilotCase],
    *,
    weights: PilotLossWeights = PilotLossWeights(),
) -> PilotAggregateMetrics:
    if not cases:
        raise ValueError("pilot evaluation requires at least one complete case")
    seen: set[str] = set()
    for case in cases:
        if not case.case_id:
            raise ValueError("pilot evaluation cases require stable case_id values")
        if case.case_id in seen:
            raise ValueError(f"duplicate pilot evaluation case_id: {case.case_id}")
        seen.add(case.case_id)
        if case.sigma is not None and not 0.0 <= float(case.sigma) <= 1.0:
            raise ValueError(f"pilot case {case.case_id!r} sigma is outside [0,1]")

    was_training = student_block.training
    student_block.eval()
    metrics: list[PilotCaseMetrics] = []
    try:
        with torch.no_grad():
            for case in cases:
                metrics.append(_metrics_from_loss(case, pilot_loss(
                    teacher_block, student_block, case, weights=weights
                )))
    finally:
        student_block.train(was_training)

    by_modality: dict[str, Mapping[str, float | int]] = {}
    for modality in sorted({row.modality_label for row in metrics}):
        selected = [row for row in metrics if row.modality_label == modality]
        by_modality[modality] = {
            "case_count": len(selected),
            "mean_attention_normalized_mse": _mean(selected, "attention_normalized_mse"),
            "mean_block_normalized_mse": _mean(selected, "block_normalized_mse"),
            "mean_attention_cosine": _mean(selected, "attention_cosine"),
            "mean_block_cosine": _mean(selected, "block_cosine"),
        }
    return PilotAggregateMetrics(
        case_count=len(metrics),
        mean_total=_mean(metrics, "total"),
        mean_attention_normalized_mse=_mean(metrics, "attention_normalized_mse"),
        mean_block_normalized_mse=_mean(metrics, "block_normalized_mse"),
        mean_attention_cosine=_mean(metrics, "attention_cosine"),
        mean_block_cosine=_mean(metrics, "block_cosine"),
        worst_attention_normalized_mse=max(row.attention_normalized_mse for row in metrics),
        worst_block_normalized_mse=max(row.block_normalized_mse for row in metrics),
        minimum_attention_cosine=min(row.attention_cosine for row in metrics),
        minimum_block_cosine=min(row.block_cosine for row in metrics),
        by_modality=by_modality,
        cases=tuple(metrics),
    )


def validate_optimizer_matches_trainable(
    student_block: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    trainable = {id(p): name for name, p in student_block.named_parameters() if p.requires_grad}
    optimized: dict[int, int] = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter in group["params"]:
            pid = id(parameter)
            if pid in optimized:
                raise RuntimeError("pilot optimizer contains the same parameter more than once")
            optimized[pid] = group_index
    missing = [name for pid, name in trainable.items() if pid not in optimized]
    stale = [pid for pid in optimized if pid not in trainable]
    if missing or stale:
        raise RuntimeError(
            "pilot optimizer does not exactly cover the current trainable stage; "
            f"missing={missing}, stale_parameter_count={len(stale)}. "
            "Rebuild the optimizer after each freeze/unfreeze stage transition."
        )


def _apply_and_validate_stage(
    student_block: nn.Module,
    optimizer: torch.optim.Optimizer,
    stage: PilotTrainStage,
) -> None:
    if stage not in _PILOT_STAGES:
        raise ValueError(f"unsupported pilot stage: {stage!r}")
    set_pilot_block_stage(student_block, stage)
    validate_optimizer_matches_trainable(student_block, optimizer)


def train_pilot_stage(
    teacher_block: nn.Module,
    student_block: nn.Module,
    cases: Sequence[PilotCase],
    optimizer: torch.optim.Optimizer,
    *,
    stage: PilotTrainStage,
    epochs: int,
    weights: PilotLossWeights = PilotLossWeights(),
    max_grad_norm: float | None = None,
) -> tuple[PilotTrainingEvent, ...]:
    if epochs <= 0:
        raise ValueError("pilot stage epochs must be positive")
    if not cases:
        raise ValueError("pilot stage requires at least one training case")
    _apply_and_validate_stage(student_block, optimizer, stage)
    student_block.train(True)
    events: list[PilotTrainingEvent] = []
    for epoch in range(epochs):
        for case in cases:
            report = pilot_train_step(
                teacher_block,
                student_block,
                case,
                optimizer,
                weights=weights,
                max_grad_norm=max_grad_norm,
            )
            events.append(PilotTrainingEvent(stage=stage, epoch=epoch, case_id=case.case_id, report=report))
    return tuple(events)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("resume checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_pilot_resume_checkpoint(
    path: str | Path,
    *,
    student_block: nn.Module,
    optimizer: torch.optim.Optimizer,
    identity: PilotRunIdentity,
    stage: PilotTrainStage,
    step: int,
    extra: Mapping[str, Any] | None = None,
) -> str:
    if step < 0:
        raise ValueError("pilot resume step must be non-negative")
    _apply_and_validate_stage(student_block, optimizer, stage)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": RESUME_SCHEMA,
        "identity": asdict(identity),
        "stage": stage,
        "step": int(step),
        "student_state_dict": student_block.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": _capture_rng_state(),
        "extra": dict(extra or {}),
    }
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, tmp)
        with open(tmp, "rb+") as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return sha256_file(path)


def load_pilot_resume_checkpoint(
    path: str | Path,
    *,
    student_block: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_identity: PilotRunIdentity,
    expected_stage: PilotTrainStage | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a trusted local training checkpoint and reject run/stage mismatches.

    ``torch.load(..., weights_only=False)`` is intentional because optimizer and Python/
    NumPy RNG states are not tensor-only. Do not load untrusted resume files.
    The caller must construct the optimizer for the saved stage; this function enforces
    that invariant before accepting its serialized optimizer state.
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != RESUME_SCHEMA:
        raise RuntimeError("not a Keyless Stage-A pilot resume checkpoint")
    if payload.get("identity") != asdict(expected_identity):
        raise RuntimeError("pilot resume identity does not match the requested run")
    stage = payload.get("stage")
    if stage not in _PILOT_STAGES:
        raise RuntimeError(f"pilot resume checkpoint has invalid stage: {stage!r}")
    if expected_stage is not None and stage != expected_stage:
        raise RuntimeError(
            f"pilot resume stage {stage!r} does not match requested stage {expected_stage!r}"
        )
    set_pilot_block_stage(student_block, stage)
    validate_optimizer_matches_trainable(student_block, optimizer)
    student_block.load_state_dict(payload["student_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    validate_optimizer_matches_trainable(student_block, optimizer)
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return {
        "stage": stage,
        "step": int(payload["step"]),
        "extra": dict(payload.get("extra") or {}),
        "checkpoint_sha256": sha256_file(path),
    }


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return sha256_file(path)