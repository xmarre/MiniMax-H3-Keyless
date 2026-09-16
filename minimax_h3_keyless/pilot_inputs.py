from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .checkpoint import sha256_file
from .pilot import PilotLossWeights
from .pilot_campaign import _require_sha256, canonical_json_sha256
from .pilot_capture_set import CaptureArtifactRef
from .pilot_runner import StageATrainStage, validate_stage_a_train_plan


CAPTURE_REGISTRY_SCHEMA = "minimax_h3_keyless_stage_a_capture_registry_v1"
TRAIN_PLAN_SCHEMA = "minimax_h3_keyless_stage_a_train_plan_v1"
CANONICAL_STAGE_A_COVERAGE_TAGS = ("short", "long", "reference", "audio", "mixed-grid")


@dataclass(frozen=True)
class StageACaptureRegistry:
    dataset_manifest_sha256: str
    artifacts: tuple[CaptureArtifactRef, ...]
    registry_file_sha256: str


@dataclass(frozen=True)
class StageARunPlan:
    stages: tuple[StageATrainStage, ...]
    loss_weights: PilotLossWeights
    same_input_atol: float
    same_input_rtol: float
    plan_identity_sha256: str
    plan_file_sha256: str


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must decode to an object: {path}")
    return value


def _resolve_relative(base: Path, value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def load_stage_a_capture_registry(path: str | Path) -> StageACaptureRegistry:
    path = Path(path)
    value = _read_json_object(path)
    if value.get("schema") != CAPTURE_REGISTRY_SCHEMA:
        raise ValueError(f"capture registry schema must be {CAPTURE_REGISTRY_SCHEMA!r}")
    dataset_sha = _require_sha256(
        "capture registry dataset manifest SHA-256",
        value.get("dataset_manifest_sha256", ""),
    )
    rows = value.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("capture registry must contain at least one artifact")
    refs: list[CaptureArtifactRef] = []
    seen_receipts: set[str] = set()
    base = path.parent
    allowed = {"bundle_path", "receipt_path", "receipt_sha256"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"capture registry artifact {index} must be an object")
        unknown = set(row).difference(allowed)
        if unknown:
            raise ValueError(
                f"capture registry artifact {index} has unknown fields: {sorted(unknown)}"
            )
        bundle = row.get("bundle_path")
        receipt = row.get("receipt_path")
        receipt_sha = row.get("receipt_sha256")
        if not isinstance(bundle, str) or not bundle.strip():
            raise ValueError(f"capture registry artifact {index} has invalid bundle_path")
        if not isinstance(receipt, str) or not receipt.strip():
            raise ValueError(f"capture registry artifact {index} has invalid receipt_path")
        receipt_sha = _require_sha256(
            f"capture registry artifact {index} receipt SHA-256",
            receipt_sha or "",
        )
        if receipt_sha in seen_receipts:
            raise ValueError(f"duplicate capture receipt SHA-256 in registry: {receipt_sha}")
        seen_receipts.add(receipt_sha)
        refs.append(
            CaptureArtifactRef(
                bundle_path=_resolve_relative(base, bundle),
                receipt_path=_resolve_relative(base, receipt),
                receipt_sha256=receipt_sha,
            )
        )
    unknown_top = set(value).difference({"schema", "dataset_manifest_sha256", "artifacts"})
    if unknown_top:
        raise ValueError(f"capture registry has unknown fields: {sorted(unknown_top)}")
    return StageACaptureRegistry(
        dataset_manifest_sha256=dataset_sha,
        artifacts=tuple(refs),
        registry_file_sha256=sha256_file(path),
    )


def _finite_nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    out = float(value)
    if not math.isfinite(out) or out < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return out


def _finite_positive(name: str, value: Any) -> float:
    out = _finite_nonnegative(name, value)
    if out <= 0:
        raise ValueError(f"{name} must be positive")
    return out


def load_stage_a_run_plan(path: str | Path) -> StageARunPlan:
    """Load all Stage-A training hyperparameters from a predeclared immutable JSON plan."""
    path = Path(path)
    value = _read_json_object(path)
    if value.get("schema") != TRAIN_PLAN_SCHEMA:
        raise ValueError(f"Stage-A train-plan schema must be {TRAIN_PLAN_SCHEMA!r}")
    unknown_top = set(value).difference(
        {"schema", "stages", "loss_weights", "same_input_atol", "same_input_rtol"}
    )
    if unknown_top:
        raise ValueError(f"Stage-A train plan has unknown fields: {sorted(unknown_top)}")
    rows = value.get("stages")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Stage-A train plan must contain stages")
    stages: list[StageATrainStage] = []
    allowed_stage = {"stage", "epochs", "learning_rate", "weight_decay", "max_grad_norm"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Stage-A stage {index} must be an object")
        unknown = set(row).difference(allowed_stage)
        if unknown:
            raise ValueError(f"Stage-A stage {index} has unknown fields: {sorted(unknown)}")
        epochs = row.get("epochs")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(f"Stage-A stage {index} epochs must be a positive integer")
        learning_rate = _finite_positive(
            f"Stage-A stage {index} learning_rate", row.get("learning_rate")
        )
        weight_decay = _finite_nonnegative(
            f"Stage-A stage {index} weight_decay", row.get("weight_decay", 0.0)
        )
        max_grad_norm_raw = row.get("max_grad_norm")
        max_grad_norm = (
            None
            if max_grad_norm_raw is None
            else _finite_positive(f"Stage-A stage {index} max_grad_norm", max_grad_norm_raw)
        )
        stages.append(
            StageATrainStage(
                stage=row.get("stage"),
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
            )
        )
    stage_tuple = validate_stage_a_train_plan(stages)

    loss = value.get("loss_weights")
    if not isinstance(loss, dict):
        raise ValueError("Stage-A train plan must contain loss_weights")
    allowed_loss = {"attention_output", "block_output", "epsilon"}
    unknown_loss = set(loss).difference(allowed_loss)
    if unknown_loss:
        raise ValueError(f"Stage-A loss_weights has unknown fields: {sorted(unknown_loss)}")
    attention_output = _finite_nonnegative(
        "Stage-A attention_output loss weight", loss.get("attention_output")
    )
    block_output = _finite_nonnegative(
        "Stage-A block_output loss weight", loss.get("block_output")
    )
    epsilon = _finite_positive("Stage-A loss epsilon", loss.get("epsilon", 1e-8))
    weights = PilotLossWeights(
        attention_output=attention_output,
        block_output=block_output,
        epsilon=epsilon,
    )
    atol = _finite_nonnegative("Stage-A same_input_atol", value.get("same_input_atol", 0.0))
    rtol = _finite_nonnegative("Stage-A same_input_rtol", value.get("same_input_rtol", 0.0))
    return StageARunPlan(
        stages=stage_tuple,
        loss_weights=weights,
        same_input_atol=atol,
        same_input_rtol=rtol,
        plan_identity_sha256=canonical_json_sha256(value),
        plan_file_sha256=sha256_file(path),
    )
