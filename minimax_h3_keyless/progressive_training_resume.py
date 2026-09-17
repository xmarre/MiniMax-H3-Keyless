from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from .checkpoint import sha256_file
from .pilot import PilotStepReport, set_pilot_block_stage
from .pilot_campaign import (
    PilotTrainingEvent,
    _require_sha256,
    validate_optimizer_matches_trainable,
)


PROGRESSIVE_TRAINING_RESUME_SCHEMA = "minimax_h3_keyless_progressive_training_resume_v1"
_STAGE_NAMES = ("route", "query", "value", "norm_out")


@dataclass(frozen=True)
class ProgressiveTrainingResumeIdentity:
    """Immutable experiment identity for a mutable in-progress Stage-B checkpoint."""

    sweep_id: str
    code_commit: str
    stage_a_campaign_sha256: str
    dataset_manifest_sha256: str
    gate_manifest_sha256: str
    prefix_identity_sha256: str
    prefix_manifest_sha256: str
    capture_registry_file_sha256: str
    train_plan_identity_sha256: str
    train_plan_file_sha256: str
    block_index: int
    selected_route_mode: str
    selected_lambda_relative: float

    def __post_init__(self) -> None:
        if not isinstance(self.sweep_id, str) or not self.sweep_id.strip():
            raise ValueError("progressive resume sweep_id must be non-empty")
        if not isinstance(self.code_commit, str) or len(self.code_commit) != 40:
            raise ValueError("progressive resume code_commit must be a full 40-hex revision")
        try:
            int(self.code_commit, 16)
        except ValueError as exc:
            raise ValueError("progressive resume code_commit must be a full 40-hex revision") from exc
        for name in (
            "stage_a_campaign_sha256",
            "dataset_manifest_sha256",
            "gate_manifest_sha256",
            "prefix_identity_sha256",
            "prefix_manifest_sha256",
            "capture_registry_file_sha256",
            "train_plan_identity_sha256",
            "train_plan_file_sha256",
        ):
            object.__setattr__(self, name, _require_sha256(name, getattr(self, name)))
        if isinstance(self.block_index, bool) or not isinstance(self.block_index, int):
            raise ValueError("progressive resume block_index must be an integer")
        if not 0 <= self.block_index < 50:
            raise ValueError("progressive resume block_index must be within [0,50)")
        if self.selected_route_mode not in ("identity", "least_squares"):
            raise ValueError("progressive resume selected_route_mode is invalid")
        value = float(self.selected_lambda_relative)
        if self.selected_route_mode == "identity" and value != 0.0:
            raise ValueError("identity progressive resume requires selected_lambda_relative=0")
        if value not in (0.0, 1e-4, 1e-2):
            raise ValueError("progressive resume lambda is outside the fixed initialization grid")


@dataclass(frozen=True)
class ProgressiveTrainingResumeState:
    path: str
    checkpoint_sha256: str
    identity: ProgressiveTrainingResumeIdentity
    stage_index: int
    stage: str
    completed_epochs: int
    events: tuple[PilotTrainingEvent, ...]
    student_state_dict: Mapping[str, torch.Tensor]
    optimizer_state_dict: Mapping[str, Any]
    rng_state: Mapping[str, Any]


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": __import__("random").getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    __import__("random").setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("progressive resume contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _event_from_json(value: Any) -> PilotTrainingEvent:
    if not isinstance(value, dict) or set(value) != {"stage", "epoch", "case_id", "report"}:
        raise RuntimeError("progressive resume contains an invalid training event")
    stage = value["stage"]
    if stage not in _STAGE_NAMES:
        raise RuntimeError("progressive resume event has an invalid stage")
    epoch = value["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise RuntimeError("progressive resume event has an invalid epoch")
    case_id = value["case_id"]
    if not isinstance(case_id, str) or not case_id:
        raise RuntimeError("progressive resume event has an invalid case_id")
    report_raw = value["report"]
    report_keys = {
        "total",
        "attention_normalized_mse",
        "block_normalized_mse",
        "attention_cosine",
        "block_cosine",
        "trainable_parameters",
        "gradient_l2_norm",
    }
    if not isinstance(report_raw, dict) or set(report_raw) != report_keys:
        raise RuntimeError("progressive resume contains an invalid step report")
    try:
        report = PilotStepReport(**report_raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("progressive resume contains an invalid step report") from exc
    return PilotTrainingEvent(stage=stage, epoch=epoch, case_id=case_id, report=report)


def _fsync_directory(path: Path) -> None:
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_progressive_training_resume(
    path: str | Path,
    *,
    student_block: nn.Module,
    optimizer: torch.optim.Optimizer,
    identity: ProgressiveTrainingResumeIdentity,
    stage_index: int,
    stage: str,
    completed_epochs: int,
    events: tuple[PilotTrainingEvent, ...],
) -> str:
    """Atomically replace the mutable crash-recovery checkpoint after a full epoch.

    This file is scratch recovery state, not accepted evidence. The immutable Stage-B
    candidate checkpoint/result pair remains the acceptance authority after training
    finishes. Only trusted local resume files may be loaded because optimizer/RNG state
    requires ``torch.load(..., weights_only=False)``.
    """

    if isinstance(stage_index, bool) or not isinstance(stage_index, int) or stage_index < 0:
        raise ValueError("progressive resume stage_index must be a non-negative integer")
    if stage not in _STAGE_NAMES:
        raise ValueError(f"unsupported progressive resume stage {stage!r}")
    if isinstance(completed_epochs, bool) or not isinstance(completed_epochs, int) or completed_epochs <= 0:
        raise ValueError("progressive resume completed_epochs must be a positive integer")
    validate_optimizer_matches_trainable(student_block, optimizer)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": PROGRESSIVE_TRAINING_RESUME_SCHEMA,
        "identity": asdict(identity),
        "stage_index": stage_index,
        "stage": stage,
        "completed_epochs": completed_epochs,
        "events": [asdict(event) for event in events],
        "student_state_dict": student_block.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": _capture_rng_state(),
    }
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with tmp.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return sha256_file(path)


def load_progressive_training_resume(
    path: str | Path,
    *,
    expected_identity: ProgressiveTrainingResumeIdentity,
) -> ProgressiveTrainingResumeState:
    """Load and identity-check one trusted local crash-recovery checkpoint."""

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != PROGRESSIVE_TRAINING_RESUME_SCHEMA:
        raise RuntimeError("not a current progressive training resume checkpoint")
    if payload.get("identity") != asdict(expected_identity):
        raise RuntimeError("progressive training resume identity does not match this block run")
    stage_index = payload.get("stage_index")
    if isinstance(stage_index, bool) or not isinstance(stage_index, int) or stage_index < 0:
        raise RuntimeError("progressive training resume has an invalid stage_index")
    stage = payload.get("stage")
    if stage not in _STAGE_NAMES:
        raise RuntimeError("progressive training resume has an invalid stage")
    completed_epochs = payload.get("completed_epochs")
    if (
        isinstance(completed_epochs, bool)
        or not isinstance(completed_epochs, int)
        or completed_epochs <= 0
    ):
        raise RuntimeError("progressive training resume has invalid completed_epochs")
    event_rows = payload.get("events")
    if not isinstance(event_rows, list):
        raise RuntimeError("progressive training resume events must be a list")
    events = tuple(_event_from_json(row) for row in event_rows)
    student_state = payload.get("student_state_dict")
    optimizer_state = payload.get("optimizer_state_dict")
    rng_state = payload.get("rng_state")
    if not isinstance(student_state, dict):
        raise RuntimeError("progressive training resume is missing student state")
    if not isinstance(optimizer_state, dict):
        raise RuntimeError("progressive training resume is missing optimizer state")
    if not isinstance(rng_state, dict):
        raise RuntimeError("progressive training resume is missing RNG state")
    return ProgressiveTrainingResumeState(
        path=str(path),
        checkpoint_sha256=sha256_file(path),
        identity=expected_identity,
        stage_index=stage_index,
        stage=stage,
        completed_epochs=completed_epochs,
        events=events,
        student_state_dict=student_state,
        optimizer_state_dict=optimizer_state,
        rng_state=rng_state,
    )


def restore_progressive_training_resume(
    state: ProgressiveTrainingResumeState,
    *,
    student_block: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    restore_rng: bool = True,
) -> None:
    """Restore student state and, when continuing the same stage, optimizer state."""

    student_block.load_state_dict(state.student_state_dict, strict=True)
    set_pilot_block_stage(student_block, state.stage)
    if optimizer is not None:
        validate_optimizer_matches_trainable(student_block, optimizer)
        optimizer.load_state_dict(state.optimizer_state_dict)
        validate_optimizer_matches_trainable(student_block, optimizer)
    if restore_rng:
        _restore_rng_state(state.rng_state)


def remove_progressive_training_resume(path: str | Path) -> None:
    """Remove mutable recovery state only after immutable candidate persistence succeeds."""

    path = Path(path)
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_directory(path.parent)
