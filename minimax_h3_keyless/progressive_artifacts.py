from __future__ import annotations

import json
import os
import random
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from .checkpoint import sha256_file
from .pilot_campaign import _require_sha256, canonical_json_sha256, validate_optimizer_matches_trainable
from .progressive import ProgressivePrefix
from .progressive_capture_set import ProgressiveBlockCaptureSet
from .progressive_runner import ProgressiveBlockTrainingResult


PROGRESSIVE_RESUME_SCHEMA = "minimax_h3_keyless_progressive_resume_v1"
PROGRESSIVE_RESULT_SCHEMA = "minimax_h3_keyless_progressive_block_result_v1"
_SAFE_STEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_STAGE_NAMES = ("route", "query", "value", "norm_out")


@dataclass(frozen=True)
class ProgressiveRunIdentity:
    sweep_id: str
    code_commit: str
    stage_a_campaign_sha256: str
    dataset_manifest_sha256: str
    gate_manifest_sha256: str
    prefix_identity_sha256: str
    capture_set_identity_sha256: str
    block_index: int
    route_mode: str
    lambda_relative: float

    def __post_init__(self) -> None:
        if not _SAFE_STEM.fullmatch(self.sweep_id):
            raise ValueError(
                "progressive sweep_id must use only letters, digits, '.', '_' or '-' for artifacts"
            )
        if not isinstance(self.code_commit, str) or not self.code_commit.strip():
            raise ValueError("progressive code_commit must be non-empty")
        for name in (
            "stage_a_campaign_sha256",
            "dataset_manifest_sha256",
            "gate_manifest_sha256",
            "prefix_identity_sha256",
            "capture_set_identity_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        if isinstance(self.block_index, bool) or not isinstance(self.block_index, int):
            raise ValueError("progressive block_index must be an integer")
        if not 0 <= self.block_index < 50:
            raise ValueError("progressive block_index must be within [0,50)")
        if self.route_mode not in ("identity", "least_squares"):
            raise ValueError(f"unsupported progressive route_mode {self.route_mode!r}")
        if self.route_mode == "identity" and float(self.lambda_relative) != 0.0:
            raise ValueError("identity progressive initialization requires lambda_relative=0")
        if float(self.lambda_relative) not in (0.0, 1e-4, 1e-2):
            raise ValueError("progressive lambda_relative is outside the fixed pilot grid")


@dataclass(frozen=True)
class ProgressiveArtifactReceipt:
    checkpoint_path: str
    checkpoint_sha256: str
    result_path: str
    result_sha256: str
    result_payload_sha256: str
    capture_set_identity_sha256: str


def _capture_record_identity(record, split: str) -> dict[str, Any]:
    context = record.case.context
    receipt = context.get("progressive_capture_receipt_sha256") if isinstance(context, Mapping) else None
    bundle = context.get("progressive_capture_bundle_sha256") if isinstance(context, Mapping) else None
    if not isinstance(receipt, str) or not isinstance(bundle, str):
        raise ValueError(
            "persisted progressive evidence requires capture records loaded from immutable artifacts"
        )
    return {
        "split": split,
        "case_id": record.case.case_id,
        "sigma": record.case.sigma,
        "modality_label": record.case.modality_label,
        "receipt_sha256": _require_sha256("progressive capture receipt SHA-256", receipt),
        "bundle_sha256": _require_sha256("progressive capture bundle SHA-256", bundle),
    }


def progressive_capture_set_identity(captures: ProgressiveBlockCaptureSet) -> str:
    """Hash the ordered live-input corpus used for one progressive optimization.

    Order is intentionally retained because train-record order changes optimizer updates.
    Each row is already bound to an immutable capture bundle and sidecar hash by the
    production capture-set loader.
    """
    if not captures.artifact_refs:
        raise ValueError("persisted progressive evidence requires immutable capture artifact refs")
    rows = [
        *(_capture_record_identity(record, "train") for record in captures.train),
        *(_capture_record_identity(record, "holdout") for record in captures.holdout),
    ]
    return canonical_json_sha256(
        {
            "schema": "minimax_h3_keyless_progressive_capture_set_identity_v1",
            "target_block": captures.target_block,
            "prefix_identity_sha256": captures.prefix_identity_sha256,
            "stage_a_campaign_sha256": captures.stage_a_campaign_sha256,
            "dataset_manifest_sha256": captures.dataset_manifest_sha256,
            "gate_manifest_sha256": captures.gate_manifest_sha256,
            "code_commit": captures.code_commit,
            "comfy_commit": captures.comfy_commit,
            "execution_descriptor": captures.execution_descriptor,
            "records": rows,
        }
    )


def progressive_run_identity(
    prefix: ProgressivePrefix,
    captures: ProgressiveBlockCaptureSet,
    result: ProgressiveBlockTrainingResult,
) -> ProgressiveRunIdentity:
    target = prefix.next_block
    if target is None:
        raise ValueError("cannot build a progressive run identity from a complete prefix")
    if result.block_index != target or captures.target_block != target:
        raise ValueError("progressive result/capture block does not match the prefix target")
    if result.prefix_identity_sha256.lower() != prefix.identity_sha256.lower():
        raise ValueError("progressive result does not belong to the supplied prefix")
    if captures.prefix_identity_sha256.lower() != prefix.identity_sha256.lower():
        raise ValueError("progressive capture set does not belong to the supplied prefix")
    return ProgressiveRunIdentity(
        sweep_id=prefix.sweep_id,
        code_commit=prefix.code_commit,
        stage_a_campaign_sha256=prefix.stage_a_campaign_sha256,
        dataset_manifest_sha256=prefix.dataset_manifest_sha256,
        gate_manifest_sha256=prefix.gate_manifest_sha256,
        prefix_identity_sha256=prefix.identity_sha256,
        capture_set_identity_sha256=progressive_capture_set_identity(captures),
        block_index=target,
        route_mode=result.selected_route_mode,
        lambda_relative=result.selected_lambda_relative,
    )


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_progressive_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("progressive resume contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _paths(output_dir: str | Path, identity: ProgressiveRunIdentity, stage: str) -> tuple[Path, Path]:
    if stage not in _STAGE_NAMES:
        raise ValueError(f"unsupported progressive stage {stage!r}")
    root = Path(output_dir)
    stem = f"{identity.sweep_id}.block{identity.block_index:02d}.{stage}"
    return root / f"{stem}.resume.pt", root / f"{stem}.result.json"


def _result_payload(result: ProgressiveBlockTrainingResult) -> dict[str, Any]:
    return {
        "block_index": result.block_index,
        "prefix_identity_sha256": result.prefix_identity_sha256,
        "replay_reports": [asdict(row) for row in result.replay_reports],
        "initialization_evaluations": [asdict(row) for row in result.initialization_evaluations],
        "selected_route_mode": result.selected_route_mode,
        "selected_lambda_relative": result.selected_lambda_relative,
        "identity_baseline": asdict(result.identity_baseline),
        "least_squares_baseline": asdict(result.least_squares_baseline),
        "candidate": asdict(result.candidate),
        "candidate_attention_diagnostics": [
            asdict(row) for row in result.candidate_attention_diagnostics
        ],
        "training_events": [asdict(row) for row in result.training_events],
        "gate": asdict(result.gate),
        "final_stage": result.final_stage,
    }


def _new_temp(final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{final_path.name}.",
        suffix=".tmp",
        dir=final_path.parent,
    )
    os.close(fd)
    return Path(name)


def _publish_no_replace(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            f"progressive evidence is immutable and already exists: {destination}"
        ) from exc


def _unlink_if_same(path: Path, source: Path) -> None:
    try:
        if path.exists() and source.exists() and os.path.samefile(path, source):
            path.unlink()
    except FileNotFoundError:
        pass


def _unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def persist_progressive_block_artifacts(
    output_dir: str | Path,
    *,
    prefix: ProgressivePrefix,
    captures: ProgressiveBlockCaptureSet,
    result: ProgressiveBlockTrainingResult,
) -> ProgressiveArtifactReceipt:
    """Persist one immutable Stage-B training checkpoint/result transaction.

    The checkpoint preserves the training-form q/R/v weights, final optimizer state and
    Python/NumPy/Torch RNG state. The result sidecar binds the complete numerical evidence
    and ordered immutable live-input capture-set identity. Neither final path is replaced
    if a prior accepted or failed experiment already owns that artifact identity.
    """
    identity = progressive_run_identity(prefix, captures, result)
    stage = result.final_stage
    if stage not in _STAGE_NAMES:
        raise ValueError(f"unsupported progressive final_stage {stage!r}")
    validate_optimizer_matches_trainable(result.student_block, result.optimizer)
    checkpoint_path, result_path = _paths(output_dir, identity, stage)
    occupied = [str(path) for path in (checkpoint_path, result_path) if path.exists()]
    if occupied:
        raise FileExistsError(f"progressive evidence is immutable; occupied={occupied}")

    numerical = _result_payload(result)
    payload_sha = canonical_json_sha256(numerical)
    checkpoint_payload = {
        "schema": PROGRESSIVE_RESUME_SCHEMA,
        "identity": asdict(identity),
        "stage": stage,
        "step": len(result.training_events),
        "student_state_dict": result.student_block.state_dict(),
        "optimizer_state_dict": result.optimizer.state_dict(),
        "rng_state": _capture_rng_state(),
        "result_schema": PROGRESSIVE_RESULT_SCHEMA,
        "result_payload_sha256": payload_sha,
    }

    checkpoint_temp: Path | None = None
    result_temp: Path | None = None
    checkpoint_published = False
    result_published = False
    try:
        checkpoint_temp = _new_temp(checkpoint_path)
        with checkpoint_temp.open("wb") as handle:
            torch.save(checkpoint_payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        checkpoint_sha = sha256_file(checkpoint_temp)
        _publish_no_replace(checkpoint_temp, checkpoint_path)
        checkpoint_published = True

        receipt_payload = {
            "schema": PROGRESSIVE_RESULT_SCHEMA,
            "identity": asdict(identity),
            "stage": stage,
            "step": len(result.training_events),
            "checkpoint_filename": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha,
            "result_payload_sha256": payload_sha,
            "result": numerical,
        }
        result_temp = _new_temp(result_path)
        encoded = (json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        with result_temp.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        result_sha = sha256_file(result_temp)
        _publish_no_replace(result_temp, result_path)
        result_published = True
    except BaseException:
        if result_published and result_temp is not None:
            _unlink_if_same(result_path, result_temp)
        if checkpoint_published and checkpoint_temp is not None:
            _unlink_if_same(checkpoint_path, checkpoint_temp)
        raise
    finally:
        _unlink(result_temp)
        _unlink(checkpoint_temp)

    return ProgressiveArtifactReceipt(
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha,
        result_path=str(result_path),
        result_sha256=result_sha,
        result_payload_sha256=payload_sha,
        capture_set_identity_sha256=identity.capture_set_identity_sha256,
    )


def load_progressive_resume_checkpoint(
    path: str | Path,
    *,
    student_block: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_identity: ProgressiveRunIdentity,
    expected_stage: str,
    expected_sha256: str | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a trusted-local progressive q/R/v checkpoint with strict identity checks."""
    if expected_stage not in _STAGE_NAMES:
        raise ValueError(f"unsupported progressive expected_stage {expected_stage!r}")
    path = Path(path)
    if expected_sha256 is not None:
        expected_sha256 = _require_sha256("progressive checkpoint SHA-256", expected_sha256)
        if sha256_file(path).lower() != expected_sha256.lower():
            raise RuntimeError("progressive checkpoint SHA-256 does not match immutable evidence")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != PROGRESSIVE_RESUME_SCHEMA:
        raise RuntimeError("not a current progressive resume checkpoint")
    if payload.get("identity") != asdict(expected_identity):
        raise RuntimeError("progressive resume identity does not match requested sweep/prefix")
    if payload.get("stage") != expected_stage:
        raise RuntimeError("progressive resume stage does not match requested stage")
    set_stage = __import__("minimax_h3_keyless.pilot", fromlist=["set_pilot_block_stage"]).set_pilot_block_stage
    set_stage(student_block, expected_stage)
    validate_optimizer_matches_trainable(student_block, optimizer)
    student_block.load_state_dict(payload["student_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    validate_optimizer_matches_trainable(student_block, optimizer)
    if restore_rng:
        restore_progressive_rng_state(payload["rng_state"])
    return {
        "stage": expected_stage,
        "step": int(payload["step"]),
        "result_payload_sha256": _require_sha256(
            "progressive result payload SHA-256", payload["result_payload_sha256"]
        ),
        "checkpoint_sha256": sha256_file(path),
    }
