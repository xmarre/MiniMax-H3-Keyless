from __future__ import annotations

import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from .pilot_campaign import (
    PilotRunIdentity,
    _require_sha256,
    canonical_json_sha256,
    save_pilot_resume_checkpoint,
    write_json_atomic,
)


STAGE_A_RESULT_SCHEMA = "minimax_h3_keyless_stage_a_block_result_v3"
_SAFE_STEM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class StageAArtifactRequest:
    output_dir: str
    run_id: str
    code_commit: str
    experiment_context_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.output_dir.strip():
            raise ValueError("Stage-A artifact output_dir must be non-empty")
        if not _SAFE_STEM.fullmatch(self.run_id):
            raise ValueError(
                "Stage-A artifact run_id must use only letters, digits, '.', '_' or '-' "
                "and may not contain path separators"
            )
        if not self.code_commit.strip():
            raise ValueError("Stage-A artifact code_commit must be non-empty")
        if self.experiment_context_sha256 is not None:
            _require_sha256(
                "Stage-A experiment context SHA-256", self.experiment_context_sha256
            )

    def paths(self, block_index: int, stage: str) -> tuple[Path, Path]:
        root = Path(self.output_dir)
        stem = f"{self.run_id}.block{int(block_index):02d}.{stage}"
        return root / f"{stem}.resume.pt", root / f"{stem}.result.json"

    def assert_available(self, block_index: int, stage: str) -> None:
        checkpoint, result = self.paths(block_index, stage)
        occupied = [str(path) for path in (checkpoint, result) if path.exists()]
        if occupied:
            raise FileExistsError(
                "Stage-A evidence artifacts are immutable; choose a new run_id instead of "
                f"overwriting: {occupied}"
            )


@dataclass(frozen=True)
class StageAArtifactReceipt:
    checkpoint_path: str
    checkpoint_sha256: str
    result_path: str
    result_sha256: str


def _temporary_publish_path(final_path: Path) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{final_path.name}.",
        suffix=".publish",
        dir=final_path.parent,
    )
    os.close(fd)
    Path(name).unlink()
    return Path(name)


def _publish_no_replace(source: Path, destination: Path) -> None:
    """Atomically add a final hard link without replacing immutable evidence."""
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            "Stage-A evidence artifacts are immutable; choose a new run_id instead of "
            f"overwriting: {destination}"
        ) from exc


def _unlink_if_same_file(path: Path, source: Path) -> None:
    """Rollback only a final link that still names this transaction's source inode."""
    try:
        if path.exists() and source.exists() and os.path.samefile(path, source):
            path.unlink()
    except FileNotFoundError:
        pass


def _unlink_if_exists(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def persist_stage_a_block_artifacts(
    request: StageAArtifactRequest,
    *,
    student_block: nn.Module,
    optimizer: torch.optim.Optimizer,
    identity: PilotRunIdentity,
    stage: str,
    step: int,
    result_payload: Mapping[str, Any],
) -> StageAArtifactReceipt:
    """Persist one immutable Stage-A checkpoint/result pair without clobber races.

    The generic checkpoint and JSON writers first write transaction-private temporary
    paths. Final names are then published with same-directory hard links, whose create
    operation fails atomically when another writer already owns that immutable identity.
    The temporary links remain alive until both final names exist, so rollback removes a
    final path only when it still refers to this transaction's inode. Existing evidence
    is never overwritten or deleted.

    The resume checkpoint retains optimizer/RNG state through the trusted-local resume
    format. The deterministic numerical-result payload hash and, when supplied, the full
    experiment-context identity are embedded in that checkpoint before its full-file hash
    is written into the JSON result. This binds resumable evidence without a hash cycle.
    """
    if identity.run_id != request.run_id:
        raise ValueError("Stage-A artifact request run_id does not match pilot identity")
    if identity.code_commit != request.code_commit:
        raise ValueError("Stage-A artifact request code_commit does not match pilot identity")
    checkpoint_path, result_path = request.paths(identity.block_index, stage)
    request.assert_available(identity.block_index, stage)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(result_payload)
    payload_sha = canonical_json_sha256(payload)
    checkpoint_temp: Path | None = None
    result_temp: Path | None = None
    checkpoint_published = False
    result_published = False
    try:
        checkpoint_temp = _temporary_publish_path(checkpoint_path)
        checkpoint_sha = save_pilot_resume_checkpoint(
            checkpoint_temp,
            student_block=student_block,
            optimizer=optimizer,
            identity=identity,
            stage=stage,
            step=int(step),
            extra={
                "stage_a_result_schema": STAGE_A_RESULT_SCHEMA,
                "stage_a_result_payload_sha256": payload_sha,
                "stage_a_experiment_context_sha256": request.experiment_context_sha256,
            },
        )
        _publish_no_replace(checkpoint_temp, checkpoint_path)
        checkpoint_published = True

        receipt = {
            "schema": STAGE_A_RESULT_SCHEMA,
            "identity": asdict(identity),
            "stage": stage,
            "step": int(step),
            "experiment_context_sha256": request.experiment_context_sha256,
            "checkpoint_filename": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha,
            "result_payload_sha256": payload_sha,
            "result": payload,
        }
        result_temp = _temporary_publish_path(result_path)
        result_sha = write_json_atomic(result_temp, receipt)
        _publish_no_replace(result_temp, result_path)
        result_published = True
    except BaseException:
        if result_published and result_temp is not None:
            _unlink_if_same_file(result_path, result_temp)
        if checkpoint_published and checkpoint_temp is not None:
            _unlink_if_same_file(checkpoint_path, checkpoint_temp)
        raise
    finally:
        _unlink_if_exists(result_temp)
        _unlink_if_exists(checkpoint_temp)

    return StageAArtifactReceipt(
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha,
        result_path=str(result_path),
        result_sha256=result_sha,
    )
