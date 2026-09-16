from __future__ import annotations

import re
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


STAGE_A_RESULT_SCHEMA = "minimax_h3_keyless_stage_a_block_result_v1"
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
    """Atomically persist one Stage-A block result without overwriting prior evidence.

    The resume checkpoint retains optimizer/RNG state through the existing trusted-local
    resume format. The deterministic numerical-result payload hash and, when supplied,
    the full experiment-context identity are embedded in that checkpoint before its
    full-file hash is written into the JSON result. This binds resumable evidence without
    creating a hash cycle. If writing the JSON result fails, the newly-created checkpoint
    is removed so a partial transaction is not mistaken for a complete pilot result.
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
    checkpoint_created = False
    try:
        checkpoint_sha = save_pilot_resume_checkpoint(
            checkpoint_path,
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
        checkpoint_created = True
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
        result_sha = write_json_atomic(result_path, receipt)
    except BaseException:
        if checkpoint_created:
            try:
                checkpoint_path.unlink()
            except FileNotFoundError:
                pass
        try:
            result_path.unlink()
        except FileNotFoundError:
            pass
        raise

    return StageAArtifactReceipt(
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha,
        result_path=str(result_path),
        result_sha256=result_sha,
    )
