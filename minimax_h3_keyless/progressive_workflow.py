from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch.nn as nn

from .progressive import ProgressivePrefix, validate_progressive_model_prefix
from .progressive_acceptance import accept_progressive_block
from .progressive_artifacts import (
    ProgressiveArtifactReceipt,
    persist_progressive_block_artifacts,
)
from .progressive_authorization import (
    load_progressive_prefix_manifest,
    write_progressive_prefix_manifest,
)
from .progressive_capture_integrity import validate_progressive_capture_set_integrity
from .progressive_capture_set import ProgressiveBlockCaptureSet
from .progressive_runner import ProgressiveBlockTrainingResult


@dataclass(frozen=True)
class ProgressiveAcceptedStep:
    previous_prefix_manifest_path: str
    previous_prefix_manifest_sha256: str
    artifact: ProgressiveArtifactReceipt
    prefix: ProgressivePrefix
    prefix_manifest_path: str
    prefix_manifest_sha256: str


def _next_prefix_path(output_dir: str | Path, prefix: ProgressivePrefix) -> Path:
    accepted_after_step = len(prefix.accepted) + 1
    return Path(output_dir) / f"{prefix.sweep_id}.prefix-{accepted_after_step:02d}.json"


def persist_accept_progressive_block(
    model: nn.Module,
    prefix: ProgressivePrefix,
    captures: ProgressiveBlockCaptureSet,
    result: ProgressiveBlockTrainingResult,
    *,
    current_prefix_manifest_path: str | Path,
    current_prefix_manifest_sha256: str,
    output_dir: str | Path,
    gate_manifest: Mapping[str, object],
    fold_atol: float,
    fold_rtol: float,
) -> ProgressiveAcceptedStep:
    """Commit one Stage-B block without exposing a half-accepted live model state.

    The current prefix manifest is reloaded and hash-checked first. Every in-memory replay
    record is then rebound to its immutable capture bytes. Training evidence is persisted
    before model mutation. The folded block is installed only through the strict acceptance
    gate, and the live model is rolled back to its previous native target block if publishing
    the next immutable prefix manifest loses a race or otherwise fails. Persisted candidate
    evidence may remain after such a rollback, but it is not part of the accepted prefix and
    cannot silently advance the sweep.
    """

    current_path = Path(current_prefix_manifest_path)
    loaded_prefix, actual_current_sha = load_progressive_prefix_manifest(current_path)
    if actual_current_sha.lower() != str(current_prefix_manifest_sha256).lower():
        raise RuntimeError("current progressive prefix manifest SHA-256 changed before acceptance")
    if loaded_prefix != prefix:
        raise RuntimeError("live progressive prefix does not match the current immutable manifest")
    validate_progressive_model_prefix(model, prefix)
    validate_progressive_capture_set_integrity(captures)

    target = prefix.next_block
    if target is None:
        raise RuntimeError("progressive sweep is already complete")
    if result.block_index != target:
        raise RuntimeError("progressive training result is not for the current next block")
    next_manifest = _next_prefix_path(output_dir, prefix)
    if next_manifest.exists():
        raise FileExistsError(
            f"next progressive prefix manifest already exists: {next_manifest}"
        )

    artifact = persist_progressive_block_artifacts(
        output_dir,
        prefix=prefix,
        captures=captures,
        result=result,
    )

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise RuntimeError("progressive live model lost its core blocks before commit")
    previous_block = blocks[target]
    advanced: ProgressivePrefix | None = None
    try:
        advanced = accept_progressive_block(
            model,
            prefix,
            captures,
            result,
            artifact,
            gate_manifest=gate_manifest,
            fold_atol=float(fold_atol),
            fold_rtol=float(fold_rtol),
        )
        manifest_sha = write_progressive_prefix_manifest(
            next_manifest,
            advanced,
            previous_manifest_sha256=actual_current_sha,
        )
        reloaded, reloaded_sha = load_progressive_prefix_manifest(
            next_manifest,
            expected_previous_manifest_sha256=actual_current_sha,
        )
        if reloaded != advanced or reloaded_sha.lower() != manifest_sha.lower():
            raise RuntimeError("new progressive prefix manifest changed during validation")
    except BaseException:
        # If the immutable prefix publication did not complete, the persisted block evidence
        # remains an unaccepted experiment. Restore the exact previous live block object.
        blocks[target] = previous_block
        validate_progressive_model_prefix(model, prefix)
        raise

    assert advanced is not None
    return ProgressiveAcceptedStep(
        previous_prefix_manifest_path=str(current_path),
        previous_prefix_manifest_sha256=actual_current_sha,
        artifact=artifact,
        prefix=advanced,
        prefix_manifest_path=str(next_manifest),
        prefix_manifest_sha256=manifest_sha,
    )
