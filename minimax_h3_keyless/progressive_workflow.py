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


@dataclass(frozen=True)
class _PreparedAcceptance:
    current_path: Path
    current_sha256: str
    target_block: int
    next_manifest: Path


def _next_prefix_path(output_dir: str | Path, prefix: ProgressivePrefix) -> Path:
    accepted_after_step = len(prefix.accepted) + 1
    return Path(output_dir) / f"{prefix.sweep_id}.prefix-{accepted_after_step:02d}.json"


def _prepare_acceptance(
    model: nn.Module,
    prefix: ProgressivePrefix,
    captures: ProgressiveBlockCaptureSet,
    result: ProgressiveBlockTrainingResult,
    *,
    current_prefix_manifest_path: str | Path,
    current_prefix_manifest_sha256: str,
    output_dir: str | Path,
) -> _PreparedAcceptance:
    """Validate immutable/live state before any candidate persistence or model mutation."""

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
    return _PreparedAcceptance(
        current_path=current_path,
        current_sha256=actual_current_sha,
        target_block=target,
        next_manifest=next_manifest,
    )


def _accept_prepared_progressive_block(
    model: nn.Module,
    prefix: ProgressivePrefix,
    captures: ProgressiveBlockCaptureSet,
    result: ProgressiveBlockTrainingResult,
    artifact: ProgressiveArtifactReceipt,
    prepared: _PreparedAcceptance,
    *,
    gate_manifest: Mapping[str, object],
    fold_atol: float,
    fold_rtol: float,
) -> ProgressiveAcceptedStep:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise RuntimeError("progressive live model lost its core blocks before commit")
    previous_block = blocks[prepared.target_block]
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
            prepared.next_manifest,
            advanced,
            previous_manifest_sha256=prepared.current_sha256,
        )
        reloaded, reloaded_sha = load_progressive_prefix_manifest(
            prepared.next_manifest,
            expected_previous_manifest_sha256=prepared.current_sha256,
        )
        if reloaded != advanced or reloaded_sha.lower() != manifest_sha.lower():
            raise RuntimeError("new progressive prefix manifest changed during validation")
    except BaseException:
        # Persisted candidate evidence is immutable but not accepted until the next prefix
        # manifest publishes successfully. Restore the exact previous live target object.
        blocks[prepared.target_block] = previous_block
        validate_progressive_model_prefix(model, prefix)
        raise

    assert advanced is not None
    return ProgressiveAcceptedStep(
        previous_prefix_manifest_path=str(prepared.current_path),
        previous_prefix_manifest_sha256=prepared.current_sha256,
        artifact=artifact,
        prefix=advanced,
        prefix_manifest_path=str(prepared.next_manifest),
        prefix_manifest_sha256=manifest_sha,
    )


def accept_persisted_progressive_block(
    model: nn.Module,
    prefix: ProgressivePrefix,
    captures: ProgressiveBlockCaptureSet,
    result: ProgressiveBlockTrainingResult,
    artifact: ProgressiveArtifactReceipt,
    *,
    current_prefix_manifest_path: str | Path,
    current_prefix_manifest_sha256: str,
    output_dir: str | Path,
    gate_manifest: Mapping[str, object],
    fold_atol: float,
    fold_rtol: float,
) -> ProgressiveAcceptedStep:
    """Accept already-persisted candidate evidence and publish the next prefix atomically.

    This split is required by the production runner: failed numerical candidates are still
    immutable evidence, but they must not attempt model/prefix acceptance. A passed candidate
    can therefore be persisted first, then accepted without writing the same artifact twice.
    """

    prepared = _prepare_acceptance(
        model,
        prefix,
        captures,
        result,
        current_prefix_manifest_path=current_prefix_manifest_path,
        current_prefix_manifest_sha256=current_prefix_manifest_sha256,
        output_dir=output_dir,
    )
    return _accept_prepared_progressive_block(
        model,
        prefix,
        captures,
        result,
        artifact,
        prepared,
        gate_manifest=gate_manifest,
        fold_atol=fold_atol,
        fold_rtol=fold_rtol,
    )


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
    """Persist and accept one Stage-B block without exposing a half-accepted live state.

    The current prefix manifest and capture bytes are validated before candidate evidence is
    written. Training evidence is persisted before model mutation. The folded block is then
    installed only through the strict acceptance gate; if publishing the next immutable prefix
    manifest loses a race or otherwise fails, the live model is rolled back to its exact prior
    native target block. Persisted candidate evidence may remain after such a rollback, but it
    is not part of the accepted prefix and cannot silently advance the sweep.
    """

    prepared = _prepare_acceptance(
        model,
        prefix,
        captures,
        result,
        current_prefix_manifest_path=current_prefix_manifest_path,
        current_prefix_manifest_sha256=current_prefix_manifest_sha256,
        output_dir=output_dir,
    )
    artifact = persist_progressive_block_artifacts(
        output_dir,
        prefix=prefix,
        captures=captures,
        result=result,
    )
    return _accept_prepared_progressive_block(
        model,
        prefix,
        captures,
        result,
        artifact,
        prepared,
        gate_manifest=gate_manifest,
        fold_atol=fold_atol,
        fold_rtol=fold_rtol,
    )
