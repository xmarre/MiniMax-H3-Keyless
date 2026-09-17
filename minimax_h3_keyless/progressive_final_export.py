from __future__ import annotations

from pathlib import Path

import torch

from .checkpoint import read_safetensors_signatures, validate_deploy_checkpoint
from .export import ExportResult, canonical_metadata, export_deploy_bf16
from .live_capture import discover_clean_git_revision
from .progressive_authorization import load_progressive_prefix_manifest
from .progressive_restore import restore_progressive_model_prefix
from .teacher import load_pinned_bf16_teacher
from .teacher_compat import validate_deploy_artifact_against_teacher


def export_completed_progressive_bf16(
    *,
    teacher_path: str | Path,
    prefix_manifest_path: str | Path,
    artifact_dir: str | Path,
    output_path: str | Path,
    manifest_path: str | Path | None = None,
    command: str | None = None,
) -> ExportResult:
    """Export a completed Stage-B prefix as canonical ``h3_keyless_core50_v1`` BF16.

    Stage-B accepted checkpoints are stored block-by-block in training q/R/v form, while
    accepted-prefix reconstruction installs their already-folded deploy QV attentions. This
    function therefore reconstructs the full 0..49 accepted prefix and sends that deploy
    mapping directly through the canonical exporter; it never invents an aggregate q/R/v
    checkpoint or folds an already-folded QV projection a second time.

    The training-source revision and export-source revision are intentionally distinct
    provenance fields. ``prefix.code_commit`` records the frozen Stage-B source. The current
    repository only has to be clean; its discovered revision is recorded as ``export_commit``
    and may differ when export hardening occurs after the sweep. The complete prefix payload
    and prefix-manifest SHA-256 bind every accepted block checkpoint/result to the release.
    """

    prefix, prefix_manifest_sha256 = load_progressive_prefix_manifest(prefix_manifest_path)
    if not prefix.complete:
        raise RuntimeError(
            "canonical BF16 export requires a complete accepted progressive prefix 0..49; "
            f"accepted={len(prefix.accepted)}"
        )

    export_commit = discover_clean_git_revision(
        Path(__file__).resolve().parents[1],
        label="MiniMax-H3-Keyless canonical BF16 export source",
    )
    teacher = load_pinned_bf16_teacher(teacher_path)
    model = teacher.diffusion_model

    # Reconstruct on CPU so accepted attention materialization and final safetensors writing
    # do not require a second full GPU-resident H3 model. The immutable accepted artifacts,
    # not this transient model object, remain the authority for the completed Stage-B sweep.
    model.to(torch.device("cpu"))
    restore_progressive_model_prefix(
        model,
        prefix,
        output_dir=artifact_dir,
    )
    deploy_state = model.state_dict()

    metadata = canonical_metadata(
        training_run=f"progressive:{prefix.sweep_id}:{prefix.identity_sha256}",
        export_commit=export_commit,
    )
    manifest_extra = {
        "progressive_training": {
            "prefix_manifest_sha256": prefix_manifest_sha256,
            "prefix": prefix.identity_payload(),
        }
    }
    result = export_deploy_bf16(
        deploy_state,
        output_path,
        metadata=metadata,
        manifest_path=manifest_path,
        command=command,
        manifest_extra=manifest_extra,
        teacher_path=teacher_path,
        refuse_replace=True,
    )

    # Re-open the bytes that will become the BF16 parent of any INT8 derivative. In-memory
    # validation alone is insufficient because serialization or path mixups must fail before
    # a canonical artifact is reported to the caller.
    signatures, written_metadata = read_safetensors_signatures(result.artifact_path)
    validate_deploy_checkpoint(signatures, written_metadata)
    validate_deploy_artifact_against_teacher(teacher_path, result.artifact_path)
    return result
