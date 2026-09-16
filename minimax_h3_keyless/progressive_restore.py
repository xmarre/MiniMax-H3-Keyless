from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from .attention import KeylessAttentionTrain
from .checkpoint import sha256_file
from .pilot import build_training_student_block, set_pilot_block_stage
from .pilot_campaign import canonical_json_sha256
from .progressive import ProgressiveAcceptedBlock, ProgressivePrefix, validate_progressive_model_prefix
from .progressive_acceptance import fold_progressive_training_block
from .progressive_artifacts import (
    PROGRESSIVE_RESULT_SCHEMA,
    PROGRESSIVE_RESUME_SCHEMA,
    ProgressiveRunIdentity,
)


StudentBuilder = Callable[[nn.Module, int], nn.Module]


def _default_student_builder(native_block: nn.Module, block_index: int) -> nn.Module:
    student, _ = build_training_student_block(
        native_block,
        block_index=block_index,
        route_mode="identity",
        lambda_relative=0.0,
    )
    return student


def _prefix_before(prefix: ProgressivePrefix, count: int) -> ProgressivePrefix:
    return ProgressivePrefix(
        sweep_id=prefix.sweep_id,
        code_commit=prefix.code_commit,
        stage_a_campaign_sha256=prefix.stage_a_campaign_sha256,
        dataset_manifest_sha256=prefix.dataset_manifest_sha256,
        gate_manifest_sha256=prefix.gate_manifest_sha256,
        accepted=prefix.accepted[:count],
    )


def _artifact_paths(
    output_dir: str | Path,
    prefix: ProgressivePrefix,
    accepted: ProgressiveAcceptedBlock,
) -> tuple[Path, Path]:
    root = Path(output_dir)
    stem = f"{prefix.sweep_id}.block{accepted.block_index:02d}.{accepted.final_stage}"
    return root / f"{stem}.resume.pt", root / f"{stem}.result.json"


def _load_result(
    path: Path,
    *,
    expected_sha256: str,
    accepted: ProgressiveAcceptedBlock,
    prefix_before: ProgressivePrefix,
) -> tuple[ProgressiveRunIdentity, str]:
    actual_sha = sha256_file(path)
    if actual_sha.lower() != accepted.result_sha256.lower() or actual_sha.lower() != expected_sha256.lower():
        raise RuntimeError("accepted progressive result SHA-256 does not match immutable prefix")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid accepted progressive result JSON: {path}") from exc
    expected_keys = {
        "schema",
        "identity",
        "stage",
        "step",
        "checkpoint_filename",
        "checkpoint_sha256",
        "result_payload_sha256",
        "result",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RuntimeError("accepted progressive result has an incompatible schema")
    if value.get("schema") != PROGRESSIVE_RESULT_SCHEMA:
        raise RuntimeError("accepted progressive result schema version is unsupported")
    raw_identity = value.get("identity")
    if not isinstance(raw_identity, dict):
        raise RuntimeError("accepted progressive result is missing run identity")
    try:
        identity = ProgressiveRunIdentity(**raw_identity)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("accepted progressive result carries an invalid run identity") from exc
    expected_identity_fields = {
        "sweep_id": prefix_before.sweep_id,
        "code_commit": prefix_before.code_commit,
        "stage_a_campaign_sha256": prefix_before.stage_a_campaign_sha256,
        "dataset_manifest_sha256": prefix_before.dataset_manifest_sha256,
        "gate_manifest_sha256": prefix_before.gate_manifest_sha256,
        "prefix_identity_sha256": prefix_before.identity_sha256,
        "block_index": accepted.block_index,
    }
    mismatches = [
        name for name, expected in expected_identity_fields.items()
        if getattr(identity, name) != expected
    ]
    if mismatches:
        raise RuntimeError(
            "accepted progressive result identity does not extend the expected prior prefix: "
            + ", ".join(mismatches)
        )
    if value.get("stage") != accepted.final_stage:
        raise RuntimeError("accepted progressive result stage differs from prefix record")
    result_payload = value.get("result")
    if not isinstance(result_payload, dict):
        raise RuntimeError("accepted progressive result is missing numerical payload")
    payload_sha = canonical_json_sha256(result_payload)
    if str(value.get("result_payload_sha256", "")).lower() != payload_sha:
        raise RuntimeError("accepted progressive result payload hash does not recompute")
    if result_payload.get("block_index") != accepted.block_index:
        raise RuntimeError("accepted progressive numerical payload names the wrong block")
    gate = result_payload.get("gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise RuntimeError("accepted progressive result does not record a passed numerical gate")
    return identity, payload_sha


def _load_student_state(
    path: Path,
    *,
    expected_sha256: str,
    accepted: ProgressiveAcceptedBlock,
    identity: ProgressiveRunIdentity,
    result_payload_sha256: str,
) -> dict[str, torch.Tensor]:
    actual_sha = sha256_file(path)
    if actual_sha.lower() != accepted.checkpoint_sha256.lower() or actual_sha.lower() != expected_sha256.lower():
        raise RuntimeError("accepted progressive checkpoint SHA-256 does not match immutable prefix")
    # Progressive resume files contain optimizer and Python/NumPy RNG state, so they are
    # trusted-local artifacts rather than tensor-only interchange files. Full-file SHA-256
    # validation occurs before unpickling and the identity is checked immediately after.
    value = torch.load(path, map_location="cpu", weights_only=False)
    expected_keys = {
        "schema",
        "identity",
        "stage",
        "step",
        "student_state_dict",
        "optimizer_state_dict",
        "rng_state",
        "result_schema",
        "result_payload_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RuntimeError("accepted progressive checkpoint has an incompatible schema")
    if value.get("schema") != PROGRESSIVE_RESUME_SCHEMA:
        raise RuntimeError("accepted progressive checkpoint schema version is unsupported")
    if value.get("result_schema") != PROGRESSIVE_RESULT_SCHEMA:
        raise RuntimeError("accepted progressive checkpoint references an unsupported result schema")
    if value.get("identity") != asdict(identity):
        raise RuntimeError("accepted progressive checkpoint identity differs from result identity")
    if value.get("stage") != accepted.final_stage:
        raise RuntimeError("accepted progressive checkpoint stage differs from prefix record")
    if str(value.get("result_payload_sha256", "")).lower() != result_payload_sha256.lower():
        raise RuntimeError("accepted progressive checkpoint is not bound to its result payload")
    state = value.get("student_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("accepted progressive checkpoint is missing student state")
    if not all(isinstance(name, str) and torch.is_tensor(tensor) for name, tensor in state.items()):
        raise RuntimeError("accepted progressive student state contains invalid entries")
    return state


def restore_progressive_model_prefix(
    model: nn.Module,
    prefix: ProgressivePrefix,
    *,
    output_dir: str | Path,
    student_builder: StudentBuilder = _default_student_builder,
) -> nn.Module:
    """Reconstruct an accepted deploy prefix from native BF16 teacher blocks and artifacts.

    The input model must begin as the native core50 teacher. Blocks are restored strictly
    early-to-late. Each accepted result/checkpoint pair is hash-checked against the prefix,
    its run identity must extend the exact prior prefix, its numerical gate must record a
    pass, and the training-form q/R/v state is loaded strictly before being folded to the
    deploy QV representation. A failure leaves already-restored earlier accepted blocks in
    place and does not modify the failing or later native block.
    """

    native_prefix = _prefix_before(prefix, 0)
    validate_progressive_model_prefix(model, native_prefix)
    blocks = getattr(model, "blocks", None)
    assert blocks is not None

    for offset, accepted in enumerate(prefix.accepted):
        prior = _prefix_before(prefix, offset)
        if accepted.block_index != offset:
            raise RuntimeError("accepted progressive prefix is not contiguous early-to-late")
        checkpoint_path, result_path = _artifact_paths(output_dir, prefix, accepted)
        identity, payload_sha = _load_result(
            result_path,
            expected_sha256=accepted.result_sha256,
            accepted=accepted,
            prefix_before=prior,
        )
        if Path(str(json.loads(result_path.read_text(encoding="utf-8"))["checkpoint_filename"])).name != checkpoint_path.name:
            raise RuntimeError("accepted progressive result references the wrong checkpoint filename")
        result_json = json.loads(result_path.read_text(encoding="utf-8"))
        if str(result_json["checkpoint_sha256"]).lower() != accepted.checkpoint_sha256.lower():
            raise RuntimeError("accepted progressive result checkpoint hash differs from prefix record")
        state = _load_student_state(
            checkpoint_path,
            expected_sha256=accepted.checkpoint_sha256,
            accepted=accepted,
            identity=identity,
            result_payload_sha256=payload_sha,
        )

        native_block = blocks[offset]
        student = student_builder(native_block, offset)
        if not isinstance(getattr(student, "attn", None), KeylessAttentionTrain):
            raise RuntimeError("progressive restore student builder did not install KeylessAttentionTrain")
        if int(getattr(student.attn, "block_index", -1)) != offset:
            raise RuntimeError("progressive restore student builder installed the wrong block_index")
        set_pilot_block_stage(student, accepted.final_stage)
        student.load_state_dict(state, strict=True)
        folded = fold_progressive_training_block(student)
        blocks[offset] = folded
        validate_progressive_model_prefix(model, _prefix_before(prefix, offset + 1))

    validate_progressive_model_prefix(model, prefix)
    return model
