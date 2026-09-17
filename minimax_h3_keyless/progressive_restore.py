from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from .attention import KeylessAttentionDeploy, KeylessAttentionTrain
from .checkpoint import sha256_file
from .export import fold_query_route_weight
from .pilot import _native_attention_facts
from .pilot_campaign import PILOT_LS_LAMBDAS, canonical_json_sha256
from .progressive import ProgressiveAcceptedBlock, ProgressivePrefix, validate_progressive_model_prefix
from .progressive_artifacts import (
    PROGRESSIVE_RESULT_SCHEMA,
    PROGRESSIVE_RESUME_SCHEMA,
    ProgressiveRunIdentity,
)


StudentBuilder = Callable[[nn.Module, int], nn.Module]
_SELECTION_SPLIT = "train_complete_cases"


@dataclass(frozen=True)
class _LoadedProgressiveResult:
    identity: ProgressiveRunIdentity
    result_payload_sha256: str
    checkpoint_filename: str
    checkpoint_sha256: str
    step: int


def _build_training_attention_from_native(
    native_block: nn.Module,
    block_index: int,
) -> KeylessAttentionTrain:
    native_attention = getattr(native_block, "attn", None)
    if native_attention is None:
        raise RuntimeError("progressive restore native block has no attention module")
    hidden, heads, head_dim, eps, gate = _native_attention_facts(native_attention)
    qkv_weight = native_attention.qkv_proj.weight
    if qkv_weight.dtype != torch.bfloat16:
        raise RuntimeError(
            f"progressive restore requires the pinned BF16 native teacher, got {qkv_weight.dtype}"
        )
    return KeylessAttentionTrain(
        hidden,
        heads,
        head_dim,
        eps,
        gate_compress=gate,
        block_index=int(block_index),
        dtype=qkv_weight.dtype,
        device=qkv_weight.device,
        operations=None,
    )


def _default_student_builder(native_block: nn.Module, block_index: int) -> nn.Module:
    """Build only the training attention needed to reconstruct one accepted block.

    Progressive checkpoints retain the full copied block state for resume/audit purposes,
    but accepted restore only needs the attention state. Keeping the native block in place
    avoids duplicating its large frozen MLP/AdaLN tensors while reconstructing a prefix.
    """
    shell = nn.Module()
    shell.add_module(
        "attn",
        _build_training_attention_from_native(native_block, block_index),
    )
    return shell


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
    expected_checkpoint_filename: str,
) -> _LoadedProgressiveResult:
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
    step = value.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise RuntimeError("accepted progressive result step must be a non-negative integer")
    checkpoint_filename = value.get("checkpoint_filename")
    if (
        not isinstance(checkpoint_filename, str)
        or Path(checkpoint_filename).name != checkpoint_filename
        or checkpoint_filename != expected_checkpoint_filename
    ):
        raise RuntimeError("accepted progressive result references the wrong checkpoint filename")
    checkpoint_sha = value.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or checkpoint_sha.lower() != accepted.checkpoint_sha256.lower():
        raise RuntimeError("accepted progressive result checkpoint hash differs from prefix record")

    result_payload = value.get("result")
    if not isinstance(result_payload, dict):
        raise RuntimeError("accepted progressive result is missing numerical payload")
    payload_sha = canonical_json_sha256(result_payload)
    if str(value.get("result_payload_sha256", "")).lower() != payload_sha:
        raise RuntimeError("accepted progressive result payload hash does not recompute")
    if result_payload.get("block_index") != accepted.block_index:
        raise RuntimeError("accepted progressive numerical payload names the wrong block")
    if result_payload.get("prefix_identity_sha256") != prefix_before.identity_sha256:
        raise RuntimeError("accepted progressive numerical payload names the wrong prior prefix")
    if result_payload.get("final_stage") != accepted.final_stage:
        raise RuntimeError("accepted progressive numerical payload names the wrong final stage")
    if result_payload.get("selection_split") != _SELECTION_SPLIT:
        raise RuntimeError(
            "accepted progressive result does not record train-only initialization selection"
        )
    if result_payload.get("selected_route_mode") != identity.route_mode:
        raise RuntimeError("accepted progressive numerical payload route mode differs from run identity")
    try:
        selected_lambda = float(result_payload.get("selected_lambda_relative"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("accepted progressive numerical payload has an invalid route lambda") from exc
    if selected_lambda != float(identity.lambda_relative):
        raise RuntimeError("accepted progressive numerical payload route lambda differs from run identity")
    try:
        baseline_lambda = float(result_payload.get("least_squares_baseline_lambda_relative"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "accepted progressive numerical payload has an invalid train-selected LS baseline lambda"
        ) from exc
    if baseline_lambda not in PILOT_LS_LAMBDAS:
        raise RuntimeError(
            "accepted progressive numerical payload LS baseline lambda is outside the fixed grid"
        )
    gate = result_payload.get("gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise RuntimeError("accepted progressive result does not record a passed numerical gate")
    if gate.get("block_index") != accepted.block_index:
        raise RuntimeError("accepted progressive gate names the wrong block")

    return _LoadedProgressiveResult(
        identity=identity,
        result_payload_sha256=payload_sha,
        checkpoint_filename=checkpoint_filename,
        checkpoint_sha256=checkpoint_sha.lower(),
        step=step,
    )


def _load_student_state(
    path: Path,
    *,
    expected_sha256: str,
    accepted: ProgressiveAcceptedBlock,
    identity: ProgressiveRunIdentity,
    result_payload_sha256: str,
    expected_step: int,
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
    if value.get("step") != expected_step:
        raise RuntimeError("accepted progressive checkpoint step differs from result evidence")
    if str(value.get("result_payload_sha256", "")).lower() != result_payload_sha256.lower():
        raise RuntimeError("accepted progressive checkpoint is not bound to its result payload")
    state = value.get("student_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("accepted progressive checkpoint is missing student state")
    if not all(isinstance(name, str) and torch.is_tensor(tensor) for name, tensor in state.items()):
        raise RuntimeError("accepted progressive student state contains invalid entries")
    return state


def _attention_state_from_full_checkpoint(
    state: dict[str, torch.Tensor],
    *,
    native_block: nn.Module,
    training_attention: KeylessAttentionTrain,
) -> dict[str, torch.Tensor]:
    """Validate the full resume key topology and return only accepted attention tensors.

    Non-attention tensors are retained in resume artifacts for audit/resume, but they are
    frozen by design and are never installed during accepted-prefix reconstruction. The
    live pinned teacher remains the source of truth for those tensors.
    """
    expected_attention = set(training_attention.state_dict())
    expected_non_attention = {
        name for name in native_block.state_dict()
        if not name.startswith("attn.")
    }
    expected_full = {
        *(f"attn.{name}" for name in expected_attention),
        *expected_non_attention,
    }
    actual_full = set(state)
    if actual_full != expected_full:
        missing = sorted(expected_full - actual_full)
        unexpected = sorted(actual_full - expected_full)
        raise RuntimeError(
            "accepted progressive checkpoint state keys do not match the canonical "
            f"training block: missing={missing}, unexpected={unexpected}"
        )
    return {
        name: state[f"attn.{name}"]
        for name in sorted(expected_attention)
    }


def _fold_training_attention(training: KeylessAttentionTrain) -> KeylessAttentionDeploy:
    """Fold one accepted q/R/v attention without copying the frozen H3 block."""
    q_weight = getattr(training.q_proj, "weight", None)
    v_weight = getattr(training.v_proj, "weight", None)
    route_weight = getattr(training.query_route, "weight", None)
    if not all(torch.is_tensor(value) for value in (q_weight, v_weight, route_weight)):
        raise RuntimeError("progressive restore requires materialized q/R/v weights")
    assert q_weight is not None and v_weight is not None and route_weight is not None
    if any(getattr(value, "is_meta", False) for value in (q_weight, v_weight, route_weight)):
        raise RuntimeError("progressive restore cannot fold meta-device q/R/v weights")

    deploy = KeylessAttentionDeploy(
        training.hidden,
        training.heads,
        training.head_dim,
        float(training.q_norm.eps),
        gate_compress=training.to_gate_compress is not None,
        block_index=training.block_index,
        dtype=q_weight.dtype,
        device=q_weight.device,
        operations=None,
    )
    q_eff = fold_query_route_weight(
        q_weight.detach(),
        route_weight.detach(),
        output_dtype=q_weight.dtype,
    )
    with torch.no_grad():
        deploy.qv_proj.weight.copy_(torch.cat((q_eff, v_weight.detach()), dim=0))
        deploy.q_norm.weight.copy_(training.q_norm.weight.detach())
        deploy.route_norm.weight.copy_(training.route_norm.weight.detach())
        deploy.out_proj.weight.copy_(training.out_proj.weight.detach())
        if training.to_gate_compress is not None:
            if deploy.to_gate_compress is None:
                raise RuntimeError("progressive restore fold lost gate-compress topology")
            deploy.to_gate_compress.weight.copy_(training.to_gate_compress.weight.detach())
    deploy.eval()
    for parameter in deploy.parameters():
        parameter.requires_grad_(False)
    return deploy


def _load_folded_attention_for_accepted_block(
    native_block: nn.Module,
    prefix: ProgressivePrefix,
    accepted: ProgressiveAcceptedBlock,
    *,
    offset: int,
    output_dir: str | Path,
    student_builder: StudentBuilder,
) -> KeylessAttentionDeploy:
    prior = _prefix_before(prefix, offset)
    checkpoint_path, result_path = _artifact_paths(output_dir, prefix, accepted)
    loaded_result = _load_result(
        result_path,
        expected_sha256=accepted.result_sha256,
        accepted=accepted,
        prefix_before=prior,
        expected_checkpoint_filename=checkpoint_path.name,
    )
    state = _load_student_state(
        checkpoint_path,
        expected_sha256=accepted.checkpoint_sha256,
        accepted=accepted,
        identity=loaded_result.identity,
        result_payload_sha256=loaded_result.result_payload_sha256,
        expected_step=loaded_result.step,
    )
    student = student_builder(native_block, offset)
    training_attention = getattr(student, "attn", None)
    if not isinstance(training_attention, KeylessAttentionTrain):
        raise RuntimeError(
            "progressive restore student builder did not install KeylessAttentionTrain"
        )
    if int(getattr(training_attention, "block_index", -1)) != offset:
        raise RuntimeError(
            "progressive restore student builder installed the wrong block_index"
        )
    attention_state = _attention_state_from_full_checkpoint(
        state,
        native_block=native_block,
        training_attention=training_attention,
    )
    training_attention.load_state_dict(attention_state, strict=True)
    return _fold_training_attention(training_attention)


def load_progressive_deploy_attentions(
    model: nn.Module,
    prefix: ProgressivePrefix,
    *,
    output_dir: str | Path,
    student_builder: StudentBuilder = _default_student_builder,
) -> tuple[KeylessAttentionDeploy, ...]:
    """Materialize accepted deploy attentions without mutating the native teacher model.

    This is the safe boundary used by Comfy ModelPatcher object overlays: the shared base
    teacher remains native QKV, while each returned attention is independently reconstructed
    from the immutable accepted result/checkpoint pair.
    """
    native_prefix = _prefix_before(prefix, 0)
    validate_progressive_model_prefix(model, native_prefix)
    blocks = getattr(model, "blocks", None)
    assert blocks is not None

    out: list[KeylessAttentionDeploy] = []
    for offset, accepted in enumerate(prefix.accepted):
        if accepted.block_index != offset:
            raise RuntimeError("accepted progressive prefix is not contiguous early-to-late")
        out.append(
            _load_folded_attention_for_accepted_block(
                blocks[offset],
                prefix,
                accepted,
                offset=offset,
                output_dir=output_dir,
                student_builder=student_builder,
            )
        )
    return tuple(out)


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
    pass, and the v2 result must record train-only initialization selection before the
    training-form q/R/v attention state is loaded strictly and folded to deploy QV form.

    Only ``block.attn`` is replaced. Frozen MLP/AdaLN/non-attention tensors remain the
    original pinned-teacher objects, avoiding a full H3 block deepcopy for every restored
    prefix element. A failure leaves already-restored earlier attentions in place and does
    not modify the failing or later native attention.
    """

    native_prefix = _prefix_before(prefix, 0)
    validate_progressive_model_prefix(model, native_prefix)
    blocks = getattr(model, "blocks", None)
    assert blocks is not None

    for offset, accepted in enumerate(prefix.accepted):
        if accepted.block_index != offset:
            raise RuntimeError("accepted progressive prefix is not contiguous early-to-late")
        native_block = blocks[offset]
        folded_attention = _load_folded_attention_for_accepted_block(
            native_block,
            prefix,
            accepted,
            offset=offset,
            output_dir=output_dir,
            student_builder=student_builder,
        )

        original_attention = native_block.attn
        try:
            native_block.attn = folded_attention
            validate_progressive_model_prefix(model, _prefix_before(prefix, offset + 1))
        except BaseException:
            native_block.attn = original_attention
            raise

    validate_progressive_model_prefix(model, prefix)
    return model
