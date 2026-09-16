from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import save_file

from .checkpoint import sha256_file, validate_deploy_checkpoint, validate_training_checkpoint
from .contracts import (
    ARCHITECTURE,
    CHECKPOINT_FORMAT_VERSION,
    CORE_BLOCKS,
    HEAD_DIM,
    HEADS,
    HIDDEN_SIZE,
    INNER_DIM,
    QV_ORDER,
    ROPE_POLICY,
    TARGET_MODEL_REVISION,
    TEACHER_COMPATIBILITY_MARKER,
    TEACHER_SHA256,
    TOKEN_REFINER_BLOCKS,
)


@dataclass(frozen=True)
class ExportResult:
    artifact_path: str
    artifact_sha256: str
    artifact_bytes: int
    manifest_path: str
    manifest_sha256: str
    manifest_identity_sha256: str
    tensor_count: int
    teacher_compatibility_checked: bool


def fold_query_route_weight(
    q_weight: torch.Tensor,
    route_weight: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fold per-head nn.Linear-style route weights into Q in FP32."""
    if q_weight.shape[0] % route_weight.shape[0] != 0:
        raise ValueError("q_weight rows are not divisible by route head count")
    heads = route_weight.shape[0]
    head_dim = route_weight.shape[1]
    if route_weight.shape[2] != head_dim or q_weight.shape[0] != heads * head_dim:
        raise ValueError("incompatible q/route geometry")
    qw = q_weight.reshape(heads, head_dim, q_weight.shape[1]).float()
    rw = route_weight.float()
    folded = torch.bmm(rw, qw).reshape_as(q_weight)
    return folded.to(output_dtype or q_weight.dtype)


def fold_training_state_dict(
    state_dict: Mapping[str, torch.Tensor], *, output_dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    validate_training_checkpoint(state_dict)
    out = {k: v for k, v in state_dict.items()}
    for i in range(CORE_BLOCKS):
        p = f"blocks.{i}.attn."
        q = out.pop(p + "q_proj.weight")
        r = out.pop(p + "query_route.weight")
        v = out.pop(p + "v_proj.weight")
        q_eff = fold_query_route_weight(q, r, output_dtype=output_dtype)
        out[p + "qv_proj.weight"] = torch.cat((q_eff, v.to(output_dtype)), dim=0)
    return out


def canonical_metadata(
    *, training_run: str, export_commit: str, parent_model_sha256: str = TEACHER_SHA256,
    parent_model_revision: str = TARGET_MODEL_REVISION, manifest_sha256: str | None = None,
) -> dict[str, str]:
    md = {
        "architecture": ARCHITECTURE,
        "checkpoint_format_version": str(CHECKPOINT_FORMAT_VERSION),
        "qv_order": QV_ORDER,
        "routing_source": "value",
        "retrieval_source": "raw_projected_value",
        "routing_norm": "rmsnorm",
        "routing_norm_epsilon": "1e-5",
        "rope_policy": ROPE_POLICY,
        "hidden_size": str(HIDDEN_SIZE),
        "heads": str(HEADS),
        "head_dim": str(HEAD_DIM),
        "inner_dim": str(INNER_DIM),
        "core_blocks": str(CORE_BLOCKS),
        "token_refiner": "native_qkv",
        "token_refiner_blocks": str(TOKEN_REFINER_BLOCKS),
        "teacher_model_revision": TARGET_MODEL_REVISION,
        "teacher_model_sha256": TEACHER_SHA256,
        "parent_model_revision": parent_model_revision,
        "parent_model_sha256": parent_model_sha256,
        "training_run": training_run,
        "export_commit": export_commit,
    }
    if manifest_sha256:
        md["manifest_sha256"] = manifest_sha256
    return md


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _tensor_manifest(tensors: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(tensors):
        tensor = tensors[key]
        rows.append(
            {
                "key": key,
                "shape": [int(x) for x in tensor.shape],
                "dtype": str(tensor.dtype),
                "numel": int(tensor.numel()),
                "bytes": int(tensor.numel() * tensor.element_size()),
            }
        )
    return rows


def build_export_manifest_body(
    tensors: Mapping[str, torch.Tensor],
    *,
    artifact_filename: str,
    metadata: Mapping[str, str],
    command: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the deterministic pre-artifact manifest body hashed into checkpoint metadata."""
    clean_metadata = {k: str(v) for k, v in metadata.items() if k != "manifest_sha256"}
    rows = _tensor_manifest(tensors)
    body: dict[str, Any] = {
        "schema": "minimax_h3_keyless_export_manifest_v1",
        "artifact_filename": artifact_filename,
        "metadata": dict(sorted(clean_metadata.items())),
        "tensor_count": len(rows),
        "tensor_payload_bytes": sum(row["bytes"] for row in rows),
        "tensors": rows,
    }
    if command is not None:
        body["command"] = command
    if extra:
        body["extra"] = dict(extra)
    return body


def manifest_identity_sha256(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def export_folded_bf16(
    state_dict: Mapping[str, torch.Tensor],
    output_path: str | Path,
    *,
    metadata: Mapping[str, str],
    manifest_path: str | Path | None = None,
    command: str | None = None,
    manifest_extra: Mapping[str, Any] | None = None,
    teacher_path: str | Path | None = None,
) -> ExportResult:
    """Write folded BF16 plus a deterministic provenance/tensor receipt.

    Passing ``teacher_path`` enables the canonical full-compatibility gate: every key and
    signature must be the exact native-QKV -> Keyless-QV transformation, and every tensor
    outside the design-authorized core-attention trainable set must remain byte-identical
    to the pinned BF16 teacher. Release exports must use this gate.

    The sidecar has two layers to avoid a hash cycle: the deterministic manifest body
    is hashed first and that identity is embedded as ``manifest_sha256`` in the
    safetensors metadata. After the artifact is written, the sidecar receipt adds the
    artifact full-file SHA-256 and byte size. The final sidecar file hash is returned
    separately and is not embedded back into the artifact.
    """
    output_path = Path(output_path)
    if "teacher_compatibility" in metadata:
        raise ValueError(
            "metadata may not set reserved teacher_compatibility evidence; pass teacher_path"
        )
    folded = fold_training_state_dict(state_dict, output_dtype=torch.bfloat16)
    base_metadata = {k: str(v) for k, v in metadata.items() if k != "manifest_sha256"}
    validate_deploy_checkpoint(folded, base_metadata)

    body_extra = dict(manifest_extra or {})
    if "teacher_compatibility" in body_extra:
        raise ValueError("manifest_extra may not override reserved teacher_compatibility evidence")
    compatibility_checked = teacher_path is not None
    if teacher_path is not None:
        from .teacher_compat import validate_deploy_mapping_against_teacher

        report = validate_deploy_mapping_against_teacher(teacher_path, folded)
        if base_metadata.get("parent_model_sha256", "").lower() != report.teacher_sha256.lower():
            raise ValueError(
                "export metadata parent_model_sha256 does not match the teacher used by the compatibility gate"
            )
        if base_metadata.get("parent_model_revision") != TARGET_MODEL_REVISION:
            raise ValueError(
                "export metadata parent_model_revision does not match the pinned teacher revision"
            )
        base_metadata["teacher_compatibility"] = TEACHER_COMPATIBILITY_MARKER
        body_extra["teacher_compatibility"] = {
            **asdict(report),
            "status": "passed",
        }

    body = build_export_manifest_body(
        folded,
        artifact_filename=output_path.name,
        metadata=base_metadata,
        command=command,
        extra=body_extra or None,
    )
    identity = manifest_identity_sha256(body)
    supplied_identity = metadata.get("manifest_sha256")
    if supplied_identity is not None and supplied_identity.lower() != identity:
        raise ValueError(
            "supplied manifest_sha256 does not match the deterministic export manifest body"
        )
    final_metadata = dict(base_metadata)
    final_metadata["manifest_sha256"] = identity
    validate_deploy_checkpoint(folded, final_metadata)

    cpu = {k: v.detach().cpu().contiguous() for k, v in folded.items()}
    save_file(cpu, str(output_path), metadata=final_metadata)
    artifact_sha = sha256_file(output_path)
    artifact_bytes = output_path.stat().st_size

    receipt = dict(body)
    receipt["manifest_sha256"] = identity
    receipt["artifact_sha256"] = artifact_sha
    receipt["artifact_bytes"] = artifact_bytes
    if manifest_path is None:
        manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path = Path(manifest_path)
    manifest_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    sidecar_sha = sha256_file(manifest_path)
    return ExportResult(
        artifact_path=str(output_path),
        artifact_sha256=artifact_sha,
        artifact_bytes=artifact_bytes,
        manifest_path=str(manifest_path),
        manifest_sha256=sidecar_sha,
        manifest_identity_sha256=identity,
        tensor_count=len(folded),
        teacher_compatibility_checked=compatibility_checked,
    )
