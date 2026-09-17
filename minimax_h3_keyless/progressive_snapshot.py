from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from .attention import KeylessAttentionDeploy
from .checkpoint import (
    CheckpointValidationError,
    _require_dtype,
    _require_shape,
    _validate_pruned_adaln,
    read_safetensors_signatures,
    sha256_file,
)
from .contracts import (
    CORE_BLOCKS,
    HEAD_DIM,
    HIDDEN_SIZE,
    INNER_DIM,
    TARGET_MODEL_REVISION,
    TEACHER_SHA256,
    TOKEN_REFINER_BLOCKS,
)
from .pilot import _native_attention_facts
from .pilot_campaign import _require_sha256
from .progressive import ProgressivePrefix, validate_progressive_model_prefix
from .teacher import TEACHER_TENSOR_COUNT


PROGRESSIVE_SNAPSHOT_ARCHITECTURE = "h3_keyless_progressive_core50_v1"
PROGRESSIVE_SNAPSHOT_FORMAT_VERSION = 1
PROGRESSIVE_SNAPSHOT_MANIFEST_SCHEMA = "minimax_h3_keyless_progressive_snapshot_manifest_v1"
PROGRESSIVE_SNAPSHOT_RECEIPT_SCHEMA = "minimax_h3_keyless_progressive_snapshot_receipt_v1"


@dataclass(frozen=True)
class ProgressiveSnapshotResult:
    artifact_path: str
    artifact_sha256: str
    artifact_bytes: int
    manifest_path: str
    manifest_sha256: str
    manifest_identity_sha256: str
    prefix_identity_sha256: str
    accepted_blocks: int
    tensor_count: int


def _require_attention_dtype(
    tensors: Mapping[str, Any], key: str, expected: str = "BF16"
) -> None:
    _require_dtype(tensors, key, expected)


def validate_progressive_snapshot_tensors(
    tensors: Mapping[str, Any],
    prefix: ProgressivePrefix,
) -> None:
    """Validate the exact mixed QV/QKV topology for one accepted Stage-B prefix.

    A progressive snapshot is deliberately *not* a canonical all-Keyless release artifact.
    Blocks before the accepted boundary are folded deploy QV attentions; the boundary and
    all later blocks remain the pinned native QKV teacher. Training q/R/v tensors are
    forbidden everywhere, and the two token-refiner attentions remain native QKV.
    """

    if len(tensors) != TEACHER_TENSOR_COUNT:
        raise CheckpointValidationError(
            "progressive snapshot must preserve the pinned teacher tensor count: "
            f"expected {TEACHER_TENSOR_COUNT}, got {len(tensors)}"
        )
    _validate_pruned_adaln(tensors)
    boundary = len(prefix.accepted)
    training_suffixes = ("q_proj.weight", "v_proj.weight", "query_route.weight")

    for index in range(CORE_BLOCKS):
        p = f"blocks.{index}.attn."
        for suffix in training_suffixes:
            if p + suffix in tensors:
                raise CheckpointValidationError(
                    f"progressive snapshot contains training-only tensor {p + suffix}"
                )
        if index < boundary:
            required = {
                "qv_proj.weight": (2 * INNER_DIM, HIDDEN_SIZE),
                "q_norm.weight": (HEAD_DIM,),
                "route_norm.weight": (HEAD_DIM,),
                "out_proj.weight": (HIDDEN_SIZE, INNER_DIM),
            }
            for suffix, shape in required.items():
                _require_shape(tensors, p + suffix, shape)
                _require_attention_dtype(tensors, p + suffix)
            for suffix in ("qkv_proj.weight", "k_norm.weight"):
                if p + suffix in tensors:
                    raise CheckpointValidationError(
                        f"accepted progressive block {index} still contains native {suffix}"
                    )
        else:
            required = {
                "qkv_proj.weight": (3 * INNER_DIM, HIDDEN_SIZE),
                "q_norm.weight": (HEAD_DIM,),
                "k_norm.weight": (HEAD_DIM,),
                "out_proj.weight": (HIDDEN_SIZE, INNER_DIM),
            }
            for suffix, shape in required.items():
                _require_shape(tensors, p + suffix, shape)
                _require_attention_dtype(tensors, p + suffix)
            for suffix in ("qv_proj.weight", "route_norm.weight"):
                if p + suffix in tensors:
                    raise CheckpointValidationError(
                        f"unaccepted progressive block {index} contains Keyless {suffix}"
                    )

    for index in range(TOKEN_REFINER_BLOCKS):
        p = f"token_refiner.blocks.{index}.attn."
        for suffix, shape in {
            "qkv_proj.weight": (3 * INNER_DIM, HIDDEN_SIZE),
            "q_norm.weight": (HEAD_DIM,),
            "k_norm.weight": (HEAD_DIM,),
            "out_proj.weight": (HIDDEN_SIZE, INNER_DIM),
        }.items():
            _require_shape(tensors, p + suffix, shape)
            _require_attention_dtype(tensors, p + suffix)
        for suffix in ("qv_proj.weight", "route_norm.weight", *training_suffixes):
            if p + suffix in tensors:
                raise CheckpointValidationError(
                    "progressive snapshot token-refiner attention must remain native QKV"
                )


def progressive_snapshot_metadata(
    prefix: ProgressivePrefix,
    *,
    prefix_manifest_sha256: str,
    manifest_sha256: str | None = None,
) -> dict[str, str]:
    prefix_manifest_sha256 = _require_sha256(
        "progressive prefix manifest SHA-256", prefix_manifest_sha256
    )
    metadata = {
        "architecture": PROGRESSIVE_SNAPSHOT_ARCHITECTURE,
        "checkpoint_format_version": str(PROGRESSIVE_SNAPSHOT_FORMAT_VERSION),
        "snapshot_kind": "stage_b_mixed_qv_qkv",
        "canonical_release": "false",
        "prefix_identity_sha256": prefix.identity_sha256,
        "prefix_manifest_sha256": prefix_manifest_sha256,
        "accepted_block_count": str(len(prefix.accepted)),
        "accepted_blocks": json.dumps(list(prefix.accepted_blocks), separators=(",", ":")),
        "remaining_core": "native_qkv",
        "accepted_core": "folded_qv",
        "token_refiner": "native_qkv",
        "sweep_id": prefix.sweep_id,
        "code_commit": prefix.code_commit.lower(),
        "stage_a_campaign_sha256": prefix.stage_a_campaign_sha256,
        "dataset_manifest_sha256": prefix.dataset_manifest_sha256,
        "gate_manifest_sha256": prefix.gate_manifest_sha256,
        "teacher_model_revision": TARGET_MODEL_REVISION,
        "teacher_model_sha256": TEACHER_SHA256,
    }
    if manifest_sha256 is not None:
        metadata["manifest_sha256"] = _require_sha256(
            "progressive snapshot manifest SHA-256", manifest_sha256
        )
    return metadata


def validate_progressive_snapshot_metadata(
    metadata: Mapping[str, str],
    prefix: ProgressivePrefix,
    *,
    prefix_manifest_sha256: str,
    require_manifest_identity: bool = True,
) -> None:
    expected = progressive_snapshot_metadata(
        prefix,
        prefix_manifest_sha256=prefix_manifest_sha256,
    )
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise CheckpointValidationError(
                f"progressive snapshot metadata {key!r} must be {value!r}, "
                f"got {metadata.get(key)!r}"
            )
    if require_manifest_identity:
        _require_sha256(
            "progressive snapshot manifest SHA-256",
            metadata.get("manifest_sha256", ""),
        )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _tensor_manifest(tensors: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(tensors):
        tensor = tensors[key]
        if not torch.is_tensor(tensor) or getattr(tensor, "is_meta", False):
            raise RuntimeError(f"progressive snapshot tensor is not materialized: {key}")
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


def _manifest_body(
    tensors: Mapping[str, torch.Tensor],
    prefix: ProgressivePrefix,
    *,
    artifact_filename: str,
    prefix_manifest_sha256: str,
) -> dict[str, Any]:
    metadata = progressive_snapshot_metadata(
        prefix,
        prefix_manifest_sha256=prefix_manifest_sha256,
    )
    rows = _tensor_manifest(tensors)
    return {
        "schema": PROGRESSIVE_SNAPSHOT_MANIFEST_SCHEMA,
        "artifact_filename": artifact_filename,
        "metadata": dict(sorted(metadata.items())),
        "prefix": prefix.identity_payload(),
        "tensor_count": len(rows),
        "tensor_payload_bytes": sum(row["bytes"] for row in rows),
        "tensors": rows,
    }


def _manifest_identity(body: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


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
            f"progressive snapshot output is immutable and already exists: {destination}"
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


def export_progressive_snapshot(
    model: nn.Module,
    prefix: ProgressivePrefix,
    output_path: str | Path,
    *,
    prefix_manifest_sha256: str,
    manifest_path: str | Path | None = None,
) -> ProgressiveSnapshotResult:
    """Write an immutable full-model Stage-B mixed QV/QKV BF16 snapshot.

    This artifact exists for periodic full-model testing during the progressive sweep. It
    is explicitly marked non-canonical and cannot be loaded by the final all-Keyless loader.
    Callers should offload the model to CPU before export; this function refuses meta tensors
    but otherwise preserves each source tensor dtype rather than recasting unrelated weights.
    """

    validate_progressive_model_prefix(model, prefix)
    state = model.state_dict()
    validate_progressive_snapshot_tensors(state, prefix)

    output_path = Path(output_path)
    if manifest_path is None:
        manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path = Path(manifest_path)
    occupied = [str(path) for path in (output_path, manifest_path) if path.exists()]
    if occupied:
        raise FileExistsError(f"progressive snapshot outputs are immutable; occupied={occupied}")

    body = _manifest_body(
        state,
        prefix,
        artifact_filename=output_path.name,
        prefix_manifest_sha256=prefix_manifest_sha256,
    )
    manifest_identity = _manifest_identity(body)
    metadata = progressive_snapshot_metadata(
        prefix,
        prefix_manifest_sha256=prefix_manifest_sha256,
        manifest_sha256=manifest_identity,
    )
    cpu_state = {
        key: tensor.detach().cpu().contiguous()
        for key, tensor in state.items()
    }

    artifact_temp: Path | None = None
    manifest_temp: Path | None = None
    artifact_published = False
    manifest_published = False
    try:
        artifact_temp = _new_temp(output_path)
        save_file(cpu_state, str(artifact_temp), metadata=metadata)
        with artifact_temp.open("rb") as handle:
            os.fsync(handle.fileno())
        artifact_sha = sha256_file(artifact_temp)
        artifact_bytes = artifact_temp.stat().st_size
        _publish_no_replace(artifact_temp, output_path)
        artifact_published = True

        receipt = {
            **body,
            "schema": PROGRESSIVE_SNAPSHOT_RECEIPT_SCHEMA,
            "manifest_sha256": manifest_identity,
            "artifact_sha256": artifact_sha,
            "artifact_bytes": artifact_bytes,
        }
        manifest_temp = _new_temp(manifest_path)
        encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
        with manifest_temp.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        manifest_file_sha = sha256_file(manifest_temp)
        _publish_no_replace(manifest_temp, manifest_path)
        manifest_published = True
    except BaseException:
        if manifest_published and manifest_temp is not None:
            _unlink_if_same(manifest_path, manifest_temp)
        if artifact_published and artifact_temp is not None:
            _unlink_if_same(output_path, artifact_temp)
        raise
    finally:
        _unlink(manifest_temp)
        _unlink(artifact_temp)

    return ProgressiveSnapshotResult(
        artifact_path=str(output_path),
        artifact_sha256=artifact_sha,
        artifact_bytes=artifact_bytes,
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_file_sha,
        manifest_identity_sha256=manifest_identity,
        prefix_identity_sha256=prefix.identity_sha256,
        accepted_blocks=len(prefix.accepted),
        tensor_count=len(state),
    )


def validate_progressive_snapshot_file(
    snapshot_path: str | Path,
    prefix: ProgressivePrefix,
    *,
    prefix_manifest_sha256: str,
    manifest_path: str | Path | None = None,
) -> tuple[str, Mapping[str, str]]:
    """Validate snapshot bytes, embedded metadata, topology and optional sidecar receipt."""

    snapshot_path = Path(snapshot_path)
    tensors, metadata = read_safetensors_signatures(snapshot_path)
    validate_progressive_snapshot_metadata(
        metadata,
        prefix,
        prefix_manifest_sha256=prefix_manifest_sha256,
    )
    validate_progressive_snapshot_tensors(tensors, prefix)
    artifact_sha = sha256_file(snapshot_path)

    if manifest_path is not None:
        path = Path(manifest_path)
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"invalid progressive snapshot receipt: {path}") from exc
        if not isinstance(receipt, dict) or receipt.get("schema") != PROGRESSIVE_SNAPSHOT_RECEIPT_SCHEMA:
            raise RuntimeError("progressive snapshot receipt has an incompatible schema")
        if receipt.get("artifact_filename") != snapshot_path.name:
            raise RuntimeError("progressive snapshot receipt references a different artifact filename")
        if str(receipt.get("artifact_sha256", "")).lower() != artifact_sha.lower():
            raise RuntimeError("progressive snapshot artifact SHA-256 differs from its receipt")
        if int(receipt.get("artifact_bytes", -1)) != snapshot_path.stat().st_size:
            raise RuntimeError("progressive snapshot byte count differs from its receipt")
        if receipt.get("prefix") != prefix.identity_payload():
            raise RuntimeError("progressive snapshot receipt prefix payload differs from requested prefix")
        body = dict(receipt)
        for key in ("manifest_sha256", "artifact_sha256", "artifact_bytes"):
            body.pop(key, None)
        body["schema"] = PROGRESSIVE_SNAPSHOT_MANIFEST_SCHEMA
        recomputed = _manifest_identity(body)
        if str(receipt.get("manifest_sha256", "")).lower() != recomputed:
            raise RuntimeError("progressive snapshot receipt manifest identity does not recompute")
        if metadata.get("manifest_sha256", "").lower() != recomputed:
            raise RuntimeError("progressive snapshot embedded manifest identity differs from receipt")
    return artifact_sha, metadata


def _replace_native_prefix_with_empty_deploy(model: nn.Module, prefix: ProgressivePrefix) -> None:
    """Install the mixed snapshot module topology before strict state-dict loading."""

    empty = ProgressivePrefix(
        sweep_id=prefix.sweep_id,
        code_commit=prefix.code_commit,
        stage_a_campaign_sha256=prefix.stage_a_campaign_sha256,
        dataset_manifest_sha256=prefix.dataset_manifest_sha256,
        gate_manifest_sha256=prefix.gate_manifest_sha256,
    )
    validate_progressive_model_prefix(model, empty)
    blocks = getattr(model, "blocks", None)
    assert blocks is not None
    for index in prefix.accepted_blocks:
        native = blocks[index].attn
        hidden, heads, head_dim, eps, gate = _native_attention_facts(native)
        qkv_weight = native.qkv_proj.weight
        blocks[index].attn = KeylessAttentionDeploy(
            hidden,
            heads,
            head_dim,
            eps,
            gate_compress=gate,
            block_index=index,
            dtype=qkv_weight.dtype,
            device=qkv_weight.device,
            operations=None,
        )
    validate_progressive_model_prefix(model, prefix)


def load_progressive_snapshot_into_native_model(
    model: nn.Module,
    snapshot_path: str | Path,
    prefix: ProgressivePrefix,
    *,
    prefix_manifest_sha256: str,
    manifest_path: str | Path | None = None,
) -> nn.Module:
    """Load a validated mixed snapshot into a plain native H3 model for periodic testing."""

    validate_progressive_snapshot_file(
        snapshot_path,
        prefix,
        prefix_manifest_sha256=prefix_manifest_sha256,
        manifest_path=manifest_path,
    )
    original_attentions = tuple(model.blocks[index].attn for index in prefix.accepted_blocks)
    try:
        _replace_native_prefix_with_empty_deploy(model, prefix)
        state = load_file(str(snapshot_path), device="cpu")
        expected = model.state_dict()
        if set(state) != set(expected):
            raise RuntimeError("progressive snapshot keys differ from reconstructed mixed model")
        for key in expected:
            if state[key].shape != expected[key].shape or state[key].dtype != expected[key].dtype:
                raise RuntimeError(
                    f"progressive snapshot tensor signature differs from reconstructed model: {key}"
                )
        model.load_state_dict(state, strict=True)
        validate_progressive_model_prefix(model, prefix)
    except BaseException:
        for index, attention in zip(prefix.accepted_blocks, original_attentions):
            model.blocks[index].attn = attention
        raise
    return model
