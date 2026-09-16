from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .activation_capture import CapturedPilotCase
from .checkpoint import sha256_file
from .contracts import TARGET_MODEL_REVISION, TEACHER_SHA256
from .pilot import PilotCase
from .pilot_campaign import PILOT_BLOCKS, _require_sha256, write_json_atomic


CAPTURE_BUNDLE_SCHEMA = "minimax_h3_keyless_pilot_capture_bundle_v1"
CAPTURE_RECEIPT_SCHEMA = "minimax_h3_keyless_pilot_capture_receipt_v1"


@dataclass(frozen=True)
class CaptureBundleProvenance:
    code_commit: str
    comfy_commit: str
    dataset_manifest_sha256: str
    execution_descriptor: str
    purpose: str = "stage_a_pilot"
    teacher_model_revision: str = TARGET_MODEL_REVISION
    teacher_model_sha256: str = TEACHER_SHA256

    def __post_init__(self) -> None:
        for name in ("code_commit", "comfy_commit", "execution_descriptor", "purpose"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"capture provenance {name} must be non-empty")
        _require_sha256("dataset_manifest_sha256", self.dataset_manifest_sha256)
        if self.teacher_model_revision != TARGET_MODEL_REVISION:
            raise ValueError("capture provenance teacher revision must be the pinned BF16 parent")
        if self.teacher_model_sha256.lower() != TEACHER_SHA256:
            raise ValueError("capture provenance teacher SHA-256 must be the pinned BF16 parent")


@dataclass(frozen=True)
class CaptureBundleWriteResult:
    bundle_path: str
    bundle_sha256: str
    bundle_bytes: int
    receipt_path: str
    receipt_sha256: str
    block_indices: tuple[int, ...]
    case_id: str


def _json_safe(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return [_json_safe(v, path=f"{path}[]") for v in value]
    if isinstance(value, list):
        return [_json_safe(v, path=f"{path}[]") for v in value]
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string JSON key")
            out[key] = _json_safe(item, path=f"{path}.{key}")
        return out
    raise ValueError(f"{path} contains non-JSON provenance value {type(value)!r}")


def _safe_nested_execution_value(value: Any, *, path: str) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return tuple(_safe_nested_execution_value(v, path=f"{path}[]") for v in value)
    if isinstance(value, list):
        return [_safe_nested_execution_value(v, path=f"{path}[]") for v in value]
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise ValueError(f"{path} contains unsupported mapping key {type(key)!r}")
            out[key] = _safe_nested_execution_value(item, path=f"{path}.{key}")
        return out
    raise ValueError(f"{path} contains unsupported execution value {type(value)!r}")


def _tensor_summary(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "shape": [int(x) for x in value.shape],
        "dtype": str(value.dtype),
        "bytes": int(value.numel() * value.element_size()),
    }


def _record_summary(record: CapturedPilotCase) -> dict[str, Any]:
    return {
        "block_index": record.block_index,
        "captured_bytes": int(record.captured_bytes),
        "x": _tensor_summary(record.case.x),
        "t_emb": _tensor_summary(record.case.t_emb),
        "rope_freqs": _tensor_summary(record.case.rope_freqs),
        "position_ids": _tensor_summary(record.case.position_ids),
        "attention_input": _tensor_summary(record.attention_input),
    }


def _record_payload(record: CapturedPilotCase) -> dict[str, Any]:
    case = record.case
    if case.transformer_options:
        raise ValueError(
            "canonical Stage-A capture bundles require empty replay transformer_options; "
            "runtime option provenance belongs in case.context"
        )
    if case.x.device.type != "cpu" or case.t_emb.device.type != "cpu":
        raise ValueError("canonical Stage-A capture tensors must be CPU-resident before serialization")
    if record.attention_input.device.type != "cpu":
        raise ValueError("captured post-AdaLN attention input must be CPU-resident")
    if record.attention_input.shape != case.x.shape:
        raise ValueError("captured post-AdaLN attention input must match block-input shape")
    context = _json_safe(dict(case.context), path="case.context")
    return {
        "block_index": int(record.block_index),
        "captured_bytes": int(record.captured_bytes),
        "case_id": case.case_id,
        "sigma": case.sigma,
        "modality_label": case.modality_label,
        "x": case.x.detach().cpu(),
        "t_emb": case.t_emb.detach().cpu(),
        "mod_segments": _safe_nested_execution_value(case.mod_segments, path="mod_segments"),
        "rope_freqs": None if case.rope_freqs is None else case.rope_freqs.detach().cpu(),
        "position_ids": None if case.position_ids is None else case.position_ids.detach().cpu(),
        "attention_input": record.attention_input.detach().cpu(),
        "context": context,
    }


def _validate_records(records: Sequence[CapturedPilotCase]) -> tuple[tuple[int, ...], str]:
    if not records:
        raise ValueError("capture bundle requires at least one captured pilot block")
    indices = tuple(int(record.block_index) for record in records)
    if len(set(indices)) != len(indices):
        raise ValueError("capture bundle contains duplicate block indices")
    invalid = [index for index in indices if index not in PILOT_BLOCKS]
    if invalid:
        raise ValueError(f"capture bundle contains non-Stage-A block indices: {invalid}")
    case_ids = {record.case.case_id for record in records}
    if len(case_ids) != 1 or "" in case_ids:
        raise ValueError("capture bundle records must share one non-empty case_id")
    sigmas = {record.case.sigma for record in records}
    modalities = {record.case.modality_label for record in records}
    if len(sigmas) != 1 or len(modalities) != 1:
        raise ValueError("capture bundle records must share sigma and modality provenance")
    for record in records:
        if record.captured_bytes < 0:
            raise ValueError("capture bundle captured_bytes cannot be negative")
    return indices, next(iter(case_ids))


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_captured_pilot_bundle(
    path: str | Path,
    records: Sequence[CapturedPilotCase],
    *,
    provenance: CaptureBundleProvenance,
    receipt_path: str | Path | None = None,
) -> CaptureBundleWriteResult:
    """Persist a bounded Stage-A capture plus an independently hashable JSON receipt."""
    path = Path(path)
    indices, case_id = _validate_records(records)
    record_payloads = [_record_payload(record) for record in records]
    payload = {
        "schema": CAPTURE_BUNDLE_SCHEMA,
        "provenance": asdict(provenance),
        "records": record_payloads,
    }
    _atomic_torch_save(path, payload)
    bundle_sha = sha256_file(path)
    bundle_bytes = path.stat().st_size

    receipt_path = (
        Path(receipt_path)
        if receipt_path is not None
        else path.with_suffix(path.suffix + ".receipt.json")
    )
    receipt = {
        "schema": CAPTURE_RECEIPT_SCHEMA,
        "bundle_filename": path.name,
        "bundle_sha256": bundle_sha,
        "bundle_bytes": bundle_bytes,
        "case_id": case_id,
        "sigma": records[0].case.sigma,
        "modality_label": records[0].case.modality_label,
        "block_indices": list(indices),
        "provenance": asdict(provenance),
        "records": [_record_summary(record) for record in records],
    }
    try:
        receipt_sha = write_json_atomic(receipt_path, receipt)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    return CaptureBundleWriteResult(
        bundle_path=str(path),
        bundle_sha256=bundle_sha,
        bundle_bytes=bundle_bytes,
        receipt_path=str(receipt_path),
        receipt_sha256=receipt_sha,
        block_indices=indices,
        case_id=case_id,
    )


def _read_and_prevalidate_receipt(
    path: Path,
    receipt_path: Path,
    *,
    expected_receipt_sha256: str | None,
) -> dict[str, Any]:
    if expected_receipt_sha256 is not None:
        expected_receipt_sha256 = _require_sha256(
            "expected_receipt_sha256", expected_receipt_sha256
        )
        actual_receipt_sha = sha256_file(receipt_path)
        if actual_receipt_sha.lower() != expected_receipt_sha256.lower():
            raise ValueError(
                "capture receipt SHA-256 does not match the expected immutable identity"
            )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid capture receipt: {receipt_path}") from exc
    if not isinstance(receipt, dict):
        raise ValueError("capture receipt must decode to a JSON object")
    if receipt.get("schema") != CAPTURE_RECEIPT_SCHEMA:
        raise ValueError("capture receipt has an unsupported schema")
    if receipt.get("bundle_filename") != path.name:
        raise ValueError("capture receipt bundle_filename does not match the bundle path")
    if int(receipt.get("bundle_bytes", -1)) != path.stat().st_size:
        raise ValueError("capture receipt bundle_bytes does not match the bundle")
    actual_bundle_sha = sha256_file(path)
    if str(receipt.get("bundle_sha256", "")).lower() != actual_bundle_sha.lower():
        raise ValueError("capture receipt bundle_sha256 does not match the bundle")
    return receipt


def load_captured_pilot_bundle(
    path: str | Path,
    *,
    receipt_path: str | Path | None = None,
    expected_receipt_sha256: str | None = None,
) -> tuple[tuple[CapturedPilotCase, ...], CaptureBundleProvenance]:
    """Load a capture only after receipt/hash/provenance validation.

    The bundle hash is checked before deserialization. ``weights_only=True`` remains
    mandatory because capture bundles are data artifacts, not executable checkpoints.
    Passing ``expected_receipt_sha256`` additionally binds the sidecar itself to an
    immutable run/artifact registry identity.
    """
    path = Path(path)
    receipt_path = (
        Path(receipt_path)
        if receipt_path is not None
        else path.with_suffix(path.suffix + ".receipt.json")
    )
    receipt = _read_and_prevalidate_receipt(
        path,
        receipt_path,
        expected_receipt_sha256=expected_receipt_sha256,
    )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != CAPTURE_BUNDLE_SCHEMA:
        raise ValueError("capture bundle has an unsupported schema")
    provenance_raw = payload.get("provenance")
    if not isinstance(provenance_raw, dict):
        raise ValueError("capture bundle is missing provenance")
    provenance = CaptureBundleProvenance(**provenance_raw)
    if receipt.get("provenance") != provenance_raw:
        raise ValueError("capture receipt provenance does not match the serialized bundle")

    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("capture bundle contains no records")
    records: list[CapturedPilotCase] = []
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("capture bundle record must be a mapping")
        x = raw.get("x")
        t_emb = raw.get("t_emb")
        attention_input = raw.get("attention_input")
        if not all(torch.is_tensor(v) for v in (x, t_emb, attention_input)):
            raise ValueError("capture bundle record is missing required tensors")
        case = PilotCase(
            x=x,
            t_emb=t_emb,
            mod_segments=raw.get("mod_segments"),
            rope_freqs=raw.get("rope_freqs"),
            transformer_options={},
            case_id=str(raw.get("case_id", "")),
            sigma=raw.get("sigma"),
            modality_label=raw.get("modality_label"),
            position_ids=raw.get("position_ids"),
            context=dict(raw.get("context") or {}),
        )
        records.append(
            CapturedPilotCase(
                block_index=int(raw.get("block_index")),
                case=case,
                attention_input=attention_input,
                captured_bytes=int(raw.get("captured_bytes", -1)),
            )
        )
    indices, case_id = _validate_records(records)
    if list(indices) != receipt.get("block_indices") or case_id != receipt.get("case_id"):
        raise ValueError("capture receipt block/case identity does not match serialized records")
    if records[0].case.sigma != receipt.get("sigma"):
        raise ValueError("capture receipt sigma does not match serialized records")
    if records[0].case.modality_label != receipt.get("modality_label"):
        raise ValueError("capture receipt modality does not match serialized records")
    expected_summaries = [_record_summary(record) for record in records]
    if receipt.get("records") != expected_summaries:
        raise ValueError("capture receipt tensor summaries do not match serialized records")
    for record in records:
        if record.attention_input.shape != record.case.x.shape:
            raise ValueError("loaded capture attention input shape does not match block input")
    return tuple(records), provenance
