from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .activation_capture import CapturedPilotCase
from .checkpoint import sha256_file
from .contracts import CORE_BLOCKS, TARGET_MODEL_REVISION, TEACHER_SHA256
from .pilot import PilotCase
from .pilot_campaign import _require_sha256
from .progressive import PROGRESSIVE_PREFIX_CONTEXT_KEY


PROGRESSIVE_CAPTURE_BUNDLE_SCHEMA = "minimax_h3_keyless_progressive_capture_bundle_v1"
PROGRESSIVE_CAPTURE_RECEIPT_SCHEMA = "minimax_h3_keyless_progressive_capture_receipt_v1"
PROGRESSIVE_CAPTURE_PURPOSE = "progressive_core50_live_input"


@dataclass(frozen=True)
class ProgressiveCaptureProvenance:
    """Fixed identities required for one Phase-4 live-input capture corpus."""

    code_commit: str
    comfy_commit: str
    dataset_manifest_sha256: str
    gate_manifest_sha256: str
    stage_a_campaign_sha256: str
    prefix_identity_sha256: str
    target_block: int
    execution_descriptor: str
    purpose: str = PROGRESSIVE_CAPTURE_PURPOSE
    teacher_model_revision: str = TARGET_MODEL_REVISION
    teacher_model_sha256: str = TEACHER_SHA256

    def __post_init__(self) -> None:
        for name in ("code_commit", "comfy_commit", "execution_descriptor"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"progressive capture provenance {name} must be non-empty")
        if self.purpose != PROGRESSIVE_CAPTURE_PURPOSE:
            raise ValueError(
                f"progressive capture purpose must be {PROGRESSIVE_CAPTURE_PURPOSE!r}"
            )
        if (
            isinstance(self.target_block, bool)
            or not isinstance(self.target_block, int)
            or not 0 <= self.target_block < CORE_BLOCKS
        ):
            raise ValueError(
                f"progressive capture target_block must be within [0,{CORE_BLOCKS})"
            )
        for name in (
            "dataset_manifest_sha256",
            "gate_manifest_sha256",
            "stage_a_campaign_sha256",
            "prefix_identity_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(f"progressive capture {name}", getattr(self, name)),
            )
        if self.teacher_model_revision != TARGET_MODEL_REVISION:
            raise ValueError("progressive capture teacher revision must be the pinned BF16 parent")
        if self.teacher_model_sha256.lower() != TEACHER_SHA256:
            raise ValueError("progressive capture teacher SHA-256 must be the pinned BF16 parent")


@dataclass(frozen=True)
class ProgressiveCaptureWriteResult:
    bundle_path: str
    bundle_sha256: str
    bundle_bytes: int
    receipt_path: str
    receipt_sha256: str
    target_block: int
    case_id: str


def _json_safe(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, tuple):
        return [_json_safe(item, path=f"{path}[]") for item in value]
    if isinstance(value, list):
        return [_json_safe(item, path=f"{path}[]") for item in value]
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
        return tuple(_safe_nested_execution_value(item, path=f"{path}[]") for item in value)
    if isinstance(value, list):
        return [_safe_nested_execution_value(item, path=f"{path}[]") for item in value]
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
        "shape": [int(dim) for dim in value.shape],
        "dtype": str(value.dtype),
        "bytes": int(value.numel() * value.element_size()),
    }


def _require_bound_prefix_context(
    record: CapturedPilotCase,
    provenance: ProgressiveCaptureProvenance,
) -> None:
    context = record.case.context
    if not isinstance(context, Mapping):
        raise ValueError("progressive capture case.context must be a mapping")
    prefix = context.get(PROGRESSIVE_PREFIX_CONTEXT_KEY)
    if not isinstance(prefix, Mapping):
        raise ValueError("progressive capture is missing its bound prefix context")
    required = {
        "api": 1,
        "prefix_identity_sha256": provenance.prefix_identity_sha256,
        "next_block": provenance.target_block,
        "stage_a_campaign_sha256": provenance.stage_a_campaign_sha256,
        "dataset_manifest_sha256": provenance.dataset_manifest_sha256,
        "gate_manifest_sha256": provenance.gate_manifest_sha256,
    }
    for name, expected in required.items():
        if prefix.get(name) != expected:
            raise ValueError(
                f"progressive capture prefix context {name!r} does not match provenance"
            )
    accepted = prefix.get("accepted_blocks")
    expected_accepted = list(range(provenance.target_block))
    if accepted != expected_accepted:
        raise ValueError(
            "progressive capture prefix context does not describe the exact accepted prefix: "
            f"expected={expected_accepted}, actual={accepted!r}"
        )


def _validate_record(
    record: CapturedPilotCase,
    provenance: ProgressiveCaptureProvenance,
) -> str:
    if record.block_index != provenance.target_block:
        raise ValueError(
            "progressive capture record block does not match provenance target_block: "
            f"record={record.block_index}, target={provenance.target_block}"
        )
    case = record.case
    if not isinstance(case.case_id, str) or not case.case_id.strip():
        raise ValueError("progressive capture case_id must be non-empty")
    if case.sigma is None:
        raise ValueError("progressive capture must record sigma")
    if case.transformer_options:
        raise ValueError(
            "progressive capture bundles require empty replay transformer_options; runtime audit "
            "state belongs in case.context"
        )
    if case.x.device.type != "cpu" or case.t_emb.device.type != "cpu":
        raise ValueError("progressive capture tensors must be CPU-resident before serialization")
    if record.attention_input.device.type != "cpu":
        raise ValueError("progressive capture attention input must be CPU-resident")
    if record.attention_input.shape != case.x.shape:
        raise ValueError("progressive capture attention input must match block-input shape")
    if record.captured_bytes < 0:
        raise ValueError("progressive capture captured_bytes cannot be negative")
    _require_bound_prefix_context(record, provenance)
    return case.case_id


def _record_summary(record: CapturedPilotCase) -> dict[str, Any]:
    return {
        "block_index": int(record.block_index),
        "captured_bytes": int(record.captured_bytes),
        "x": _tensor_summary(record.case.x),
        "t_emb": _tensor_summary(record.case.t_emb),
        "rope_freqs": _tensor_summary(record.case.rope_freqs),
        "position_ids": _tensor_summary(record.case.position_ids),
        "attention_input": _tensor_summary(record.attention_input),
    }


def _record_payload(record: CapturedPilotCase) -> dict[str, Any]:
    case = record.case
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
        "context": _json_safe(dict(case.context), path="case.context"),
    }


def _new_temp_path(path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)


def _write_torch_temp(path: Path, payload: Any) -> str:
    fd, name = _new_temp_path(path)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        return name
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _write_json_temp(path: Path, value: Mapping[str, Any]) -> str:
    encoded = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    fd, name = _new_temp_path(path)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        return name
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _publish_no_replace(source: str, destination: Path) -> None:
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise FileExistsError(
            f"progressive capture evidence is immutable and already exists: {destination}"
        ) from exc


def _unlink_if_same_file(path: Path, source: str) -> None:
    try:
        if path.exists() and os.path.samefile(path, source):
            path.unlink()
    except FileNotFoundError:
        pass


def _unlink_temp(path: str | None) -> None:
    if path is None:
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def write_progressive_capture_bundle(
    path: str | Path,
    record: CapturedPilotCase,
    *,
    provenance: ProgressiveCaptureProvenance,
    receipt_path: str | Path | None = None,
) -> ProgressiveCaptureWriteResult:
    """Persist one immutable live-input block capture bound to an accepted prefix digest."""

    path = Path(path)
    receipt_path = (
        Path(receipt_path)
        if receipt_path is not None
        else path.with_suffix(path.suffix + ".receipt.json")
    )
    case_id = _validate_record(record, provenance)
    payload = {
        "schema": PROGRESSIVE_CAPTURE_BUNDLE_SCHEMA,
        "provenance": asdict(provenance),
        "record": _record_payload(record),
    }

    bundle_temp: str | None = None
    receipt_temp: str | None = None
    bundle_published = False
    receipt_published = False
    try:
        bundle_temp = _write_torch_temp(path, payload)
        _publish_no_replace(bundle_temp, path)
        bundle_published = True
        bundle_sha = sha256_file(path)
        bundle_bytes = path.stat().st_size
        receipt = {
            "schema": PROGRESSIVE_CAPTURE_RECEIPT_SCHEMA,
            "bundle_filename": path.name,
            "bundle_sha256": bundle_sha,
            "bundle_bytes": bundle_bytes,
            "target_block": provenance.target_block,
            "case_id": case_id,
            "sigma": record.case.sigma,
            "modality_label": record.case.modality_label,
            "provenance": asdict(provenance),
            "record": _record_summary(record),
        }
        receipt_temp = _write_json_temp(receipt_path, receipt)
        _publish_no_replace(receipt_temp, receipt_path)
        receipt_published = True
        receipt_sha = sha256_file(receipt_path)
    except BaseException:
        if receipt_published and receipt_temp is not None:
            _unlink_if_same_file(receipt_path, receipt_temp)
        if bundle_published and bundle_temp is not None:
            _unlink_if_same_file(path, bundle_temp)
        raise
    finally:
        _unlink_temp(receipt_temp)
        _unlink_temp(bundle_temp)

    return ProgressiveCaptureWriteResult(
        bundle_path=str(path),
        bundle_sha256=bundle_sha,
        bundle_bytes=bundle_bytes,
        receipt_path=str(receipt_path),
        receipt_sha256=receipt_sha,
        target_block=provenance.target_block,
        case_id=case_id,
    )


def _load_receipt(
    path: Path,
    receipt_path: Path,
    *,
    expected_receipt_sha256: str | None,
) -> dict[str, Any]:
    if expected_receipt_sha256 is not None:
        expected = _require_sha256(
            "expected progressive capture receipt SHA-256", expected_receipt_sha256
        )
        if sha256_file(receipt_path).lower() != expected:
            raise ValueError("progressive capture receipt SHA-256 does not match expected identity")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid progressive capture receipt: {receipt_path}") from exc
    if not isinstance(receipt, dict):
        raise ValueError("progressive capture receipt must decode to an object")
    if receipt.get("schema") != PROGRESSIVE_CAPTURE_RECEIPT_SCHEMA:
        raise ValueError("progressive capture receipt has an unsupported schema")
    if receipt.get("bundle_filename") != path.name:
        raise ValueError("progressive capture receipt bundle_filename does not match bundle path")
    if int(receipt.get("bundle_bytes", -1)) != path.stat().st_size:
        raise ValueError("progressive capture receipt bundle_bytes does not match bundle")
    actual_bundle_sha = sha256_file(path)
    if str(receipt.get("bundle_sha256", "")).lower() != actual_bundle_sha.lower():
        raise ValueError("progressive capture receipt bundle_sha256 does not match bundle")
    return receipt


def load_progressive_capture_bundle(
    path: str | Path,
    *,
    receipt_path: str | Path | None = None,
    expected_receipt_sha256: str | None = None,
) -> tuple[CapturedPilotCase, ProgressiveCaptureProvenance]:
    """Load one progressive capture only after sidecar/hash/prefix validation."""

    path = Path(path)
    receipt_path = (
        Path(receipt_path)
        if receipt_path is not None
        else path.with_suffix(path.suffix + ".receipt.json")
    )
    receipt = _load_receipt(
        path,
        receipt_path,
        expected_receipt_sha256=expected_receipt_sha256,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != PROGRESSIVE_CAPTURE_BUNDLE_SCHEMA:
        raise ValueError("progressive capture bundle has an unsupported schema")
    provenance_raw = payload.get("provenance")
    if not isinstance(provenance_raw, dict):
        raise ValueError("progressive capture bundle is missing provenance")
    provenance = ProgressiveCaptureProvenance(**provenance_raw)
    if receipt.get("provenance") != provenance_raw:
        raise ValueError("progressive capture receipt provenance differs from bundle")

    raw = payload.get("record")
    if not isinstance(raw, dict):
        raise ValueError("progressive capture bundle is missing its record")
    x = raw.get("x")
    t_emb = raw.get("t_emb")
    attention_input = raw.get("attention_input")
    if not all(torch.is_tensor(value) for value in (x, t_emb, attention_input)):
        raise ValueError("progressive capture record is missing required tensors")
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
    record = CapturedPilotCase(
        block_index=int(raw.get("block_index")),
        case=case,
        attention_input=attention_input,
        captured_bytes=int(raw.get("captured_bytes", -1)),
    )
    case_id = _validate_record(record, provenance)
    if receipt.get("target_block") != provenance.target_block:
        raise ValueError("progressive capture receipt target_block differs from provenance")
    if receipt.get("case_id") != case_id:
        raise ValueError("progressive capture receipt case_id differs from bundle")
    if receipt.get("sigma") != record.case.sigma:
        raise ValueError("progressive capture receipt sigma differs from bundle")
    if receipt.get("modality_label") != record.case.modality_label:
        raise ValueError("progressive capture receipt modality differs from bundle")
    if receipt.get("record") != _record_summary(record):
        raise ValueError("progressive capture receipt tensor summary differs from bundle")
    return record, provenance
