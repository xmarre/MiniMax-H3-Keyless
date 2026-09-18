from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from minimax_h3_keyless.activation_capture import CapturedPilotCase
from minimax_h3_keyless.capture_io import (
    CaptureBundleProvenance,
    load_captured_pilot_bundle,
    write_captured_pilot_bundle,
)
from minimax_h3_keyless.checkpoint import sha256_file
from minimax_h3_keyless.contracts import TEACHER_SHA256
from minimax_h3_keyless.pilot import PilotCase


def _case(case_id="case-7", sigma=0.5, modality="video") -> PilotCase:
    return PilotCase(
        x=torch.arange(12, dtype=torch.float32).reshape(3, 4),
        t_emb=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        mod_segments=((0, 3, torch.tensor([0, 0, 0], dtype=torch.long)),),
        rope_freqs=torch.ones(1, 3, 1, 1, 2, 2),
        transformer_options={},
        case_id=case_id,
        sigma=sigma,
        modality_label=modality,
        position_ids=torch.arange(9, dtype=torch.float64).reshape(3, 3),
        context={
            "split": "holdout",
            "minimax_h3_keyless_live_capture_v1": {
                "block_index": 0,
                "layout": {"seq_len": 3},
            },
        },
    )


def _records():
    case = _case()
    return (
        CapturedPilotCase(
            block_index=0,
            case=case,
            attention_input=case.x + 1.0,
            captured_bytes=512,
        ),
        CapturedPilotCase(
            block_index=25,
            case=replace(
                case,
                x=case.x + 4.0,
                context={
                    **case.context,
                    "minimax_h3_keyless_live_capture_v1": {
                        "block_index": 25,
                        "layout": {"seq_len": 3},
                    },
                },
            ),
            attention_input=case.x + 5.0,
            captured_bytes=1024,
        ),
    )


def _provenance(**updates):
    values = {
        "code_commit": "0123456789abcdef",
        "comfy_commit": "fedcba9876543210",
        "dataset_manifest_sha256": "a" * 64,
        "execution_descriptor": "ComfyUI fixed Stage-A capture run",
    }
    values.update(updates)
    return CaptureBundleProvenance(**values)


def test_capture_bundle_roundtrip_is_hash_bound_and_preserves_replay_payload(tmp_path: Path) -> None:
    path = tmp_path / "capture.pt"
    written = write_captured_pilot_bundle(path, _records(), provenance=_provenance())
    assert written.bundle_sha256 == sha256_file(path)
    assert written.receipt_sha256 == sha256_file(written.receipt_path)
    assert written.block_indices == (0, 25)
    assert written.case_id == "case-7"

    loaded, provenance = load_captured_pilot_bundle(
        path,
        expected_receipt_sha256=written.receipt_sha256,
    )
    assert provenance == _provenance()
    assert [record.block_index for record in loaded] == [0, 25]
    assert [record.captured_bytes for record in loaded] == [512, 1024]
    torch.testing.assert_close(loaded[0].case.x, _records()[0].case.x)
    torch.testing.assert_close(loaded[1].case.x, _records()[1].case.x)
    torch.testing.assert_close(loaded[1].attention_input, _records()[1].attention_input)
    torch.testing.assert_close(loaded[0].case.mod_segments[0][2], torch.tensor([0, 0, 0]))
    assert loaded[0].case.context["split"] == "holdout"
    assert loaded[0].case.transformer_options == {}


def test_capture_bundle_hash_is_checked_before_deserialization(tmp_path: Path) -> None:
    path = tmp_path / "capture.pt"
    written = write_captured_pilot_bundle(path, _records(), provenance=_provenance())
    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="bundle_bytes|bundle_sha256"):
        load_captured_pilot_bundle(
            path,
            expected_receipt_sha256=written.receipt_sha256,
        )


def test_expected_receipt_identity_detects_sidecar_replacement(tmp_path: Path) -> None:
    path = tmp_path / "capture.pt"
    written = write_captured_pilot_bundle(path, _records(), provenance=_provenance())
    receipt_path = Path(written.receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["execution_note"] = "replacement"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt SHA-256"):
        load_captured_pilot_bundle(
            path,
            expected_receipt_sha256=written.receipt_sha256,
        )


def test_receipt_tensor_summary_tamper_is_detected_even_without_external_receipt_hash(tmp_path: Path) -> None:
    path = tmp_path / "capture.pt"
    written = write_captured_pilot_bundle(path, _records(), provenance=_provenance())
    receipt_path = Path(written.receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["records"][0]["x"]["shape"] = [999]
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="tensor summaries"):
        load_captured_pilot_bundle(path)


def test_capture_bundle_rejects_noncanonical_replay_options_and_non_json_context(tmp_path: Path) -> None:
    path = tmp_path / "capture.pt"
    record = _records()[0]
    with pytest.raises(ValueError, match="empty replay transformer_options"):
        write_captured_pilot_bundle(
            path,
            (replace(record, case=replace(record.case, transformer_options={"provider": object()})),),
            provenance=_provenance(),
        )
    assert not path.exists()

    with pytest.raises(ValueError, match="non-JSON provenance"):
        write_captured_pilot_bundle(
            path,
            (replace(record, case=replace(record.case, context={"opaque": object()})),),
            provenance=_provenance(),
        )
    assert not path.exists()


def test_capture_publish_never_overwrites_existing_bundle_or_receipt(tmp_path: Path) -> None:
    path = tmp_path / "capture.pt"
    first = write_captured_pilot_bundle(path, _records(), provenance=_provenance())
    receipt_path = Path(first.receipt_path)
    original_bundle = path.read_bytes()
    original_receipt = receipt_path.read_bytes()

    changed = tuple(
        replace(record, attention_input=record.attention_input + 100.0)
        for record in _records()
    )
    with pytest.raises(FileExistsError, match="immutable"):
        write_captured_pilot_bundle(path, changed, provenance=_provenance())

    assert path.read_bytes() == original_bundle
    assert receipt_path.read_bytes() == original_receipt
    assert sha256_file(path) == first.bundle_sha256
    assert sha256_file(receipt_path) == first.receipt_sha256


def test_preexisting_receipt_is_preserved_and_new_bundle_link_is_rolled_back(tmp_path: Path) -> None:
    path = tmp_path / "capture.pt"
    receipt_path = path.with_suffix(path.suffix + ".receipt.json")
    sentinel = b"preexisting immutable receipt\n"
    receipt_path.write_bytes(sentinel)

    with pytest.raises(FileExistsError, match="immutable"):
        write_captured_pilot_bundle(path, _records(), provenance=_provenance())

    assert not path.exists()
    assert receipt_path.read_bytes() == sentinel


def test_capture_provenance_cannot_claim_a_different_teacher() -> None:
    with pytest.raises(ValueError, match="teacher SHA-256"):
        _provenance(teacher_model_sha256="0" * 64)
    assert TEACHER_SHA256 != "0" * 64


def test_capture_bundle_forwards_explicit_map_location(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "capture.pt"
    written = write_captured_pilot_bundle(path, _records(), provenance=_provenance())
    original_load = torch.load
    seen = {}

    def wrapped_load(*args, **kwargs):
        seen["map_location"] = kwargs.get("map_location")
        kwargs["map_location"] = "cpu"
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", wrapped_load)
    loaded, _ = load_captured_pilot_bundle(
        path,
        expected_receipt_sha256=written.receipt_sha256,
        map_location="cuda:0",
    )
    assert seen["map_location"] == "cuda:0"
    assert loaded[0].attention_input.device.type == "cpu"
