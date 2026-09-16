from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

import minimax_h3_keyless.export as export_mod
from minimax_h3_keyless.checkpoint import sha256_file
from minimax_h3_keyless.export import (
    build_export_manifest_body,
    export_folded_bf16,
    manifest_identity_sha256,
)


def test_manifest_identity_is_mapping_order_independent() -> None:
    tensors_a = {"z": torch.ones(2), "a": torch.zeros(3, dtype=torch.bfloat16)}
    tensors_b = {"a": tensors_a["a"], "z": tensors_a["z"]}
    metadata_a = {"training_run": "pilot", "export_commit": "abc"}
    metadata_b = {"export_commit": "abc", "training_run": "pilot"}
    body_a = build_export_manifest_body(
        tensors_a, artifact_filename="student.safetensors", metadata=metadata_a
    )
    body_b = build_export_manifest_body(
        tensors_b, artifact_filename="student.safetensors", metadata=metadata_b
    )
    assert body_a == body_b
    assert [row["key"] for row in body_a["tensors"]] == ["a", "z"]
    assert manifest_identity_sha256(body_a) == manifest_identity_sha256(body_b)


def test_export_receipt_binds_metadata_manifest_and_artifact_sha(tmp_path: Path, monkeypatch) -> None:
    folded = {
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16),
        "bias": torch.tensor([1.0, 2.0], dtype=torch.float32),
    }
    monkeypatch.setattr(export_mod, "fold_training_state_dict", lambda state, output_dtype: folded)
    monkeypatch.setattr(export_mod, "validate_deploy_checkpoint", lambda tensors, metadata: None)
    path = tmp_path / "student.safetensors"
    result = export_folded_bf16(
        {"ignored": torch.ones(1)},
        path,
        metadata={"training_run": "pilot", "export_commit": "deadbeef"},
        command="test export",
    )
    assert result.artifact_sha256 == sha256_file(path)
    assert result.artifact_bytes == path.stat().st_size
    assert result.tensor_count == 2
    with safe_open(str(path), framework="pt", device="cpu") as f:
        md = dict(f.metadata() or {})
    assert md["manifest_sha256"] == result.manifest_identity_sha256
    receipt = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert receipt["artifact_sha256"] == result.artifact_sha256
    assert receipt["artifact_bytes"] == result.artifact_bytes
    assert receipt["manifest_sha256"] == result.manifest_identity_sha256
    assert result.manifest_sha256 == sha256_file(result.manifest_path)


def test_export_rejects_predeclared_wrong_manifest_identity(tmp_path: Path, monkeypatch) -> None:
    folded = {"weight": torch.ones(2, 2, dtype=torch.bfloat16)}
    monkeypatch.setattr(export_mod, "fold_training_state_dict", lambda state, output_dtype: folded)
    with torch.no_grad():
        try:
            export_folded_bf16(
                {},
                tmp_path / "bad.safetensors",
                metadata={"manifest_sha256": "0" * 64},
            )
        except ValueError as exc:
            assert "does not match" in str(exc)
        else:
            raise AssertionError("wrong predeclared manifest identity must fail")
