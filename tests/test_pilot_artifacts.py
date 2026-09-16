from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from minimax_h3_keyless.attention import KeylessAttentionTrain
from minimax_h3_keyless.checkpoint import sha256_file
from minimax_h3_keyless.pilot import set_pilot_block_stage
from minimax_h3_keyless.pilot_artifacts import (
    STAGE_A_RESULT_SCHEMA,
    StageAArtifactRequest,
    persist_stage_a_block_artifacts,
)
from minimax_h3_keyless.pilot_campaign import PilotRunIdentity, canonical_json_sha256


class TinyStudentBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = KeylessAttentionTrain(4, 2, 2, 1e-5, block_index=25, dtype=torch.float32)
        self.mlp = nn.Linear(4, 4, bias=False)
        set_pilot_block_stage(self, "route")


def _student_and_optimizer():
    student = TinyStudentBlock()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    return student, optimizer


def _identity(run_id="pilot-001") -> PilotRunIdentity:
    return PilotRunIdentity(
        run_id=run_id,
        code_commit="deadbeef",
        dataset_manifest_sha256="a" * 64,
        gate_manifest_sha256="b" * 64,
        block_index=25,
        route_mode="least_squares",
        lambda_relative=1e-4,
    )


def test_stage_a_artifact_transaction_binds_resume_hash_and_refuses_overwrite(tmp_path: Path) -> None:
    student, optimizer = _student_and_optimizer()
    request = StageAArtifactRequest(str(tmp_path), "pilot-001", "deadbeef")
    payload = {"gate": {"passed": False}, "metric": 1.25}
    receipt = persist_stage_a_block_artifacts(
        request,
        student_block=student,
        optimizer=optimizer,
        identity=_identity(),
        stage="route",
        step=7,
        result_payload=payload,
    )
    assert receipt.checkpoint_sha256 == sha256_file(receipt.checkpoint_path)
    assert receipt.result_sha256 == sha256_file(receipt.result_path)
    result = json.loads(Path(receipt.result_path).read_text(encoding="utf-8"))
    assert result["schema"] == STAGE_A_RESULT_SCHEMA
    assert result["checkpoint_sha256"] == receipt.checkpoint_sha256
    assert result["identity"]["run_id"] == "pilot-001"
    assert result["result"]["gate"]["passed"] is False
    expected_payload_sha = canonical_json_sha256(payload)
    assert result["result_payload_sha256"] == expected_payload_sha
    checkpoint = torch.load(receipt.checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["extra"]["stage_a_result_payload_sha256"] == expected_payload_sha
    assert checkpoint["extra"]["stage_a_result_schema"] == STAGE_A_RESULT_SCHEMA
    with pytest.raises(FileExistsError, match="immutable"):
        persist_stage_a_block_artifacts(
            request,
            student_block=student,
            optimizer=optimizer,
            identity=_identity(),
            stage="route",
            step=8,
            result_payload={},
        )


def test_stage_a_artifact_request_rejects_path_traversal_and_identity_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id"):
        StageAArtifactRequest(str(tmp_path), "../escape", "deadbeef")
    request = StageAArtifactRequest(str(tmp_path), "pilot-002", "deadbeef")
    student, optimizer = _student_and_optimizer()
    with pytest.raises(ValueError, match="run_id"):
        persist_stage_a_block_artifacts(
            request,
            student_block=student,
            optimizer=optimizer,
            identity=_identity("pilot-001"),
            stage="route",
            step=0,
            result_payload={},
        )
