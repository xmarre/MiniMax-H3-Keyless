from __future__ import annotations

import copy
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from minimax_h3_keyless.attention import KeylessAttentionTrain
from minimax_h3_keyless.pilot import PilotStepReport, set_pilot_block_stage
from minimax_h3_keyless.pilot_campaign import PilotTrainingEvent
from minimax_h3_keyless.progressive_training_resume import (
    ProgressiveTrainingResumeIdentity,
    ProgressiveTrainingResumeRequest,
    load_progressive_training_resume,
    remove_progressive_training_resume,
    restore_progressive_training_resume,
    save_progressive_training_resume,
)


class TinyStudent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = KeylessAttentionTrain(
            4,
            2,
            2,
            1e-5,
            block_index=0,
            dtype=torch.float32,
        )
        self.mlp = nn.Linear(4, 4, bias=False)
        set_pilot_block_stage(self, "route")


def _identity() -> ProgressiveTrainingResumeIdentity:
    return ProgressiveTrainingResumeIdentity(
        sweep_id="resume-test",
        code_commit="a" * 40,
        stage_a_campaign_sha256="1" * 64,
        dataset_manifest_sha256="2" * 64,
        gate_manifest_sha256="3" * 64,
        prefix_identity_sha256="4" * 64,
        prefix_manifest_sha256="5" * 64,
        capture_registry_file_sha256="6" * 64,
        train_plan_identity_sha256="7" * 64,
        train_plan_file_sha256="8" * 64,
        block_index=0,
        selected_route_mode="identity",
        selected_lambda_relative=0.0,
    )


def _event(epoch: int = 0, case_id: str = "case-a") -> PilotTrainingEvent:
    return PilotTrainingEvent(
        stage="route",
        epoch=epoch,
        case_id=case_id,
        report=PilotStepReport(
            total=0.5,
            attention_normalized_mse=0.2,
            block_normalized_mse=0.3,
            attention_cosine=0.9,
            block_cosine=0.8,
            trainable_parameters=12,
            gradient_l2_norm=1.25,
        ),
    )


def _optimizer(student: TinyStudent) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=1e-3,
    )


def test_resume_request_requires_explicit_hash_bound_inputs() -> None:
    request = ProgressiveTrainingResumeRequest(
        path="scratch.pt",
        prefix_manifest_sha256="1" * 64,
        capture_registry_file_sha256="2" * 64,
        train_plan_identity_sha256="3" * 64,
        train_plan_file_sha256="4" * 64,
        resume=True,
    )
    assert request.resume is True
    with pytest.raises(ValueError, match="SHA-256"):
        replace(request, prefix_manifest_sha256="bad")


def test_save_load_restore_binds_identity_student_optimizer_events_and_rng(tmp_path: Path) -> None:
    torch.manual_seed(1001)
    np.random.seed(1002)
    random.seed(1003)
    student = TinyStudent()
    optimizer = _optimizer(student)

    # Populate Adam state so a successful restore proves more than parameter-group shape.
    loss = student.attn.query_route.weight.square().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    path = tmp_path / "block00.training-resume.pt"
    weight_at_save = student.attn.query_route.weight.detach().clone()
    torch_state_at_save = torch.get_rng_state().clone()
    expected_generator = torch.Generator()
    expected_generator.set_state(torch_state_at_save)
    expected_next = torch.rand(5, generator=expected_generator)

    digest = save_progressive_training_resume(
        path,
        student_block=student,
        optimizer=optimizer,
        identity=_identity(),
        stage_index=0,
        stage="route",
        completed_epochs=1,
        events=(_event(),),
    )
    assert len(digest) == 64

    state = load_progressive_training_resume(path, expected_identity=_identity())
    assert state.checkpoint_sha256 == digest
    assert state.stage_index == 0
    assert state.stage == "route"
    assert state.completed_epochs == 1
    assert state.events == (_event(),)

    with torch.no_grad():
        student.attn.query_route.weight.add_(10.0)
    torch.manual_seed(9999)
    restored_optimizer = _optimizer(student)
    restore_progressive_training_resume(
        state,
        student_block=student,
        optimizer=restored_optimizer,
        restore_rng=True,
    )
    torch.testing.assert_close(student.attn.query_route.weight, weight_at_save)
    torch.testing.assert_close(torch.rand(5), expected_next)
    assert restored_optimizer.state_dict()["state"]


def test_resume_load_rejects_different_immutable_identity(tmp_path: Path) -> None:
    student = TinyStudent()
    optimizer = _optimizer(student)
    path = tmp_path / "resume.pt"
    save_progressive_training_resume(
        path,
        student_block=student,
        optimizer=optimizer,
        identity=_identity(),
        stage_index=0,
        stage="route",
        completed_epochs=1,
        events=(_event(),),
    )
    wrong = replace(_identity(), prefix_manifest_sha256="f" * 64)
    with pytest.raises(RuntimeError, match="identity does not match"):
        load_progressive_training_resume(path, expected_identity=wrong)


def test_resume_checkpoint_is_mutable_scratch_and_removable_after_persistence(tmp_path: Path) -> None:
    student = TinyStudent()
    optimizer = _optimizer(student)
    path = tmp_path / "resume.pt"
    first = save_progressive_training_resume(
        path,
        student_block=student,
        optimizer=optimizer,
        identity=_identity(),
        stage_index=0,
        stage="route",
        completed_epochs=1,
        events=(_event(),),
    )
    with torch.no_grad():
        student.attn.query_route.weight.add_(0.25)
    second = save_progressive_training_resume(
        path,
        student_block=student,
        optimizer=optimizer,
        identity=_identity(),
        stage_index=0,
        stage="route",
        completed_epochs=2,
        events=(_event(0, "case-a"), _event(1, "case-a")),
    )
    assert first != second
    assert path.is_file()
    remove_progressive_training_resume(path)
    assert not path.exists()
    remove_progressive_training_resume(path)
