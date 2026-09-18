from __future__ import annotations

from types import SimpleNamespace

import pytest

from minimax_h3_keyless.pilot import PilotStepReport
from minimax_h3_keyless.pilot_campaign import PilotTrainingEvent
from minimax_h3_keyless.pilot_runner import StageATrainStage
from minimax_h3_keyless.progressive_runner import _validate_resume_progress
from minimax_h3_keyless.progressive_training_resume import ProgressiveTrainingResumeState


def _report() -> PilotStepReport:
    return PilotStepReport(
        total=0.5,
        attention_normalized_mse=0.2,
        block_normalized_mse=0.3,
        attention_cosine=0.9,
        block_cosine=0.8,
        trainable_parameters=12,
        gradient_l2_norm=1.25,
    )


def _event(stage: str, epoch: int, case_id: str) -> PilotTrainingEvent:
    return PilotTrainingEvent(
        stage=stage,
        epoch=epoch,
        case_id=case_id,
        report=_report(),
    )


def _plan() -> tuple[StageATrainStage, ...]:
    return (
        StageATrainStage("route", 2, 1e-3),
        StageATrainStage("query", 2, 1e-3),
    )


def _state(events: tuple[PilotTrainingEvent, ...]) -> ProgressiveTrainingResumeState:
    return ProgressiveTrainingResumeState(
        path="resume.pt",
        checkpoint_sha256="a" * 64,
        identity=SimpleNamespace(),
        stage_index=1,
        stage="query",
        completed_epochs=1,
        events=events,
        student_state_dict={},
        optimizer_state_dict={},
        rng_state={},
    )


def _expected_events(case_ids: tuple[str, ...]) -> tuple[PilotTrainingEvent, ...]:
    rows: list[PilotTrainingEvent] = []
    for epoch in range(2):
        rows.extend(_event("route", epoch, case_id) for case_id in case_ids)
    rows.extend(_event("query", 0, case_id) for case_id in case_ids)
    return tuple(rows)


def test_resume_progress_accepts_exact_stage_epoch_capture_traversal() -> None:
    case_ids = ("train-a", "train-b")
    _validate_resume_progress(
        _state(_expected_events(case_ids)),
        _plan(),
        train_case_ids=case_ids,
    )


def test_resume_progress_rejects_same_count_with_reordered_capture_cases() -> None:
    case_ids = ("train-a", "train-b")
    rows = list(_expected_events(case_ids))
    rows[2], rows[3] = rows[3], rows[2]
    with pytest.raises(RuntimeError, match="exact capture traversal"):
        _validate_resume_progress(
            _state(tuple(rows)),
            _plan(),
            train_case_ids=case_ids,
        )


def test_resume_progress_rejects_same_count_with_foreign_case_id() -> None:
    case_ids = ("train-a", "train-b")
    rows = list(_expected_events(case_ids))
    rows[-1] = _event("query", 0, "other-registry-case")
    with pytest.raises(RuntimeError, match="exact capture traversal"):
        _validate_resume_progress(
            _state(tuple(rows)),
            _plan(),
            train_case_ids=case_ids,
        )


def test_resume_progress_rejects_same_count_with_wrong_epoch() -> None:
    case_ids = ("train-a", "train-b")
    rows = list(_expected_events(case_ids))
    rows[1] = _event("route", 1, "train-b")
    with pytest.raises(RuntimeError, match="exact capture traversal"):
        _validate_resume_progress(
            _state(tuple(rows)),
            _plan(),
            train_case_ids=case_ids,
        )
