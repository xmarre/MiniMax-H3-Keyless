from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
import torch.nn as nn

from .initialization import RouteInitMode
from .pilot import (
    PilotLossWeights,
    build_training_student_block,
    set_pilot_block_stage,
)
from .pilot_campaign import (
    PILOT_BLOCKS,
    PILOT_LS_LAMBDAS,
    PilotAggregateMetrics,
    PilotTrainingEvent,
    evaluate_pilot_cases,
    train_pilot_stage,
)
from .pilot_capture_set import StageACaptureSet
from .pilot_gates import (
    StageABlockGateResult,
    StageACampaignGateResult,
    evaluate_stage_a_block_gate,
    evaluate_stage_a_campaign_gate,
    stage_a_policy_from_gate_manifest,
)
from .pilot_replay import CapturedReplayReport, pilot_case_to_device, verify_captured_pilot_replay


StudentBuilder = Callable[[nn.Module, int, RouteInitMode, float], tuple[nn.Module, object]]
OptimizerFactory = Callable[[Sequence[nn.Parameter], float, float], torch.optim.Optimizer]


@dataclass(frozen=True)
class StageATrainStage:
    stage: str
    epochs: int
    learning_rate: float
    weight_decay: float = 0.0
    max_grad_norm: float | None = None

    def __post_init__(self) -> None:
        if self.stage not in ("route", "query", "value", "norm_out"):
            raise ValueError(f"unsupported Stage-A training stage: {self.stage!r}")
        if self.epochs <= 0:
            raise ValueError("Stage-A stage epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Stage-A learning rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("Stage-A weight decay must be non-negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("Stage-A max_grad_norm must be positive when specified")


@dataclass(frozen=True)
class StageAInitializationEvaluation:
    route_mode: RouteInitMode
    lambda_relative: float
    metrics: PilotAggregateMetrics


@dataclass(frozen=True)
class StageABlockPilotResult:
    block_index: int
    replay_reports: tuple[CapturedReplayReport, ...]
    initialization_evaluations: tuple[StageAInitializationEvaluation, ...]
    selected_route_mode: RouteInitMode
    selected_lambda_relative: float
    identity_baseline: PilotAggregateMetrics
    least_squares_baseline: PilotAggregateMetrics
    candidate: PilotAggregateMetrics
    training_events: tuple[PilotTrainingEvent, ...]
    gate: StageABlockGateResult


@dataclass(frozen=True)
class StageACampaignPilotResult:
    blocks: Mapping[int, StageABlockPilotResult]
    gate: StageACampaignGateResult


def validate_stage_a_train_plan(plan: Sequence[StageATrainStage]) -> tuple[StageATrainStage, ...]:
    if not plan:
        raise ValueError("Stage-A training plan must contain at least the route stage")
    ranks = {"route": 0, "query": 1, "value": 2, "norm_out": 3}
    if plan[0].stage != "route":
        raise ValueError("Stage-A training plan must start with route calibration")
    seen: set[str] = set()
    previous = -1
    for spec in plan:
        if spec.stage in seen:
            raise ValueError(f"Stage-A training plan repeats stage {spec.stage!r}")
        rank = ranks[spec.stage]
        if rank <= previous:
            raise ValueError("Stage-A training stages must follow route→query→value→norm_out order")
        if spec.stage == "norm_out" and "value" not in seen:
            raise ValueError("Stage-A norm/out escalation requires value to be unfrozen first")
        seen.add(spec.stage)
        previous = rank
    return tuple(plan)


def _default_builder(
    teacher_block: nn.Module,
    block_index: int,
    route_mode: RouteInitMode,
    lambda_relative: float,
) -> tuple[nn.Module, object]:
    return build_training_student_block(
        teacher_block,
        block_index=block_index,
        route_mode=route_mode,
        lambda_relative=lambda_relative,
    )


def _default_optimizer(
    parameters: Sequence[nn.Parameter], learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )


def _move_cases(records, device: str | torch.device):
    return tuple(pilot_case_to_device(record.case, device) for record in records)


def select_stage_a_initialization(
    evaluations: Sequence[StageAInitializationEvaluation],
) -> StageAInitializationEvaluation:
    """Select initialization by held-out attention error with deterministic tie breaks."""
    if not evaluations:
        raise ValueError("Stage-A initialization selection requires evaluations")
    identities = [row for row in evaluations if row.route_mode == "identity"]
    if len(identities) != 1:
        raise ValueError("Stage-A initialization grid must contain exactly one identity baseline")
    ls_lambdas = {
        row.lambda_relative for row in evaluations if row.route_mode == "least_squares"
    }
    if ls_lambdas != set(PILOT_LS_LAMBDAS):
        raise ValueError(
            f"Stage-A initialization grid must contain LS lambdas {PILOT_LS_LAMBDAS}"
        )
    return min(
        evaluations,
        key=lambda row: (
            float(row.metrics.mean_attention_normalized_mse),
            float(row.metrics.mean_block_normalized_mse),
            0 if row.route_mode == "identity" else 1,
            float(row.lambda_relative),
        ),
    )


def _evaluate_initialization_grid(
    teacher_block: nn.Module,
    holdout_cases,
    *,
    block_index: int,
    device: str | torch.device,
    weights: PilotLossWeights,
    student_builder: StudentBuilder,
) -> tuple[
    tuple[StageAInitializationEvaluation, ...],
    StageAInitializationEvaluation,
    StageAInitializationEvaluation,
]:
    evaluations: list[StageAInitializationEvaluation] = []
    grid: tuple[tuple[RouteInitMode, float], ...] = (
        ("identity", 0.0),
        *(("least_squares", value) for value in PILOT_LS_LAMBDAS),
    )
    for route_mode, lambda_relative in grid:
        student, _ = student_builder(
            teacher_block,
            block_index,
            route_mode,
            float(lambda_relative),
        )
        student.to(device)
        student.eval()
        metrics = evaluate_pilot_cases(
            teacher_block,
            student,
            holdout_cases,
            weights=weights,
        )
        evaluations.append(
            StageAInitializationEvaluation(
                route_mode=route_mode,
                lambda_relative=float(lambda_relative),
                metrics=metrics,
            )
        )
        del student

    selected = select_stage_a_initialization(evaluations)
    identity = next(row for row in evaluations if row.route_mode == "identity")
    best_ls = min(
        (row for row in evaluations if row.route_mode == "least_squares"),
        key=lambda row: (
            float(row.metrics.mean_attention_normalized_mse),
            float(row.metrics.mean_block_normalized_mse),
            float(row.lambda_relative),
        ),
    )
    return tuple(evaluations), selected, best_ls


def run_stage_a_block_pilot(
    teacher_block: nn.Module,
    capture_set: StageACaptureSet,
    *,
    block_index: int,
    device: str | torch.device,
    gate_manifest: Mapping[str, object],
    train_plan: Sequence[StageATrainStage],
    loss_weights: PilotLossWeights = PilotLossWeights(),
    same_input_atol: float = 0.0,
    same_input_rtol: float = 0.0,
    student_builder: StudentBuilder = _default_builder,
    optimizer_factory: OptimizerFactory = _default_optimizer,
) -> StageABlockPilotResult:
    """Run one bounded Stage-A depth pilot from immutable live captures.

    The initialization grid is evaluated before training on the fixed holdout corpus.
    Training then starts from the selected identity/LS initialization and follows the
    predeclared monotonic freeze schedule. A fresh optimizer is deliberately created
    after each stage transition so stale frozen parameters cannot remain in its groups.
    The returned gate result may fail; callers must not weaken the fixed gate manifest
    in response to a failed run.
    """
    if block_index not in PILOT_BLOCKS:
        raise ValueError(f"Stage-A pilot block must be one of {PILOT_BLOCKS}")
    plan = validate_stage_a_train_plan(train_plan)
    policy = stage_a_policy_from_gate_manifest(gate_manifest)
    device = torch.device(device)
    teacher_block.to(device)
    teacher_block.eval()
    for parameter in teacher_block.parameters():
        parameter.requires_grad_(False)

    train_records = capture_set.records(block_index, "train")
    holdout_records = capture_set.records(block_index, "holdout")
    replay_reports = tuple(
        verify_captured_pilot_replay(
            teacher_block,
            record,
            device=device,
            same_input_atol=same_input_atol,
            same_input_rtol=same_input_rtol,
        )
        for record in (*train_records, *holdout_records)
    )
    train_cases = _move_cases(train_records, device)
    holdout_cases = _move_cases(holdout_records, device)

    evaluations, selected, best_ls = _evaluate_initialization_grid(
        teacher_block,
        holdout_cases,
        block_index=block_index,
        device=device,
        weights=loss_weights,
        student_builder=student_builder,
    )
    identity = next(row for row in evaluations if row.route_mode == "identity")

    student, _ = student_builder(
        teacher_block,
        block_index,
        selected.route_mode,
        selected.lambda_relative,
    )
    student.to(device)
    events: list[PilotTrainingEvent] = []
    for spec in plan:
        set_pilot_block_stage(student, spec.stage)
        parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError(f"Stage-A stage {spec.stage!r} exposed no trainable parameters")
        optimizer = optimizer_factory(parameters, spec.learning_rate, spec.weight_decay)
        events.extend(
            train_pilot_stage(
                teacher_block,
                student,
                train_cases,
                optimizer,
                stage=spec.stage,
                epochs=spec.epochs,
                weights=loss_weights,
                max_grad_norm=spec.max_grad_norm,
            )
        )

    candidate = evaluate_pilot_cases(
        teacher_block,
        student,
        holdout_cases,
        weights=loss_weights,
    )
    gate = evaluate_stage_a_block_gate(
        block_index=block_index,
        candidate=candidate,
        identity_baseline=identity.metrics,
        least_squares_baseline=best_ls.metrics,
        training_events=events,
        policy=policy,
    )
    return StageABlockPilotResult(
        block_index=block_index,
        replay_reports=replay_reports,
        initialization_evaluations=evaluations,
        selected_route_mode=selected.route_mode,
        selected_lambda_relative=selected.lambda_relative,
        identity_baseline=identity.metrics,
        least_squares_baseline=best_ls.metrics,
        candidate=candidate,
        training_events=tuple(events),
        gate=gate,
    )


def run_stage_a_campaign(
    teacher_blocks: Mapping[int, nn.Module],
    capture_set: StageACaptureSet,
    *,
    device: str | torch.device,
    gate_manifest: Mapping[str, object],
    train_plan: Sequence[StageATrainStage],
    loss_weights: PilotLossWeights = PilotLossWeights(),
    same_input_atol: float = 0.0,
    same_input_rtol: float = 0.0,
    student_builder: StudentBuilder = _default_builder,
    optimizer_factory: OptimizerFactory = _default_optimizer,
) -> StageACampaignPilotResult:
    if set(teacher_blocks) != set(PILOT_BLOCKS):
        raise ValueError(f"Stage-A campaign requires exactly teacher blocks {PILOT_BLOCKS}")
    plan = validate_stage_a_train_plan(train_plan)
    block_results = {
        block_index: run_stage_a_block_pilot(
            teacher_blocks[block_index],
            capture_set,
            block_index=block_index,
            device=device,
            gate_manifest=gate_manifest,
            train_plan=plan,
            loss_weights=loss_weights,
            same_input_atol=same_input_atol,
            same_input_rtol=same_input_rtol,
            student_builder=student_builder,
            optimizer_factory=optimizer_factory,
        )
        for block_index in PILOT_BLOCKS
    }
    campaign_gate = evaluate_stage_a_campaign_gate(
        {index: result.gate for index, result in block_results.items()}
    )
    return StageACampaignPilotResult(blocks=block_results, gate=campaign_gate)
