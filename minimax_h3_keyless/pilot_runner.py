from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import torch
import torch.nn as nn

from .attention import KeylessAttentionTrain
from .initialization import RouteInitMode
from .pilot import PilotLossWeights, build_training_student_block, set_pilot_block_stage
from .pilot_artifacts import (
    StageAArtifactReceipt,
    StageAArtifactRequest,
    persist_stage_a_block_artifacts,
)
from .pilot_attention_diagnostics import (
    PilotAttentionDiagnostic,
    compare_captured_native_keyless_attention,
)
from .pilot_campaign import (
    PILOT_BLOCKS,
    PILOT_LS_LAMBDAS,
    PilotAggregateMetrics,
    PilotRunIdentity,
    PilotTrainingEvent,
    evaluate_pilot_cases,
    train_pilot_stage,
    validate_pilot_gate_manifest,
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
from .route_fit import (
    RouteActivationFit,
    RouteActivationFitDiagnostics,
    collect_route_activation_statistics,
    solve_route_activation_fit,
)


StudentBuilder = Callable[[nn.Module, int, RouteInitMode, float], tuple[nn.Module, object]]
OptimizerFactory = Callable[[Sequence[nn.Parameter], float, float], torch.optim.Optimizer]
_SELECTION_SPLIT = "train_complete_cases"


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
            raise ValueError("Stage-A weight_decay must be non-negative")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("Stage-A max_grad_norm must be positive when specified")


@dataclass(frozen=True)
class StageAInitializationEvaluation:
    route_mode: RouteInitMode
    lambda_relative: float
    metrics: PilotAggregateMetrics
    route_fit_diagnostics: RouteActivationFitDiagnostics | None = None
    attention_diagnostics: tuple[PilotAttentionDiagnostic, ...] = ()


@dataclass(frozen=True)
class StageABlockPilotResult:
    block_index: int
    replay_reports: tuple[CapturedReplayReport, ...]
    initialization_evaluations: tuple[StageAInitializationEvaluation, ...]
    selected_route_mode: RouteInitMode
    selected_lambda_relative: float
    identity_baseline: PilotAggregateMetrics
    least_squares_baseline: PilotAggregateMetrics
    least_squares_baseline_lambda_relative: float
    candidate: PilotAggregateMetrics
    candidate_attention_diagnostics: tuple[PilotAttentionDiagnostic, ...]
    training_events: tuple[PilotTrainingEvent, ...]
    gate: StageABlockGateResult
    artifact: StageAArtifactReceipt | None = None


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
    if route_mode == "least_squares":
        route_mode = "identity"
        lambda_relative = 0.0
    return build_training_student_block(
        teacher_block,
        block_index=block_index,
        route_mode=route_mode,
        lambda_relative=lambda_relative,
    )


def _default_optimizer(
    parameters: Sequence[nn.Parameter], learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(parameters, lr=float(learning_rate), weight_decay=float(weight_decay))


def _move_cases(records, device: str | torch.device):
    return tuple(pilot_case_to_device(record.case, device) for record in records)


def _install_activation_route_fit(student: nn.Module, fit: RouteActivationFit) -> None:
    attention = getattr(student, "attn", None)
    route = getattr(attention, "query_route", None)
    weight = getattr(route, "weight", None)
    if not torch.is_tensor(weight):
        raise RuntimeError("Stage-A student does not expose a materialized query_route weight")
    if tuple(weight.shape) != tuple(fit.storage_weight.shape):
        raise RuntimeError(
            "Stage-A activation route fit geometry does not match the student query_route"
        )
    with torch.no_grad():
        weight.copy_(fit.storage_weight.to(device=weight.device, dtype=weight.dtype))


def _build_student(
    teacher_block: nn.Module,
    *,
    block_index: int,
    route_mode: RouteInitMode,
    lambda_relative: float,
    route_fits: Mapping[float, RouteActivationFit],
    student_builder: StudentBuilder,
) -> nn.Module:
    if route_mode == "identity":
        student, _ = student_builder(teacher_block, block_index, "identity", 0.0)
        return student
    fit = route_fits.get(float(lambda_relative))
    if fit is None:
        raise RuntimeError(
            f"missing activation-derived Stage-A route fit for lambda={lambda_relative:g}"
        )
    student, _ = student_builder(teacher_block, block_index, "identity", 0.0)
    _install_activation_route_fit(student, fit)
    return student


def _evaluate_attention_diagnostics(
    teacher_block: nn.Module,
    student_block: nn.Module,
    records,
) -> tuple[PilotAttentionDiagnostic, ...]:
    teacher_attention = getattr(teacher_block, "attn", None)
    student_attention = getattr(student_block, "attn", None)
    if teacher_attention is None:
        raise RuntimeError("Stage-A teacher block is missing attention for diagnostics")
    if not isinstance(student_attention, KeylessAttentionTrain):
        raise RuntimeError("Stage-A student block is missing KeylessAttentionTrain diagnostics target")
    diagnostics = tuple(
        compare_captured_native_keyless_attention(
            teacher_attention,
            student_attention,
            record,
        )
        for record in records
    )
    if not diagnostics:
        raise RuntimeError("Stage-A attention diagnostics require at least one capture")
    case_ids = tuple(row.case_id for row in diagnostics)
    expected = tuple(record.case.case_id for record in records)
    if case_ids != expected:
        raise RuntimeError("Stage-A attention diagnostic case ordering diverged from captures")
    return diagnostics


def select_stage_a_initialization(
    evaluations: Sequence[StageAInitializationEvaluation],
) -> StageAInitializationEvaluation:
    if not evaluations:
        raise ValueError("Stage-A initialization selection requires evaluations")
    identities = [row for row in evaluations if row.route_mode == "identity"]
    if len(identities) != 1:
        raise ValueError("Stage-A initialization grid must contain exactly one identity baseline")
    ls_rows = [row for row in evaluations if row.route_mode == "least_squares"]
    ls_lambdas = {row.lambda_relative for row in ls_rows}
    if len(ls_rows) != len(PILOT_LS_LAMBDAS) or ls_lambdas != set(PILOT_LS_LAMBDAS):
        raise ValueError(f"Stage-A initialization grid must contain LS lambdas {PILOT_LS_LAMBDAS}")
    if any(row.route_fit_diagnostics is None for row in ls_rows):
        raise ValueError("Stage-A LS initialization rows require captured-activation diagnostics")
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
    cases,
    records,
    *,
    block_index: int,
    device: str | torch.device,
    weights: PilotLossWeights,
    route_fits: Mapping[float, RouteActivationFit],
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
        student = _build_student(
            teacher_block,
            block_index=block_index,
            route_mode=route_mode,
            lambda_relative=float(lambda_relative),
            route_fits=route_fits,
            student_builder=student_builder,
        )
        student.to(device)
        student.eval()
        metrics = evaluate_pilot_cases(teacher_block, student, cases, weights=weights)
        attention_diagnostics = _evaluate_attention_diagnostics(
            teacher_block, student, records
        )
        fit = None if route_mode == "identity" else route_fits[float(lambda_relative)]
        evaluations.append(
            StageAInitializationEvaluation(
                route_mode=route_mode,
                lambda_relative=float(lambda_relative),
                metrics=metrics,
                route_fit_diagnostics=None if fit is None else fit.diagnostics,
                attention_diagnostics=attention_diagnostics,
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


def _evaluate_untrained_baseline(
    teacher_block: nn.Module,
    cases,
    *,
    block_index: int,
    device: str | torch.device,
    weights: PilotLossWeights,
    route_mode: RouteInitMode,
    lambda_relative: float,
    route_fits: Mapping[float, RouteActivationFit],
    student_builder: StudentBuilder,
) -> PilotAggregateMetrics:
    student = _build_student(
        teacher_block,
        block_index=block_index,
        route_mode=route_mode,
        lambda_relative=lambda_relative,
        route_fits=route_fits,
        student_builder=student_builder,
    )
    student.to(device)
    student.eval()
    try:
        return evaluate_pilot_cases(teacher_block, student, cases, weights=weights)
    finally:
        del student


def _result_payload(
    *,
    block_index: int,
    replay_reports: tuple[CapturedReplayReport, ...],
    evaluations: tuple[StageAInitializationEvaluation, ...],
    selected: StageAInitializationEvaluation,
    identity_baseline: PilotAggregateMetrics,
    least_squares_baseline: PilotAggregateMetrics,
    least_squares_baseline_lambda_relative: float,
    candidate: PilotAggregateMetrics,
    candidate_attention_diagnostics: tuple[PilotAttentionDiagnostic, ...],
    events: tuple[PilotTrainingEvent, ...],
    gate: StageABlockGateResult,
) -> dict:
    return {
        "block_index": int(block_index),
        "selection_split": _SELECTION_SPLIT,
        "replay_reports": [asdict(row) for row in replay_reports],
        "initialization_evaluations": [asdict(row) for row in evaluations],
        "selected_route_mode": selected.route_mode,
        "selected_lambda_relative": selected.lambda_relative,
        "identity_baseline": asdict(identity_baseline),
        "least_squares_baseline": asdict(least_squares_baseline),
        "least_squares_baseline_lambda_relative": float(
            least_squares_baseline_lambda_relative
        ),
        "candidate": asdict(candidate),
        "candidate_attention_diagnostics": [
            asdict(row) for row in candidate_attention_diagnostics
        ],
        "training_events": [asdict(row) for row in events],
        "gate": asdict(gate),
    }


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
    artifact_request: StageAArtifactRequest | None = None,
) -> StageABlockPilotResult:
    if block_index not in PILOT_BLOCKS:
        raise ValueError(f"Stage-A pilot block must be one of {PILOT_BLOCKS}")
    plan = validate_stage_a_train_plan(train_plan)
    policy = stage_a_policy_from_gate_manifest(gate_manifest)
    gate_sha = validate_pilot_gate_manifest(gate_manifest)
    if artifact_request is not None:
        artifact_request.assert_available(block_index, plan[-1].stage)
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

    route_statistics = collect_route_activation_statistics(teacher_block.attn, train_records)
    route_fits = {
        float(lambda_relative): solve_route_activation_fit(
            route_statistics,
            lambda_relative=float(lambda_relative),
        )
        for lambda_relative in PILOT_LS_LAMBDAS
    }
    train_cases = _move_cases(train_records, device)
    holdout_cases = _move_cases(holdout_records, device)

    # Initialization/model-selection decisions are training-data decisions.  The fixed
    # complete-case holdout remains untouched until the two untrained baselines and the
    # trained candidate are evaluated for the Stage-A exit gate.
    evaluations, selected, best_ls = _evaluate_initialization_grid(
        teacher_block,
        train_cases,
        train_records,
        block_index=block_index,
        device=device,
        weights=loss_weights,
        route_fits=route_fits,
        student_builder=student_builder,
    )
    identity_baseline = _evaluate_untrained_baseline(
        teacher_block,
        holdout_cases,
        block_index=block_index,
        device=device,
        weights=loss_weights,
        route_mode="identity",
        lambda_relative=0.0,
        route_fits=route_fits,
        student_builder=student_builder,
    )
    least_squares_baseline = _evaluate_untrained_baseline(
        teacher_block,
        holdout_cases,
        block_index=block_index,
        device=device,
        weights=loss_weights,
        route_mode="least_squares",
        lambda_relative=best_ls.lambda_relative,
        route_fits=route_fits,
        student_builder=student_builder,
    )

    student = _build_student(
        teacher_block,
        block_index=block_index,
        route_mode=selected.route_mode,
        lambda_relative=selected.lambda_relative,
        route_fits=route_fits,
        student_builder=student_builder,
    )
    student.to(device)
    events: list[PilotTrainingEvent] = []
    optimizer: torch.optim.Optimizer | None = None
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
    assert optimizer is not None

    candidate = evaluate_pilot_cases(teacher_block, student, holdout_cases, weights=loss_weights)
    candidate_attention_diagnostics = _evaluate_attention_diagnostics(
        teacher_block, student, holdout_records
    )
    event_tuple = tuple(events)
    gate = evaluate_stage_a_block_gate(
        block_index=block_index,
        candidate=candidate,
        identity_baseline=identity_baseline,
        least_squares_baseline=least_squares_baseline,
        training_events=event_tuple,
        policy=policy,
    )
    artifact = None
    if artifact_request is not None:
        run_identity = PilotRunIdentity(
            run_id=artifact_request.run_id,
            code_commit=artifact_request.code_commit,
            dataset_manifest_sha256=capture_set.dataset_manifest_sha256,
            gate_manifest_sha256=gate_sha,
            block_index=block_index,
            route_mode=selected.route_mode,
            lambda_relative=selected.lambda_relative,
        )
        artifact = persist_stage_a_block_artifacts(
            artifact_request,
            student_block=student,
            optimizer=optimizer,
            identity=run_identity,
            stage=plan[-1].stage,
            step=len(event_tuple),
            result_payload=_result_payload(
                block_index=block_index,
                replay_reports=replay_reports,
                evaluations=evaluations,
                selected=selected,
                identity_baseline=identity_baseline,
                least_squares_baseline=least_squares_baseline,
                least_squares_baseline_lambda_relative=best_ls.lambda_relative,
                candidate=candidate,
                candidate_attention_diagnostics=candidate_attention_diagnostics,
                events=event_tuple,
                gate=gate,
            ),
        )
    return StageABlockPilotResult(
        block_index=block_index,
        replay_reports=replay_reports,
        initialization_evaluations=evaluations,
        selected_route_mode=selected.route_mode,
        selected_lambda_relative=selected.lambda_relative,
        identity_baseline=identity_baseline,
        least_squares_baseline=least_squares_baseline,
        least_squares_baseline_lambda_relative=best_ls.lambda_relative,
        candidate=candidate,
        candidate_attention_diagnostics=candidate_attention_diagnostics,
        training_events=event_tuple,
        gate=gate,
        artifact=artifact,
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
    artifact_request: StageAArtifactRequest | None = None,
) -> StageACampaignPilotResult:
    if set(teacher_blocks) != set(PILOT_BLOCKS):
        raise ValueError(f"Stage-A campaign requires exactly teacher blocks {PILOT_BLOCKS}")
    plan = validate_stage_a_train_plan(train_plan)
    if artifact_request is not None:
        for block_index in PILOT_BLOCKS:
            artifact_request.assert_available(block_index, plan[-1].stage)
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
            artifact_request=artifact_request,
        )
        for block_index in PILOT_BLOCKS
    }
    campaign_gate = evaluate_stage_a_campaign_gate(
        {index: result.gate for index, result in block_results.items()}
    )
    return StageACampaignPilotResult(blocks=block_results, gate=campaign_gate)
