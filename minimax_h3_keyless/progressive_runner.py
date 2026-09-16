from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
import torch.nn as nn

from .attention import KeylessAttentionTrain
from .initialization import RouteInitMode
from .pilot import PilotLossWeights, build_training_student_block, set_pilot_block_stage
from .pilot_attention_diagnostics import (
    PilotAttentionDiagnostic,
    compare_captured_native_keyless_attention,
)
from .pilot_campaign import (
    PILOT_LS_LAMBDAS,
    PilotAggregateMetrics,
    PilotTrainingEvent,
    evaluate_pilot_cases,
    train_pilot_stage,
    validate_pilot_gate_manifest,
)
from .pilot_gates import StageAGatePolicy, stage_a_policy_from_gate_manifest
from .pilot_replay import CapturedReplayReport, pilot_case_to_device, verify_captured_pilot_replay
from .pilot_runner import (
    StageAInitializationEvaluation,
    StageATrainStage,
    select_stage_a_initialization,
    validate_stage_a_train_plan,
)
from .progressive import ProgressivePrefix
from .progressive_capture_set import ProgressiveBlockCaptureSet
from .progressive_gates import ProgressiveBlockGateResult, evaluate_progressive_block_gate
from .route_fit import (
    RouteActivationFit,
    collect_route_activation_statistics,
    solve_route_activation_fit,
)


StudentBuilder = Callable[[nn.Module, int, RouteInitMode, float], tuple[nn.Module, object]]
OptimizerFactory = Callable[[Sequence[nn.Parameter], float, float], torch.optim.Optimizer]


@dataclass
class ProgressiveBlockTrainingResult:
    """Unaccepted Stage-B candidate plus all local evidence used to decide its gate.

    Returning the candidate separately from ``ProgressivePrefix.advance`` is deliberate:
    a failed experiment cannot mutate the accepted model/prefix merely by completing a
    training call. Persistence and installation are a later explicit acceptance step.
    """

    block_index: int
    prefix_identity_sha256: str
    replay_reports: tuple[CapturedReplayReport, ...]
    initialization_evaluations: tuple[StageAInitializationEvaluation, ...]
    selected_route_mode: RouteInitMode
    selected_lambda_relative: float
    identity_baseline: PilotAggregateMetrics
    least_squares_baseline: PilotAggregateMetrics
    candidate: PilotAggregateMetrics
    candidate_attention_diagnostics: tuple[PilotAttentionDiagnostic, ...]
    training_events: tuple[PilotTrainingEvent, ...]
    gate: ProgressiveBlockGateResult
    final_stage: str
    student_block: nn.Module


def _default_builder(
    teacher_block: nn.Module,
    block_index: int,
    route_mode: RouteInitMode,
    lambda_relative: float,
) -> tuple[nn.Module, object]:
    if route_mode != "identity" or float(lambda_relative) != 0.0:
        raise ValueError("progressive structural builder only accepts identity initialization")
    return build_training_student_block(
        teacher_block,
        block_index=block_index,
        route_mode="identity",
        lambda_relative=0.0,
    )


def _default_optimizer(
    parameters: Sequence[nn.Parameter], learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )


def _require_native_teacher_block(
    teacher_block: nn.Module,
    *,
    require_bf16: bool,
) -> None:
    attention = getattr(teacher_block, "attn", None)
    if attention is None or isinstance(attention, KeylessAttentionTrain):
        raise RuntimeError("progressive local teacher must contain the original native QKV attention")
    required = ("qkv_proj", "q_norm", "k_norm", "out_proj", "heads", "head_dim")
    missing = [name for name in required if getattr(attention, name, None) is None]
    if missing:
        raise RuntimeError(f"progressive native teacher attention is missing {missing}")
    if getattr(attention, "qv_proj", None) is not None or getattr(attention, "query_route", None) is not None:
        raise RuntimeError("progressive local teacher must not already be a Keyless/QV attention")
    weight = getattr(attention.qkv_proj, "weight", None)
    if not torch.is_tensor(weight) or weight.ndim != 2 or getattr(weight, "is_meta", False):
        raise RuntimeError("progressive native teacher requires a materialized QKV weight")
    if not weight.is_floating_point():
        raise RuntimeError("progressive native teacher QKV weight must be floating point")
    if require_bf16 and weight.dtype != torch.bfloat16:
        raise RuntimeError(
            f"production progressive teacher must be BF16, got {weight.dtype}"
        )


def _validate_capture_context(
    captures: ProgressiveBlockCaptureSet,
    prefix: ProgressivePrefix,
) -> int:
    target = prefix.next_block
    if target is None:
        raise RuntimeError("progressive prefix is already complete")
    expected = {
        "target_block": target,
        "prefix_identity_sha256": prefix.identity_sha256.lower(),
        "stage_a_campaign_sha256": prefix.stage_a_campaign_sha256.lower(),
        "dataset_manifest_sha256": prefix.dataset_manifest_sha256.lower(),
        "gate_manifest_sha256": prefix.gate_manifest_sha256.lower(),
    }
    actual = {
        "target_block": captures.target_block,
        "prefix_identity_sha256": captures.prefix_identity_sha256.lower(),
        "stage_a_campaign_sha256": captures.stage_a_campaign_sha256.lower(),
        "dataset_manifest_sha256": captures.dataset_manifest_sha256.lower(),
        "gate_manifest_sha256": captures.gate_manifest_sha256.lower(),
    }
    mismatches = [name for name, value in expected.items() if actual[name] != value]
    if mismatches:
        raise RuntimeError(
            "progressive capture set is not bound to the requested accepted prefix: "
            + ", ".join(mismatches)
        )
    if captures.code_commit.lower() != prefix.code_commit.lower():
        raise RuntimeError("progressive capture set source revision differs from sweep revision")
    if not captures.train or not captures.holdout:
        raise RuntimeError("progressive target requires non-empty train and holdout captures")
    return target


def _move_cases(records, device: torch.device):
    return tuple(pilot_case_to_device(record.case, device) for record in records)


def _install_route_fit(student: nn.Module, fit: RouteActivationFit) -> None:
    attention = getattr(student, "attn", None)
    route = getattr(attention, "query_route", None)
    weight = getattr(route, "weight", None)
    if not torch.is_tensor(weight) or getattr(weight, "is_meta", False):
        raise RuntimeError("progressive student does not expose a materialized query_route weight")
    if tuple(weight.shape) != tuple(fit.storage_weight.shape):
        raise RuntimeError("progressive route fit geometry does not match student query_route")
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
    student, _ = student_builder(teacher_block, block_index, "identity", 0.0)
    attention = getattr(student, "attn", None)
    if not isinstance(attention, KeylessAttentionTrain):
        raise RuntimeError("progressive student builder must install KeylessAttentionTrain")
    if int(getattr(attention, "block_index", -1)) != block_index:
        raise RuntimeError("progressive student builder installed the wrong block_index")
    if route_mode == "identity":
        return student
    fit = route_fits.get(float(lambda_relative))
    if fit is None:
        raise RuntimeError(
            f"missing progressive activation-derived route fit for lambda={lambda_relative:g}"
        )
    _install_route_fit(student, fit)
    return student


def _attention_diagnostics(
    teacher_block: nn.Module,
    student_block: nn.Module,
    records,
) -> tuple[PilotAttentionDiagnostic, ...]:
    teacher_attention = getattr(teacher_block, "attn", None)
    student_attention = getattr(student_block, "attn", None)
    if teacher_attention is None or not isinstance(student_attention, KeylessAttentionTrain):
        raise RuntimeError("progressive attention diagnostic topology is invalid")
    out = tuple(
        compare_captured_native_keyless_attention(
            teacher_attention,
            student_attention,
            record,
        )
        for record in records
    )
    if not out:
        raise RuntimeError("progressive attention diagnostics require holdout captures")
    if tuple(row.case_id for row in out) != tuple(record.case.case_id for record in records):
        raise RuntimeError("progressive attention diagnostic case ordering diverged from captures")
    return out


def _evaluate_initializations(
    teacher_block: nn.Module,
    *,
    block_index: int,
    holdout_cases,
    holdout_records,
    device: torch.device,
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
        metrics = evaluate_pilot_cases(
            teacher_block,
            student,
            holdout_cases,
            weights=weights,
        )
        diagnostics = _attention_diagnostics(teacher_block, student, holdout_records)
        fit = None if route_mode == "identity" else route_fits[float(lambda_relative)]
        evaluations.append(
            StageAInitializationEvaluation(
                route_mode=route_mode,
                lambda_relative=float(lambda_relative),
                metrics=metrics,
                route_fit_diagnostics=None if fit is None else fit.diagnostics,
                attention_diagnostics=diagnostics,
            )
        )
        del student

    rows = tuple(evaluations)
    selected = select_stage_a_initialization(rows)
    best_ls = min(
        (row for row in rows if row.route_mode == "least_squares"),
        key=lambda row: (
            float(row.metrics.mean_attention_normalized_mse),
            float(row.metrics.mean_block_normalized_mse),
            float(row.lambda_relative),
        ),
    )
    return rows, selected, best_ls


def run_progressive_block_training(
    teacher_block: nn.Module,
    captures: ProgressiveBlockCaptureSet,
    prefix: ProgressivePrefix,
    *,
    device: str | torch.device,
    gate_manifest: Mapping[str, object],
    train_plan: Sequence[StageATrainStage],
    loss_weights: PilotLossWeights = PilotLossWeights(),
    same_input_atol: float = 0.0,
    same_input_rtol: float = 0.0,
    student_builder: StudentBuilder = _default_builder,
    optimizer_factory: OptimizerFactory = _default_optimizer,
    require_bf16_teacher: bool = True,
) -> ProgressiveBlockTrainingResult:
    """Train one early-to-late Stage-B block without mutating the accepted prefix/model.

    Captures must come from the current partially converted model under ``prefix``. The
    frozen original QKV block is replayed on those exact inputs and must reproduce the
    recorded post-AdaLN execution point before route fitting or optimization begins.
    The returned Keyless block is only a candidate; callers must persist and explicitly
    accept it after ``result.gate.passed`` before replacing the live model block.
    """

    block_index = _validate_capture_context(captures, prefix)
    validate_pilot_gate_manifest(gate_manifest)
    if captures.gate_manifest_sha256.lower() != prefix.gate_manifest_sha256.lower():
        raise RuntimeError("progressive gate manifest identity differs from capture policy")
    policy: StageAGatePolicy = stage_a_policy_from_gate_manifest(gate_manifest)
    plan = validate_stage_a_train_plan(train_plan)
    _require_native_teacher_block(teacher_block, require_bf16=require_bf16_teacher)

    device = torch.device(device)
    teacher_block.to(device)
    teacher_block.eval()
    for parameter in teacher_block.parameters():
        parameter.requires_grad_(False)

    train_records = captures.records("train")
    holdout_records = captures.records("holdout")
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

    evaluations, selected, best_ls = _evaluate_initializations(
        teacher_block,
        block_index=block_index,
        holdout_cases=holdout_cases,
        holdout_records=holdout_records,
        device=device,
        weights=loss_weights,
        route_fits=route_fits,
        student_builder=student_builder,
    )
    identity = next(row for row in evaluations if row.route_mode == "identity")

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
    for spec in plan:
        set_pilot_block_stage(student, spec.stage)
        parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError(f"progressive stage {spec.stage!r} exposed no trainable parameters")
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
    candidate_diagnostics = _attention_diagnostics(teacher_block, student, holdout_records)
    event_tuple = tuple(events)
    gate = evaluate_progressive_block_gate(
        block_index=block_index,
        candidate=candidate,
        identity_baseline=identity.metrics,
        least_squares_baseline=best_ls.metrics,
        training_events=event_tuple,
        policy=policy,
    )
    student.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)

    return ProgressiveBlockTrainingResult(
        block_index=block_index,
        prefix_identity_sha256=prefix.identity_sha256,
        replay_reports=replay_reports,
        initialization_evaluations=evaluations,
        selected_route_mode=selected.route_mode,
        selected_lambda_relative=selected.lambda_relative,
        identity_baseline=identity.metrics,
        least_squares_baseline=best_ls.metrics,
        candidate=candidate,
        candidate_attention_diagnostics=candidate_diagnostics,
        training_events=event_tuple,
        gate=gate,
        final_stage=plan[-1].stage,
        student_block=student,
    )
