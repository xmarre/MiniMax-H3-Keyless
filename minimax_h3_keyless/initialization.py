from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .attention import KeylessAttentionTrain


RouteInitMode = Literal["identity", "least_squares"]
PilotTrainStage = Literal["route", "query", "value", "norm_out"]


@dataclass(frozen=True)
class RouteInitializationReport:
    mode: RouteInitMode
    lambda_relative: float | None
    lambda_actual: tuple[float, ...] | None


def _require_shape(name: str, tensor: torch.Tensor, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{name}: expected shape {expected}, got {tuple(tensor.shape)}")


def initialize_training_attention_from_native(
    student: KeylessAttentionTrain,
    *,
    qkv_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    gate_compress_weight: torch.Tensor | None = None,
    route_mode: RouteInitMode = "identity",
    lambda_relative: float = 0.0,
    route_storage_weight: torch.Tensor | None = None,
    lambda_actual: tuple[float, ...] | None = None,
) -> RouteInitializationReport:
    """Initialize one block-local Keyless student from its exact native-QKV teacher.

    Q and raw retrieval V are copied directly. ``route_norm`` starts from teacher
    ``k_norm`` only as a scale prior. Least-squares route weights are deliberately not
    inferred from projection weights here: Stage A must fit them from captured post-AdaLN
    train activations and supply the storage-orientation result explicitly.
    """
    inner = student.inner_dim
    hidden = student.hidden
    heads = student.heads
    head_dim = student.head_dim
    _require_shape("qkv_weight", qkv_weight, (3 * inner, hidden))
    _require_shape("q_norm_weight", q_norm_weight, (head_dim,))
    _require_shape("k_norm_weight", k_norm_weight, (head_dim,))
    _require_shape("out_proj_weight", out_proj_weight, (hidden, inner))

    q_storage, _, v_storage = qkv_weight.split(inner, dim=0)
    report_lambda_actual: tuple[float, ...] | None = None

    with torch.no_grad():
        student.q_proj.weight.copy_(q_storage)
        student.v_proj.weight.copy_(v_storage)
        student.q_norm.weight.copy_(q_norm_weight)
        student.route_norm.weight.copy_(k_norm_weight)
        student.out_proj.weight.copy_(out_proj_weight)

        if route_mode == "identity":
            if lambda_relative != 0.0:
                raise ValueError("lambda_relative is only meaningful for least_squares initialization")
            if route_storage_weight is not None or lambda_actual is not None:
                raise ValueError("identity initialization must not supply an activation LS route")
            student.query_route.reset_identity()
        elif route_mode == "least_squares":
            if lambda_relative < 0:
                raise ValueError("lambda_relative must be non-negative")
            if route_storage_weight is None or lambda_actual is None:
                raise ValueError(
                    "least_squares initialization requires a route fitted from captured train activations"
                )
            _require_shape(
                "route_storage_weight",
                route_storage_weight,
                (heads, head_dim, head_dim),
            )
            if len(lambda_actual) != heads:
                raise ValueError("lambda_actual must contain one value per attention head")
            checked_lambdas = tuple(float(value) for value in lambda_actual)
            if any(not torch.isfinite(torch.tensor(value)) or value < 0.0 for value in checked_lambdas):
                raise ValueError("lambda_actual values must be finite and non-negative")
            student.query_route.weight.copy_(
                route_storage_weight.to(
                    device=student.query_route.weight.device,
                    dtype=student.query_route.weight.dtype,
                )
            )
            report_lambda_actual = checked_lambdas
        else:
            raise ValueError(f"unsupported route initialization mode: {route_mode!r}")

        if student.to_gate_compress is None:
            if gate_compress_weight is not None:
                raise ValueError("teacher has gate-compress weight but student gate compression is disabled")
        else:
            if gate_compress_weight is None:
                raise ValueError("student gate compression is enabled but teacher weight was not provided")
            _require_shape("gate_compress_weight", gate_compress_weight, (inner, hidden))
            student.to_gate_compress.weight.copy_(gate_compress_weight)

    return RouteInitializationReport(
        mode=route_mode,
        lambda_relative=None if route_mode == "identity" else float(lambda_relative),
        lambda_actual=report_lambda_actual,
    )


def set_pilot_trainable_stage(
    student: KeylessAttentionTrain,
    stage: PilotTrainStage,
) -> tuple[str, ...]:
    """Apply the Stage-A freeze schedule to one Keyless attention block.

    ``norm_out`` is the final escalation and intentionally leaves gate-compress frozen;
    copied MLP/AdaLN/non-attention tensors live outside this module and are unaffected.
    """
    stages = {
        "route": ("query_route.weight", "route_norm.weight"),
        "query": ("query_route.weight", "route_norm.weight", "q_proj.weight"),
        "value": (
            "query_route.weight",
            "route_norm.weight",
            "q_proj.weight",
            "v_proj.weight",
        ),
        "norm_out": (
            "query_route.weight",
            "route_norm.weight",
            "q_proj.weight",
            "v_proj.weight",
            "q_norm.weight",
            "out_proj.weight",
        ),
    }
    if stage not in stages:
        raise ValueError(f"unsupported pilot trainable stage: {stage!r}")
    enabled = set(stages[stage])
    seen = set()
    for name, parameter in student.named_parameters():
        parameter.requires_grad_(name in enabled)
        if parameter.requires_grad:
            seen.add(name)
    missing = enabled.difference(seen)
    if missing:
        raise RuntimeError(f"student is missing expected pilot parameters: {sorted(missing)}")
    return tuple(name for name in stages[stage])
