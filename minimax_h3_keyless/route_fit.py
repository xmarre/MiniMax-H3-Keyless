from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation_capture import CapturedPilotCase


@dataclass(frozen=True)
class RouteActivationStatistics:
    """Train-split sufficient statistics for one native H3 attention block.

    ``gram`` is sum(V^T V) and ``cross`` is sum(V^T K), both in mathematical
    per-head activation orientation and stored as CPU float64 tensors. The raw captured
    rows are never concatenated, and holdout rows must not be supplied here.
    """

    gram: torch.Tensor
    cross: torch.Tensor
    rows: int
    heads: int
    head_dim: int
    smallest_singular_value: tuple[float, ...]
    largest_singular_value: tuple[float, ...]
    numerical_rank: tuple[int, ...]
    full_rank_condition_number: tuple[float | None, ...]


@dataclass(frozen=True)
class RouteActivationFitDiagnostics:
    rows: int
    lambda_relative: float
    lambda_actual: tuple[float, ...]
    smallest_singular_value: tuple[float, ...]
    largest_singular_value: tuple[float, ...]
    numerical_rank: tuple[int, ...]
    full_rank_condition_number: tuple[float | None, ...]


@dataclass(frozen=True)
class RouteActivationFit:
    """Storage-orientation per-head route weight plus persisted numerical diagnostics."""

    storage_weight: torch.Tensor
    diagnostics: RouteActivationFitDiagnostics


def _native_projection_geometry(native_attention: nn.Module) -> tuple[torch.Tensor, int, int, int]:
    qkv = getattr(native_attention, "qkv_proj", None)
    weight = getattr(qkv, "weight", None)
    if not torch.is_tensor(weight) or weight.ndim != 2:
        raise RuntimeError("Stage-A activation route fitting requires a materialized native qkv weight")
    if getattr(weight, "is_meta", False):
        raise RuntimeError("Stage-A activation route fitting cannot use meta-device qkv weights")
    if not weight.is_floating_point():
        raise RuntimeError(
            f"Stage-A activation route fitting requires a floating teacher projection, got {weight.dtype}"
        )
    bias = getattr(qkv, "bias", None)
    if bias is not None:
        raise RuntimeError("Stage-A activation route fitting assumes native H3 bias-free qkv projection")
    heads = int(getattr(native_attention, "heads", 0))
    head_dim = int(getattr(native_attention, "head_dim", 0))
    inner = heads * head_dim
    if heads <= 0 or head_dim <= 0 or weight.shape[0] != 3 * inner:
        raise RuntimeError("native H3 attention geometry is incompatible with Stage-A route fitting")
    return weight, heads, head_dim, int(weight.shape[1])


def collect_route_activation_statistics(
    native_attention: nn.Module,
    records: Sequence[CapturedPilotCase],
    *,
    chunk_rows: int = 512,
) -> RouteActivationStatistics:
    """Accumulate LS sufficient statistics from captured post-AdaLN train activations.

    The production Stage-A caller has already validated the pinned BF16 teacher. This
    helper accepts other floating dtypes only so small synthetic contract tests can run
    on CPU without weakening the production loader/runner gate.

    The teacher K/V projections are evaluated from the exact captured attention input.
    Accumulation stays bounded: only one row chunk plus two ``[heads,d,d]`` FP32
    accumulators live on the teacher device. The final small matrices are promoted to
    CPU float64 for eigendiagnostics and solves.
    """
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError("route-fit chunk_rows must be a positive integer")
    if not records:
        raise ValueError("Stage-A activation route fitting requires train captures")

    weight, heads, head_dim, hidden = _native_projection_geometry(native_attention)
    inner = heads * head_dim
    k_weight = weight[inner : 2 * inner]
    v_weight = weight[2 * inner :]
    device = weight.device
    gram = torch.zeros((heads, head_dim, head_dim), dtype=torch.float32, device=device)
    cross = torch.zeros_like(gram)
    rows = 0

    with torch.no_grad():
        for record in records:
            hidden_input = record.attention_input
            if not torch.is_tensor(hidden_input) or hidden_input.shape[-1] != hidden:
                raise ValueError(
                    "captured post-AdaLN attention input does not match teacher hidden width"
                )
            flat = hidden_input.reshape(-1, hidden)
            if flat.shape[0] == 0:
                raise ValueError("captured post-AdaLN attention input contains zero rows")
            rows += int(flat.shape[0])
            for start in range(0, int(flat.shape[0]), chunk_rows):
                stop = min(int(flat.shape[0]), start + chunk_rows)
                x = flat[start:stop].to(device=device, dtype=weight.dtype)
                k = F.linear(x, k_weight).reshape(-1, heads, head_dim).float()
                v = F.linear(x, v_weight).reshape(-1, heads, head_dim).float()
                if not torch.isfinite(k).all() or not torch.isfinite(v).all():
                    raise RuntimeError("non-finite teacher K/V activation during Stage-A route fitting")
                gram.add_(torch.einsum("rhd,rhe->hde", v, v))
                cross.add_(torch.einsum("rhd,rhe->hde", v, k))

    gram64 = (0.5 * (gram + gram.transpose(-1, -2))).double().cpu()
    cross64 = cross.double().cpu()
    if not torch.isfinite(gram64).all() or not torch.isfinite(cross64).all():
        raise RuntimeError("non-finite Stage-A route sufficient statistics")

    eigvals = torch.linalg.eigvalsh(gram64).clamp_min(0.0)
    singular = eigvals.sqrt()
    smallest = singular[:, 0]
    largest = singular[:, -1]
    eps = torch.finfo(torch.float32).eps
    tolerance = largest * (head_dim * eps)
    rank = (singular > tolerance.unsqueeze(-1)).sum(dim=-1)
    conditions: list[float | None] = []
    for head in range(heads):
        if int(rank[head].item()) == head_dim and float(smallest[head].item()) > 0.0:
            conditions.append(float((largest[head] / smallest[head]).item()))
        else:
            conditions.append(None)

    return RouteActivationStatistics(
        gram=gram64,
        cross=cross64,
        rows=rows,
        heads=heads,
        head_dim=head_dim,
        smallest_singular_value=tuple(float(x) for x in smallest.tolist()),
        largest_singular_value=tuple(float(x) for x in largest.tolist()),
        numerical_rank=tuple(int(x) for x in rank.tolist()),
        full_rank_condition_number=tuple(conditions),
    )


def solve_route_activation_fit(
    statistics: RouteActivationStatistics,
    *,
    lambda_relative: float,
) -> RouteActivationFit:
    """Solve ``W = (V^T V + lambda I)^-1 V^T K`` in storage orientation.

    ``KeylessAttentionTrain.query_route`` applies its stored weight as ``q @ W.T``.
    Therefore this is exactly the storage transpose of the design's mathematical
    ``R = K^T V (V^T V + lambda I)^-1``.
    """
    if not isinstance(lambda_relative, (int, float)) or isinstance(lambda_relative, bool):
        raise ValueError("lambda_relative must be a non-negative finite number")
    lambda_relative = float(lambda_relative)
    if not torch.isfinite(torch.tensor(lambda_relative)) or lambda_relative < 0.0:
        raise ValueError("lambda_relative must be a non-negative finite number")
    gram = statistics.gram
    cross = statistics.cross
    if gram.shape != cross.shape or gram.shape != (
        statistics.heads,
        statistics.head_dim,
        statistics.head_dim,
    ):
        raise ValueError("route activation statistics have inconsistent geometry")

    scale = torch.diagonal(gram, dim1=-2, dim2=-1).mean(dim=-1)
    lambdas = scale * lambda_relative
    eye = torch.eye(statistics.head_dim, dtype=gram.dtype).expand(statistics.heads, -1, -1)
    if lambda_relative == 0.0:
        rtol = statistics.head_dim * torch.finfo(torch.float32).eps
        inverse = torch.linalg.pinv(gram, hermitian=True, rtol=rtol)
        storage = inverse @ cross
    else:
        storage = torch.linalg.solve(gram + lambdas[:, None, None] * eye, cross)
    if not torch.isfinite(storage).all():
        raise RuntimeError("Stage-A activation LS route solve produced non-finite weights")

    diagnostics = RouteActivationFitDiagnostics(
        rows=statistics.rows,
        lambda_relative=lambda_relative,
        lambda_actual=tuple(float(x) for x in lambdas.tolist()),
        smallest_singular_value=statistics.smallest_singular_value,
        largest_singular_value=statistics.largest_singular_value,
        numerical_rank=statistics.numerical_rank,
        full_rank_condition_number=statistics.full_rank_condition_number,
    )
    return RouteActivationFit(storage_weight=storage, diagnostics=diagnostics)
