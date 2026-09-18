from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BilinearResidual:
    relative_frobenius: float
    residual_frobenius: float
    teacher_frobenius: float
    value_rank: int
    value_condition: float


def fixed_value_bilinear_residual(
    q_weight_math: torch.Tensor,
    k_weight_math: torch.Tensor,
    v_weight_math: torch.Tensor,
    *, rcond: float | None = None,
) -> BilinearResidual:
    """Best fixed-V raw bilinear residual without materializing a hidden×hidden matrix.

    Inputs are mathematical projection matrices [hidden, head_dim] A/B/C. Computes
    min_M ||A B^T - A M C^T||_F / ||A B^T||_F. This is the relevant raw linear
    subspace diagnostic only; H3's q/k RMSNorm + partial RoPE still make exact runtime
    conversion a separate question.
    """
    A = q_weight_math.double()
    B = k_weight_math.double()
    C = v_weight_math.double()
    if A.shape != B.shape or B.shape != C.shape:
        raise ValueError("A/B/C must have identical [hidden, head_dim] shape")

    X = torch.linalg.lstsq(C, B, rcond=rcond).solution
    D = B - C @ X
    ga = A.T @ A
    residual_sq = torch.trace(ga @ (D.T @ D)).clamp_min(0.0)
    teacher_sq = torch.trace(ga @ (B.T @ B)).clamp_min(0.0)
    residual = torch.sqrt(residual_sq)
    teacher = torch.sqrt(teacher_sq)
    rel = residual / teacher if teacher > 0 else torch.tensor(float("nan"), dtype=torch.float64)
    s = torch.linalg.svdvals(C)
    tol = max(C.shape) * torch.finfo(s.dtype).eps * s.max()
    rank = int((s > tol).sum().item())
    cond = float((s.max() / s.min()).item()) if s.min() > 0 else float("inf")
    return BilinearResidual(float(rel), float(residual), float(teacher), rank, cond)


def regularized_ls_row_route(
    k_weight_math: torch.Tensor,
    v_weight_math: torch.Tensor,
    *, lambda_relative: float,
) -> tuple[torch.Tensor, float]:
    """Return the storage weight for the pre-q_norm query factorization.

    Let B/C be mathematical teacher K/V matrices [hidden, head_dim]. ``PerHeadLinear``
    applies a stored matrix W as ``q @ W.T``. The design's mathematical query factor is
    therefore ``R = W.T``. For the fixed-V raw least-squares initialization,

        R = B.T C (C.T C + lambda I)^-1

    so the storage tensor must be

        W = (C.T C + lambda I)^-1 C.T B.

    Equivalently W minimizes ``||C @ W - B||``. Returning the transpose here would
    silently optimize the wrong bilinear map even though its shapes remain valid.
    """
    B = k_weight_math.double()
    C = v_weight_math.double()
    if B.shape != C.shape:
        raise ValueError("B and C must have identical shapes")
    if lambda_relative < 0:
        raise ValueError("lambda_relative must be non-negative")
    gram = C.T @ C
    scale = float(torch.diagonal(gram).mean().item())
    lam = float(lambda_relative) * scale
    if lam == 0.0:
        storage_weight = torch.linalg.lstsq(C, B).solution
        return storage_weight, lam
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    storage_weight = torch.linalg.solve(gram + lam * eye, C.T @ B)
    return storage_weight, lam


def split_qkv_storage_weight(
    qkv_weight: torch.Tensor, *, heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inner = heads * head_dim
    if qkv_weight.shape[0] != 3 * inner:
        raise ValueError(f"expected {3*inner} qkv rows, got {qkv_weight.shape[0]}")
    q, k, v = qkv_weight.split(inner, dim=0)
    return q.T, k.T, v.T
