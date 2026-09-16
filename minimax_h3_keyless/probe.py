from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open

from .checkpoint import sha256_file
from .contracts import HEAD_DIM, HEADS, HIDDEN_SIZE, INNER_DIM, TEACHER_SHA256
from .diagnostics import fixed_value_bilinear_residual, regularized_ls_row_route


@dataclass(frozen=True)
class MatrixSpectrum:
    rank: int
    tolerance: float
    condition: float
    singular_values: tuple[float, ...]


@dataclass(frozen=True)
class NativeHeadSlices:
    q_math: torch.Tensor
    k_math: torch.Tensor
    v_math: torch.Tensor
    q_sha256: str
    k_sha256: str
    v_sha256: str


@dataclass(frozen=True)
class HeadProbe:
    layer: int
    head: int
    q: MatrixSpectrum
    k: MatrixSpectrum
    v: MatrixSpectrum
    principal_angles_degrees_k_v: tuple[float, ...]
    key_projection_relative_residual: float
    fixed_value_bilinear_relative_residual: float
    fixed_value_bilinear_residual_frobenius: float
    teacher_bilinear_frobenius: float
    q_slice_sha256: str
    k_slice_sha256: str
    v_slice_sha256: str
    least_squares: tuple[dict[str, float], ...]


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).flatten()
    return hashlib.sha256(bytes(raw)).hexdigest()


def _svd_with_tolerance(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, int]:
    m = matrix.double()
    u, singular, _ = torch.linalg.svd(m, full_matrices=False)
    if singular.numel() == 0:
        return u, singular, 0.0, 0
    tolerance = float(max(m.shape) * torch.finfo(singular.dtype).eps * singular.max())
    rank = int((singular > tolerance).sum().item())
    return u, singular, tolerance, rank


def _matrix_spectrum(matrix: torch.Tensor) -> MatrixSpectrum:
    _, singular, tolerance, rank = _svd_with_tolerance(matrix)
    full_rank = min(matrix.shape)
    if singular.numel() == 0:
        condition = float("nan")
    elif rank < full_rank:
        condition = float("inf")
    else:
        condition = float((singular.max() / singular.min()).item())
    return MatrixSpectrum(
        rank=rank,
        tolerance=tolerance,
        condition=condition,
        singular_values=tuple(float(x) for x in singular.cpu().tolist()),
    )


def _principal_angles_degrees(a: torch.Tensor, b: torch.Tensor) -> tuple[float, ...]:
    ua, _, _, rank_a = _svd_with_tolerance(a)
    ub, _, _, rank_b = _svd_with_tolerance(b)
    if rank_a == 0 or rank_b == 0:
        return ()
    cosines = torch.linalg.svdvals(ua[:, :rank_a].T @ ub[:, :rank_b]).clamp(0.0, 1.0)
    return tuple(float(x) for x in torch.rad2deg(torch.acos(cosines)).cpu().tolist())


def read_native_qkv_head_slices(
    checkpoint: str | Path,
    *,
    layer: int,
    head: int,
    hidden_size: int = HIDDEN_SIZE,
    heads: int = HEADS,
    head_dim: int = HEAD_DIM,
) -> NativeHeadSlices:
    if not 0 <= head < heads:
        raise ValueError(f"head index {head} outside [0,{heads})")
    inner = heads * head_dim
    key = f"blocks.{layer}.attn.qkv_proj.weight"
    with safe_open(str(checkpoint), framework="pt", device="cpu") as f:
        if key not in f.keys():
            raise KeyError(f"missing teacher tensor {key}")
        sl = f.get_slice(key)
        shape = tuple(int(x) for x in sl.get_shape())
        expected = (3 * inner, hidden_size)
        if shape != expected:
            raise ValueError(f"{key}: expected shape {expected}, got {shape}")
        offset = head * head_dim
        q_storage = sl[offset:offset + head_dim, :].contiguous()
        k_storage = sl[inner + offset:inner + offset + head_dim, :].contiguous()
        v_storage = sl[2 * inner + offset:2 * inner + offset + head_dim, :].contiguous()
    return NativeHeadSlices(
        q_math=q_storage.T.contiguous(),
        k_math=k_storage.T.contiguous(),
        v_math=v_storage.T.contiguous(),
        q_sha256=_tensor_sha256(q_storage),
        k_sha256=_tensor_sha256(k_storage),
        v_sha256=_tensor_sha256(v_storage),
    )


def probe_native_head(
    slices: NativeHeadSlices,
    *,
    layer: int,
    head: int,
    lambda_relatives: Iterable[float] = (0.0, 1e-4, 1e-2),
) -> HeadProbe:
    q = slices.q_math.double()
    k = slices.k_math.double()
    v = slices.v_math.double()
    if q.shape != k.shape or k.shape != v.shape:
        raise ValueError("Q/K/V mathematical head matrices must share a shape")

    projection = torch.linalg.lstsq(v, k).solution
    key_residual = k - v @ projection
    key_norm = torch.linalg.vector_norm(k)
    key_relative = float(torch.linalg.vector_norm(key_residual) / key_norm) if key_norm > 0 else float("nan")
    bilinear = fixed_value_bilinear_residual(q, k, v)

    ls_rows: list[dict[str, float]] = []
    for lambda_relative in lambda_relatives:
        storage_weight, lambda_actual = regularized_ls_row_route(
            k,
            v,
            lambda_relative=float(lambda_relative),
        )
        fitted = v @ storage_weight.T
        denom = torch.linalg.vector_norm(k)
        relative = (
            float(torch.linalg.vector_norm(fitted - k) / denom)
            if denom > 0
            else float("nan")
        )
        ls_rows.append(
            {
                "lambda_relative": float(lambda_relative),
                "lambda_actual": float(lambda_actual),
                "key_fit_relative_residual": relative,
            }
        )

    return HeadProbe(
        layer=int(layer),
        head=int(head),
        q=_matrix_spectrum(q),
        k=_matrix_spectrum(k),
        v=_matrix_spectrum(v),
        principal_angles_degrees_k_v=_principal_angles_degrees(k, v),
        key_projection_relative_residual=key_relative,
        fixed_value_bilinear_relative_residual=bilinear.relative_frobenius,
        fixed_value_bilinear_residual_frobenius=bilinear.residual_frobenius,
        teacher_bilinear_frobenius=bilinear.teacher_frobenius,
        q_slice_sha256=slices.q_sha256,
        k_slice_sha256=slices.k_sha256,
        v_slice_sha256=slices.v_sha256,
        least_squares=tuple(ls_rows),
    )


def run_teacher_probe(
    checkpoint: str | Path,
    *,
    layers: Iterable[int] = (0, 25, 49),
    heads: Iterable[int] = (0,),
    expected_sha256: str = TEACHER_SHA256,
    verify_full_sha256: bool = True,
) -> dict:
    path = Path(checkpoint)
    actual_sha = sha256_file(path) if verify_full_sha256 else None
    if actual_sha is not None and actual_sha.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"teacher SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )
    probes = []
    for layer in layers:
        for head in heads:
            slices = read_native_qkv_head_slices(path, layer=int(layer), head=int(head))
            probes.append(asdict(probe_native_head(slices, layer=int(layer), head=int(head))))
    return {
        "schema": "minimax_h3_keyless_teacher_weight_probe_v1",
        "checkpoint": path.name,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha,
        "full_sha256_verified": actual_sha is not None,
        "geometry": {
            "hidden_size": HIDDEN_SIZE,
            "heads": HEADS,
            "head_dim": HEAD_DIM,
            "inner_dim": INNER_DIM,
        },
        "scope": "raw_weight_subspace_diagnostic_only",
        "probes": probes,
    }


def write_probe_json(result: dict, output: str | Path) -> None:
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
