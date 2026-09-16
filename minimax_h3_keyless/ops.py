from __future__ import annotations

import torch
import torch.nn.functional as F

from .contracts import ROPE_ROT_DIM, RoutingSpecV1


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    work = x.float()
    inv = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + eps)
    out = work * inv
    return (out * weight.float()).to(dtype=x.dtype)


def apply_h3_split_half_rope(
    x: torch.Tensor,
    rope_freqs: torch.Tensor,
    *,
    rot_dim: int = ROPE_ROT_DIM,
) -> torch.Tensor:
    """Apply H3 split-half RoPE without mutating x.

    x may be [S,H,D] or [1,S,H,D]. rope_freqs is Comfy H3's rotation table
    [1,S,1,rot_dim/2,2,2]. Channels [rot_dim:D] are copied through.
    """
    squeeze = False
    if x.ndim == 3:
        x4 = x.unsqueeze(0)
        squeeze = True
    elif x.ndim == 4:
        x4 = x
    else:
        raise ValueError(f"expected rank-3 or rank-4 tensor, got shape {tuple(x.shape)}")
    if rot_dim % 2:
        raise ValueError("rot_dim must be even")
    if rot_dim > x4.shape[-1]:
        raise ValueError(f"rot_dim {rot_dim} exceeds head dim {x4.shape[-1]}")
    half = rot_dim // 2
    if rope_freqs.ndim != 6 or rope_freqs.shape[-3] != half or tuple(rope_freqs.shape[-2:]) != (2, 2):
        raise ValueError(
            f"invalid H3 RoPE table shape {tuple(rope_freqs.shape)}; expected [1,S,1,{half},2,2]"
        )
    if rope_freqs.shape[1] != x4.shape[1]:
        raise ValueError(f"RoPE row count {rope_freqs.shape[1]} != tensor rows {x4.shape[1]}")

    rot = rope_freqs.to(device=x4.device, dtype=x4.dtype)
    c = rot[..., 0, 0]
    s = rot[..., 1, 0]
    a = x4[..., :half]
    b = x4[..., half:rot_dim]
    first = a * c - b * s
    second = a * s + b * c
    out = torch.cat((first, second, x4[..., rot_dim:]), dim=-1)
    return out.squeeze(0) if squeeze else out


def normalized_positioned(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    rope_freqs: torch.Tensor | None,
) -> torch.Tensor:
    y = rms_norm(x, weight, eps)
    if rope_freqs is not None:
        y = apply_h3_split_half_rope(y, rope_freqs, rot_dim=rope_freqs.shape[-3] * 2)
    return y


def materialize_route(v: torch.Tensor, spec: RoutingSpecV1) -> torch.Tensor:
    route = normalized_positioned(v, spec.norm_weight, spec.norm_epsilon, spec.rope_freqs)
    for preprocessor in spec.preprocessors:
        updated = preprocessor.fn(route)
        if updated.shape != route.shape:
            raise RuntimeError(
                f"routing preprocessor {preprocessor.identity!r} changed route shape "
                f"{tuple(route.shape)} -> {tuple(updated.shape)}"
            )
        route = updated
    return route


def _broadcast_attention_bias(
    q: torch.Tensor,
    route: torch.Tensor,
    mask: torch.Tensor | None,
    log_measure: torch.Tensor | None,
) -> torch.Tensor | None:
    bias = None
    if mask is not None:
        bias = mask.to(device=q.device, dtype=torch.float32)
    if log_measure is not None:
        if log_measure.ndim != 1 or log_measure.shape[0] != route.shape[0]:
            raise ValueError(
                f"log_measure must have shape [{route.shape[0]}], got {tuple(log_measure.shape)}"
            )
        lm = log_measure.to(device=q.device, dtype=torch.float32).view(1, 1, 1, -1)
        bias = lm if bias is None else bias + lm
    return bias


def dense_reference_attention(
    q: torch.Tensor,
    route: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
    log_measure: torch.Tensor | None = None,
) -> torch.Tensor:
    """Explicit FP32 oracle. Inputs/output use H3 [S,H,D] layout."""
    if q.ndim != 3 or route.ndim != 3 or v.ndim != 3:
        raise ValueError("q/route/v must all be [rows, heads, head_dim]")
    if route.shape != v.shape:
        raise ValueError("route and retrieval V must have identical physical domains")
    if q.shape[1:] != route.shape[1:]:
        raise ValueError("query and route head geometry must match")
    scale = (q.shape[-1] ** -0.5) if scale is None else float(scale)
    qh = q.float().permute(1, 0, 2)
    rh = route.float().permute(1, 2, 0)
    logits = torch.matmul(qh, rh) * scale
    if mask is not None:
        logits = logits + mask.to(device=logits.device, dtype=logits.dtype)
    if log_measure is not None:
        if log_measure.ndim != 1 or log_measure.shape[0] != route.shape[0]:
            raise ValueError("log_measure shape does not match route/value rows")
        logits = logits + log_measure.float().to(logits.device).view(1, 1, -1)
    probs = torch.softmax(logits, dim=-1)
    out = torch.matmul(probs, v.float().permute(1, 0, 2))
    return out.permute(1, 0, 2).to(dtype=v.dtype)


def torch_sdpa_attention(
    q: torch.Tensor,
    route: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    mask: torch.Tensor | None = None,
    log_measure: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materialized implementation using PyTorch SDPA, [S,H,D] -> [S,H,D]."""
    bias = _broadcast_attention_bias(q, route, mask, log_measure)
    q4 = q.permute(1, 0, 2).unsqueeze(0)
    r4 = route.permute(1, 0, 2).unsqueeze(0)
    v4 = v.permute(1, 0, 2).unsqueeze(0)
    out = F.scaled_dot_product_attention(q4, r4, v4, attn_mask=bias, dropout_p=0.0, scale=scale)
    return out.squeeze(0).permute(1, 0, 2)
