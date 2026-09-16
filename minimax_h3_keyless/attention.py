from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from .contracts import (
    PROVIDER_KEY,
    TELEMETRY_KEY,
    ROPE_POLICY,
    RowDomain,
    RoutingPreprocessor,
    RoutingSpecV1,
    get_keyless_provider,
)
from .ops import materialize_route, normalized_positioned, torch_sdpa_attention


class PerHeadLinear(nn.Module):
    """Bias-free head-local D->D linear layer with PyTorch Linear weight orientation."""

    def __init__(self, heads: int, head_dim: int, *, device=None, dtype=None) -> None:
        super().__init__()
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.weight = nn.Parameter(torch.empty(heads, head_dim, head_dim, device=device, dtype=dtype))

    def reset_identity(self) -> None:
        with torch.no_grad():
            self.weight.zero_()
            eye = torch.eye(self.head_dim, device=self.weight.device, dtype=self.weight.dtype)
            self.weight.copy_(eye.unsqueeze(0).expand(self.heads, -1, -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.heads, self.head_dim):
            raise ValueError(
                f"expected [...,{self.heads},{self.head_dim}], got {tuple(x.shape)}"
            )
        return torch.einsum("...hd,hed->...he", x, self.weight)


def _linear(operations: Any, in_features: int, out_features: int, *, device, dtype) -> nn.Module:
    if operations is None:
        return nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
    return operations.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)


def _rmsnorm(operations: Any, dim: int, eps: float, *, device, dtype) -> nn.Module:
    if operations is None:
        return nn.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
    return operations.RMSNorm(dim, eps=eps, dtype=dtype, device=device)


def _layout_identity(transformer_options: Mapping[str, Any]) -> str | None:
    layout = transformer_options.get("minimax_h3_layout")
    if layout is None:
        return None
    for name in ("semantic_digest", "identity", "digest"):
        value = getattr(layout, name, None)
        if callable(value):
            value = value()
        if value is not None:
            return str(value)
    return f"{type(layout).__module__}.{type(layout).__qualname__}:{id(layout)}"


def _routing_preprocessors(transformer_options: Mapping[str, Any]) -> tuple[RoutingPreprocessor, ...]:
    items = transformer_options.get("minimax_h3_keyless_routing_preprocessors_v1", ())
    out: list[RoutingPreprocessor] = []
    for item in items:
        if isinstance(item, RoutingPreprocessor):
            out.append(item)
            continue
        identity = getattr(item, "identity", None)
        if identity is None or not callable(item):
            raise RuntimeError("invalid Keyless routing preprocessor; expected callable with identity")
        out.append(RoutingPreprocessor(str(identity), item))
    return tuple(out)


def _domain_from_option(value: Any) -> RowDomain | None:
    if value is None or isinstance(value, RowDomain):
        return value
    if isinstance(value, slice) and value.step in (None, 1):
        if value.stop is None:
            raise RuntimeError("Keyless slice row-domain requires an explicit stop")
        return RowDomain(start=value.start or 0, stop=value.stop)
    if isinstance(value, (tuple, list)) and all(isinstance(v, int) for v in value):
        return RowDomain(indices=tuple(value))
    raise RuntimeError(f"unsupported Keyless row-domain descriptor: {value!r}")


class _KeylessAttentionBase(nn.Module):
    def __init__(self, hidden: int, heads: int, head_dim: int, eps: float, *,
                 gate_compress: bool = False, block_index: int = -1,
                 dtype=None, device=None, operations=None) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.heads * self.head_dim
        self.block_index = int(block_index)
        self.q_norm = _rmsnorm(operations, head_dim, eps, device=device, dtype=dtype)
        self.route_norm = _rmsnorm(operations, head_dim, eps, device=device, dtype=dtype)
        self.out_proj = _linear(operations, self.inner_dim, hidden, device=device, dtype=dtype)
        self.to_gate_compress = (
            _linear(operations, hidden, self.inner_dim, device=device, dtype=dtype)
            if gate_compress else None
        )

    def _routing_spec(self, rope_freqs, transformer_options: Mapping[str, Any]) -> RoutingSpecV1:
        return RoutingSpecV1(
            api=1,
            block_index=self.block_index,
            norm_weight=self.route_norm.weight,
            norm_epsilon=float(self.route_norm.eps),
            rope_freqs=rope_freqs,
            rope_policy=ROPE_POLICY,
            layout_identity=_layout_identity(transformer_options),
            value_domain=_domain_from_option(transformer_options.get("minimax_h3_keyless_value_domain_v1")),
            routing_position_domain=_domain_from_option(
                transformer_options.get("minimax_h3_keyless_routing_position_domain_v1")
            ),
            preprocessors=_routing_preprocessors(transformer_options),
        )

    def _materialized_attention(
        self,
        q: torch.Tensor,
        v: torch.Tensor,
        routing: RoutingSpecV1,
        *,
        transformer_options: Mapping[str, Any],
        mask: torch.Tensor | None = None,
        log_measure: torch.Tensor | None = None,
    ) -> torch.Tensor:
        route = materialize_route(v, routing)
        scale = self.head_dim ** -0.5
        if mask is None and log_measure is None:
            try:
                from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

                qc = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
                rc = AttentionTensorContainer(route.transpose(0, 1).unsqueeze(0))
                vc = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
                out = optimized_attention(
                    qc, rc, vc, self.heads, mask=None, skip_reshape=True,
                    transformer_options=dict(transformer_options),
                )
                return out.squeeze(0)
            except ImportError:
                pass
        out = torch_sdpa_attention(q, route, v, scale=scale, mask=mask, log_measure=log_measure)
        return out.reshape(out.shape[0], self.inner_dim)

    def _dispatch(
        self,
        q: torch.Tensor,
        v: torch.Tensor,
        rope_freqs: torch.Tensor | None,
        transformer_options: Mapping[str, Any],
    ) -> torch.Tensor:
        routing = self._routing_spec(rope_freqs, transformer_options)
        qn = normalized_positioned(q, self.q_norm.weight, self.q_norm.eps, rope_freqs)
        provider = get_keyless_provider(transformer_options)
        mask = transformer_options.get("minimax_h3_keyless_mask_v1")
        log_measure = transformer_options.get("minimax_h3_keyless_log_measure_v1")
        exact_blocks = transformer_options.get("minimax_h3_keyless_exact_blocks_v1")
        query_domain = _domain_from_option(transformer_options.get("minimax_h3_keyless_query_domain_v1"))
        value_domain = routing.value_domain

        def dense_fallback(*, q=qn, v=v, routing=routing, mask=mask, log_measure=log_measure):
            telemetry = transformer_options.get(TELEMETRY_KEY)
            if callable(telemetry):
                telemetry({
                    "api": 1,
                    "mode": "materialized_route",
                    "block_index": self.block_index,
                    "layout_identity": routing.layout_identity,
                })
            return self._materialized_attention(
                q, v, routing, transformer_options=transformer_options,
                mask=mask, log_measure=log_measure,
            )

        if provider is None:
            return dense_fallback()
        out = provider(
            q=qn,
            v=v,
            heads=self.heads,
            scale=self.head_dim ** -0.5,
            routing=routing,
            mask=mask,
            log_measure=log_measure,
            exact_blocks=exact_blocks,
            query_domain=query_domain,
            value_domain=value_domain,
            dense_fallback=dense_fallback,
            transformer_options=transformer_options,
        )
        if not torch.is_tensor(out):
            raise RuntimeError(f"{PROVIDER_KEY} returned non-tensor {type(out)!r}")
        return out


class KeylessAttentionDeploy(_KeylessAttentionBase):
    """Canonical deployable QV form: packed [Q_effective; V], no K and no R."""

    def __init__(self, hidden: int, heads: int, head_dim: int, eps: float, **kwargs) -> None:
        operations = kwargs.get("operations")
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        super().__init__(hidden, heads, head_dim, eps, **kwargs)
        self.qv_proj = _linear(operations, hidden, 2 * self.inner_dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, rope_freqs=None, transformer_options=None) -> torch.Tensor:
        transformer_options = {} if transformer_options is None else transformer_options
        s = x.shape[0]
        q, v = self.qv_proj(x).split(self.inner_dim, dim=-1)
        q = q.view(s, self.heads, self.head_dim)
        v = v.view(s, self.heads, self.head_dim)
        out = self._dispatch(q, v, rope_freqs, transformer_options)
        if out.ndim == 3:
            out = out.reshape(s, self.inner_dim)
        if out.shape != (s, self.inner_dim):
            raise RuntimeError(f"Keyless backend returned invalid shape {tuple(out.shape)}")
        return self.out_proj(out)


class KeylessAttentionTrain(_KeylessAttentionBase):
    """Training/resume form with foldable per-head query routing factors."""

    def __init__(self, hidden: int, heads: int, head_dim: int, eps: float, **kwargs) -> None:
        operations = kwargs.get("operations")
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")
        super().__init__(hidden, heads, head_dim, eps, **kwargs)
        self.q_proj = _linear(operations, hidden, self.inner_dim, device=device, dtype=dtype)
        self.query_route = PerHeadLinear(heads, head_dim, device=device, dtype=dtype)
        self.v_proj = _linear(operations, hidden, self.inner_dim, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, rope_freqs=None, transformer_options=None) -> torch.Tensor:
        transformer_options = {} if transformer_options is None else transformer_options
        s = x.shape[0]
        q = self.q_proj(x).view(s, self.heads, self.head_dim)
        q = self.query_route(q)
        v = self.v_proj(x).view(s, self.heads, self.head_dim)
        out = self._dispatch(q, v, rope_freqs, transformer_options)
        if out.ndim == 3:
            out = out.reshape(s, self.inner_dim)
        if out.shape != (s, self.inner_dim):
            raise RuntimeError(f"Keyless backend returned invalid shape {tuple(out.shape)}")
        return self.out_proj(out)
