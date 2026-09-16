from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Protocol

import torch

ARCHITECTURE = "h3_keyless_core50_v1"
CONTRACT_KEY = "minimax_h3_keyless_contract_v1"
PROVIDER_KEY = "minimax_h3_keyless_provider_v1"
TELEMETRY_KEY = "minimax_h3_keyless_telemetry_v1"
CHECKPOINT_FORMAT_VERSION = 1
QV_ORDER = "q_effective;v"
ROPE_POLICY = "h3_split_half_96_v1"
QUANTIZATION_RECIPE = "minimax_h3_keyless_core50_200_v1"
TEACHER_COMPATIBILITY_MARKER = "pinned_exact_copy_v1"

TARGET_MODEL_REPO = "xmarre/MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-ComfyUI"
TARGET_MODEL_REVISION = "f26363f0d42fbd46ef59008fd5e4d946ea0e9426"
TEACHER_FILENAME = "MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy.safetensors"
TEACHER_SHA256 = "78b88298e241231b3bd95d752abde711efc9dd6517669a8a934faeb70baf6a98"
QKV_INT8_FILENAME = "MiniMax-H3-Pruned-Ref-Delta-Fused-r1024-comfy-int8-convrot.safetensors"
QKV_INT8_SHA256 = "00be5b0f995cc5a628921790f69cb22e138776c1e12235e3eab521941bb4b8c2"

HIDDEN_SIZE = 5376
HEADS = 56
HEAD_DIM = 128
INNER_DIM = HEADS * HEAD_DIM
CORE_BLOCKS = 50
TOKEN_REFINER_BLOCKS = 2
ROPE_ROT_DIM = 96
NORM_EPS = 1e-5
ADALN_CURVE_GRID = 1025
ADALN_BASIS_DIM = 8
ADALN_BLOCK_OUT = 6 * HIDDEN_SIZE * 3


@dataclass(frozen=True)
class ProjectionInfoV1:
    block_index: int
    key: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    object_identity: tuple[int, int]


@dataclass(frozen=True)
class KeylessContractV1:
    api: int = 1
    architecture: str = ARCHITECTURE
    core_blocks: int = CORE_BLOCKS
    token_refiner: str = "native_qkv"
    token_refiner_blocks: int = TOKEN_REFINER_BLOCKS
    heads: int = HEADS
    head_dim: int = HEAD_DIM
    inner_dim: int = INNER_DIM
    hidden_size: int = HIDDEN_SIZE
    routing_source: str = "value"
    retrieval_source: str = "raw_projected_value"
    routing_norm: str = "rmsnorm"
    routing_norm_epsilon: float = NORM_EPS
    rope_policy: str = ROPE_POLICY
    qv_order: str = QV_ORDER
    projection_attr: str = "qv_proj"
    checkpoint_format_version: int = CHECKPOINT_FORMAT_VERSION
    provenance_identity: str | None = None

    def identity(self) -> tuple[Any, ...]:
        return (
            self.api,
            self.architecture,
            self.checkpoint_format_version,
            self.qv_order,
            self.heads,
            self.head_dim,
            self.inner_dim,
            self.routing_source,
            self.retrieval_source,
            self.rope_policy,
            self.provenance_identity,
        )

    def projection_key(self, block_index: int) -> str:
        if not 0 <= block_index < self.core_blocks:
            raise IndexError(block_index)
        return f"blocks.{block_index}.attn.qv_proj.weight"

    def projection_info(self, attention: Any, block_index: int) -> ProjectionInfoV1:
        projection = getattr(attention, self.projection_attr, None)
        if projection is None:
            raise RuntimeError(
                f"Keyless block {block_index} does not expose {self.projection_attr!r}"
            )
        weight = getattr(projection, "weight", None)
        if weight is None:
            raise RuntimeError(f"Keyless block {block_index} projection has no weight")
        return ProjectionInfoV1(
            block_index=block_index,
            key=self.projection_key(block_index),
            shape=tuple(int(x) for x in weight.shape),
            dtype=str(weight.dtype),
            device=str(weight.device),
            object_identity=(id(projection), id(weight)),
        )


@dataclass(frozen=True)
class RowDomain:
    """Immutable identity for a physical or logical row domain."""

    start: int | None = None
    stop: int | None = None
    indices: tuple[int, ...] | None = None
    identity: str | None = None

    def __post_init__(self) -> None:
        if self.indices is not None and (self.start is not None or self.stop is not None):
            raise ValueError("row domain cannot contain both slice bounds and explicit indices")
        if (self.start is None) != (self.stop is None):
            raise ValueError("row-domain start and stop must be provided together")
        if self.start is not None and self.stop is not None and self.stop < self.start:
            raise ValueError("row-domain stop must be >= start")


@dataclass(frozen=True)
class RoutingPreprocessor:
    """Routing-only transformation. It must never be applied to retrieval V."""

    identity: str
    fn: Callable[[torch.Tensor], torch.Tensor] = field(compare=False, repr=False)


Selector = slice | torch.Tensor | tuple[int, ...] | list[int]


def _selector_indices(selector: Selector, rows: int, device: torch.device) -> torch.Tensor:
    if isinstance(selector, slice):
        start, stop, step = selector.indices(rows)
        if step != 1:
            raise ValueError("Keyless row-domain slices must be contiguous (step=1)")
        idx = torch.arange(start, stop, device=device, dtype=torch.long)
    elif torch.is_tensor(selector):
        if selector.dtype == torch.bool:
            if selector.ndim != 1 or selector.numel() != rows:
                raise ValueError("boolean row selector must match the current value domain")
            idx = selector.nonzero(as_tuple=False).flatten().to(device=device)
        else:
            idx = selector.to(device=device, dtype=torch.long).flatten()
    else:
        idx = torch.tensor(tuple(selector), device=device, dtype=torch.long)
    if idx.numel() and (bool((idx < 0).any()) or bool((idx >= rows).any())):
        raise IndexError(f"row selector contains an index outside [0,{rows})")
    return idx


def _compose_row_domain(
    domain: RowDomain | None,
    local_indices: tuple[int, ...],
    *,
    current_rows: int,
    identity: str | None,
) -> RowDomain:
    """Map a selection through an existing domain instead of resetting to local indices."""
    inherited_identity = None if domain is None else domain.identity
    next_identity = inherited_identity if identity is None else identity
    if domain is None:
        mapped = local_indices
    elif domain.indices is not None:
        if len(domain.indices) != current_rows:
            raise ValueError(
                "explicit row-domain length must match the current physical value rows"
            )
        mapped = tuple(domain.indices[i] for i in local_indices)
    elif domain.start is not None:
        assert domain.stop is not None
        if domain.stop - domain.start != current_rows:
            raise ValueError(
                "slice row-domain length must match the current physical value rows"
            )
        mapped = tuple(domain.start + i for i in local_indices)
    else:
        mapped = local_indices
    return RowDomain(indices=mapped, identity=next_identity)


@dataclass(frozen=True)
class RoutingSpecV1:
    api: int
    block_index: int
    norm_weight: torch.Tensor = field(compare=False, repr=False)
    norm_epsilon: float
    rope_freqs: torch.Tensor | None = field(default=None, compare=False, repr=False)
    rope_policy: str = ROPE_POLICY
    layout_identity: str | None = None
    value_domain: RowDomain | None = None
    routing_position_domain: RowDomain | None = None
    preprocessors: tuple[RoutingPreprocessor, ...] = ()

    def __post_init__(self) -> None:
        if self.api != 1:
            raise ValueError(f"unsupported routing spec api: {self.api}")
        if self.rope_policy != ROPE_POLICY:
            raise ValueError(f"unsupported rope policy: {self.rope_policy}")
        if self.norm_epsilon <= 0:
            raise ValueError("routing norm epsilon must be positive")

    def materialize(self, v: torch.Tensor) -> torch.Tensor:
        from .ops import materialize_route
        return materialize_route(v, self)

    def select_value_rows(
        self,
        v: torch.Tensor,
        selector: Selector,
        *,
        log_measure: torch.Tensor | None = None,
        identity: str | None = None,
    ) -> tuple[torch.Tensor, "RoutingSpecV1", torch.Tensor | None]:
        """Select V once and carry the same row selection into routing positions/measure.

        Domain coordinates are composed through prior selections. This is required for
        repeated sparse gathers: a second selector is local to the current tensor but
        provider/cache identities must continue to describe the original logical rows.
        """
        if v.ndim < 1:
            raise ValueError("V must have a row dimension")
        rows = int(v.shape[0])
        idx = _selector_indices(selector, rows, v.device)
        selected_v = v.index_select(0, idx)
        selected_rope = self.rope_freqs
        if selected_rope is not None:
            if selected_rope.ndim < 2 or selected_rope.shape[1] != rows:
                raise ValueError("rope_freqs must align exactly with the current value rows")
            selected_rope = selected_rope.index_select(1, idx.to(selected_rope.device))
        selected_measure = log_measure
        if selected_measure is not None:
            if selected_measure.ndim != 1 or selected_measure.shape[0] != rows:
                raise ValueError("log_measure must align exactly with the pre-selection V domain")
            selected_measure = selected_measure.index_select(0, idx.to(selected_measure.device))
        local_indices = tuple(int(x) for x in idx.detach().cpu().tolist())
        value_domain = _compose_row_domain(
            self.value_domain,
            local_indices,
            current_rows=rows,
            identity=identity,
        )
        routing_position_domain = _compose_row_domain(
            self.routing_position_domain,
            local_indices,
            current_rows=rows,
            identity=identity,
        )
        return selected_v, replace(
            self,
            rope_freqs=selected_rope,
            value_domain=value_domain,
            routing_position_domain=routing_position_domain,
        ), selected_measure


class KeylessProviderV1(Protocol):
    api: int

    def __call__(
        self,
        *,
        q: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        scale: float,
        routing: RoutingSpecV1,
        mask: torch.Tensor | None,
        log_measure: torch.Tensor | None,
        exact_blocks: Any,
        query_domain: RowDomain | None,
        value_domain: RowDomain | None,
        dense_fallback: Callable[..., torch.Tensor],
        transformer_options: Mapping[str, Any],
    ) -> torch.Tensor: ...


def get_keyless_provider(transformer_options: Mapping[str, Any] | None) -> KeylessProviderV1 | None:
    if not transformer_options:
        return None
    provider = transformer_options.get(PROVIDER_KEY)
    if provider is None:
        return None
    api = getattr(provider, "api", None)
    if api != 1:
        raise RuntimeError(f"{PROVIDER_KEY} must expose api=1, got {api!r}")
    if not callable(provider):
        raise RuntimeError(f"{PROVIDER_KEY} must be callable")
    return provider


def contract_from_metadata(metadata: Mapping[str, str]) -> KeylessContractV1:
    return KeylessContractV1(
        provenance_identity=metadata.get("provenance_identity")
        or metadata.get("manifest_sha256")
        or metadata.get("parent_model_sha256")
    )
