from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

import torch

ARCHITECTURE = "h3_keyless_core50_v1"
CONTRACT_KEY = "minimax_h3_keyless_contract_v1"
PROVIDER_KEY = "minimax_h3_keyless_provider_v1"
CHECKPOINT_FORMAT_VERSION = 1
QV_ORDER = "q_effective;v"
ROPE_POLICY = "h3_split_half_96_v1"
QUANTIZATION_RECIPE = "minimax_h3_keyless_core50_200_v1"

HIDDEN_SIZE = 5376
HEADS = 56
HEAD_DIM = 128
INNER_DIM = HEADS * HEAD_DIM
CORE_BLOCKS = 50
TOKEN_REFINER_BLOCKS = 2
ROPE_ROT_DIM = 96
NORM_EPS = 1e-5


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
