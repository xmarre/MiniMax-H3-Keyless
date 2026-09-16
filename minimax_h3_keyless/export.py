from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import save_file

from .checkpoint import validate_deploy_checkpoint, validate_training_checkpoint
from .contracts import (
    ARCHITECTURE,
    CHECKPOINT_FORMAT_VERSION,
    CORE_BLOCKS,
    HEAD_DIM,
    HEADS,
    HIDDEN_SIZE,
    INNER_DIM,
    QV_ORDER,
    ROPE_POLICY,
    TARGET_MODEL_REVISION,
    TEACHER_SHA256,
    TOKEN_REFINER_BLOCKS,
)


def fold_query_route_weight(
    q_weight: torch.Tensor,
    route_weight: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fold per-head nn.Linear-style route weights into Q in FP32."""
    if q_weight.shape[0] % route_weight.shape[0] != 0:
        raise ValueError("q_weight rows are not divisible by route head count")
    heads = route_weight.shape[0]
    head_dim = route_weight.shape[1]
    if route_weight.shape[2] != head_dim or q_weight.shape[0] != heads * head_dim:
        raise ValueError("incompatible q/route geometry")
    qw = q_weight.reshape(heads, head_dim, q_weight.shape[1]).float()
    rw = route_weight.float()
    folded = torch.bmm(rw, qw).reshape_as(q_weight)
    return folded.to(output_dtype or q_weight.dtype)


def fold_training_state_dict(
    state_dict: Mapping[str, torch.Tensor], *, output_dtype: torch.dtype = torch.bfloat16
) -> dict[str, torch.Tensor]:
    validate_training_checkpoint(state_dict)
    out = {k: v for k, v in state_dict.items()}
    for i in range(CORE_BLOCKS):
        p = f"blocks.{i}.attn."
        q = out.pop(p + "q_proj.weight")
        r = out.pop(p + "query_route.weight")
        v = out.pop(p + "v_proj.weight")
        q_eff = fold_query_route_weight(q, r, output_dtype=output_dtype)
        out[p + "qv_proj.weight"] = torch.cat((q_eff, v.to(output_dtype)), dim=0)
    return out


def canonical_metadata(
    *, training_run: str, export_commit: str, parent_model_sha256: str = TEACHER_SHA256,
    parent_model_revision: str = TARGET_MODEL_REVISION, manifest_sha256: str | None = None,
) -> dict[str, str]:
    md = {
        "architecture": ARCHITECTURE,
        "checkpoint_format_version": str(CHECKPOINT_FORMAT_VERSION),
        "qv_order": QV_ORDER,
        "routing_source": "value",
        "retrieval_source": "raw_projected_value",
        "routing_norm": "rmsnorm",
        "routing_norm_epsilon": "1e-5",
        "rope_policy": ROPE_POLICY,
        "hidden_size": str(HIDDEN_SIZE),
        "heads": str(HEADS),
        "head_dim": str(HEAD_DIM),
        "inner_dim": str(INNER_DIM),
        "core_blocks": str(CORE_BLOCKS),
        "token_refiner": "native_qkv",
        "token_refiner_blocks": str(TOKEN_REFINER_BLOCKS),
        "teacher_model_revision": TARGET_MODEL_REVISION,
        "teacher_model_sha256": TEACHER_SHA256,
        "parent_model_revision": parent_model_revision,
        "parent_model_sha256": parent_model_sha256,
        "training_run": training_run,
        "export_commit": export_commit,
    }
    if manifest_sha256:
        md["manifest_sha256"] = manifest_sha256
    return md


def export_folded_bf16(
    state_dict: Mapping[str, torch.Tensor], output_path: str | Path, *, metadata: Mapping[str, str]
) -> None:
    folded = fold_training_state_dict(state_dict, output_dtype=torch.bfloat16)
    validate_deploy_checkpoint(folded, metadata)
    cpu = {k: v.detach().cpu().contiguous() for k, v in folded.items()}
    save_file(cpu, str(output_path), metadata=dict(metadata))
