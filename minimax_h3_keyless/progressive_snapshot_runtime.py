from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

from .checkpoint import _dtype_name, read_safetensors_signatures
from .progressive import ProgressivePrefix, validate_progressive_model_prefix
from .progressive_snapshot import (
    _replace_native_prefix_with_empty_deploy,
    validate_progressive_snapshot_file,
)


def load_progressive_snapshot_streaming(
    model: nn.Module,
    snapshot_path: str | Path,
    prefix: ProgressivePrefix,
    *,
    prefix_manifest_sha256: str,
    manifest_path: str | Path | None = None,
) -> nn.Module:
    """Load a validated mixed Stage-B snapshot with bounded host memory.

    ``safetensors.torch.load_file`` would materialize the complete snapshot beside the
    already-loaded H3 constructor model. The production verification path instead validates
    all key/shape/dtype signatures first and then copies one tensor at a time from mmap-backed
    safetensors storage. Peak extra host memory is therefore bounded by the current tensor,
    not another full H3 state dict.

    The supplied model must be a disposable fresh native teacher shell. If filesystem/device
    I/O fails after copying begins, callers must discard that model rather than treating its
    partially updated state as usable. No accepted-prefix artifact is mutated by this loader.
    """

    snapshot_path = Path(snapshot_path)
    validate_progressive_snapshot_file(
        snapshot_path,
        prefix,
        prefix_manifest_sha256=prefix_manifest_sha256,
        manifest_path=manifest_path,
    )

    _replace_native_prefix_with_empty_deploy(model, prefix)
    expected = model.state_dict()
    signatures, _ = read_safetensors_signatures(snapshot_path)
    if set(signatures) != set(expected):
        missing = sorted(set(expected) - set(signatures))
        unexpected = sorted(set(signatures) - set(expected))
        raise RuntimeError(
            "progressive snapshot keys differ from reconstructed mixed model: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    for key, target in expected.items():
        signature = signatures[key]
        if signature.shape != tuple(target.shape) or signature.dtype.upper() != _dtype_name(target):
            raise RuntimeError(
                "progressive snapshot tensor signature differs from reconstructed model: "
                f"{key}: snapshot=({signature.shape},{signature.dtype}), "
                f"model=({tuple(target.shape)},{_dtype_name(target)})"
            )

    with torch.no_grad(), safe_open(str(snapshot_path), framework="pt", device="cpu") as handle:
        for key, target in expected.items():
            source = handle.get_tensor(key)
            if target.device.type == "cpu" and source.dtype == target.dtype:
                target.copy_(source)
            else:
                target.copy_(source.to(device=target.device, dtype=target.dtype))
            del source

    validate_progressive_model_prefix(model, prefix)
    return model
