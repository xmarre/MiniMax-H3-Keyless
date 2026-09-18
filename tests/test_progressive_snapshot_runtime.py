from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

import minimax_h3_keyless.progressive_snapshot as snapshot
import minimax_h3_keyless.progressive_snapshot_runtime as runtime
from minimax_h3_keyless.attention import KeylessAttentionDeploy
from minimax_h3_keyless.contracts import CORE_BLOCKS
from minimax_h3_keyless.progressive import ProgressiveAcceptedBlock, ProgressivePrefix


def _prefix(count: int = 2) -> ProgressivePrefix:
    return ProgressivePrefix(
        sweep_id="snapshot-stream-test",
        code_commit="a" * 40,
        stage_a_campaign_sha256="1" * 64,
        dataset_manifest_sha256="2" * 64,
        gate_manifest_sha256="3" * 64,
        accepted=tuple(
            ProgressiveAcceptedBlock(
                block_index=index,
                final_stage="route",
                checkpoint_sha256=f"{index + 1:064x}",
                result_sha256=f"{index + 101:064x}",
            )
            for index in range(count)
        ),
    )


class TinyNativeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = 2
        self.head_dim = 2
        self.qkv_proj = nn.Linear(4, 12, bias=False, dtype=torch.bfloat16)
        self.q_norm = nn.RMSNorm(2, eps=1e-5, dtype=torch.bfloat16)
        self.k_norm = nn.RMSNorm(2, eps=1e-5, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.to_gate_compress = None


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = TinyNativeAttention()
        self.mlp = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)


class TinyProgressiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(TinyBlock() for _ in range(CORE_BLOCKS))


def _mixed_source(prefix: ProgressivePrefix) -> TinyProgressiveModel:
    model = TinyProgressiveModel()
    snapshot._replace_native_prefix_with_empty_deploy(model, prefix)
    torch.manual_seed(1301)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, dtype=torch.float32).to(parameter.dtype))
    return model


def _write_snapshot(path: Path, source: TinyProgressiveModel) -> None:
    save_file(
        {
            key: value.detach().cpu().contiguous()
            for key, value in source.state_dict().items()
        },
        str(path),
    )


def test_streaming_reload_reconstructs_exact_mixed_state_tensor_by_tensor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prefix = _prefix(2)
    source = _mixed_source(prefix)
    path = tmp_path / "tiny-stream.safetensors"
    _write_snapshot(path, source)

    monkeypatch.setattr(
        runtime,
        "validate_progressive_snapshot_file",
        lambda *args, **kwargs: ("fixture-sha", {}),
    )

    real_safe_open = runtime.safe_open
    fetched: list[str] = []

    class CountingOpen:
        def __init__(self, *args, **kwargs) -> None:
            self._inner = real_safe_open(*args, **kwargs)
            self._handle = None

        def __enter__(self):
            self._handle = self._inner.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._inner.__exit__(exc_type, exc, tb)

        def get_tensor(self, key: str):
            assert self._handle is not None
            fetched.append(key)
            return self._handle.get_tensor(key)

    monkeypatch.setattr(runtime, "safe_open", CountingOpen)

    target = TinyProgressiveModel()
    expected_key_order = list(source.state_dict())
    runtime.load_progressive_snapshot_streaming(
        target,
        path,
        prefix,
        prefix_manifest_sha256="4" * 64,
    )

    assert fetched == expected_key_order
    assert isinstance(target.blocks[0].attn, KeylessAttentionDeploy)
    assert isinstance(target.blocks[1].attn, KeylessAttentionDeploy)
    assert isinstance(target.blocks[2].attn, TinyNativeAttention)
    for key, expected in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], expected)


def test_streaming_reload_rejects_signature_mismatch_before_tensor_reads(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prefix = _prefix(1)
    source = _mixed_source(prefix)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in source.state_dict().items()
    }
    missing_key = next(iter(state))
    state.pop(missing_key)
    path = tmp_path / "missing-key.safetensors"
    save_file(state, str(path))

    monkeypatch.setattr(
        runtime,
        "validate_progressive_snapshot_file",
        lambda *args, **kwargs: ("fixture-sha", {}),
    )

    opened_for_payload = False

    def forbidden_safe_open(*args, **kwargs):
        nonlocal opened_for_payload
        opened_for_payload = True
        raise AssertionError("payload reads must not begin before signature validation passes")

    monkeypatch.setattr(runtime, "safe_open", forbidden_safe_open)
    target = TinyProgressiveModel()
    with pytest.raises(RuntimeError, match="snapshot keys differ"):
        runtime.load_progressive_snapshot_streaming(
            target,
            path,
            prefix,
            prefix_manifest_sha256="4" * 64,
        )
    assert opened_for_payload is False
