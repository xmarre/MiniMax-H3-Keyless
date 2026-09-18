from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import minimax_h3_keyless.progressive_overlay as overlay_module
from minimax_h3_keyless.attention import KeylessAttentionDeploy
from minimax_h3_keyless.live_capture import mark_pinned_stage_a_teacher
from minimax_h3_keyless.progressive import ProgressiveAcceptedBlock, ProgressivePrefix
from minimax_h3_keyless.progressive_authorization import write_progressive_prefix_manifest
from minimax_h3_keyless.progressive_overlay import (
    PROGRESSIVE_OVERLAY_ATTACHMENT_KEY,
    prepare_progressive_overlay,
    require_progressive_overlay,
)


class NativeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = nn.Linear(4, 12, bias=False, dtype=torch.bfloat16)
        self.q_norm = nn.RMSNorm(2, eps=1e-5, dtype=torch.bfloat16)
        self.k_norm = nn.RMSNorm(2, eps=1e-5, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.heads = 2
        self.head_dim = 2


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = NativeAttention()


class Core50(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(50))


class FakePatcher:
    def __init__(self, inner: Core50) -> None:
        self.model = SimpleNamespace(
            diffusion_model=inner,
            model_sampling=SimpleNamespace(multiplier=1000.0),
        )
        self.attachments = {}
        self.patches = {}
        self.object_patches = {}
        self.weight_wrapper_patches = {}
        self.injections = {}
        self.hook_patches = {}
        self.current_hooks = None
        self.forced_hooks = None
        self.callbacks = {}
        self.wrappers = {}
        self.model_options = {"transformer_options": {}}
        self.load_device = torch.device("cpu")

    def clone(self):
        clone = FakePatcher(self.model.diffusion_model)
        clone.model = self.model
        clone.attachments = dict(self.attachments)
        clone.patches = {key: list(value) for key, value in self.patches.items()}
        clone.object_patches = dict(self.object_patches)
        clone.weight_wrapper_patches = dict(self.weight_wrapper_patches)
        clone.injections = dict(self.injections)
        clone.hook_patches = dict(self.hook_patches)
        clone.current_hooks = self.current_hooks
        clone.forced_hooks = self.forced_hooks
        clone.callbacks = dict(self.callbacks)
        clone.wrappers = dict(self.wrappers)
        clone.model_options = {
            "transformer_options": dict(self.model_options.get("transformer_options", {}))
        }
        clone.load_device = self.load_device
        return clone

    def add_object_patch(self, name: str, value) -> None:
        self.object_patches[name] = value

    def set_attachments(self, key: str, value) -> None:
        self.attachments[key] = value


def _prefix(count: int = 2) -> ProgressivePrefix:
    accepted = tuple(
        ProgressiveAcceptedBlock(
            block_index=index,
            final_stage="route",
            checkpoint_sha256=f"{index + 1:064x}",
            result_sha256=f"{index + 101:064x}",
        )
        for index in range(count)
    )
    return ProgressivePrefix(
        sweep_id="overlay-test",
        code_commit="e" * 40,
        stage_a_campaign_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        gate_manifest_sha256="c" * 64,
        accepted=accepted,
    )


def _write_manifest(tmp_path: Path, prefix: ProgressivePrefix) -> Path:
    path = tmp_path / f"{prefix.sweep_id}.prefix-{len(prefix.accepted):02d}.json"
    write_progressive_prefix_manifest(path, prefix, previous_manifest_sha256=None)
    return path


def _deploy(index: int) -> KeylessAttentionDeploy:
    return KeylessAttentionDeploy(
        4,
        2,
        2,
        1e-5,
        block_index=index,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )


def test_overlay_uses_clone_object_patches_without_mutating_shared_teacher(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prefix = _prefix(2)
    manifest = _write_manifest(tmp_path, prefix)
    inner = Core50()
    base = FakePatcher(inner)
    mark_pinned_stage_a_teacher(base)
    original_attention0 = inner.blocks[0].attn
    original_attention1 = inner.blocks[1].attn

    monkeypatch.setattr(
        overlay_module,
        "validate_loaded_native_teacher_model",
        lambda model: None,
    )
    calls = []

    def materialize(model, requested_prefix, *, output_dir):
        calls.append((model, requested_prefix, Path(output_dir)))
        return (_deploy(0), _deploy(1))

    monkeypatch.setattr(
        overlay_module,
        "load_progressive_deploy_attentions",
        materialize,
    )

    installation = prepare_progressive_overlay(
        base,
        manifest,
        code_commit="e" * 40,
    )
    clone = installation.patcher

    assert clone is not base
    assert clone.model is base.model
    assert base.object_patches == {}
    assert inner.blocks[0].attn is original_attention0
    assert inner.blocks[1].attn is original_attention1
    assert set(clone.object_patches) == {
        "diffusion_model.blocks.0.attn",
        "diffusion_model.blocks.1.attn",
    }
    assert isinstance(
        clone.object_patches["diffusion_model.blocks.0.attn"],
        KeylessAttentionDeploy,
    )
    assert calls == [(inner, prefix, manifest.parent)]
    marker = require_progressive_overlay(
        clone,
        prefix=prefix,
        prefix_manifest_sha256=installation.prefix_manifest_sha256,
    )
    assert marker.accepted_blocks == (0, 1)
    assert PROGRESSIVE_OVERLAY_ATTACHMENT_KEY not in base.attachments


def test_overlay_allows_empty_authorized_prefix_without_object_patches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prefix = _prefix(0)
    manifest = _write_manifest(tmp_path, prefix)
    base = FakePatcher(Core50())
    mark_pinned_stage_a_teacher(base)
    monkeypatch.setattr(
        overlay_module,
        "validate_loaded_native_teacher_model",
        lambda model: None,
    )
    monkeypatch.setattr(
        overlay_module,
        "load_progressive_deploy_attentions",
        lambda model, requested_prefix, *, output_dir: (),
    )

    installation = prepare_progressive_overlay(
        base,
        manifest,
        code_commit="e" * 40,
    )

    assert installation.patcher.object_patches == {}
    marker = require_progressive_overlay(
        installation.patcher,
        prefix=prefix,
        prefix_manifest_sha256=installation.prefix_manifest_sha256,
    )
    assert marker.accepted_blocks == ()


def test_overlay_rejects_code_revision_mismatch_before_materialization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prefix = _prefix(1)
    manifest = _write_manifest(tmp_path, prefix)
    base = FakePatcher(Core50())
    mark_pinned_stage_a_teacher(base)
    monkeypatch.setattr(
        overlay_module,
        "validate_loaded_native_teacher_model",
        lambda model: None,
    )
    called = False

    def materialize(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("must fail before reading accepted artifacts")

    monkeypatch.setattr(
        overlay_module,
        "load_progressive_deploy_attentions",
        materialize,
    )

    with pytest.raises(RuntimeError, match="code revision differs"):
        prepare_progressive_overlay(
            base,
            manifest,
            code_commit="d" * 40,
        )

    assert called is False
    assert base.object_patches == {}


def test_overlay_rejects_preexisting_model_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prefix = _prefix(0)
    manifest = _write_manifest(tmp_path, prefix)
    base = FakePatcher(Core50())
    mark_pinned_stage_a_teacher(base)
    base.object_patches["diffusion_model.blocks.0.attn"] = object()
    monkeypatch.setattr(
        overlay_module,
        "validate_loaded_native_teacher_model",
        lambda model: None,
    )

    with pytest.raises(RuntimeError, match="object_patches is non-empty"):
        prepare_progressive_overlay(
            base,
            manifest,
            code_commit="e" * 40,
        )
