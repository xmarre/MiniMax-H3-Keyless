from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import minimax_h3_keyless.progressive_live_capture as live_module
from minimax_h3_keyless.attention import KeylessAttentionDeploy
from minimax_h3_keyless.live_capture import mark_pinned_stage_a_teacher
from minimax_h3_keyless.pilot_campaign import (
    DATASET_SCHEMA,
    validate_pilot_dataset_manifest,
)
from minimax_h3_keyless.progressive import ProgressiveAcceptedBlock, ProgressivePrefix
from minimax_h3_keyless.progressive_capture_io import (
    ProgressiveCaptureProvenance,
    ProgressiveCaptureWriteResult,
)
from minimax_h3_keyless.progressive_live_capture import (
    PROGRESSIVE_CAPTURE_WRAPPER_KEY,
    ProgressiveLiveCaptureController,
    ProgressiveLiveCaptureSpec,
    build_progressive_capture_spec,
)
from minimax_h3_keyless.progressive_overlay import (
    PROGRESSIVE_OVERLAY_ATTACHMENT_KEY,
    ProgressiveOverlayAttachment,
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

    def set_attachments(self, key: str, value) -> None:
        self.attachments[key] = value


class FakeExecutor:
    def __init__(self, controller, inner, *, output="ok") -> None:
        self.wrappers = [controller]
        self.idx = 0
        self.class_obj = inner
        self.output = output
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.output


def _prefix(count: int = 0, *, dataset_sha: str = "b" * 64) -> ProgressivePrefix:
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
        sweep_id="progressive-live",
        code_commit="e" * 40,
        stage_a_campaign_sha256="a" * 64,
        dataset_manifest_sha256=dataset_sha,
        gate_manifest_sha256="c" * 64,
        accepted=accepted,
    )


def _patcher_for_prefix(prefix: ProgressivePrefix, manifest_sha: str) -> FakePatcher:
    patcher = FakePatcher(Core50())
    mark_pinned_stage_a_teacher(patcher)
    patcher.attachments[PROGRESSIVE_OVERLAY_ATTACHMENT_KEY] = ProgressiveOverlayAttachment(
        api=1,
        prefix_identity_sha256=prefix.identity_sha256,
        prefix_manifest_sha256=manifest_sha,
        code_commit=prefix.code_commit,
        accepted_blocks=prefix.accepted_blocks,
    )
    for index in prefix.accepted_blocks:
        patcher.object_patches[f"diffusion_model.blocks.{index}.attn"] = KeylessAttentionDeploy(
            4,
            2,
            2,
            1e-5,
            block_index=index,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
    return patcher


def _spec(tmp_path: Path, prefix: ProgressivePrefix, manifest_sha: str) -> ProgressiveLiveCaptureSpec:
    target = prefix.next_block
    assert target is not None
    return ProgressiveLiveCaptureSpec(
        dataset_manifest_sha256=prefix.dataset_manifest_sha256,
        prefix_identity_sha256=prefix.identity_sha256,
        prefix_manifest_sha256=manifest_sha,
        source_case_id="case-a",
        split="train",
        modality_label="video_audio",
        target_sigma=0.5,
        target_block=target,
        output_path=str(tmp_path / "capture.pt"),
        max_capture_bytes=1024 * 1024,
    )


def _provenance(prefix: ProgressivePrefix) -> ProgressiveCaptureProvenance:
    target = prefix.next_block
    assert target is not None
    return ProgressiveCaptureProvenance(
        code_commit=prefix.code_commit,
        comfy_commit="d" * 40,
        dataset_manifest_sha256=prefix.dataset_manifest_sha256,
        gate_manifest_sha256=prefix.gate_manifest_sha256,
        stage_a_campaign_sha256=prefix.stage_a_campaign_sha256,
        prefix_identity_sha256=prefix.identity_sha256,
        target_block=target,
        execution_descriptor="unit-test",
    )


def test_progressive_target_capture_runs_through_exact_empty_prefix_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prefix = _prefix(0)
    manifest_sha = "f" * 64
    patcher = _patcher_for_prefix(prefix, manifest_sha)
    controller = ProgressiveLiveCaptureController(
        _spec(tmp_path, prefix, manifest_sha),
        prefix,
        _provenance(prefix),
    )
    controller.bind(patcher, wrapper_type="diffusion_model")
    patcher.wrappers = {
        "diffusion_model": {PROGRESSIVE_CAPTURE_WRAPPER_KEY: [controller]}
    }

    record = object()

    class FakeCapture:
        def __enter__(self):
            return SimpleNamespace(records=lambda: (record,))

        def __exit__(self, exc_type, exc, tb):
            return None

    sessions = []

    def fake_session(model, requested_prefix, **kwargs):
        sessions.append((model, requested_prefix, kwargs))
        return FakeCapture()

    monkeypatch.setattr(live_module, "progressive_capture_session", fake_session)
    writes = []

    def fake_write(path, captured_record, *, provenance, receipt_path=None):
        writes.append((path, captured_record, provenance, receipt_path))
        return ProgressiveCaptureWriteResult(
            bundle_path=str(path),
            bundle_sha256="1" * 64,
            bundle_bytes=123,
            receipt_path=str(receipt_path),
            receipt_sha256="2" * 64,
            target_block=0,
            case_id="case-a",
        )

    monkeypatch.setattr(live_module, "write_progressive_capture_bundle", fake_write)
    executor = FakeExecutor(controller, patcher.model.diffusion_model)
    runtime_options = {
        "wrappers": {
            "diffusion_model": {PROGRESSIVE_CAPTURE_WRAPPER_KEY: [controller]}
        },
        "minimax_h3_sigma_shift_video": 12.0,
    }

    output = controller(
        executor,
        [torch.zeros(1), torch.zeros(1)],
        torch.tensor([500.0]),
        torch.zeros(1),
        runtime_options,
    )

    assert output == "ok"
    assert controller.captured is True
    assert controller.matched_forward_count == 1
    assert len(sessions) == 1
    assert sessions[0][0] is patcher.model.diffusion_model
    assert sessions[0][1] == prefix
    assert sessions[0][2]["case_id"] == "case-a"
    assert len(writes) == 1
    assert writes[0][1] is record
    forwarded_options = executor.calls[0][0][3]
    assert "wrappers" not in forwarded_options
    assert forwarded_options["minimax_h3_sigma_shift_video"] == 12.0

    with pytest.raises(RuntimeError, match="executed more than once"):
        controller(
            executor,
            [torch.zeros(1), torch.zeros(1)],
            torch.tensor([500.0]),
            torch.zeros(1),
            runtime_options,
        )


def test_progressive_capture_requires_object_overlay_to_be_physically_applied(
    tmp_path: Path,
) -> None:
    prefix = _prefix(1)
    manifest_sha = "f" * 64
    patcher = _patcher_for_prefix(prefix, manifest_sha)
    controller = ProgressiveLiveCaptureController(
        _spec(tmp_path, prefix, manifest_sha),
        prefix,
        _provenance(prefix),
    )
    controller.bind(patcher, wrapper_type="diffusion_model")
    patcher.wrappers = {
        "diffusion_model": {PROGRESSIVE_CAPTURE_WRAPPER_KEY: [controller]}
    }
    executor = FakeExecutor(controller, patcher.model.diffusion_model)

    with pytest.raises(RuntimeError, match="accepted progressive block 0"):
        controller(
            executor,
            None,
            torch.tensor([400.0]),
            None,
            {"wrappers": [controller]},
        )


def _dataset_manifest() -> dict:
    sigmas = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]
    cases = []
    coverage = ["short", "long", "reference", "audio", "mixed-grid"]
    for index in range(16):
        cases.append(
            {
                "case_id": f"case-{index}",
                "split": "train" if index < 12 else "holdout",
                "prompt": f"prompt {index}",
                "seed": index,
                "schedule": {"name": "fixed"},
                "modality_label": "video_audio" if index % 2 == 0 else "video",
                "resolution": [720, 1280],
                "duration_seconds": 6.0,
                "sigmas": sigmas,
                "coverage_tags": coverage if index == 0 else [],
                "assets": [],
            }
        )
    return {"schema": DATASET_SCHEMA, "cases": cases}


def test_progressive_capture_spec_is_bound_to_prefix_dataset_case_and_target(
    tmp_path: Path,
) -> None:
    manifest = _dataset_manifest()
    dataset_sha = validate_pilot_dataset_manifest(
        manifest,
        required_coverage_tags=("short", "long", "reference", "audio", "mixed-grid"),
    )
    manifest_path = tmp_path / "dataset.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    prefix = _prefix(3, dataset_sha=dataset_sha)

    spec = build_progressive_capture_spec(
        manifest_path,
        prefix,
        "f" * 64,
        case_id="case-14",
        target_sigma=0.625,
        output_root=tmp_path / "output",
        output_subdir="captures/progressive",
        max_capture_mib=64,
    )

    assert spec.target_block == 3
    assert spec.split == "holdout"
    assert spec.modality_label == "video_audio"
    assert spec.prefix_identity_sha256 == prefix.identity_sha256
    assert "block-03" in Path(spec.output_path).name
    assert prefix.identity_sha256[:12] in Path(spec.output_path).name
    assert spec.max_capture_bytes == 64 * 1024 * 1024
