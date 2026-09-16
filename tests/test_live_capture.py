from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import minimax_h3_keyless.live_capture as live_capture
from minimax_h3_keyless.capture_io import CaptureBundleProvenance, CaptureBundleWriteResult
from minimax_h3_keyless.live_capture import (
    STAGE_A_CAPTURE_WRAPPER_KEY,
    StageACaptureSpec,
    StageALiveCaptureController,
    build_stage_a_capture_spec,
    mark_pinned_stage_a_teacher,
    require_pinned_stage_a_teacher,
    runtime_video_sigma,
)
from minimax_h3_keyless.pilot_campaign import DATASET_SCHEMA


def _spec(tmp_path: Path, *, sigma=0.5) -> StageACaptureSpec:
    return StageACaptureSpec(
        dataset_manifest_sha256="a" * 64,
        source_case_id="case-a",
        split="train",
        modality_label="video_audio",
        target_sigma=sigma,
        output_path=str(tmp_path / "case-a.capture.pt"),
        max_capture_bytes=1024 * 1024,
        sigma_tolerance=1e-6,
    )


def _provenance() -> CaptureBundleProvenance:
    return CaptureBundleProvenance(
        code_commit="1" * 40,
        comfy_commit="2" * 40,
        dataset_manifest_sha256="a" * 64,
        execution_descriptor="unit-test",
    )


class _FakePatcher:
    def __init__(self, inner):
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

    def set_attachments(self, key, value):
        self.attachments[key] = value


class _FakeExecutor:
    def __init__(self, controller, inner, *, output="ok"):
        self.wrappers = [controller]
        self.idx = 0
        self.class_obj = inner
        self.output = output
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.output


def test_runtime_video_sigma_uses_native_h3_flow_multiplier() -> None:
    assert runtime_video_sigma(torch.tensor([500.0])) == pytest.approx(0.5)
    assert runtime_video_sigma(torch.tensor([0.0, 0.0])) == 0.0
    with pytest.raises(RuntimeError, match="timestep rows differ"):
        runtime_video_sigma(torch.tensor([500.0, 501.0]))
    with pytest.raises(RuntimeError, match="invalid H3 video sigma"):
        runtime_video_sigma(torch.tensor([1200.0]))


def test_stage_a_teacher_marker_survives_explicit_validation() -> None:
    patcher = _FakePatcher(object())
    mark_pinned_stage_a_teacher(patcher)
    marker = require_pinned_stage_a_teacher(patcher)
    assert marker.teacher_model_sha256 == live_capture.TEACHER_SHA256


def test_target_capture_strips_only_its_wrapper_and_persists_once(tmp_path: Path, monkeypatch) -> None:
    inner = object()
    patcher = _FakePatcher(inner)
    mark_pinned_stage_a_teacher(patcher)
    controller = StageALiveCaptureController(_spec(tmp_path), _provenance())
    controller.bind(patcher, wrapper_type="diffusion_model")
    patcher.wrappers = {"diffusion_model": {STAGE_A_CAPTURE_WRAPPER_KEY: [controller]}}
    monkeypatch.setattr(live_capture, "validate_loaded_native_teacher_model", lambda model: None)

    captured = SimpleNamespace(records=lambda: ("r0", "r25", "r49"))

    class FakeCapture:
        def __init__(self, model, **kwargs):
            assert model is inner
            assert kwargs["sigma"] == pytest.approx(0.5)
            assert kwargs["block_indices"] == live_capture.PILOT_BLOCKS
            assert kwargs["context"]["stage_a_observed_video_sigma"] == pytest.approx(0.5)

        def __enter__(self):
            return captured

        def __exit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setattr(live_capture, "PilotActivationCapture", FakeCapture)
    writes = []

    def fake_write(path, records, *, provenance, receipt_path=None):
        writes.append((path, records, provenance, receipt_path))
        return CaptureBundleWriteResult(
            bundle_path=str(path),
            bundle_sha256="b" * 64,
            bundle_bytes=123,
            receipt_path=str(receipt_path),
            receipt_sha256="c" * 64,
            block_indices=(0, 25, 49),
            case_id="case-a",
        )

    monkeypatch.setattr(live_capture, "write_captured_pilot_bundle", fake_write)
    executor = _FakeExecutor(controller, inner)
    runtime_options = {
        "wrappers": {"diffusion_model": {STAGE_A_CAPTURE_WRAPPER_KEY: [controller]}},
        "minimax_h3_sigma_shift_video": 12.0,
    }
    output = controller(
        executor,
        [torch.zeros(1), torch.zeros(1)],
        torch.tensor([500.0]),
        torch.zeros(1),
        runtime_options,
        minimax_payload={"refs": []},
    )
    assert output == "ok"
    assert controller.captured is True
    assert len(writes) == 1
    assert len(executor.calls) == 1
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


def test_capture_rejects_any_other_runtime_wrapper(tmp_path: Path, monkeypatch) -> None:
    inner = object()
    patcher = _FakePatcher(inner)
    mark_pinned_stage_a_teacher(patcher)
    controller = StageALiveCaptureController(_spec(tmp_path), _provenance())
    controller.bind(patcher, wrapper_type="diffusion_model")
    patcher.wrappers = {"diffusion_model": {STAGE_A_CAPTURE_WRAPPER_KEY: [controller]}}
    monkeypatch.setattr(live_capture, "validate_loaded_native_teacher_model", lambda model: None)
    executor = _FakeExecutor(controller, inner)
    executor.wrappers.append(lambda *args: None)
    with pytest.raises(RuntimeError, match="sole DIFFUSION_MODEL wrapper"):
        controller(
            executor,
            None,
            torch.tensor([500.0]),
            None,
            {"wrappers": [controller, executor.wrappers[1]]},
        )


def test_non_target_sigma_passes_through_without_capture(tmp_path: Path, monkeypatch) -> None:
    inner = object()
    patcher = _FakePatcher(inner)
    mark_pinned_stage_a_teacher(patcher)
    controller = StageALiveCaptureController(_spec(tmp_path), _provenance())
    controller.bind(patcher, wrapper_type="diffusion_model")
    patcher.wrappers = {"diffusion_model": {STAGE_A_CAPTURE_WRAPPER_KEY: [controller]}}
    monkeypatch.setattr(live_capture, "validate_loaded_native_teacher_model", lambda model: None)
    executor = _FakeExecutor(controller, inner, output="native")
    output = controller(executor, None, torch.tensor([400.0]), None, {"wrappers": [controller]})
    assert output == "native"
    assert controller.matched_forward_count == 0
    assert controller.captured is False


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


def test_capture_spec_is_bound_to_validated_manifest_case_and_sigma(tmp_path: Path) -> None:
    manifest_path = tmp_path / "dataset.json"
    manifest_path.write_text(json.dumps(_dataset_manifest()), encoding="utf-8")
    output_root = tmp_path / "output"
    spec = build_stage_a_capture_spec(
        manifest_path,
        case_id="case-14",
        target_sigma=0.625,
        output_root=output_root,
        output_subdir="captures/stage-a",
        max_capture_mib=64,
    )
    assert spec.split == "holdout"
    assert spec.modality_label == "video_audio"
    assert spec.target_sigma == pytest.approx(0.625)
    assert Path(spec.output_path).parent == (output_root / "captures/stage-a").resolve()
    assert spec.max_capture_bytes == 64 * 1024 * 1024

    with pytest.raises(ValueError, match="not declared"):
        build_stage_a_capture_spec(
            manifest_path,
            case_id="case-14",
            target_sigma=0.3,
            output_root=output_root,
        )
    with pytest.raises(ValueError, match="output_subdir"):
        build_stage_a_capture_spec(
            manifest_path,
            case_id="case-14",
            target_sigma=0.625,
            output_root=output_root,
            output_subdir="../escape",
        )
