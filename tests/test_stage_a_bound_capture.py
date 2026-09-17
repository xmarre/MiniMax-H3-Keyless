from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import minimax_h3_keyless.live_capture as live_capture
import minimax_h3_keyless.stage_a_bound_capture as bound_capture
from minimax_h3_keyless.capture_io import CaptureBundleProvenance, CaptureBundleWriteResult
from minimax_h3_keyless.pilot_campaign import DATASET_SCHEMA
from minimax_h3_keyless.stage_a_execution_binding import (
    WORKFLOW_CONTEXT_KEY,
    canonical_stage_a_workflow_prompt_sha256,
)


def _prompt(*, capture_node_id: str = "20", seed: int = 123) -> dict:
    return {
        "2": {"class_type": "MiniMaxH3StageATeacherLoader", "inputs": {"model_name": "h3.safetensors"}},
        "7": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {"prompt": "fixed prompt", "width": 960, "height": 704, "length": 124},
        },
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        capture_node_id: {
            "class_type": "MiniMaxH3StageACapture",
            "inputs": {
                "model": ["2", 0],
                "dataset_manifest_path": "/manifest.json",
                "case_id": "case-14",
                "target_sigma": 0.625,
                "output_subdir": "captures/stage-a",
                "max_capture_mib": 64,
                "sigma_tolerance": 1e-6,
            },
        },
    }


def _manifest(workflow_sha: str) -> dict:
    sigmas = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]
    coverage = ["short", "long", "reference", "audio", "mixed-grid"]
    cases = []
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
                "workflow_prompt_sha256": workflow_sha,
            }
        )
    return {"schema": DATASET_SCHEMA, "cases": cases}


def test_bound_capture_spec_requires_predeclared_executed_workflow(tmp_path: Path) -> None:
    prompt = _prompt()
    workflow_sha = canonical_stage_a_workflow_prompt_sha256(prompt, capture_node_id="20")
    manifest = _manifest(workflow_sha)
    manifest_path = tmp_path / "dataset.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    spec = bound_capture.build_stage_a_capture_spec(
        manifest_path,
        case_id="case-14",
        target_sigma=0.625,
        output_root=tmp_path / "output",
        workflow_prompt=prompt,
        capture_node_id="20",
        asset_path_resolver=None,
        output_subdir="captures/stage-a",
        max_capture_mib=64,
    )
    assert spec.workflow_prompt_sha256 == workflow_sha
    assert spec.split == "holdout"

    with pytest.raises(ValueError, match="executed Comfy API prompt does not match"):
        bound_capture.build_stage_a_capture_spec(
            manifest_path,
            case_id="case-14",
            target_sigma=0.625,
            output_root=tmp_path / "other-output",
            workflow_prompt=_prompt(seed=999),
            capture_node_id="20",
            asset_path_resolver=None,
            max_capture_mib=64,
        )


def test_bound_controller_persists_workflow_identity_in_capture_context(tmp_path: Path, monkeypatch) -> None:
    workflow_sha = "a" * 64
    spec = bound_capture.StageACaptureSpec(
        dataset_manifest_sha256="b" * 64,
        source_case_id="case-a",
        split="train",
        modality_label="video_audio",
        target_sigma=0.5,
        output_path=str(tmp_path / "case-a.capture.pt"),
        max_capture_bytes=1024 * 1024,
        workflow_prompt_sha256=workflow_sha,
    )
    provenance = CaptureBundleProvenance(
        code_commit="1" * 40,
        comfy_commit="2" * 40,
        dataset_manifest_sha256="b" * 64,
        execution_descriptor="unit-test",
    )

    class FakePatcher:
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

        def set_attachments(self, key, value):
            self.attachments[key] = value

    inner = object()
    patcher = FakePatcher(inner)
    live_capture.mark_pinned_stage_a_teacher(patcher)
    controller = bound_capture.StageALiveCaptureController(spec, provenance)
    controller.bind(patcher, wrapper_type="diffusion_model")
    patcher.wrappers = {
        "diffusion_model": {live_capture.STAGE_A_CAPTURE_WRAPPER_KEY: [controller]}
    }
    monkeypatch.setattr(live_capture, "validate_loaded_native_teacher_model", lambda model: None)

    captured = SimpleNamespace(records=lambda: ("r0", "r25", "r49"))

    class FakeCapture:
        def __init__(self, model, **kwargs):
            assert model is inner
            assert kwargs["context"][WORKFLOW_CONTEXT_KEY] == workflow_sha

        def __enter__(self):
            return captured

        def __exit__(self, exc_type, exc, tb):
            return None

    monkeypatch.setattr(bound_capture, "PilotActivationCapture", FakeCapture)
    monkeypatch.setattr(
        bound_capture,
        "write_captured_pilot_bundle",
        lambda *args, **kwargs: CaptureBundleWriteResult(
            bundle_path=spec.output_path,
            bundle_sha256="c" * 64,
            bundle_bytes=123,
            receipt_path=spec.receipt_path,
            receipt_sha256="d" * 64,
            block_indices=(0, 25, 49),
            case_id="case-a",
        ),
    )

    class FakeExecutor:
        wrappers = [controller]
        idx = 0
        class_obj = inner

        def __call__(self, *args, **kwargs):
            return "ok"

    assert controller(
        FakeExecutor(),
        [torch.zeros(1), torch.zeros(1)],
        torch.tensor([500.0]),
        torch.zeros(1),
        {"wrappers": [controller]},
    ) == "ok"


def test_comfy_entrypoint_requests_prompt_and_unique_id_hidden_inputs() -> None:
    path = Path(__file__).resolve().parents[1] / "__init__.py"
    package_name = "_minimax_h3_keyless_stage_a_binding_entry"
    spec = importlib.util.spec_from_file_location(
        package_name,
        path,
        submodule_search_locations=[str(path.parent)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    try:
        spec.loader.exec_module(module)
        inputs = module.MiniMaxH3StageACapture.INPUT_TYPES()
        assert inputs["hidden"] == {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}
    finally:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)
