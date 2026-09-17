from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from minimax_h3_keyless.progressive import ProgressivePrefix


_ENTRYPOINT_LOAD_INDEX = 0


def _load_entrypoint():
    global _ENTRYPOINT_LOAD_INDEX
    _ENTRYPOINT_LOAD_INDEX += 1
    path = Path(__file__).resolve().parents[1] / "__init__.py"
    package_name = f"_minimax_h3_keyless_comfy_entry_{_ENTRYPOINT_LOAD_INDEX}"
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
    finally:
        # Comfy loads a custom-node __init__.py as a real package. Mirror that behavior for
        # the test, then remove the synthetic package tree so repeated loads cannot share
        # module/class state through sys.modules.
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)
    return module


def test_entrypoint_registers_progressive_stage_b_nodes() -> None:
    module = _load_entrypoint()
    assert module.NODE_CLASS_MAPPINGS["MiniMaxH3ProgressiveOverlay"] is module.MiniMaxH3ProgressiveOverlay
    assert module.NODE_CLASS_MAPPINGS["MiniMaxH3ProgressiveCapture"] is module.MiniMaxH3ProgressiveCapture
    assert module.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3ProgressiveOverlay"] == (
        "MiniMax H3 Progressive Prefix Overlay"
    )
    assert module.NODE_DISPLAY_NAME_MAPPINGS["MiniMaxH3ProgressiveCapture"] == (
        "MiniMax H3 Progressive Capture"
    )


def test_progressive_overlay_node_uses_clean_plugin_revision_and_optional_artifact_root(
    monkeypatch,
) -> None:
    module = _load_entrypoint()
    calls = {}

    monkeypatch.setattr(
        module,
        "discover_clean_git_revision",
        lambda path, *, label: "e" * 40,
    )

    def fake_prepare(model, prefix_manifest_path, *, code_commit, artifact_dir):
        calls.update(
            model=model,
            prefix_manifest_path=prefix_manifest_path,
            code_commit=code_commit,
            artifact_dir=artifact_dir,
        )
        return SimpleNamespace(patcher="overlay-model")

    monkeypatch.setattr(module, "prepare_progressive_overlay", fake_prepare)
    model = object()
    output = module.MiniMaxH3ProgressiveOverlay().apply(
        model,
        "/tmp/prefix.json",
        "   ",
    )

    assert output == ("overlay-model",)
    assert calls == {
        "model": model,
        "prefix_manifest_path": "/tmp/prefix.json",
        "code_commit": "e" * 40,
        "artifact_dir": None,
    }


def test_progressive_capture_node_revalidates_overlay_then_clones_and_binds_provenance(
    monkeypatch,
) -> None:
    module = _load_entrypoint()
    prefix = ProgressivePrefix(
        sweep_id="node-test",
        code_commit="e" * 40,
        stage_a_campaign_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        gate_manifest_sha256="c" * 64,
    )
    manifest_sha = "f" * 64
    clone = SimpleNamespace(name="clone")

    class FakeModel:
        def clone(self):
            return clone

    model = FakeModel()
    overlay_checks = []
    monkeypatch.setattr(
        module,
        "load_progressive_prefix_manifest",
        lambda path: (prefix, manifest_sha),
    )
    monkeypatch.setattr(
        module,
        "require_progressive_overlay",
        lambda candidate, **kwargs: overlay_checks.append((candidate, kwargs)),
    )

    spec = SimpleNamespace(target_block=0)
    spec_calls = {}

    def fake_spec(dataset_manifest_path, supplied_prefix, prefix_manifest_sha256, **kwargs):
        spec_calls.update(
            dataset_manifest_path=dataset_manifest_path,
            prefix=supplied_prefix,
            prefix_manifest_sha256=prefix_manifest_sha256,
            kwargs=kwargs,
        )
        return spec

    monkeypatch.setattr(module, "build_progressive_capture_spec", fake_spec)
    monkeypatch.setattr(
        module,
        "build_progressive_execution_descriptor",
        lambda candidate, supplied_prefix: "progressive-node-test",
    )

    folder_paths = ModuleType("folder_paths")
    folder_paths.__file__ = "/tmp/comfy/folder_paths.py"
    folder_paths.get_output_directory = lambda: "/tmp/comfy-output"
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    def fake_revision(path, *, label):
        return "e" * 40 if label == "MiniMax-H3-Keyless" else "d" * 40

    monkeypatch.setattr(module, "discover_clean_git_revision", fake_revision)

    controller_calls = {}

    class FakeController:
        def __init__(self, supplied_spec, supplied_prefix, provenance):
            controller_calls.update(
                spec=supplied_spec,
                prefix=supplied_prefix,
                provenance=provenance,
            )

    installed = []
    monkeypatch.setattr(module, "ProgressiveLiveCaptureController", FakeController)
    monkeypatch.setattr(
        module,
        "install_progressive_capture_wrapper",
        lambda candidate, controller: installed.append((candidate, controller)),
    )

    output = module.MiniMaxH3ProgressiveCapture().apply(
        model,
        "/tmp/prefix.json",
        "/tmp/dataset.json",
        "case-14",
        0.625,
        "captures/progressive",
        4096,
        1e-6,
    )

    assert output == (clone,)
    assert [candidate for candidate, _ in overlay_checks] == [model, clone]
    for _, kwargs in overlay_checks:
        assert kwargs == {
            "prefix": prefix,
            "prefix_manifest_sha256": manifest_sha,
        }
    assert spec_calls["dataset_manifest_path"] == "/tmp/dataset.json"
    assert spec_calls["prefix"] == prefix
    assert spec_calls["prefix_manifest_sha256"] == manifest_sha
    assert spec_calls["kwargs"] == {
        "case_id": "case-14",
        "target_sigma": 0.625,
        "output_root": "/tmp/comfy-output",
        "output_subdir": "captures/progressive",
        "max_capture_mib": 4096,
        "sigma_tolerance": 1e-6,
    }

    provenance = controller_calls["provenance"]
    assert provenance.code_commit == "e" * 40
    assert provenance.comfy_commit == "d" * 40
    assert provenance.dataset_manifest_sha256 == prefix.dataset_manifest_sha256
    assert provenance.gate_manifest_sha256 == prefix.gate_manifest_sha256
    assert provenance.stage_a_campaign_sha256 == prefix.stage_a_campaign_sha256
    assert provenance.prefix_identity_sha256 == prefix.identity_sha256
    assert provenance.target_block == 0
    assert provenance.execution_descriptor == "progressive-node-test"
    assert len(installed) == 1
    assert installed[0][0] is clone
    assert isinstance(installed[0][1], FakeController)
