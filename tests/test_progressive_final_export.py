from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import minimax_h3_keyless.progressive_final_export as final_export
from minimax_h3_keyless.export import ExportResult
from minimax_h3_keyless.progressive import ProgressiveAcceptedBlock, ProgressivePrefix


def _prefix(count: int) -> ProgressivePrefix:
    return ProgressivePrefix(
        sweep_id="final-export-test",
        code_commit="e" * 40,
        stage_a_campaign_sha256="a" * 64,
        dataset_manifest_sha256="b" * 64,
        gate_manifest_sha256="c" * 64,
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


def test_final_bf16_export_rejects_incomplete_prefix_before_runtime_or_teacher(monkeypatch) -> None:
    prefix = _prefix(49)
    monkeypatch.setattr(
        final_export,
        "load_progressive_prefix_manifest",
        lambda path: (prefix, "d" * 64),
    )
    monkeypatch.setattr(
        final_export,
        "discover_clean_git_revision",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("incomplete prefix must fail before export-source discovery")
        ),
    )
    monkeypatch.setattr(
        final_export,
        "load_pinned_bf16_teacher",
        lambda path: (_ for _ in ()).throw(
            AssertionError("incomplete prefix must fail before allocating the teacher")
        ),
    )

    with pytest.raises(RuntimeError, match="complete accepted progressive prefix"):
        final_export.export_completed_progressive_bf16(
            teacher_path="teacher.safetensors",
            prefix_manifest_path="prefix-49.json",
            artifact_dir="artifacts",
            output_path="final.safetensors",
        )


def test_final_bf16_export_reconstructs_completed_prefix_and_binds_training_and_export_revisions(
    monkeypatch,
) -> None:
    prefix = _prefix(50)
    prefix_manifest_sha = "d" * 64
    calls: list[tuple] = []
    deploy_state = {"deploy.weight": torch.ones(2, 2, dtype=torch.bfloat16)}

    class FakeModel:
        def to(self, device):
            calls.append(("to", str(device)))
            return self

        def state_dict(self):
            calls.append(("state_dict",))
            return deploy_state

    model = FakeModel()
    monkeypatch.setattr(
        final_export,
        "load_progressive_prefix_manifest",
        lambda path: (
            calls.append(("prefix", str(path))) or (prefix, prefix_manifest_sha)
        ),
    )
    # Export hardening is allowed to happen at a later clean revision than the frozen
    # Stage-B training source. Both identities must be retained rather than conflated.
    monkeypatch.setattr(
        final_export,
        "discover_clean_git_revision",
        lambda path, *, label: "f" * 40,
    )
    monkeypatch.setattr(
        final_export,
        "load_pinned_bf16_teacher",
        lambda path: (
            calls.append(("teacher", str(path)))
            or SimpleNamespace(diffusion_model=model)
        ),
    )

    def fake_restore(candidate, supplied_prefix, *, output_dir):
        calls.append(("restore", candidate, supplied_prefix, str(output_dir)))
        assert calls.index(("to", "cpu")) < len(calls) - 1
        return candidate

    monkeypatch.setattr(final_export, "restore_progressive_model_prefix", fake_restore)

    result = ExportResult(
        artifact_path="final.safetensors",
        artifact_sha256="1" * 64,
        artifact_bytes=123,
        manifest_path="final.safetensors.manifest.json",
        manifest_sha256="2" * 64,
        manifest_identity_sha256="3" * 64,
        tensor_count=1,
        teacher_compatibility_checked=True,
    )
    export_call = {}

    def fake_export(state, output_path, **kwargs):
        export_call.update(state=state, output_path=str(output_path), kwargs=kwargs)
        return result

    monkeypatch.setattr(final_export, "export_deploy_bf16", fake_export)
    monkeypatch.setattr(
        final_export,
        "read_safetensors_signatures",
        lambda path: ({"deploy.weight": object()}, {"architecture": "fixture"}),
    )
    postwrite = []
    monkeypatch.setattr(
        final_export,
        "validate_deploy_checkpoint",
        lambda signatures, metadata: postwrite.append(("checkpoint", signatures, metadata)),
    )
    monkeypatch.setattr(
        final_export,
        "validate_deploy_artifact_against_teacher",
        lambda teacher_path, artifact_path: postwrite.append(
            ("teacher", str(teacher_path), str(artifact_path))
        ),
    )

    actual = final_export.export_completed_progressive_bf16(
        teacher_path="teacher.safetensors",
        prefix_manifest_path="prefix-50.json",
        artifact_dir="artifacts",
        output_path="final.safetensors",
        manifest_path="final.receipt.json",
        command="canonical export command",
    )

    assert actual is result
    assert ("to", "cpu") in calls
    assert ("restore", model, prefix, "artifacts") in calls
    assert export_call["state"] is deploy_state
    assert export_call["output_path"] == "final.safetensors"
    kwargs = export_call["kwargs"]
    assert kwargs["metadata"]["training_run"] == (
        f"progressive:{prefix.sweep_id}:{prefix.identity_sha256}"
    )
    assert kwargs["metadata"]["export_commit"] == "f" * 40
    assert prefix.code_commit == "e" * 40
    assert kwargs["manifest_path"] == "final.receipt.json"
    assert kwargs["command"] == "canonical export command"
    assert kwargs["teacher_path"] == "teacher.safetensors"
    assert kwargs["refuse_replace"] is True
    progressive = kwargs["manifest_extra"]["progressive_training"]
    assert progressive["prefix_manifest_sha256"] == prefix_manifest_sha
    assert progressive["prefix"] == prefix.identity_payload()
    assert postwrite == [
        ("checkpoint", {"deploy.weight": postwrite[0][1]["deploy.weight"]}, {"architecture": "fixture"}),
        ("teacher", "teacher.safetensors", "final.safetensors"),
    ]
