from __future__ import annotations

from pathlib import Path

import pytest
import torch

from minimax_h3_keyless.activation_capture import CapturedPilotCase
from minimax_h3_keyless.pilot import PilotCase
from minimax_h3_keyless.pilot_campaign import DATASET_SCHEMA, canonical_json_sha256
from minimax_h3_keyless.progressive import (
    PROGRESSIVE_PREFIX_CONTEXT_KEY,
    ProgressiveAcceptedBlock,
    ProgressivePrefix,
)
from minimax_h3_keyless.progressive_capture_io import (
    ProgressiveCaptureProvenance,
    write_progressive_capture_bundle,
)
from minimax_h3_keyless.progressive_capture_set import (
    ProgressiveCaptureArtifactRef,
    load_progressive_block_capture_set,
    load_progressive_block_capture_set_lazy,
)


def _manifest():
    def case(case_id: str, split: str):
        return {
            "case_id": case_id,
            "split": split,
            "prompt": f"prompt-{case_id}",
            "seed": 1 if split == "train" else 2,
            "schedule": {"name": "fixture"},
            "modality_label": "video",
            "resolution": [64, 64],
            "duration_seconds": 1.0,
            "sigmas": [0.2, 0.8],
            "coverage_tags": [],
            "assets": [],
        }

    return {
        "schema": DATASET_SCHEMA,
        "cases": [case("train-a", "train"), case("holdout-a", "holdout")],
    }


def _accepted(result_byte: str = "2") -> ProgressiveAcceptedBlock:
    return ProgressiveAcceptedBlock(
        block_index=0,
        final_stage="route",
        checkpoint_sha256="1" * 64,
        result_sha256=result_byte * 64,
    )


def _prefix(dataset_sha: str, *, result_byte: str = "2", code_commit: str = "deadbeef"):
    return ProgressivePrefix(
        sweep_id="sweep-001",
        code_commit=code_commit,
        stage_a_campaign_sha256="a" * 64,
        dataset_manifest_sha256=dataset_sha,
        gate_manifest_sha256="c" * 64,
        accepted=(_accepted(result_byte),),
    )


def _record(prefix: ProgressivePrefix, case_id: str, sigma: float) -> CapturedPilotCase:
    torch.manual_seed(int(round(sigma * 1000)) + len(case_id))
    x = torch.randn(3, 4)
    case = PilotCase(
        x=x,
        t_emb=torch.zeros(1, 2),
        mod_segments=((0, 3, 0),),
        rope_freqs=None,
        transformer_options={},
        case_id=case_id,
        sigma=sigma,
        modality_label="video",
        context={PROGRESSIVE_PREFIX_CONTEXT_KEY: prefix.capture_context()},
    )
    return CapturedPilotCase(
        block_index=1,
        case=case,
        attention_input=x + 0.5,
        captured_bytes=256,
    )


def _artifact(
    root: Path,
    prefix: ProgressivePrefix,
    case_id: str,
    sigma: float,
    *,
    code_commit: str | None = None,
) -> ProgressiveCaptureArtifactRef:
    provenance = ProgressiveCaptureProvenance(
        code_commit=prefix.code_commit if code_commit is None else code_commit,
        comfy_commit="comfy-deadbeef",
        dataset_manifest_sha256=prefix.dataset_manifest_sha256,
        gate_manifest_sha256=prefix.gate_manifest_sha256,
        stage_a_campaign_sha256=prefix.stage_a_campaign_sha256,
        prefix_identity_sha256=prefix.identity_sha256,
        target_block=1,
        execution_descriptor="native-progressive-fixture",
    )
    path = root / f"{case_id}-{sigma}.capture.pt"
    written = write_progressive_capture_bundle(
        path,
        _record(prefix, case_id, sigma),
        provenance=provenance,
    )
    return ProgressiveCaptureArtifactRef(
        bundle_path=written.bundle_path,
        receipt_path=written.receipt_path,
        receipt_sha256=written.receipt_sha256,
    )


def _all_artifacts(root: Path, prefix: ProgressivePrefix):
    return tuple(
        _artifact(root, prefix, case_id, sigma)
        for case_id in ("train-a", "holdout-a")
        for sigma in (0.2, 0.8)
    )


def test_progressive_capture_set_requires_exact_fixed_dataset_and_annotates_split(tmp_path: Path) -> None:
    manifest = _manifest()
    prefix = _prefix(canonical_json_sha256(manifest))
    artifacts = _all_artifacts(tmp_path, prefix)

    captures = load_progressive_block_capture_set(
        artifacts,
        manifest,
        prefix,
        minimum_cases=2,
        minimum_sigma_strata=2,
    )
    assert captures.target_block == 1
    assert captures.prefix_identity_sha256 == prefix.identity_sha256
    assert len(captures.train) == 2
    assert len(captures.holdout) == 2
    assert {row.case.case_id for row in captures.train} == {
        "train-a::sigma=0.20000000000000001",
        "train-a::sigma=0.80000000000000004",
    }
    assert all(row.case.context["progressive_split"] == "train" for row in captures.train)
    assert all(
        row.case.context["progressive_source_case_id"] == "holdout-a"
        for row in captures.holdout
    )
    assert all(
        len(row.case.context["progressive_capture_receipt_sha256"]) == 64
        for row in (*captures.train, *captures.holdout)
    )


def test_lazy_progressive_capture_set_reloads_only_requested_artifact(tmp_path: Path) -> None:
    manifest = _manifest()
    prefix = _prefix(canonical_json_sha256(manifest))
    artifacts = _all_artifacts(tmp_path, prefix)
    captures = load_progressive_block_capture_set_lazy(
        artifacts,
        manifest,
        prefix,
        minimum_cases=2,
        minimum_sigma_strata=2,
    )

    assert len(captures.train) == 2
    assert len(captures.holdout) == 2
    first = captures.train[0]
    assert first.case.context["progressive_source_case_id"] == "train-a"
    assert first.case.context["progressive_split"] == "train"

    # Index construction must not retain the activation tensors as an eager fallback.
    # Removing the immutable source after indexing makes a subsequent access fail rather
    # than returning a cached record.
    Path(captures.executions[0].artifact.bundle_path).unlink()
    with pytest.raises(FileNotFoundError):
        _ = captures.train[0]


def test_progressive_capture_set_rejects_missing_execution(tmp_path: Path) -> None:
    manifest = _manifest()
    prefix = _prefix(canonical_json_sha256(manifest))
    artifacts = _all_artifacts(tmp_path, prefix)
    with pytest.raises(ValueError, match="does not exactly cover"):
        load_progressive_block_capture_set(
            artifacts[:-1],
            manifest,
            prefix,
            minimum_cases=2,
            minimum_sigma_strata=2,
        )


def test_progressive_capture_set_rejects_capture_from_different_prefix(tmp_path: Path) -> None:
    manifest = _manifest()
    dataset_sha = canonical_json_sha256(manifest)
    prefix = _prefix(dataset_sha, result_byte="2")
    other = _prefix(dataset_sha, result_byte="3")
    artifacts = list(_all_artifacts(tmp_path / "expected", prefix))
    artifacts[-1] = _artifact(tmp_path / "other", other, "holdout-a", 0.8)

    with pytest.raises(ValueError, match="prefix_identity_sha256"):
        load_progressive_block_capture_set(
            artifacts,
            manifest,
            prefix,
            minimum_cases=2,
            minimum_sigma_strata=2,
        )


def test_progressive_capture_set_rejects_capture_from_different_sweep_code(tmp_path: Path) -> None:
    manifest = _manifest()
    prefix = _prefix(canonical_json_sha256(manifest))
    artifacts = list(_all_artifacts(tmp_path / "expected", prefix))
    artifacts[-1] = _artifact(
        tmp_path / "wrong-code",
        prefix,
        "holdout-a",
        0.8,
        code_commit="cafebabe",
    )

    with pytest.raises(ValueError, match="code_commit"):
        load_progressive_block_capture_set(
            artifacts,
            manifest,
            prefix,
            minimum_cases=2,
            minimum_sigma_strata=2,
        )
