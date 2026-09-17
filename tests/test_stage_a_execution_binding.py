from __future__ import annotations

from pathlib import Path

import pytest

from minimax_h3_keyless.stage_a_execution_binding import (
    WORKFLOW_CONTEXT_KEY,
    canonical_stage_a_workflow_prompt_sha256,
    require_manifest_case_workflow_prompt_sha256,
    require_record_workflow_prompt_sha256,
    validate_stage_a_manifest_assets_for_workflow,
)


def _prompt(*, text: str = "a lighthouse", seed: int = 123, asset: str = "ref.png") -> dict:
    return {
        "4": {
            "class_type": "LoadImage",
            "inputs": {"image": asset},
            "_meta": {"title": "Reference image"},
        },
        "7": {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": {
                "prompt": text,
                "ref_image_1": ["4", 0],
                "width": 960,
                "height": 704,
                "length": 124,
            },
        },
        "10": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "20": {
            "class_type": "MiniMaxH3StageACapture",
            "inputs": {
                "model": ["2", 0],
                "dataset_manifest_path": "/evidence/stage_a.json",
                "case_id": "train-lighthouse",
                "target_sigma": 0.5,
                "output_subdir": "captures/a",
                "max_capture_mib": 8192,
                "sigma_tolerance": 1e-6,
            },
            "_meta": {"title": "Capture A"},
        },
    }


def test_workflow_prompt_hash_ignores_capture_bookkeeping_and_ui_metadata() -> None:
    first = _prompt()
    second = _prompt()
    second["20"]["inputs"].update(
        {
            "dataset_manifest_path": "/different/location.json",
            "case_id": "other-case-label",
            "target_sigma": 0.875,
            "output_subdir": "captures/b",
            "max_capture_mib": 16384,
            "sigma_tolerance": 1e-5,
        }
    )
    second["20"]["_meta"] = {"title": "Different UI title"}
    second["4"]["_meta"] = {"title": "Renamed UI node"}

    assert canonical_stage_a_workflow_prompt_sha256(
        first, capture_node_id="20"
    ) == canonical_stage_a_workflow_prompt_sha256(second, capture_node_id=20)


def test_workflow_prompt_hash_changes_for_semantic_generation_inputs() -> None:
    baseline = canonical_stage_a_workflow_prompt_sha256(_prompt(), capture_node_id="20")
    assert canonical_stage_a_workflow_prompt_sha256(
        _prompt(text="a stormy lighthouse"), capture_node_id="20"
    ) != baseline
    assert canonical_stage_a_workflow_prompt_sha256(
        _prompt(seed=124), capture_node_id="20"
    ) != baseline
    assert canonical_stage_a_workflow_prompt_sha256(
        _prompt(asset="different.png"), capture_node_id="20"
    ) != baseline


def test_workflow_prompt_hash_requires_exact_capture_node_identity() -> None:
    with pytest.raises(ValueError, match="absent"):
        canonical_stage_a_workflow_prompt_sha256(_prompt(), capture_node_id="999")
    wrong = _prompt()
    wrong["20"]["class_type"] = "OtherCapture"
    with pytest.raises(ValueError, match="not 'MiniMaxH3StageACapture'"):
        canonical_stage_a_workflow_prompt_sha256(wrong, capture_node_id="20")


def test_manifest_workflow_hash_and_record_context_are_strict_sha256() -> None:
    digest = canonical_stage_a_workflow_prompt_sha256(_prompt(), capture_node_id="20")
    case = {"case_id": "case-a", "workflow_prompt_sha256": digest}
    assert require_manifest_case_workflow_prompt_sha256(case) == digest
    with pytest.raises(ValueError, match="must predeclare"):
        require_manifest_case_workflow_prompt_sha256({"case_id": "case-a"})

    record = type(
        "Record",
        (),
        {"case": type("Case", (), {"context": {WORKFLOW_CONTEXT_KEY: digest}})()},
    )()
    assert require_record_workflow_prompt_sha256(record) == digest
    record.case.context.clear()
    with pytest.raises(ValueError, match=WORKFLOW_CONTEXT_KEY):
        require_record_workflow_prompt_sha256(record)


def test_manifest_assets_must_be_prompt_referenced_and_match_resolved_bytes(tmp_path: Path) -> None:
    asset = tmp_path / "ref.png"
    asset.write_bytes(b"stage-a-reference-bytes")
    import hashlib

    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    case = {
        "case_id": "case-a",
        "assets": [{"path_or_uri": "ref.png", "sha256": digest}],
    }
    prompt = _prompt(asset="ref.png")

    validate_stage_a_manifest_assets_for_workflow(
        case,
        prompt,
        asset_path_resolver=lambda value: tmp_path / value,
    )

    with pytest.raises(ValueError, match="not referenced literally"):
        validate_stage_a_manifest_assets_for_workflow(
            case,
            _prompt(asset="other.png"),
            asset_path_resolver=lambda value: tmp_path / value,
        )

    asset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_stage_a_manifest_assets_for_workflow(
            case,
            prompt,
            asset_path_resolver=lambda value: tmp_path / value,
        )
