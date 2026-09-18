from __future__ import annotations

import copy

import pytest

from minimax_h3_keyless.real_h3_replay_capture import (
    REAL_H3_REPLAY_CAPTURE_CLASS_TYPE,
    REAL_H3_REPLAY_EVIDENCE_SCHEMA,
    RealH3ReplayCaptureSpec,
    canonical_real_h3_replay_workflow_prompt_sha256,
    real_h3_replay_evidence_identity,
)


def _prompt():
    return {
        "1": {
            "class_type": "MiniMaxH3StageATeacherLoader",
            "inputs": {"model_name": "teacher.safetensors"},
            "_meta": {"title": "teacher"},
        },
        "2": {
            "class_type": REAL_H3_REPLAY_CAPTURE_CLASS_TYPE,
            "inputs": {
                "model": ["1", 0],
                "target_sigma": 1.0,
                "output_subdir": "one",
                "max_capture_mib": 8192,
                "sigma_tolerance": 1e-6,
            },
            "_meta": {"title": "capture"},
        },
        "3": {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["2", 0], "steps": 19, "scheduler": "simple"},
        },
    }


def test_workflow_hash_ignores_only_capture_storage_knobs():
    a = _prompt()
    b = copy.deepcopy(a)
    b["2"]["inputs"]["output_subdir"] = "other"
    b["2"]["inputs"]["max_capture_mib"] = 4096
    b["2"]["inputs"]["sigma_tolerance"] = 5e-6
    assert canonical_real_h3_replay_workflow_prompt_sha256(
        a, capture_node_id="2"
    ) == canonical_real_h3_replay_workflow_prompt_sha256(
        b, capture_node_id=2
    )

    c = copy.deepcopy(a)
    c["2"]["inputs"]["target_sigma"] = 0.5
    assert canonical_real_h3_replay_workflow_prompt_sha256(
        a, capture_node_id="2"
    ) != canonical_real_h3_replay_workflow_prompt_sha256(
        c, capture_node_id="2"
    )


def test_workflow_hash_rejects_stage_a_dataset_capture_node():
    prompt = _prompt()
    prompt["2"]["class_type"] = "MiniMaxH3StageACapture"
    with pytest.raises(ValueError, match="is not"):
        canonical_real_h3_replay_workflow_prompt_sha256(
            prompt,
            capture_node_id="2",
        )


def test_evidence_identity_is_stable_and_sigma_bound():
    workflow = canonical_real_h3_replay_workflow_prompt_sha256(
        _prompt(),
        capture_node_id="2",
    )
    first = real_h3_replay_evidence_identity(
        workflow_prompt_sha256=workflow,
        target_sigma=1.0,
    )
    second = real_h3_replay_evidence_identity(
        workflow_prompt_sha256=workflow,
        target_sigma=1.0,
    )
    other = real_h3_replay_evidence_identity(
        workflow_prompt_sha256=workflow,
        target_sigma=0.5,
    )
    assert first == second
    assert first != other
    assert len(first) == 64
    assert REAL_H3_REPLAY_EVIDENCE_SCHEMA in (
        "minimax_h3_keyless_real_h3_replay_evidence_v1",
    )


def test_spec_has_no_training_case_or_split_semantics(tmp_path):
    workflow = canonical_real_h3_replay_workflow_prompt_sha256(
        _prompt(),
        capture_node_id="2",
    )
    evidence = real_h3_replay_evidence_identity(
        workflow_prompt_sha256=workflow,
        target_sigma=1.0,
    )
    spec = RealH3ReplayCaptureSpec(
        evidence_identity_sha256=evidence,
        workflow_prompt_sha256=workflow,
        target_sigma=1.0,
        output_path=str(tmp_path / "capture.pt"),
        max_capture_bytes=1024,
    )
    assert not hasattr(spec, "case_id")
    assert not hasattr(spec, "source_case_id")
    assert not hasattr(spec, "split")


@pytest.mark.parametrize("sigma", [-0.1, 1.1, float("inf")])
def test_evidence_identity_rejects_invalid_sigma(sigma):
    workflow = canonical_real_h3_replay_workflow_prompt_sha256(
        _prompt(),
        capture_node_id="2",
    )
    with pytest.raises(ValueError, match="target sigma"):
        real_h3_replay_evidence_identity(
            workflow_prompt_sha256=workflow,
            target_sigma=sigma,
        )
