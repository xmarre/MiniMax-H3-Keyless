from __future__ import annotations

import json
from pathlib import Path

import pytest

from minimax_h3_keyless.pilot_inputs import (
    CAPTURE_REGISTRY_SCHEMA,
    TRAIN_PLAN_SCHEMA,
    load_stage_a_capture_registry,
    load_stage_a_run_plan,
)


def test_capture_registry_resolves_relative_paths_and_rejects_duplicate_receipts(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    value = {
        "schema": CAPTURE_REGISTRY_SCHEMA,
        "dataset_manifest_sha256": "a" * 64,
        "artifacts": [
            {
                "bundle_path": "captures/a.pt",
                "receipt_path": "captures/a.receipt.json",
                "receipt_sha256": "b" * 64,
            }
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    registry = load_stage_a_capture_registry(path)
    assert registry.artifacts[0].bundle_path == str((tmp_path / "captures/a.pt").resolve())
    assert registry.artifacts[0].receipt_path == str(
        (tmp_path / "captures/a.receipt.json").resolve()
    )
    assert len(registry.registry_file_sha256) == 64

    value["artifacts"].append(dict(value["artifacts"][0]))
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate capture receipt"):
        load_stage_a_capture_registry(path)


def _plan():
    return {
        "schema": TRAIN_PLAN_SCHEMA,
        "stages": [
            {
                "stage": "route",
                "epochs": 2,
                "learning_rate": 1e-3,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
            },
            {
                "stage": "value",
                "epochs": 1,
                "learning_rate": 1e-4,
            },
        ],
        "loss_weights": {
            "attention_output": 1.0,
            "block_output": 0.5,
            "epsilon": 1e-8,
        },
        "same_input_atol": 0.0,
        "same_input_rtol": 0.0,
    }


def test_train_plan_is_fully_predeclared_and_hash_bound(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    value = _plan()
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    plan = load_stage_a_run_plan(path)
    assert [stage.stage for stage in plan.stages] == ["route", "value"]
    assert plan.stages[0].epochs == 2
    assert plan.loss_weights.block_output == 0.5
    assert len(plan.plan_identity_sha256) == 64
    assert len(plan.plan_file_sha256) == 64


def test_train_plan_rejects_nan_unknown_fields_and_noninteger_epochs(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    value = _plan()
    value["stages"][0]["learning_rate"] = float("nan")
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_stage_a_run_plan(path)

    value = _plan()
    value["stages"][0]["epochs"] = 1.5
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="positive integer"):
        load_stage_a_run_plan(path)

    value = _plan()
    value["surprise"] = True
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_stage_a_run_plan(path)
