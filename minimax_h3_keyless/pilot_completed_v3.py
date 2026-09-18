"""Strict Stage-A v3 completed-evidence loader with holdout isolation.

Stage-A v3 records initialization selection only on complete training cases.  The
identity and regularized-LS baselines, the trained candidate, and the exit gate are
then evaluated on the untouched complete-case holdout.  This module is separate
from the v2 parser so older evidence remains inspectable but cannot be silently
reinterpreted under the stronger v3 semantics.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping

import torch

from .checkpoint import sha256_file
from .pilot_artifacts import (
    STAGE_A_RESULT_SCHEMA,
    StageAArtifactReceipt,
    StageAArtifactRequest,
)
from .pilot_campaign import (
    RESUME_SCHEMA,
    PilotRunIdentity,
    _require_sha256,
    canonical_json_sha256,
)
from .pilot_completed import (
    CompletedStageABlockEvidence,
    _aggregate,
    _attention_diagnostics,
    _exact_keys,
    _finite,
    _gate,
    _initialization,
    _object,
    _replay,
    _training_event,
)
from .pilot_gates import evaluate_stage_a_block_gate, stage_a_policy_from_gate_manifest
from .pilot_runner import select_stage_a_initialization


_SELECTION_SPLIT = "train_complete_cases"


def _case_ids(metrics) -> tuple[str, ...]:
    return tuple(case.case_id for case in metrics.cases)


def load_completed_stage_a_block_evidence(
    request: StageAArtifactRequest,
    *,
    block_index: int,
    final_stage: str,
    expected_dataset_manifest_sha256: str,
    expected_gate_manifest_sha256: str,
    gate_manifest: Mapping[str, object],
) -> CompletedStageABlockEvidence | None:
    """Validate and reuse one v3 block pilot without leaking holdout into selection."""
    if request.experiment_context_sha256 is None:
        raise ValueError("resumable Stage-A artifacts require experiment_context_sha256")
    expected_dataset = _require_sha256(
        "expected Stage-A dataset manifest SHA-256", expected_dataset_manifest_sha256
    )
    expected_gate = _require_sha256(
        "expected Stage-A gate manifest SHA-256", expected_gate_manifest_sha256
    )
    checkpoint_path, result_path = request.paths(block_index, final_stage)
    exists = (checkpoint_path.exists(), result_path.exists())
    if not any(exists):
        return None
    if not all(exists):
        raise RuntimeError(
            f"partial Stage-A evidence for block {block_index}: checkpoint={exists[0]}, result={exists[1]}"
        )

    try:
        top = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid Stage-A result JSON: {result_path}") from exc
    top = _object(top, "Stage-A block result")
    top_keys = {
        "schema", "identity", "stage", "step", "experiment_context_sha256",
        "checkpoint_filename", "checkpoint_sha256", "result_payload_sha256", "result",
    }
    _exact_keys(top, top_keys, "Stage-A block result")
    if top["schema"] != STAGE_A_RESULT_SCHEMA:
        raise RuntimeError("not a current Stage-A block result artifact")
    if top["stage"] != final_stage:
        raise RuntimeError("completed Stage-A artifact final stage does not match the current plan")
    if isinstance(top["step"], bool) or not isinstance(top["step"], int) or top["step"] < 0:
        raise RuntimeError("completed Stage-A artifact step must be a non-negative integer")
    if top["experiment_context_sha256"] != request.experiment_context_sha256:
        raise RuntimeError("completed Stage-A artifact experiment context does not match this campaign")
    if top["checkpoint_filename"] != checkpoint_path.name:
        raise RuntimeError("completed Stage-A artifact checkpoint filename is inconsistent")
    checkpoint_sha = _require_sha256("Stage-A checkpoint SHA-256", top["checkpoint_sha256"])
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise RuntimeError("completed Stage-A checkpoint SHA-256 does not match its result artifact")

    result_payload = _object(top["result"], "Stage-A numerical result payload")
    result_payload_sha = _require_sha256(
        "Stage-A result payload SHA-256", top["result_payload_sha256"]
    )
    if canonical_json_sha256(result_payload) != result_payload_sha:
        raise RuntimeError("completed Stage-A numerical result payload hash is invalid")

    identity_row = _object(top["identity"], "Stage-A run identity")
    try:
        identity = PilotRunIdentity(**identity_row)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("completed Stage-A run identity is invalid") from exc
    if identity.run_id != request.run_id or identity.code_commit != request.code_commit:
        raise RuntimeError("completed Stage-A run/code identity does not match this campaign")
    if identity.block_index != block_index:
        raise RuntimeError("completed Stage-A run identity has the wrong block index")
    if identity.dataset_manifest_sha256.lower() != expected_dataset:
        raise RuntimeError("completed Stage-A dataset identity does not match this campaign")
    if identity.gate_manifest_sha256.lower() != expected_gate:
        raise RuntimeError("completed Stage-A gate identity does not match this campaign")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        if not isinstance(checkpoint, dict) or checkpoint.get("schema") != RESUME_SCHEMA:
            raise RuntimeError("completed Stage-A checkpoint has the wrong resume schema")
        if checkpoint.get("identity") != asdict(identity):
            raise RuntimeError("completed Stage-A checkpoint identity differs from result JSON")
        if checkpoint.get("stage") != final_stage or checkpoint.get("step") != top["step"]:
            raise RuntimeError("completed Stage-A checkpoint stage/step differs from result JSON")
        extra = checkpoint.get("extra")
        if not isinstance(extra, dict):
            raise RuntimeError("completed Stage-A checkpoint is missing evidence metadata")
        if extra.get("stage_a_result_schema") != STAGE_A_RESULT_SCHEMA:
            raise RuntimeError("completed Stage-A checkpoint result schema marker is invalid")
        if extra.get("stage_a_result_payload_sha256") != result_payload_sha:
            raise RuntimeError("completed Stage-A checkpoint is not bound to its numerical result")
        if extra.get("stage_a_experiment_context_sha256") != request.experiment_context_sha256:
            raise RuntimeError("completed Stage-A checkpoint experiment context is invalid")
    finally:
        del checkpoint

    payload_keys = {
        "block_index", "selection_split", "replay_reports", "initialization_evaluations",
        "selected_route_mode", "selected_lambda_relative", "identity_baseline",
        "least_squares_baseline", "least_squares_baseline_lambda_relative", "candidate",
        "candidate_attention_diagnostics", "training_events", "gate",
    }
    _exact_keys(result_payload, payload_keys, "Stage-A numerical result payload")
    if result_payload["block_index"] != block_index:
        raise RuntimeError("completed Stage-A numerical result has the wrong block index")
    if result_payload["selection_split"] != _SELECTION_SPLIT:
        raise RuntimeError(
            "completed Stage-A initialization selection was not bound to complete training cases"
        )

    replays_raw = result_payload["replay_reports"]
    evaluations_raw = result_payload["initialization_evaluations"]
    events_raw = result_payload["training_events"]
    if not isinstance(replays_raw, list) or not replays_raw:
        raise RuntimeError("completed Stage-A result has no replay evidence")
    if not isinstance(evaluations_raw, list):
        raise RuntimeError("completed Stage-A result initialization_evaluations must be a list")
    if not isinstance(events_raw, list) or not events_raw:
        raise RuntimeError("completed Stage-A result has no training events")
    tuple(_replay(row, block_index) for row in replays_raw)
    evaluations = tuple(_initialization(row) for row in evaluations_raw)
    selected = select_stage_a_initialization(evaluations)
    selected_lambda = _finite(
        "completed Stage-A selected lambda", result_payload["selected_lambda_relative"]
    )
    if result_payload["selected_route_mode"] != selected.route_mode or selected_lambda != selected.lambda_relative:
        raise RuntimeError("completed Stage-A selected initialization does not match its training grid")
    if identity.route_mode != selected.route_mode or identity.lambda_relative != selected.lambda_relative:
        raise RuntimeError("completed Stage-A run identity does not match selected initialization")

    selection_case_ids = _case_ids(evaluations[0].metrics) if evaluations else ()
    if not selection_case_ids:
        raise RuntimeError("completed Stage-A initialization grid has no training cases")
    for evaluation in evaluations[1:]:
        if _case_ids(evaluation.metrics) != selection_case_ids:
            raise RuntimeError("completed Stage-A initialization rows do not share one training case set")

    best_ls = min(
        (row for row in evaluations if row.route_mode == "least_squares"),
        key=lambda row: (
            row.metrics.mean_attention_normalized_mse,
            row.metrics.mean_block_normalized_mse,
            row.lambda_relative,
        ),
    )
    baseline_lambda = _finite(
        "completed Stage-A least-squares baseline lambda",
        result_payload["least_squares_baseline_lambda_relative"],
    )
    if baseline_lambda != best_ls.lambda_relative:
        raise RuntimeError(
            "completed Stage-A holdout LS baseline does not use the train-selected LS lambda"
        )

    identity_baseline = _aggregate(result_payload["identity_baseline"])
    ls_baseline = _aggregate(result_payload["least_squares_baseline"])
    candidate = _aggregate(result_payload["candidate"])
    holdout_case_ids = _case_ids(candidate)
    if set(selection_case_ids).intersection(holdout_case_ids):
        raise RuntimeError("completed Stage-A training-selection and holdout case IDs overlap")
    _attention_diagnostics(result_payload["candidate_attention_diagnostics"], candidate.cases)
    events = tuple(_training_event(row) for row in events_raw)
    stored_gate = _gate(result_payload["gate"])
    if stored_gate.block_index != block_index:
        raise RuntimeError("completed Stage-A stored gate has the wrong block index")
    recomputed_gate = evaluate_stage_a_block_gate(
        block_index=block_index,
        candidate=candidate,
        identity_baseline=identity_baseline,
        least_squares_baseline=ls_baseline,
        training_events=events,
        policy=stage_a_policy_from_gate_manifest(gate_manifest),
    )
    if asdict(stored_gate) != asdict(recomputed_gate):
        raise RuntimeError("completed Stage-A gate does not recompute from its bound evidence")

    return CompletedStageABlockEvidence(
        block_index=block_index,
        identity=identity,
        stage=final_stage,
        step=top["step"],
        gate=recomputed_gate,
        artifact=StageAArtifactReceipt(
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha,
            result_path=str(result_path),
            result_sha256=sha256_file(result_path),
        ),
    )
