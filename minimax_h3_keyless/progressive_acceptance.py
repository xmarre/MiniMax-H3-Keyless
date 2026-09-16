from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn

from .attention import KeylessAttentionDeploy, KeylessAttentionTrain
from .checkpoint import sha256_file
from .export import fold_query_route_weight
from .pilot import _run_block_capture_attention
from .pilot_campaign import canonical_json_sha256, validate_pilot_gate_manifest
from .pilot_gates import stage_a_policy_from_gate_manifest
from .pilot_replay import pilot_case_to_device
from .progressive import ProgressivePrefix, validate_progressive_model_prefix
from .progressive_artifacts import (
    PROGRESSIVE_RESUME_SCHEMA,
    PROGRESSIVE_RESULT_SCHEMA,
    ProgressiveArtifactReceipt,
    ProgressiveRunIdentity,
    _result_payload,
    progressive_run_identity,
)
from .progressive_capture_set import ProgressiveBlockCaptureSet
from .progressive_gates import evaluate_progressive_block_gate
from .progressive_runner import ProgressiveBlockTrainingResult


def fold_progressive_training_block(student_block: nn.Module) -> nn.Module:
    """Return a frozen deploy-form copy of one accepted q/R/v training block."""
    training = getattr(student_block, "attn", None)
    if not isinstance(training, KeylessAttentionTrain):
        raise RuntimeError("progressive fold requires a KeylessAttentionTrain block")
    q_weight = getattr(training.q_proj, "weight", None)
    v_weight = getattr(training.v_proj, "weight", None)
    route_weight = getattr(training.query_route, "weight", None)
    if not all(torch.is_tensor(value) for value in (q_weight, v_weight, route_weight)):
        raise RuntimeError("progressive fold requires materialized q/R/v weights")
    assert q_weight is not None and v_weight is not None and route_weight is not None
    if getattr(q_weight, "is_meta", False) or getattr(v_weight, "is_meta", False):
        raise RuntimeError("progressive fold cannot use meta-device q/v weights")

    block = copy.deepcopy(student_block)
    deploy = KeylessAttentionDeploy(
        training.hidden,
        training.heads,
        training.head_dim,
        float(training.q_norm.eps),
        gate_compress=training.to_gate_compress is not None,
        block_index=training.block_index,
        dtype=q_weight.dtype,
        device=q_weight.device,
        operations=None,
    )
    q_eff = fold_query_route_weight(
        q_weight.detach(),
        route_weight.detach(),
        output_dtype=q_weight.dtype,
    )
    with torch.no_grad():
        deploy.qv_proj.weight.copy_(torch.cat((q_eff, v_weight.detach()), dim=0))
        deploy.q_norm.weight.copy_(training.q_norm.weight.detach())
        deploy.route_norm.weight.copy_(training.route_norm.weight.detach())
        deploy.out_proj.weight.copy_(training.out_proj.weight.detach())
        if training.to_gate_compress is not None:
            if deploy.to_gate_compress is None:
                raise RuntimeError("progressive fold lost gate-compress topology")
            deploy.to_gate_compress.weight.copy_(training.to_gate_compress.weight.detach())
    block.attn = deploy
    block.eval()
    for parameter in block.parameters():
        parameter.requires_grad_(False)
    return block


def _verify_fold_equivalence(
    training_block: nn.Module,
    deploy_block: nn.Module,
    captures: ProgressiveBlockCaptureSet,
    *,
    atol: float,
    rtol: float,
) -> None:
    if atol < 0 or rtol < 0:
        raise ValueError("progressive fold tolerances must be non-negative")
    try:
        device = next(training_block.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("progressive training block has no parameters") from exc
    was_training = training_block.training
    training_block.eval()
    deploy_block.eval()
    try:
        with torch.no_grad():
            for record in captures.records("holdout"):
                case = pilot_case_to_device(record.case, device)
                train_out, train_h, train_attn = _run_block_capture_attention(training_block, case)
                deploy_out, deploy_h, deploy_attn = _run_block_capture_attention(deploy_block, case)
                torch.testing.assert_close(
                    deploy_h,
                    train_h,
                    atol=0.0,
                    rtol=0.0,
                    msg=f"progressive fold changed block input for {case.case_id!r}",
                )
                torch.testing.assert_close(
                    deploy_attn,
                    train_attn,
                    atol=float(atol),
                    rtol=float(rtol),
                    msg=f"progressive folded attention diverged for {case.case_id!r}",
                )
                torch.testing.assert_close(
                    deploy_out,
                    train_out,
                    atol=float(atol),
                    rtol=float(rtol),
                    msg=f"progressive folded block diverged for {case.case_id!r}",
                )
                if not torch.isfinite(deploy_out).all() or not torch.isfinite(deploy_attn).all():
                    raise RuntimeError("progressive folded deploy block produced non-finite output")
    finally:
        training_block.train(was_training)


def _validate_persisted_candidate(
    *,
    prefix: ProgressivePrefix,
    captures: ProgressiveBlockCaptureSet,
    result: ProgressiveBlockTrainingResult,
    receipt: ProgressiveArtifactReceipt,
    gate_manifest: Mapping[str, object],
) -> ProgressiveRunIdentity:
    expected_identity = progressive_run_identity(prefix, captures, result)
    gate_sha = validate_pilot_gate_manifest(gate_manifest)
    if gate_sha.lower() != prefix.gate_manifest_sha256.lower():
        raise RuntimeError("acceptance gate manifest differs from fixed progressive policy")
    recomputed_gate = evaluate_progressive_block_gate(
        block_index=result.block_index,
        candidate=result.candidate,
        identity_baseline=result.identity_baseline,
        least_squares_baseline=result.least_squares_baseline,
        training_events=result.training_events,
        policy=stage_a_policy_from_gate_manifest(gate_manifest),
    )
    if asdict(recomputed_gate) != asdict(result.gate):
        raise RuntimeError("progressive candidate gate does not recompute from runtime evidence")
    if not recomputed_gate.passed:
        raise RuntimeError("progressive candidate failed its frozen numerical gate")

    checkpoint_path = Path(receipt.checkpoint_path)
    result_path = Path(receipt.result_path)
    if sha256_file(checkpoint_path).lower() != receipt.checkpoint_sha256.lower():
        raise RuntimeError("progressive acceptance checkpoint hash mismatch")
    if sha256_file(result_path).lower() != receipt.result_sha256.lower():
        raise RuntimeError("progressive acceptance result hash mismatch")

    try:
        stored_result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("invalid persisted progressive result JSON") from exc
    if not isinstance(stored_result, dict) or stored_result.get("schema") != PROGRESSIVE_RESULT_SCHEMA:
        raise RuntimeError("persisted progressive result has incompatible schema")
    if stored_result.get("identity") != asdict(expected_identity):
        raise RuntimeError("persisted progressive result identity differs from runtime candidate")
    if stored_result.get("stage") != result.final_stage:
        raise RuntimeError("persisted progressive result stage differs from runtime candidate")
    if int(stored_result.get("step", -1)) != len(result.training_events):
        raise RuntimeError("persisted progressive result step count differs from runtime candidate")
    if stored_result.get("checkpoint_filename") != checkpoint_path.name:
        raise RuntimeError("persisted progressive result references a different checkpoint filename")
    if str(stored_result.get("checkpoint_sha256", "")).lower() != receipt.checkpoint_sha256.lower():
        raise RuntimeError("persisted progressive result checkpoint hash claim is inconsistent")

    numerical = _result_payload(result)
    runtime_payload_sha = canonical_json_sha256(numerical)
    stored_payload = stored_result.get("result")
    if not isinstance(stored_payload, dict):
        raise RuntimeError("persisted progressive result is missing numerical evidence")
    stored_payload_sha = canonical_json_sha256(stored_payload)
    if stored_payload_sha != runtime_payload_sha:
        raise RuntimeError("persisted progressive numerical evidence differs from runtime candidate")
    if str(stored_result.get("result_payload_sha256", "")).lower() != runtime_payload_sha:
        raise RuntimeError("persisted progressive result payload hash claim is inconsistent")
    if receipt.result_payload_sha256.lower() != runtime_payload_sha:
        raise RuntimeError("progressive artifact receipt payload hash differs from runtime candidate")
    if receipt.capture_set_identity_sha256.lower() != expected_identity.capture_set_identity_sha256.lower():
        raise RuntimeError("progressive artifact receipt capture-set identity is inconsistent")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != PROGRESSIVE_RESUME_SCHEMA:
        raise RuntimeError("persisted progressive checkpoint has incompatible schema")
    if checkpoint.get("identity") != asdict(expected_identity):
        raise RuntimeError("persisted progressive checkpoint identity differs from runtime candidate")
    if checkpoint.get("stage") != result.final_stage:
        raise RuntimeError("persisted progressive checkpoint stage differs from runtime candidate")
    if int(checkpoint.get("step", -1)) != len(result.training_events):
        raise RuntimeError("persisted progressive checkpoint step count differs from runtime candidate")
    if str(checkpoint.get("result_payload_sha256", "")).lower() != runtime_payload_sha:
        raise RuntimeError("persisted progressive checkpoint is not bound to runtime numerical evidence")
    stored_state = checkpoint.get("student_state_dict")
    if not isinstance(stored_state, dict):
        raise RuntimeError("persisted progressive checkpoint is missing student state")
    live_state = result.student_block.state_dict()
    if set(stored_state) != set(live_state):
        raise RuntimeError("persisted progressive checkpoint student keys differ from runtime candidate")
    for name in sorted(live_state):
        saved = stored_state[name]
        live = live_state[name]
        if not torch.is_tensor(saved) or not torch.equal(saved, live.detach().cpu()):
            raise RuntimeError(
                f"persisted progressive checkpoint tensor differs from runtime candidate: {name}"
            )
    return expected_identity


def accept_progressive_block(
    model: nn.Module,
    prefix: ProgressivePrefix,
    captures: ProgressiveBlockCaptureSet,
    result: ProgressiveBlockTrainingResult,
    receipt: ProgressiveArtifactReceipt,
    *,
    gate_manifest: Mapping[str, object],
    fold_atol: float,
    fold_rtol: float,
) -> ProgressivePrefix:
    """Atomically install one passed, persisted, folded block and advance the prefix.

    Every validation and fold-parity check occurs before the live model is mutated. If
    post-install prefix validation fails, the original native target block is restored.
    """
    validate_progressive_model_prefix(model, prefix)
    target = prefix.next_block
    if target is None:
        raise RuntimeError("cannot accept another block into a complete progressive prefix")
    if result.block_index != target:
        raise RuntimeError("progressive candidate is not the current next block")
    _validate_persisted_candidate(
        prefix=prefix,
        captures=captures,
        result=result,
        receipt=receipt,
        gate_manifest=gate_manifest,
    )

    folded = fold_progressive_training_block(result.student_block)
    _verify_fold_equivalence(
        result.student_block,
        folded,
        captures,
        atol=float(fold_atol),
        rtol=float(fold_rtol),
    )
    advanced = prefix.advance(
        block_index=target,
        final_stage=result.final_stage,
        checkpoint_sha256=receipt.checkpoint_sha256,
        result_sha256=receipt.result_sha256,
    )

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise RuntimeError("progressive live model lost its core blocks before acceptance")
    original = blocks[target]
    try:
        blocks[target] = folded
        validate_progressive_model_prefix(model, advanced)
    except BaseException:
        blocks[target] = original
        raise
    return advanced
