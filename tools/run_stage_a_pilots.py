#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from minimax_h3_keyless.checkpoint import sha256_file
from minimax_h3_keyless.contracts import TEACHER_SHA256
from minimax_h3_keyless.pilot_artifacts import StageAArtifactRequest
from minimax_h3_keyless.pilot_campaign import (
    load_json_manifest,
    validate_pilot_gate_manifest,
    write_json_atomic,
)
from minimax_h3_keyless.pilot_capture_lazy import load_stage_a_capture_set_lazy
from minimax_h3_keyless.pilot_completed import load_completed_stage_a_block_evidence
from minimax_h3_keyless.pilot_gates import evaluate_stage_a_campaign_gate
from minimax_h3_keyless.pilot_inputs import (
    CANONICAL_STAGE_A_COVERAGE_TAGS,
    load_stage_a_capture_registry,
    load_stage_a_run_plan,
    stage_a_experiment_context_sha256,
)
from minimax_h3_keyless.pilot_runner import run_stage_a_block_pilot
from minimax_h3_keyless.teacher import load_pinned_bf16_teacher


CAMPAIGN_RESULT_SCHEMA = "minimax_h3_keyless_stage_a_campaign_result_v1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed blocks 0/25/49 Stage-A Keyless pilots from persisted native-BF16 "
            "capture bundles. This command validates the pinned teacher, fixed dataset, capture "
            "registry, gate manifest and predeclared train plan before training."
        )
    )
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--capture-registry", required=True)
    parser.add_argument("--gate-manifest", required=True)
    parser.add_argument("--train-plan", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--device", required=True, help="Explicit torch device, e.g. cuda:0")
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help=(
            "Reuse fully completed block artifacts from this same run after validating their "
            "checkpoint/result hashes, fixed experiment context, initialization selection and gate."
        ),
    )
    args = parser.parse_args()

    dataset = load_json_manifest(args.dataset_manifest)
    gate_manifest = load_json_manifest(args.gate_manifest)
    gate_sha = validate_pilot_gate_manifest(gate_manifest)
    registry = load_stage_a_capture_registry(args.capture_registry)
    plan = load_stage_a_run_plan(args.train_plan)

    # Stage-A captures are potentially multi-GiB at native H3 sequence lengths. Index and
    # hash-validate the whole corpus without retaining activation tensors, then materialize
    # only the requested pilot depth inside each sequential block run.
    capture_set = load_stage_a_capture_set_lazy(
        registry.artifacts,
        dataset,
        required_coverage_tags=CANONICAL_STAGE_A_COVERAGE_TAGS,
    )
    if registry.dataset_manifest_sha256.lower() != capture_set.dataset_manifest_sha256.lower():
        raise RuntimeError(
            "capture registry dataset_manifest_sha256 does not match the validated Stage-A dataset"
        )
    experiment_context_sha = stage_a_experiment_context_sha256(
        dataset_manifest_sha256=capture_set.dataset_manifest_sha256,
        gate_manifest_sha256=gate_sha,
        capture_registry_file_sha256=registry.registry_file_sha256,
        train_plan_identity_sha256=plan.plan_identity_sha256,
        capture_code_commit=capture_set.code_commit,
        capture_comfy_commit=capture_set.comfy_commit,
        capture_execution_descriptor=capture_set.execution_descriptor,
    )

    artifact_request = StageAArtifactRequest(
        output_dir=args.output_dir,
        run_id=args.run_id,
        code_commit=args.code_commit,
        experiment_context_sha256=experiment_context_sha,
    )
    campaign_result_path = Path(args.output_dir) / f"{args.run_id}.campaign.result.json"
    if campaign_result_path.exists():
        raise FileExistsError(
            f"Stage-A campaign result is immutable; choose a new run_id: {campaign_result_path}"
        )

    final_stage = plan.stages[-1].stage
    gates = {}
    artifacts = {}
    resumed_blocks: list[int] = []
    pending_blocks: list[int] = []
    for block_index in (0, 25, 49):
        completed = None
        if args.resume_completed:
            completed = load_completed_stage_a_block_evidence(
                artifact_request,
                block_index=block_index,
                final_stage=final_stage,
                expected_dataset_manifest_sha256=capture_set.dataset_manifest_sha256,
                expected_gate_manifest_sha256=gate_sha,
                gate_manifest=gate_manifest,
            )
        if completed is None:
            artifact_request.assert_available(block_index, final_stage)
            pending_blocks.append(block_index)
        else:
            gates[block_index] = completed.gate
            artifacts[block_index] = completed.artifact
            resumed_blocks.append(block_index)

    teacher_sha = sha256_file(args.teacher)
    if teacher_sha.lower() != TEACHER_SHA256:
        raise RuntimeError(
            f"teacher SHA-256 mismatch: expected {TEACHER_SHA256}, got {teacher_sha}"
        )
    executed_blocks: list[int] = []
    if pending_blocks:
        teacher = load_pinned_bf16_teacher(args.teacher)
        teacher_sha = teacher.signature_report.teacher_sha256 or teacher_sha
        teacher_blocks = teacher.pilot_blocks
        for block_index in pending_blocks:
            result = run_stage_a_block_pilot(
                teacher_blocks[block_index],
                capture_set,
                block_index=block_index,
                device=args.device,
                gate_manifest=gate_manifest,
                train_plan=plan.stages,
                loss_weights=plan.loss_weights,
                same_input_atol=plan.same_input_atol,
                same_input_rtol=plan.same_input_rtol,
                artifact_request=artifact_request,
            )
            if result.artifact is None:
                raise RuntimeError("Stage-A block runner did not persist its requested evidence")
            gates[block_index] = result.gate
            artifacts[block_index] = result.artifact
            executed_blocks.append(block_index)

    campaign_gate = evaluate_stage_a_campaign_gate(gates)
    payload = {
        "schema": CAMPAIGN_RESULT_SCHEMA,
        "run_id": args.run_id,
        "code_commit": args.code_commit,
        "experiment_context_sha256": experiment_context_sha,
        "teacher_sha256": teacher_sha,
        "dataset_manifest_sha256": capture_set.dataset_manifest_sha256,
        "gate_manifest_sha256": gate_sha,
        "capture_registry_file_sha256": registry.registry_file_sha256,
        "train_plan_identity_sha256": plan.plan_identity_sha256,
        "train_plan_file_sha256": plan.plan_file_sha256,
        "capture_code_commit": capture_set.code_commit,
        "capture_comfy_commit": capture_set.comfy_commit,
        "capture_execution_descriptor": capture_set.execution_descriptor,
        "resumed_blocks": resumed_blocks,
        "executed_blocks": executed_blocks,
        "block_artifacts": {
            str(index): asdict(artifacts[index]) for index in (0, 25, 49)
        },
        "gate": asdict(campaign_gate),
    }
    campaign_sha = write_json_atomic(campaign_result_path, payload)
    print(f"Stage-A campaign result: {campaign_result_path}")
    print(f"Stage-A campaign result SHA-256: {campaign_sha}")
    print(f"Stage-A campaign gate passed: {campaign_gate.passed}")
    return 0 if campaign_gate.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
