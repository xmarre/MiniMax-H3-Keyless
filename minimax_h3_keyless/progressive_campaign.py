from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .checkpoint import sha256_file
from .live_capture import discover_clean_git_revision
from .pilot_campaign import load_json_manifest, validate_pilot_dataset_manifest
from .pilot_inputs import CANONICAL_STAGE_A_COVERAGE_TAGS, StageARunPlan, load_stage_a_run_plan
from .progressive import ProgressivePrefix
from .progressive_artifacts import ProgressiveArtifactReceipt, persist_progressive_block_artifacts
from .progressive_authorization import load_progressive_prefix_manifest
from .progressive_capture_registry import (
    ProgressiveCaptureRegistry,
    load_progressive_capture_registry,
)
from .progressive_capture_set import (
    ProgressiveLazyCaptureSet,
    load_progressive_block_capture_set_lazy,
)
from .progressive_gates import (
    ProgressiveExecutionPolicy,
    progressive_execution_policy_from_gate_manifest,
)
from .progressive_restore import restore_progressive_model_prefix
from .progressive_runner import ProgressiveBlockTrainingResult, run_progressive_block_training
from .progressive_training_resume import (
    ProgressiveTrainingResumeRequest,
    remove_progressive_training_resume,
)
from .progressive_workflow import ProgressiveAcceptedStep, accept_persisted_progressive_block
from .stage_a_campaign_result import StageACampaignEvidence, load_stage_a_campaign_evidence
from .teacher import LoadedNativeTeacher, load_pinned_bf16_teacher


@dataclass(frozen=True)
class ProgressiveBlockRunInputs:
    """Fully bound immutable inputs for one Stage-B target block run."""

    prefix: ProgressivePrefix
    prefix_manifest_path: str
    prefix_manifest_sha256: str
    registry: ProgressiveCaptureRegistry
    stage_a: StageACampaignEvidence
    dataset_manifest: Mapping[str, Any]
    gate_manifest: Mapping[str, object]
    train_plan: StageARunPlan
    execution_policy: ProgressiveExecutionPolicy


@dataclass(frozen=True)
class ProgressiveBlockRunOutcome:
    """Persisted candidate evidence plus optional accepted-prefix advancement."""

    block_index: int
    gate_passed: bool
    artifact: ProgressiveArtifactReceipt
    accepted: ProgressiveAcceptedStep | None


def _require_equal(name: str, left: str, right: str) -> None:
    if str(left).lower() != str(right).lower():
        raise RuntimeError(f"progressive {name} identity mismatch: {left!r} != {right!r}")


def load_progressive_block_run_inputs(
    *,
    stage_a_result_path: str | Path,
    current_prefix_manifest_path: str | Path,
    capture_registry_path: str | Path,
    dataset_manifest_path: str | Path,
    gate_manifest_path: str | Path,
    train_plan_path: str | Path,
) -> ProgressiveBlockRunInputs:
    """Bind Stage-A authorization, current prefix, captures and fixed recipe before runtime work.

    The Stage-B target is not allowed to substitute a different dataset, gate policy or
    train plan after observing pilot results. Registry identity must name the exact current
    prefix manifest and next block. This function performs no model allocation.
    """

    dataset = load_json_manifest(dataset_manifest_path)
    dataset_sha = validate_pilot_dataset_manifest(
        dataset,
        required_coverage_tags=CANONICAL_STAGE_A_COVERAGE_TAGS,
    )
    gate_manifest = load_json_manifest(gate_manifest_path)
    execution_policy = progressive_execution_policy_from_gate_manifest(gate_manifest)
    plan = load_stage_a_run_plan(train_plan_path)
    prefix_path = Path(current_prefix_manifest_path)
    prefix, prefix_manifest_sha = load_progressive_prefix_manifest(prefix_path)
    if prefix.complete:
        raise RuntimeError("progressive sweep is already complete")

    stage_a = load_stage_a_campaign_evidence(
        stage_a_result_path,
        gate_manifest=gate_manifest,
        require_passed=True,
    )
    _require_equal("Stage-A campaign", stage_a.sha256, prefix.stage_a_campaign_sha256)
    _require_equal("dataset", dataset_sha, prefix.dataset_manifest_sha256)
    _require_equal("Stage-A dataset", stage_a.dataset_manifest_sha256, dataset_sha)
    _require_equal("Stage-A gate", stage_a.gate_manifest_sha256, prefix.gate_manifest_sha256)
    _require_equal("train-plan semantic", stage_a.train_plan_identity_sha256, plan.plan_identity_sha256)
    _require_equal("train-plan file", stage_a.train_plan_file_sha256, plan.plan_file_sha256)

    registry = load_progressive_capture_registry(capture_registry_path)
    _require_equal("registry prefix manifest", registry.prefix_manifest_sha256, prefix_manifest_sha)
    _require_equal("registry prefix", registry.prefix_identity_sha256, prefix.identity_sha256)
    _require_equal("registry Stage-A campaign", registry.stage_a_campaign_sha256, prefix.stage_a_campaign_sha256)
    _require_equal("registry dataset", registry.dataset_manifest_sha256, dataset_sha)
    _require_equal("registry gate", registry.gate_manifest_sha256, prefix.gate_manifest_sha256)
    _require_equal("registry code", registry.code_commit, prefix.code_commit)
    if registry.target_block != prefix.next_block:
        raise RuntimeError(
            "progressive registry target does not match current prefix next block: "
            f"registry={registry.target_block}, prefix={prefix.next_block}"
        )

    return ProgressiveBlockRunInputs(
        prefix=prefix,
        prefix_manifest_path=str(prefix_path),
        prefix_manifest_sha256=prefix_manifest_sha,
        registry=registry,
        stage_a=stage_a,
        dataset_manifest=dataset,
        gate_manifest=gate_manifest,
        train_plan=plan,
        execution_policy=execution_policy,
    )


def require_progressive_runtime_provenance(inputs: ProgressiveBlockRunInputs) -> str:
    """Require clean Keyless/Comfy revisions to match the fixed progressive capture runtime."""

    runtime_code_commit = discover_clean_git_revision(
        Path(__file__).resolve().parents[1],
        label="MiniMax-H3-Keyless progressive training source",
    )
    _require_equal("runtime code", runtime_code_commit, inputs.prefix.code_commit)
    try:
        import comfy
    except ImportError as exc:
        raise RuntimeError("progressive training requires the ComfyUI runtime on PYTHONPATH") from exc
    comfy_file = getattr(comfy, "__file__", None)
    if not comfy_file:
        raise RuntimeError("cannot resolve ComfyUI source root from comfy.__file__")
    runtime_comfy_commit = discover_clean_git_revision(
        Path(comfy_file).resolve().parents[1],
        label="ComfyUI progressive training source",
    )
    _require_equal("runtime Comfy", runtime_comfy_commit, inputs.registry.comfy_commit)
    return runtime_comfy_commit


def load_progressive_training_captures(
    inputs: ProgressiveBlockRunInputs,
) -> ProgressiveLazyCaptureSet:
    """Create the bounded production capture view for the current target block."""

    captures = load_progressive_block_capture_set_lazy(
        inputs.registry.artifacts,
        inputs.dataset_manifest,
        inputs.prefix,
        required_coverage_tags=CANONICAL_STAGE_A_COVERAGE_TAGS,
    )
    _require_equal("capture registry file", sha256_file(inputs.registry.registry_path), inputs.registry.registry_file_sha256)
    _require_equal("capture code", captures.code_commit, inputs.registry.code_commit)
    _require_equal("capture Comfy", captures.comfy_commit, inputs.registry.comfy_commit)
    if captures.execution_descriptor != inputs.registry.execution_descriptor:
        raise RuntimeError("progressive capture execution descriptor differs from registry")
    return captures


def _training_resume_path(
    artifact_dir: str | Path,
    prefix: ProgressivePrefix,
    target: int,
    requested: str | Path | None,
) -> Path:
    if requested is not None:
        path = Path(requested)
        if not str(path):
            raise ValueError("progressive training resume path must be non-empty")
        return path
    return Path(artifact_dir) / f"{prefix.sweep_id}.block{target:02d}.training-resume.pt"


def run_progressive_block_campaign(
    inputs: ProgressiveBlockRunInputs,
    *,
    teacher_path: str | Path,
    artifact_dir: str | Path,
    device: str,
    resume: bool = False,
    resume_path: str | Path | None = None,
) -> ProgressiveBlockRunOutcome:
    """Train, persist and conditionally accept exactly the current Stage-B block.

    Candidate evidence is persisted whether the frozen numerical gate passes or fails. A
    failed candidate never mutates the accepted model and never publishes a new prefix.
    A passed candidate is accepted from those already-persisted bytes, so evidence is not
    written twice and a prefix-publication failure can roll the live target block back.

    A mutable crash-recovery checkpoint is maintained after every complete training epoch.
    Starting fresh refuses to overwrite an existing recovery file; continuation requires
    explicit ``resume=True``. The recovery file is deleted only after the immutable candidate
    checkpoint/result transaction has succeeded, regardless of whether the candidate gate
    passes. It is therefore never the authority for an accepted block.
    """

    require_progressive_runtime_provenance(inputs)
    captures = load_progressive_training_captures(inputs)
    teacher: LoadedNativeTeacher = load_pinned_bf16_teacher(teacher_path)
    model = teacher.diffusion_model
    restore_progressive_model_prefix(
        model,
        inputs.prefix,
        output_dir=artifact_dir,
    )
    target = inputs.prefix.next_block
    if target is None:
        raise RuntimeError("progressive sweep became complete before training")
    blocks = getattr(model, "blocks", None)
    if blocks is None or len(blocks) <= target:
        raise RuntimeError("loaded H3 model does not expose the progressive target block")

    recovery_path = _training_resume_path(
        artifact_dir,
        inputs.prefix,
        target,
        resume_path,
    )
    resume_request = ProgressiveTrainingResumeRequest(
        path=str(recovery_path),
        prefix_manifest_sha256=inputs.prefix_manifest_sha256,
        capture_registry_file_sha256=inputs.registry.registry_file_sha256,
        train_plan_identity_sha256=inputs.train_plan.plan_identity_sha256,
        train_plan_file_sha256=inputs.train_plan.plan_file_sha256,
        resume=bool(resume),
    )
    result: ProgressiveBlockTrainingResult = run_progressive_block_training(
        blocks[target],
        captures,
        inputs.prefix,
        device=device,
        gate_manifest=inputs.gate_manifest,
        train_plan=inputs.train_plan.stages,
        loss_weights=inputs.train_plan.loss_weights,
        same_input_atol=inputs.train_plan.same_input_atol,
        same_input_rtol=inputs.train_plan.same_input_rtol,
        resume_request=resume_request,
    )
    artifact = persist_progressive_block_artifacts(
        artifact_dir,
        prefix=inputs.prefix,
        captures=captures,
        result=result,
    )
    remove_progressive_training_resume(recovery_path)
    if not result.gate.passed:
        return ProgressiveBlockRunOutcome(
            block_index=target,
            gate_passed=False,
            artifact=artifact,
            accepted=None,
        )

    accepted = accept_persisted_progressive_block(
        model,
        inputs.prefix,
        captures,
        result,
        artifact,
        current_prefix_manifest_path=inputs.prefix_manifest_path,
        current_prefix_manifest_sha256=inputs.prefix_manifest_sha256,
        output_dir=artifact_dir,
        gate_manifest=inputs.gate_manifest,
        fold_atol=inputs.execution_policy.fold_atol,
        fold_rtol=inputs.execution_policy.fold_rtol,
    )
    return ProgressiveBlockRunOutcome(
        block_index=target,
        gate_passed=True,
        artifact=artifact,
        accepted=accepted,
    )
