from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .checkpoint import sha256_file
from .contracts import TEACHER_SHA256
from .pilot_artifacts import STAGE_A_RESULT_SCHEMA, StageAArtifactReceipt, StageAArtifactRequest
from .pilot_campaign import PILOT_BLOCKS, _require_sha256, validate_pilot_gate_manifest
from .pilot_completed import CompletedStageABlockEvidence, load_completed_stage_a_block_evidence
from .pilot_gates import StageACampaignGateResult, evaluate_stage_a_campaign_gate


STAGE_A_CAMPAIGN_RESULT_SCHEMA = "minimax_h3_keyless_stage_a_campaign_result_v2"
_STAGE_NAMES = ("route", "query", "value", "norm_out")


@dataclass(frozen=True)
class StageACampaignEvidence:
    path: str
    sha256: str
    run_id: str
    code_commit: str
    final_stage: str
    experiment_context_sha256: str
    teacher_sha256: str
    dataset_manifest_sha256: str
    gate_manifest_sha256: str
    block_evidence: Mapping[int, CompletedStageABlockEvidence]
    gate: StageACampaignGateResult


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys.difference(value))
    unknown = sorted(set(value).difference(keys))
    if missing or unknown:
        raise RuntimeError(f"{label} key mismatch: missing={missing}, unknown={unknown}")


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{label} must be a non-empty string")
    return value


def _int_list(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a list")
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise RuntimeError(f"{label} must contain only integer block indices")
        out.append(item)
    if len(set(out)) != len(out):
        raise RuntimeError(f"{label} contains duplicate block indices")
    return tuple(out)


def _artifact_receipt(value: Any, block_index: int) -> StageAArtifactReceipt:
    row = _object(value, f"Stage-A campaign block {block_index} artifact")
    keys = {"checkpoint_path", "checkpoint_sha256", "result_path", "result_sha256"}
    _exact_keys(row, keys, f"Stage-A campaign block {block_index} artifact")
    checkpoint_path = _nonempty_string(
        row["checkpoint_path"], f"Stage-A campaign block {block_index} checkpoint_path"
    )
    result_path = _nonempty_string(
        row["result_path"], f"Stage-A campaign block {block_index} result_path"
    )
    return StageAArtifactReceipt(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=_require_sha256(
            f"Stage-A campaign block {block_index} checkpoint SHA-256",
            row["checkpoint_sha256"],
        ),
        result_path=result_path,
        result_sha256=_require_sha256(
            f"Stage-A campaign block {block_index} result SHA-256",
            row["result_sha256"],
        ),
    )


def _json_normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


def load_stage_a_campaign_evidence(
    path: str | Path,
    *,
    gate_manifest: Mapping[str, object],
    require_passed: bool = True,
) -> StageACampaignEvidence:
    """Validate the immutable Phase-3 exit before authorizing a core50 sweep.

    Every block result is reparsed through the strict Stage-A completed-evidence path,
    which verifies its checkpoint/result hashes, experiment identity, initialization grid,
    attention diagnostics and recomputed block gate. The campaign gate is then recomputed
    from those three block gates. A JSON field claiming ``passed=true`` is never trusted
    on its own.
    """
    path = Path(path)
    try:
        top = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid Stage-A campaign result JSON: {path}") from exc
    top = _object(top, "Stage-A campaign result")
    keys = {
        "schema",
        "stage_a_block_result_schema",
        "run_id",
        "code_commit",
        "training_comfy_commit",
        "experiment_context_sha256",
        "teacher_sha256",
        "dataset_manifest_sha256",
        "gate_manifest_sha256",
        "capture_registry_file_sha256",
        "train_plan_identity_sha256",
        "train_plan_file_sha256",
        "capture_code_commit",
        "capture_comfy_commit",
        "capture_execution_descriptor",
        "final_stage",
        "resumed_blocks",
        "executed_blocks",
        "block_artifacts",
        "gate",
    }
    _exact_keys(top, keys, "Stage-A campaign result")
    if top["schema"] != STAGE_A_CAMPAIGN_RESULT_SCHEMA:
        raise RuntimeError("not a current Stage-A campaign result artifact")
    if top["stage_a_block_result_schema"] != STAGE_A_RESULT_SCHEMA:
        raise RuntimeError("Stage-A campaign references an incompatible block-result schema")

    run_id = _nonempty_string(top["run_id"], "Stage-A campaign run_id")
    code_commit = _nonempty_string(top["code_commit"], "Stage-A campaign code_commit").lower()
    _nonempty_string(top["training_comfy_commit"], "Stage-A campaign training_comfy_commit")
    _nonempty_string(top["capture_code_commit"], "Stage-A campaign capture_code_commit")
    _nonempty_string(top["capture_comfy_commit"], "Stage-A campaign capture_comfy_commit")
    _nonempty_string(
        top["capture_execution_descriptor"], "Stage-A campaign capture_execution_descriptor"
    )
    final_stage = _nonempty_string(top["final_stage"], "Stage-A campaign final_stage")
    if final_stage not in _STAGE_NAMES:
        raise RuntimeError(f"unsupported Stage-A campaign final_stage: {final_stage!r}")

    sha_fields = (
        "experiment_context_sha256",
        "teacher_sha256",
        "dataset_manifest_sha256",
        "gate_manifest_sha256",
        "capture_registry_file_sha256",
        "train_plan_identity_sha256",
        "train_plan_file_sha256",
    )
    shas = {
        name: _require_sha256(f"Stage-A campaign {name}", top[name]) for name in sha_fields
    }
    if shas["teacher_sha256"].lower() != TEACHER_SHA256.lower():
        raise RuntimeError(
            "Stage-A campaign does not identify the pinned canonical BF16 teacher"
        )
    expected_gate_sha = validate_pilot_gate_manifest(gate_manifest)
    if shas["gate_manifest_sha256"].lower() != expected_gate_sha.lower():
        raise RuntimeError("Stage-A campaign gate manifest identity does not match supplied policy")

    resumed = _int_list(top["resumed_blocks"], "Stage-A campaign resumed_blocks")
    executed = _int_list(top["executed_blocks"], "Stage-A campaign executed_blocks")
    if set(resumed).intersection(executed):
        raise RuntimeError("Stage-A campaign marks a block as both resumed and executed")
    if set(resumed).union(executed) != set(PILOT_BLOCKS):
        raise RuntimeError(
            f"Stage-A campaign completion accounting must cover exactly {PILOT_BLOCKS}"
        )

    raw_artifacts = _object(top["block_artifacts"], "Stage-A campaign block_artifacts")
    expected_keys = {str(index) for index in PILOT_BLOCKS}
    _exact_keys(raw_artifacts, expected_keys, "Stage-A campaign block_artifacts")
    stored_artifacts = {
        index: _artifact_receipt(raw_artifacts[str(index)], index) for index in PILOT_BLOCKS
    }

    request = StageAArtifactRequest(
        output_dir=str(path.parent),
        run_id=run_id,
        code_commit=code_commit,
        experiment_context_sha256=shas["experiment_context_sha256"],
    )
    completed: dict[int, CompletedStageABlockEvidence] = {}
    for index in PILOT_BLOCKS:
        evidence = load_completed_stage_a_block_evidence(
            request,
            block_index=index,
            final_stage=final_stage,
            expected_dataset_manifest_sha256=shas["dataset_manifest_sha256"],
            expected_gate_manifest_sha256=shas["gate_manifest_sha256"],
            gate_manifest=gate_manifest,
        )
        if evidence is None:
            raise RuntimeError(f"Stage-A campaign is missing completed block evidence for {index}")
        stored = stored_artifacts[index]
        actual = evidence.artifact
        if stored.checkpoint_sha256.lower() != actual.checkpoint_sha256.lower():
            raise RuntimeError(f"Stage-A campaign block {index} checkpoint hash is inconsistent")
        if stored.result_sha256.lower() != actual.result_sha256.lower():
            raise RuntimeError(f"Stage-A campaign block {index} result hash is inconsistent")
        if Path(stored.checkpoint_path).name != Path(actual.checkpoint_path).name:
            raise RuntimeError(f"Stage-A campaign block {index} checkpoint filename is inconsistent")
        if Path(stored.result_path).name != Path(actual.result_path).name:
            raise RuntimeError(f"Stage-A campaign block {index} result filename is inconsistent")
        completed[index] = evidence

    recomputed_gate = evaluate_stage_a_campaign_gate(
        {index: evidence.gate for index, evidence in completed.items()}
    )
    if _json_normalized(top["gate"]) != _json_normalized(asdict(recomputed_gate)):
        raise RuntimeError("Stage-A campaign gate does not recompute from bound block evidence")
    if require_passed and not recomputed_gate.passed:
        raise RuntimeError(
            "Stage-A campaign did not pass all three fixed depth pilots; core50 sweep is forbidden"
        )

    return StageACampaignEvidence(
        path=str(path),
        sha256=sha256_file(path),
        run_id=run_id,
        code_commit=code_commit,
        final_stage=final_stage,
        experiment_context_sha256=shas["experiment_context_sha256"],
        teacher_sha256=shas["teacher_sha256"],
        dataset_manifest_sha256=shas["dataset_manifest_sha256"],
        gate_manifest_sha256=shas["gate_manifest_sha256"],
        block_evidence=completed,
        gate=recomputed_gate,
    )
