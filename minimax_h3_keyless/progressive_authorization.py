from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .checkpoint import sha256_file
from .immutable_io import write_json_no_replace
from .pilot_campaign import (
    canonical_json_sha256,
    validate_pilot_dataset_manifest,
    validate_pilot_gate_manifest,
)
from .pilot_inputs import CANONICAL_STAGE_A_COVERAGE_TAGS
from .progressive import ProgressiveAcceptedBlock, ProgressivePrefix
from .progressive_gates import progressive_execution_policy_from_gate_manifest
from .stage_a_campaign_result import StageACampaignEvidence, load_stage_a_campaign_evidence


PROGRESSIVE_AUTHORIZATION_SCHEMA = "minimax_h3_keyless_progressive_authorization_v1"
PROGRESSIVE_PREFIX_MANIFEST_SCHEMA = "minimax_h3_keyless_progressive_prefix_manifest_v1"
_FULL_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class ProgressiveAuthorization:
    stage_a: StageACampaignEvidence
    prefix: ProgressivePrefix
    dataset_manifest_sha256: str
    gate_manifest_sha256: str


def authorize_progressive_sweep(
    stage_a_result_path: str | Path,
    *,
    dataset_manifest: Mapping[str, Any],
    gate_manifest: Mapping[str, object],
    sweep_id: str,
    code_commit: str,
) -> ProgressiveAuthorization:
    """Create the empty Stage-B prefix only from a passed, hash-bound Stage-A campaign.

    This is the production authorization boundary between the 0/25/49 pilot and the
    early-to-late core50 sweep. A caller cannot authorize Stage B by constructing a
    ``ProgressivePrefix`` around an unvalidated campaign JSON: the campaign's three block
    artifacts and recomputed gate are validated first, then the supplied fixed dataset and
    gate policy are required to match the identities embedded in that evidence. The gate
    policy must already contain every Stage-B execution tolerance needed by the sweep; an
    authorization that could only fail later at the first block is not valid.
    """

    if not isinstance(sweep_id, str) or not sweep_id.strip():
        raise ValueError("progressive sweep_id must be non-empty")
    if not isinstance(code_commit, str) or not _FULL_COMMIT.fullmatch(code_commit):
        raise ValueError("progressive code_commit must be a full 40-hex git revision")

    dataset_sha = validate_pilot_dataset_manifest(
        dataset_manifest,
        required_coverage_tags=CANONICAL_STAGE_A_COVERAGE_TAGS,
    )
    gate_sha = validate_pilot_gate_manifest(gate_manifest)
    # Authorization is the earliest Stage-B boundary. Require the full immutable Stage-B
    # execution policy here rather than creating a prefix that is guaranteed to fail only
    # after expensive capture/training work begins.
    progressive_execution_policy_from_gate_manifest(gate_manifest)
    stage_a = load_stage_a_campaign_evidence(
        stage_a_result_path,
        gate_manifest=gate_manifest,
        require_passed=True,
    )
    if stage_a.dataset_manifest_sha256.lower() != dataset_sha.lower():
        raise RuntimeError(
            "passed Stage-A campaign dataset identity does not match the supplied fixed dataset"
        )
    if stage_a.gate_manifest_sha256.lower() != gate_sha.lower():
        raise RuntimeError(
            "passed Stage-A campaign gate identity does not match the supplied fixed policy"
        )
    if not stage_a.gate.passed:
        # ``require_passed=True`` already enforces this. Keep the local invariant explicit
        # so alternate evidence providers cannot weaken authorization accidentally.
        raise RuntimeError("progressive sweep authorization requires a passed Stage-A campaign")

    prefix = ProgressivePrefix(
        sweep_id=sweep_id,
        code_commit=code_commit.lower(),
        stage_a_campaign_sha256=stage_a.sha256.lower(),
        dataset_manifest_sha256=dataset_sha.lower(),
        gate_manifest_sha256=gate_sha.lower(),
    )
    return ProgressiveAuthorization(
        stage_a=stage_a,
        prefix=prefix,
        dataset_manifest_sha256=dataset_sha.lower(),
        gate_manifest_sha256=gate_sha.lower(),
    )


def _accepted_from_json(value: Any) -> tuple[ProgressiveAcceptedBlock, ...]:
    if not isinstance(value, list):
        raise RuntimeError("progressive prefix manifest accepted must be a list")
    rows: list[ProgressiveAcceptedBlock] = []
    expected_keys = {"block_index", "final_stage", "checkpoint_sha256", "result_sha256"}
    for index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise RuntimeError(
                f"progressive prefix manifest accepted row {index} has an invalid schema"
            )
        rows.append(
            ProgressiveAcceptedBlock(
                block_index=row["block_index"],
                final_stage=row["final_stage"],
                checkpoint_sha256=row["checkpoint_sha256"],
                result_sha256=row["result_sha256"],
            )
        )
    return tuple(rows)


def progressive_prefix_manifest_payload(
    prefix: ProgressivePrefix,
    *,
    previous_manifest_sha256: str | None,
) -> dict[str, Any]:
    if previous_manifest_sha256 is not None:
        from .pilot_campaign import _require_sha256

        previous_manifest_sha256 = _require_sha256(
            "previous progressive prefix manifest SHA-256",
            previous_manifest_sha256,
        )
    return {
        "schema": PROGRESSIVE_PREFIX_MANIFEST_SCHEMA,
        "prefix_identity_sha256": prefix.identity_sha256,
        "previous_manifest_sha256": previous_manifest_sha256,
        "prefix": prefix.identity_payload(),
    }


def write_progressive_prefix_manifest(
    path: str | Path,
    prefix: ProgressivePrefix,
    *,
    previous_manifest_sha256: str | None,
) -> str:
    """Atomically publish one immutable accepted-prefix snapshot."""
    return write_json_no_replace(
        path,
        progressive_prefix_manifest_payload(
            prefix,
            previous_manifest_sha256=previous_manifest_sha256,
        ),
    )


def load_progressive_prefix_manifest(
    path: str | Path,
    *,
    expected_previous_manifest_sha256: str | None = None,
) -> tuple[ProgressivePrefix, str]:
    """Load an immutable prefix snapshot and recompute its hash-chained identity."""
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid progressive prefix manifest: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("progressive prefix manifest must decode to an object")
    expected_top = {
        "schema",
        "prefix_identity_sha256",
        "previous_manifest_sha256",
        "prefix",
    }
    if set(value) != expected_top or value.get("schema") != PROGRESSIVE_PREFIX_MANIFEST_SCHEMA:
        raise RuntimeError("progressive prefix manifest has an incompatible schema")

    previous = value.get("previous_manifest_sha256")
    if previous is not None:
        from .pilot_campaign import _require_sha256

        previous = _require_sha256("progressive previous manifest SHA-256", previous)
    if expected_previous_manifest_sha256 is not None:
        from .pilot_campaign import _require_sha256

        expected_previous_manifest_sha256 = _require_sha256(
            "expected previous progressive manifest SHA-256",
            expected_previous_manifest_sha256,
        )
        if previous != expected_previous_manifest_sha256:
            raise RuntimeError("progressive prefix manifest does not extend the expected prior manifest")

    raw_prefix = value.get("prefix")
    if not isinstance(raw_prefix, dict):
        raise RuntimeError("progressive prefix manifest is missing prefix payload")
    expected_prefix_keys = {
        "api",
        "sweep_id",
        "code_commit",
        "stage_a_campaign_sha256",
        "dataset_manifest_sha256",
        "gate_manifest_sha256",
        "accepted",
    }
    if set(raw_prefix) != expected_prefix_keys or raw_prefix.get("api") != 1:
        raise RuntimeError("progressive prefix payload has an incompatible schema")
    prefix = ProgressivePrefix(
        sweep_id=raw_prefix["sweep_id"],
        code_commit=raw_prefix["code_commit"],
        stage_a_campaign_sha256=raw_prefix["stage_a_campaign_sha256"],
        dataset_manifest_sha256=raw_prefix["dataset_manifest_sha256"],
        gate_manifest_sha256=raw_prefix["gate_manifest_sha256"],
        accepted=_accepted_from_json(raw_prefix["accepted"]),
    )
    claimed_identity = value.get("prefix_identity_sha256")
    if claimed_identity != prefix.identity_sha256:
        raise RuntimeError("progressive prefix manifest identity does not recompute")
    if raw_prefix != prefix.identity_payload():
        raise RuntimeError("progressive prefix manifest payload is not canonical")
    return prefix, sha256_file(path)


def progressive_authorization_receipt(
    authorization: ProgressiveAuthorization,
    *,
    stage_a_result_path: str | Path,
    prefix_manifest_path: str | Path,
    prefix_manifest_sha256: str,
) -> dict[str, Any]:
    """Build a standalone audit record for the Phase-3 -> Stage-B authorization boundary."""
    return {
        "schema": PROGRESSIVE_AUTHORIZATION_SCHEMA,
        "stage_a_result_path": str(stage_a_result_path),
        "stage_a_result_sha256": authorization.stage_a.sha256,
        "stage_a_gate_passed": bool(authorization.stage_a.gate.passed),
        "dataset_manifest_sha256": authorization.dataset_manifest_sha256,
        "gate_manifest_sha256": authorization.gate_manifest_sha256,
        "prefix_identity_sha256": authorization.prefix.identity_sha256,
        "prefix_manifest_path": str(prefix_manifest_path),
        "prefix_manifest_sha256": prefix_manifest_sha256,
        "authorization_identity_sha256": canonical_json_sha256(
            {
                "stage_a_result_sha256": authorization.stage_a.sha256,
                "dataset_manifest_sha256": authorization.dataset_manifest_sha256,
                "gate_manifest_sha256": authorization.gate_manifest_sha256,
                "prefix_identity_sha256": authorization.prefix.identity_sha256,
                "prefix_manifest_sha256": prefix_manifest_sha256,
            }
        ),
    }
