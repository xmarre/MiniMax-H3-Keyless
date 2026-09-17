from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .checkpoint import sha256_file
from .pilot_campaign import _require_sha256, canonical_json_sha256


WORKFLOW_PROMPT_FIELD = "workflow_prompt_sha256"
WORKFLOW_CONTEXT_KEY = "stage_a_workflow_prompt_sha256"
_CAPTURE_CLASS_TYPE = "MiniMaxH3StageACapture"
_CAPTURE_ONLY_INPUTS = (
    "dataset_manifest_path",
    "case_id",
    "target_sigma",
    "output_subdir",
    "max_capture_mib",
    "sigma_tolerance",
)


def _json_clone(value: Any, *, label: str) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be JSON-serializable") from exc


def canonical_stage_a_workflow_prompt_sha256(
    prompt: Mapping[str, Any],
    *,
    capture_node_id: str | int,
) -> str:
    """Hash the executed API prompt while removing Stage-A capture bookkeeping.

    Comfy injects ``PROMPT`` as the original API prompt sent to the server. That prompt
    contains the generation graph and concrete widget values (prompt text, loader file
    names, seed/sampler settings, geometry, etc.). The Stage-A capture node itself adds
    bookkeeping inputs that do not change H3 generation semantics and vary across capture
    destinations/sigma observations. Those inputs, plus UI-only ``_meta`` dictionaries,
    are normalized before hashing so one fixed case graph has one portable semantic hash.
    """
    if not isinstance(prompt, Mapping) or not prompt:
        raise ValueError("Stage-A workflow binding requires a non-empty Comfy API prompt")
    node_id = str(capture_node_id)
    normalized = _json_clone(prompt, label="Stage-A Comfy API prompt")
    if not isinstance(normalized, dict):
        raise ValueError("Stage-A Comfy API prompt must decode to an object")

    for node in normalized.values():
        if isinstance(node, dict):
            node.pop("_meta", None)

    node = normalized.get(node_id)
    if not isinstance(node, dict):
        raise ValueError(
            f"Stage-A capture node id {node_id!r} is absent from the executed API prompt"
        )
    if node.get("class_type") != _CAPTURE_CLASS_TYPE:
        raise ValueError(
            f"Stage-A workflow node {node_id!r} is not {_CAPTURE_CLASS_TYPE!r}"
        )
    inputs = node.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Stage-A capture node prompt entry is missing its input mapping")
    for name in _CAPTURE_ONLY_INPUTS:
        if name in inputs:
            inputs[name] = f"<stage-a-capture:{name}>"

    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def require_manifest_case_workflow_prompt_sha256(
    case: Mapping[str, Any],
    *,
    case_id: str | None = None,
) -> str:
    label = case_id or str(case.get("case_id", "<unknown>"))
    value = case.get(WORKFLOW_PROMPT_FIELD)
    try:
        return _require_sha256(
            f"Stage-A case {label!r} {WORKFLOW_PROMPT_FIELD}",
            value,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Stage-A case {label!r} must predeclare a 64-hex {WORKFLOW_PROMPT_FIELD}"
        ) from exc


def _collect_prompt_strings(value: Any, out: set[str]) -> None:
    if isinstance(value, str):
        out.add(value)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _collect_prompt_strings(item, out)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _collect_prompt_strings(item, out)


def validate_stage_a_manifest_assets_for_workflow(
    case: Mapping[str, Any],
    prompt: Mapping[str, Any],
    *,
    asset_path_resolver: Callable[[str], str | Path] | None,
) -> None:
    """Bind manifest asset identities to file literals in the executed Comfy prompt.

    Canonical Stage-A evidence is intentionally file-backed. The API prompt must contain
    every declared ``path_or_uri`` literally and the path resolved by Comfy must hash to the
    manifest SHA-256. Remote/dynamic assets without a locally resolvable immutable file are
    therefore rejected for the canonical Stage-A corpus instead of being operator-attested.
    """
    assets = case.get("assets", [])
    if not isinstance(assets, list):
        raise ValueError("Stage-A case assets must be a list")
    if not assets:
        return
    if asset_path_resolver is None:
        raise ValueError(
            "Stage-A cases with assets require a Comfy asset-path resolver for byte verification"
        )

    prompt_strings: set[str] = set()
    _collect_prompt_strings(prompt, prompt_strings)
    case_id = str(case.get("case_id", "<unknown>"))
    for index, asset in enumerate(assets):
        if not isinstance(asset, Mapping):
            raise ValueError(f"Stage-A case {case_id!r} asset {index} must be an object")
        declared = asset.get("path_or_uri")
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError(
                f"Stage-A case {case_id!r} asset {index} has no non-empty path_or_uri"
            )
        if declared not in prompt_strings:
            raise ValueError(
                f"Stage-A case {case_id!r} asset {declared!r} is not referenced literally "
                "by the executed Comfy API prompt"
            )
        expected = _require_sha256(
            f"Stage-A case {case_id!r} asset {declared!r} SHA-256",
            asset.get("sha256"),
        )
        try:
            resolved = Path(asset_path_resolver(declared))
        except Exception as exc:
            raise ValueError(
                f"Stage-A case {case_id!r} asset {declared!r} cannot be resolved by Comfy"
            ) from exc
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Stage-A case {case_id!r} asset does not resolve to a file: {resolved}"
            )
        actual = sha256_file(resolved)
        if actual.lower() != expected.lower():
            raise ValueError(
                f"Stage-A case {case_id!r} asset {declared!r} SHA-256 mismatch: "
                f"expected={expected}, actual={actual}"
            )


def require_record_workflow_prompt_sha256(record: Any) -> str:
    case = getattr(record, "case", None)
    context = getattr(case, "context", None)
    if not isinstance(context, Mapping):
        raise ValueError("Stage-A capture record is missing its immutable context mapping")
    value = context.get(WORKFLOW_CONTEXT_KEY)
    try:
        return _require_sha256("Stage-A capture workflow prompt SHA-256", value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Stage-A capture record is missing a valid {WORKFLOW_CONTEXT_KEY}"
        ) from exc


def _manifest_case_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Stage-A workflow binding requires a validated manifest case list")
    out: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("Stage-A workflow binding encountered a non-object case")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Stage-A workflow binding encountered a case without case_id")
        out[case_id] = case
    return out


def validate_stage_a_record_workflow_bindings(
    records: Sequence[Any],
    manifest: Mapping[str, Any],
    *,
    expected_split: str | None = None,
) -> None:
    if not records:
        raise ValueError("Stage-A workflow binding requires captured records")
    cases = _manifest_case_map(manifest)
    for record in records:
        case = getattr(record, "case", None)
        context = getattr(case, "context", None)
        if not isinstance(context, Mapping):
            raise ValueError("Stage-A capture record is missing immutable context")
        source_case_id = context.get("stage_a_source_case_id")
        if not isinstance(source_case_id, str) or source_case_id not in cases:
            raise ValueError("Stage-A capture record source case is absent from the manifest")
        manifest_case = cases[source_case_id]
        split = manifest_case.get("split")
        captured_split = context.get("stage_a_split")
        if captured_split != split:
            raise ValueError(
                f"Stage-A capture split for {source_case_id!r} differs from its manifest"
            )
        if expected_split is not None and split != expected_split:
            raise ValueError(
                f"Stage-A {expected_split!r} record set contains case {source_case_id!r} "
                f"from split {split!r}"
            )
        expected = require_manifest_case_workflow_prompt_sha256(
            manifest_case,
            case_id=source_case_id,
        )
        actual = require_record_workflow_prompt_sha256(record)
        if actual.lower() != expected.lower():
            raise ValueError(
                f"Stage-A capture workflow identity for {source_case_id!r} differs from "
                f"the fixed manifest: expected={expected}, actual={actual}"
            )


class WorkflowBoundStageACaptureSet:
    """Validate per-case workflow identity whenever Stage-A records are materialized."""

    def __init__(self, capture_set: Any, manifest: Mapping[str, Any]) -> None:
        expected_dataset = canonical_json_sha256(manifest)
        actual_dataset = getattr(capture_set, "dataset_manifest_sha256", None)
        if not isinstance(actual_dataset, str) or actual_dataset.lower() != expected_dataset.lower():
            raise ValueError(
                "Stage-A workflow-bound capture set manifest identity does not match its corpus"
            )
        self._capture_set = capture_set
        self._manifest = _json_clone(manifest, label="Stage-A dataset manifest")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._capture_set, name)

    def records(self, block_index: int, split: str):
        records = self._capture_set.records(block_index, split)
        validate_stage_a_record_workflow_bindings(
            records,
            self._manifest,
            expected_split=split,
        )
        return records

    def cases(self, block_index: int, split: str):
        return tuple(record.case for record in self.records(block_index, split))
