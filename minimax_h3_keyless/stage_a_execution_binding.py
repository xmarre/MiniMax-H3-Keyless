from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .checkpoint import sha256_file
from .pilot_campaign import _require_sha256


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

    Comfy injects ``PROMPT`` as the original API prompt sent to the server.  That prompt
    contains the generation graph and concrete widget values (prompt text, loader file
    names, seed/sampler settings, geometry, etc.).  The Stage-A capture node itself adds
    bookkeeping inputs that do not change H3 generation semantics and vary across capture
    destinations/sigma observations.  Those inputs, plus UI-only ``_meta`` dictionaries,
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

    Canonical Stage-A evidence is intentionally file-backed.  The API prompt must contain
    every declared ``path_or_uri`` literally and the path resolved by Comfy must hash to the
    manifest SHA-256.  Remote/dynamic assets without a locally resolvable immutable file are
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
