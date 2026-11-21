"""Append-only case registry support for the v0.4 Patient/EHR tool sandbox."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from world_model_patient_ehr_v01.contracts import (
    InteractionContractError,
    credential_findings,
    read_json,
    sha256_file,
)


CASE_REGISTRY_VERSION = "ward-patient-ehr-case-registry-0.4.10"
SUPPORTED_CASE_REGISTRY_VERSIONS = {
    "ward-patient-ehr-case-registry-0.4.1",
    "ward-patient-ehr-case-registry-0.4.2",
    "ward-patient-ehr-case-registry-0.4.3",
    "ward-patient-ehr-case-registry-0.4.4",
    "ward-patient-ehr-case-registry-0.4.5",
    "ward-patient-ehr-case-registry-0.4.6",
    "ward-patient-ehr-case-registry-0.4.7",
    "ward-patient-ehr-case-registry-0.4.8",
    "ward-patient-ehr-case-registry-0.4.9",
    CASE_REGISTRY_VERSION,
}
READY_STATUS = "tool_sandbox_ready"
PATH_FIELDS = {
    "scenario": "scenario_path",
    "freeze_manifest": "freeze_manifest_path",
    "interface": "action_interface_path",
    "ontology": "action_ontology_path",
    "graph": "action_graph_path",
    "tool_map": "tool_map_path",
    "prompt_contract": "prompt_contract_path",
}

CURRENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_REGISTRY = (
    CURRENT_ROOT / "outputs/world_model_patient_ehr_v04/CASE_REGISTRY.v0.4.10.json"
)


def _is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value or "")))


def validate_case_registry(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if registry.get("registry_version") not in SUPPORTED_CASE_REGISTRY_VERSIONS:
        errors.append("wrong_case_registry_version")
    if credential_findings(registry):
        errors.append("credential_like_material_present")
    if registry.get("existing_case_records_immutable") is not True:
        errors.append("existing_case_records_immutable_boundary_invalid")
    if registry.get("new_cases_append_only") is not True:
        errors.append("new_cases_append_only_boundary_invalid")
    if registry.get("environment_code_change_required_for_new_case") is not False:
        errors.append("case_reuse_boundary_invalid")
    if registry.get("official_scoring_allowed") is not False:
        errors.append("official_scoring_boundary_invalid")
    if registry.get("single_total_score_absent") is not True:
        errors.append("single_total_score_boundary_invalid")
    cases = registry.get("cases")
    if not isinstance(cases, list) or not cases:
        return [*errors, "cases_must_be_nonempty_list"]
    seen: set[str] = set()
    for index, item in enumerate(cases):
        if not isinstance(item, dict):
            errors.append(f"invalid_case_entry:{index}")
            continue
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{2,119}", case_id
        ):
            errors.append(f"invalid_case_id:{index}")
            continue
        if case_id in seen:
            errors.append(f"duplicate_case_id:{case_id}")
        seen.add(case_id)
        if item.get("status") != READY_STATUS:
            continue
        artifacts = item.get("artifacts")
        if not isinstance(artifacts, dict):
            errors.append(f"missing_artifacts:{case_id}")
            continue
        for artifact_name, path_field in PATH_FIELDS.items():
            path_value = artifacts.get(path_field)
            sha_value = artifacts.get(f"{artifact_name}_sha256")
            if (
                not isinstance(path_value, str)
                or not path_value
                or Path(path_value).is_absolute()
                or ".." in Path(path_value).parts
            ):
                errors.append(f"invalid_artifact_path:{case_id}:{artifact_name}")
            if not _is_sha256(sha_value):
                errors.append(f"invalid_artifact_sha256:{case_id}:{artifact_name}")
    return sorted(set(errors))


def _workspace_path(relative_path: str, *, workspace_root: Path) -> Path:
    root = workspace_root.resolve()
    candidate = (root / relative_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise InteractionContractError("case registry path escapes workspace root")
    return candidate


def load_case_bundle(
    *,
    case_id: str,
    registry_path: Path = DEFAULT_CASE_REGISTRY,
    workspace_root: Path = CURRENT_ROOT,
) -> dict[str, Any]:
    registry = read_json(registry_path)
    errors = validate_case_registry(registry)
    if errors:
        raise InteractionContractError(
            "invalid v0.4 case registry: " + "; ".join(errors)
        )
    matches = [
        item
        for item in registry["cases"]
        if isinstance(item, dict) and item.get("case_id") == case_id
    ]
    if not matches:
        raise InteractionContractError(f"case_id is not registered: {case_id}")
    entry = matches[0]
    if entry.get("status") != READY_STATUS:
        raise InteractionContractError(
            f"case_id is registered but not tool-sandbox ready: {case_id}"
        )
    artifacts = entry["artifacts"]
    resolved: dict[str, Path] = {}
    for artifact_name, path_field in PATH_FIELDS.items():
        path = _workspace_path(
            artifacts[path_field],
            workspace_root=workspace_root,
        )
        if not path.is_file():
            raise InteractionContractError(
                f"registered artifact is missing: {case_id}:{artifact_name}"
            )
        if sha256_file(path) != artifacts[f"{artifact_name}_sha256"]:
            raise InteractionContractError(
                f"registered artifact hash mismatch: {case_id}:{artifact_name}"
            )
        resolved[artifact_name] = path
    scenario = read_json(resolved["scenario"])
    tool_map = read_json(resolved["tool_map"])
    if scenario.get("case_id") != case_id or tool_map.get("case_id") != case_id:
        raise InteractionContractError("registered case_id binding mismatch")
    return {
        "case_id": case_id,
        "status": READY_STATUS,
        "artifacts": resolved,
        "registry_path": registry_path.resolve(),
        "registry_sha256": sha256_file(registry_path),
        "registry_version": registry["registry_version"],
        "entry": copy.deepcopy(entry),
    }


def list_registered_cases(
    registry_path: Path = DEFAULT_CASE_REGISTRY,
) -> dict[str, Any]:
    registry = read_json(registry_path)
    errors = validate_case_registry(registry)
    if errors:
        raise InteractionContractError(
            "invalid v0.4 case registry: " + "; ".join(errors)
        )
    return {
        "registry_version": registry["registry_version"],
        "registry_sha256": sha256_file(registry_path),
        "cases": [
            {
                "case_id": item["case_id"],
                "status": item["status"],
            }
            for item in registry["cases"]
        ],
        "credentials_echoed": False,
        "official_scoring_allowed": False,
        "single_total_score_absent": True,
    }


__all__ = [
    "CASE_REGISTRY_VERSION",
    "DEFAULT_CASE_REGISTRY",
    "READY_STATUS",
    "list_registered_cases",
    "load_case_bundle",
    "validate_case_registry",
]
