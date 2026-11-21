"""Contracts and validation for deterministic Patient/EHR interaction."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


SCENARIO_VERSION = "ward-patient-ehr-case-0.1.0"
ENVIRONMENT_VERSION = "ward-patient-ehr-environment-0.1.0"
TRACE_VERSION = "ward-patient-ehr-interaction-trace-0.1.0"
RUN_MANIFEST_VERSION = "ward-patient-ehr-run-manifest-0.1.0"

ACTION_TYPES = {
    "request_patient_information",
    "review_document",
    "order_laboratory",
    "order_imaging",
    "review_result",
    "request_consultation",
    "start_treatment",
    "stop_treatment",
    "modify_treatment",
    "perform_procedure",
    "monitor_or_defer",
    "discharge",
    "transfer",
    "escalate_care",
    "end_episode",
}
STAGES = {"S1", "S2", "S3"}
GOLD_SOURCE_TYPES = {
    "patient_question_exception",
    "ehr_evidence_access",
    "ehr_observed_result_implies_order",
    "ehr_observed_action",
    "ehr_derived_terminal_gold",
}
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "client_secret",
    "password",
    "bearer_token",
}
_SENSITIVE_VALUE = re.compile(r"(?:\bBearer\s+\S+|\bsk-[A-Za-z0-9_-]{10,})", re.I)


class InteractionContractError(ValueError):
    """Raised when an interaction scenario or action violates its contract."""


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise InteractionContractError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def walk(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, (*path, str(index)))


def credential_findings(value: Any) -> list[str]:
    findings: list[str] = []
    for path, child in walk(value):
        key = path[-1].lower() if path else ""
        if key in _SENSITIVE_KEYS and child not in (None, "", False):
            findings.append(".".join(path))
        if isinstance(child, str) and _SENSITIVE_VALUE.search(child):
            findings.append(".".join(path))
    return sorted(set(findings))


def _strings(value: Any, label: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise InteractionContractError(f"{label} must be a list of nonempty strings")
    cleaned = [item.strip() for item in value]
    if len(cleaned) != len(set(cleaned)):
        raise InteractionContractError(f"{label} contains duplicates")
    if nonempty and not cleaned:
        raise InteractionContractError(f"{label} must not be empty")
    return cleaned


def validate_scenario(scenario: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if scenario.get("scenario_version") != SCENARIO_VERSION:
        errors.append("wrong_scenario_version")
    case_id = scenario.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        errors.append("missing_case_id")
    if credential_findings(scenario):
        errors.append("credential_like_material_present")
    freeze = scenario.get("scorer_freeze")
    if not isinstance(freeze, dict):
        errors.append("missing_scorer_freeze")
    else:
        manifest_hash = freeze.get("manifest_sha256")
        if not isinstance(manifest_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", manifest_hash
        ):
            errors.append("invalid_freeze_manifest_sha256")
    if scenario.get("official_scoring_allowed") is not False:
        errors.append("official_scoring_boundary_invalid")
    if scenario.get("single_total_score_absent") is not True:
        errors.append("single_total_score_boundary_invalid")

    initial = scenario.get("initial_state")
    if not isinstance(initial, dict):
        errors.append("missing_initial_state")
    else:
        if initial.get("stage") not in STAGES:
            errors.append("invalid_initial_stage")
        try:
            _strings(initial.get("visible_evidence_ids"), "initial visible evidence")
        except InteractionContractError as exc:
            errors.append(str(exc))

    evidence_store = scenario.get("evidence_store")
    if not isinstance(evidence_store, dict):
        errors.append("evidence_store_not_object")
        evidence_store = {}
    for evidence_id, evidence in evidence_store.items():
        if not isinstance(evidence_id, str) or not evidence_id:
            errors.append("invalid_evidence_id")
            continue
        if not isinstance(evidence, dict):
            errors.append(f"invalid_evidence_object:{evidence_id}")
            continue
        if evidence.get("evidence_id") != evidence_id:
            errors.append(f"evidence_id_mismatch:{evidence_id}")
        if not isinstance(evidence.get("content"), str) or not evidence["content"]:
            errors.append(f"empty_evidence_content:{evidence_id}")

    transitions = scenario.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        errors.append("transitions_must_be_nonempty_list")
        transitions = []
    transition_ids: set[str] = set()
    action_ids: set[str] = set()
    released: set[str] = set()
    terminal_count = 0
    for index, transition in enumerate(transitions):
        if not isinstance(transition, dict):
            errors.append(f"invalid_transition:{index}")
            continue
        if transition.get("sequence_index") != index:
            errors.append(f"transition_sequence_mismatch:{index}")
        transition_id = str(transition.get("transition_id", ""))
        if not transition_id or transition_id in transition_ids:
            errors.append(f"invalid_or_duplicate_transition_id:{index}")
        transition_ids.add(transition_id)
        if transition.get("stage_before") not in STAGES:
            errors.append(f"invalid_stage_before:{index}")
        if transition.get("stage_after") not in STAGES:
            errors.append(f"invalid_stage_after:{index}")
        action = transition.get("gold_action")
        if not isinstance(action, dict):
            errors.append(f"missing_gold_action:{index}")
            continue
        action_id = str(action.get("action_id", ""))
        if not action_id or action_id in action_ids:
            errors.append(f"invalid_or_duplicate_action_id:{index}")
        action_ids.add(action_id)
        if action.get("action_type") not in ACTION_TYPES:
            errors.append(f"invalid_action_type:{index}")
        if not isinstance(action.get("target"), str) or not action["target"]:
            errors.append(f"invalid_action_target:{index}")
        if action.get("gold_source_type") not in GOLD_SOURCE_TYPES:
            errors.append(f"invalid_gold_source_type:{index}")
        for alias_key in ("accepted_action_type_aliases", "accepted_target_aliases"):
            try:
                _strings(action.get(alias_key, []), f"{alias_key}:{index}")
            except InteractionContractError as exc:
                errors.append(str(exc))
        try:
            release_ids = _strings(
                transition.get("release_evidence_ids"),
                f"release_evidence_ids:{index}",
                nonempty=True,
            )
        except InteractionContractError as exc:
            errors.append(str(exc))
            release_ids = []
        for evidence_id in release_ids:
            if evidence_id not in evidence_store:
                errors.append(f"unknown_released_evidence:{index}:{evidence_id}")
            if evidence_id in released:
                errors.append(f"evidence_released_twice:{evidence_id}")
            released.add(evidence_id)
        terminal = transition.get("terminal")
        if not isinstance(terminal, bool):
            errors.append(f"terminal_not_boolean:{index}")
        terminal_count += terminal is True
    if transitions and terminal_count != 1:
        errors.append("exactly_one_terminal_transition_required")
    if transitions and transitions[-1].get("terminal") is not True:
        errors.append("last_transition_must_be_terminal")
    return errors


def load_bound_scenario(
    scenario_path: Path, freeze_manifest_path: Path
) -> tuple[dict[str, Any], str, str]:
    scenario = read_json(scenario_path)
    errors = validate_scenario(scenario)
    if errors:
        raise InteractionContractError("invalid scenario: " + "; ".join(errors))
    scenario_sha = sha256_file(scenario_path)
    manifest_sha = sha256_file(freeze_manifest_path)
    expected = scenario["scorer_freeze"]["manifest_sha256"]
    if manifest_sha != expected:
        raise InteractionContractError(
            "scenario is not bound to the supplied scorer freeze manifest"
        )
    manifest = read_json(freeze_manifest_path)
    if manifest.get("freeze_id") != scenario["scorer_freeze"].get("freeze_id"):
        raise InteractionContractError("scenario freeze_id mismatch")
    if manifest.get("official_scoring_allowed") is not False:
        raise InteractionContractError("freeze manifest cannot authorize official scoring")
    return scenario, scenario_sha, manifest_sha


def validate_run_manifest(
    manifest: dict[str, Any],
    *,
    trace_path: Path,
    scenario_path: Path,
    freeze_manifest_path: Path,
) -> list[str]:
    errors: list[str] = []
    if manifest.get("manifest_version") != RUN_MANIFEST_VERSION:
        errors.append("wrong_run_manifest_version")
    if manifest.get("official_scoring_allowed") is not False:
        errors.append("official_scoring_boundary_invalid")
    if manifest.get("single_total_score_absent") is not True:
        errors.append("single_total_score_boundary_invalid")
    if credential_findings(manifest):
        errors.append("credential_like_material_present")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, dict):
        return [*errors, "missing_bindings"]
    expected = {
        "trace_sha256": sha256_file(trace_path),
        "scenario_sha256": sha256_file(scenario_path),
        "scorer_freeze_manifest_sha256": sha256_file(freeze_manifest_path),
    }
    for key, value in expected.items():
        if bindings.get(key) != value:
            errors.append(f"{key}_mismatch")
    trace = read_json(trace_path)
    if trace.get("run_id") != manifest.get("run_id"):
        errors.append("trace_run_id_mismatch")
    if trace.get("case_id") != manifest.get("case_id"):
        errors.append("trace_case_id_mismatch")
    if trace.get("agent_id") != manifest.get("agent", {}).get("agent_id"):
        errors.append("trace_agent_id_mismatch")
    if trace.get("scorer_freeze", {}).get("manifest_sha256") != expected[
        "scorer_freeze_manifest_sha256"
    ]:
        errors.append("trace_freeze_binding_mismatch")
    return errors
