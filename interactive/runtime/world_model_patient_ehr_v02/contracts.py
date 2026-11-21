"""Contracts for the reusable Patient/EHR wide-action interface v0.2."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from world_model_patient_ehr_v01.contracts import (
    InteractionContractError,
    credential_findings,
    read_json,
    sha256_file,
)

from .ontology import validate_ontology


ACTION_INTERFACE_VERSION = "ward-patient-ehr-action-interface-0.2.0"
ENVIRONMENT_VERSION = "ward-patient-ehr-environment-0.2.0"
TRACE_VERSION = "ward-patient-ehr-interaction-trace-0.2.0"
RUN_MANIFEST_VERSION = "ward-patient-ehr-run-manifest-0.2.0"


def validate_action_interface(interface: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if interface.get("interface_version") != ACTION_INTERFACE_VERSION:
        errors.append("wrong_action_interface_version")
    if credential_findings(interface):
        errors.append("credential_like_material_present")
    if not isinstance(interface.get("case_id"), str) or not interface["case_id"]:
        errors.append("missing_case_id")
    scenario_sha = interface.get("scenario_sha256")
    if not isinstance(scenario_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", scenario_sha
    ):
        errors.append("invalid_scenario_sha256")
    transitions = interface.get("transition_concepts")
    if not isinstance(transitions, dict) or not transitions:
        errors.append("transition_concepts_must_be_nonempty_object")
    elif any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value
        for key, value in transitions.items()
    ):
        errors.append("invalid_transition_concept_binding")
    capabilities = interface.get("public_capabilities")
    if (
        not isinstance(capabilities, list)
        or not capabilities
        or any(not isinstance(item, str) or not item for item in capabilities)
    ):
        errors.append("invalid_public_capabilities")
    if interface.get("gold_action_visible") is not False:
        errors.append("gold_visibility_boundary_invalid")
    if interface.get("future_evidence_visible") is not False:
        errors.append("future_evidence_boundary_invalid")
    if interface.get("official_scoring_allowed") is not False:
        errors.append("official_scoring_boundary_invalid")
    if interface.get("single_total_score_absent") is not True:
        errors.append("single_total_score_boundary_invalid")
    return errors


def load_bound_action_interface(
    *,
    scenario_path: Path,
    interface_path: Path,
    ontology_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    interface = read_json(interface_path)
    ontology = read_json(ontology_path)
    errors = validate_action_interface(interface)
    errors.extend(validate_ontology(ontology))
    if errors:
        raise InteractionContractError(
            "invalid v0.2 action interface: " + "; ".join(errors)
        )
    scenario_sha = sha256_file(scenario_path)
    if interface["scenario_sha256"] != scenario_sha:
        raise InteractionContractError("action interface scenario binding mismatch")
    ontology_sha = sha256_file(ontology_path)
    if interface.get("ontology_sha256") != ontology_sha:
        raise InteractionContractError("action interface ontology binding mismatch")
    scenario = read_json(scenario_path)
    if interface["case_id"] != scenario.get("case_id"):
        raise InteractionContractError("action interface case_id mismatch")
    transition_ids = {
        transition["transition_id"] for transition in scenario["transitions"]
    }
    if set(interface["transition_concepts"]) != transition_ids:
        raise InteractionContractError(
            "action interface must bind every scenario transition exactly once"
        )
    unknown_concepts = sorted(
        set(interface["transition_concepts"].values()) - set(ontology["concepts"])
    )
    if unknown_concepts:
        raise InteractionContractError("action interface references unknown concepts")
    return interface, ontology, sha256_file(interface_path), ontology_sha

