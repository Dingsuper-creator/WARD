"""Existing public tools backed by the clinical event lifecycle resolver."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from world_model_patient_ehr_v01.contracts import (
    InteractionContractError,
    canonical_hash,
    credential_findings,
    read_json,
    sha256_file,
)
from world_model_patient_ehr_v04.tool_facade import (
    TOOL_CALL_TRACE_VERSION,
    TOOL_NAMES,
    validate_tool_map as validate_v04_tool_map,
)
from world_model_patient_ehr_v05.tool_facade import (
    StateBasedClinicalToolSandbox,
)

from .environment import (
    ACTION_ENVELOPE_VERSION,
    EVENT_FRAME_VERSION,
    LIFECYCLE_ADAPTER_VERSION,
    LIFECYCLE_POLICY_VERSION,
    LifecyclePatientEHREnvironment,
)


TOOL_MAP_VERSION = "ward-patient-ehr-tool-map-0.5.3"


def _resolve_bound_path(
    value: Any,
    *,
    project_root: Path,
    label: str,
    expected_sha256: Any,
    errors: list[str],
) -> Path:
    if not isinstance(value, str) or not value:
        errors.append(f"{label}_path_missing")
        return Path()
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    if not path.is_file():
        errors.append(f"{label}_path_not_found")
    elif expected_sha256 != sha256_file(path):
        errors.append(f"{label}_sha256_mismatch")
    return path


def load_lifecycle_tool_map(
    *,
    tool_map_path: Path,
    scenario_path: Path,
    freeze_manifest_path: Path,
    interface_path: Path,
    ontology_path: Path,
    graph_path: Path,
) -> tuple[dict[str, Any], str, Path, Path]:
    tool_map = read_json(tool_map_path)
    checkable = copy.deepcopy(tool_map)
    checkable["tool_map_version"] = (
        "ward-patient-ehr-tool-map-0.4.2"
    )
    errors = validate_v04_tool_map(checkable)
    if tool_map.get("tool_map_version") != TOOL_MAP_VERSION:
        errors.append("wrong_tool_map_version")
    if tool_map.get("state_based_adapter_version") != (
        LIFECYCLE_ADAPTER_VERSION
    ):
        errors.append("wrong_lifecycle_adapter_version")
    if tool_map.get("action_envelope_version") != ACTION_ENVELOPE_VERSION:
        errors.append("wrong_action_envelope_version")
    if tool_map.get("event_frame_version") != EVENT_FRAME_VERSION:
        errors.append("wrong_event_frame_version")
    if tool_map.get("lifecycle_policy_version") != (
        LIFECYCLE_POLICY_VERSION
    ):
        errors.append("wrong_lifecycle_policy_version")
    if credential_findings(tool_map):
        errors.append("credential_like_material_present")
    expected = {
        "scenario_sha256": sha256_file(scenario_path),
        "freeze_manifest_sha256": sha256_file(freeze_manifest_path),
        "action_interface_sha256": sha256_file(interface_path),
        "action_ontology_sha256": sha256_file(ontology_path),
        "action_graph_sha256": sha256_file(graph_path),
    }
    for key, value in expected.items():
        if tool_map.get(key) != value:
            errors.append(f"{key}_mismatch")

    project_root = Path(__file__).resolve().parents[2]
    state_routes_path = _resolve_bound_path(
        tool_map.get("state_routes_path"),
        project_root=project_root,
        label="state_routes",
        expected_sha256=tool_map.get("state_routes_sha256"),
        errors=errors,
    )
    lifecycle_policy_path = _resolve_bound_path(
        tool_map.get("lifecycle_policy_path"),
        project_root=project_root,
        label="lifecycle_policy",
        expected_sha256=tool_map.get("lifecycle_policy_sha256"),
        errors=errors,
    )
    if errors:
        raise InteractionContractError(
            "invalid v0.5.3 lifecycle tool map: " + "; ".join(errors)
        )
    return (
        tool_map,
        sha256_file(tool_map_path),
        state_routes_path,
        lifecycle_policy_path,
    )


class LifecycleClinicalToolSandbox(StateBasedClinicalToolSandbox):
    """Unchanged tool schema with generic event lifecycle interpretation."""

    def __init__(
        self,
        *,
        scenario_path: Path,
        freeze_manifest_path: Path,
        interface_path: Path,
        ontology_path: Path,
        graph_path: Path,
        tool_map_path: Path,
        run_id: str,
        agent_id: str,
    ) -> None:
        (
            self._tool_map,
            self._tool_map_sha,
            state_routes_path,
            lifecycle_policy_path,
        ) = load_lifecycle_tool_map(
            tool_map_path=tool_map_path,
            scenario_path=scenario_path,
            freeze_manifest_path=freeze_manifest_path,
            interface_path=interface_path,
            ontology_path=ontology_path,
            graph_path=graph_path,
        )
        self._env = LifecyclePatientEHREnvironment(
            state_routes_path=state_routes_path,
            lifecycle_policy_path=lifecycle_policy_path,
            scenario_path=scenario_path,
            freeze_manifest_path=freeze_manifest_path,
            interface_path=interface_path,
            ontology_path=ontology_path,
            graph_path=graph_path,
            run_id=run_id,
            agent_id=agent_id,
        )
        self._run_id = run_id
        self._agent_id = agent_id
        self._memory = self._empty_memory()
        self._tool_entries: list[dict[str, Any]] = []
        self._previous_tool_entry_hash: str | None = None
        self._evidence_generation = 0

    def step_tool_call(
        self,
        tool_call: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        observation, outcome, done = (
            super(StateBasedClinicalToolSandbox, self).step_tool_call(
                tool_call
            )
        )
        env_entry = self._env._trace_entries[-1]
        route_audit = env_entry["action_assessment"].get(
            "state_based_resolution_audit",
            {},
        )
        event_frame = copy.deepcopy(route_audit.get("event_frame", {}))
        public_audit = {
            "adapter_version": LIFECYCLE_ADAPTER_VERSION,
            "action_envelope_version": ACTION_ENVELOPE_VERSION,
            "event_frame_version": EVENT_FRAME_VERSION,
            "lifecycle_policy_version": LIFECYCLE_POLICY_VERSION,
            "lifecycle_policy_sha256": self._env._lifecycle_policy_sha,
            "routing_method": route_audit.get("routing_method"),
            "candidate_count": route_audit.get("candidate_count", 0),
            "available_candidate_count": route_audit.get(
                "available_candidate_count",
                0,
            ),
            "top_candidate_count": route_audit.get("top_candidate_count", 0),
            "ambiguity_detected": bool(
                route_audit.get("ambiguity_detected", False)
            ),
            "selected": route_audit.get("selected_transition_id") is not None,
            "selected_feature_hits": list(
                route_audit.get("selected_feature_hits", [])
            ),
            "command_features": list(
                route_audit.get("command_features", [])
            ),
            "command_feature_sources": copy.deepcopy(
                route_audit.get("command_feature_sources", {})
            ),
            "background_features": list(
                route_audit.get("background_features", [])
            ),
            "event_frame": event_frame,
            "phase_conflict": bool(
                event_frame.get("phase_conflict", False)
            ),
            "background_fields_used_for_routing": False,
            "background_fields_used_for_event_inference": False,
            "hidden_identifier_required_from_agent": False,
            "hidden_transition_identifier_exposed": False,
            "state_based_outcome_validation": True,
        }
        entry = self._tool_entries[-1]
        entry.pop("tool_entry_sha256", None)
        entry["state_based_resolution_audit"] = public_audit
        entry_hash = canonical_hash(entry)
        entry["tool_entry_sha256"] = entry_hash
        self._previous_tool_entry_hash = entry_hash
        observation[
            "state_based_adapter_version"
        ] = LIFECYCLE_ADAPTER_VERSION
        return observation, outcome, done

    def export_tool_trace(self) -> dict[str, Any]:
        trace = super(StateBasedClinicalToolSandbox, self).export_tool_trace()
        trace["state_based_adapter_version"] = LIFECYCLE_ADAPTER_VERSION
        trace["action_envelope_version"] = ACTION_ENVELOPE_VERSION
        trace["event_frame_version"] = EVENT_FRAME_VERSION
        trace["lifecycle_policy_version"] = LIFECYCLE_POLICY_VERSION
        trace["lifecycle_policy_sha256"] = self._env._lifecycle_policy_sha
        trace["state_based_outcome_validation"] = True
        trace["role_separated_routing"] = True
        trace["clinical_event_lifecycle"] = True
        trace["phase_conflicts_audited"] = True
        trace["background_fields_used_for_routing"] = False
        trace["background_fields_used_for_event_inference"] = False
        trace["hidden_identifier_required_from_agent"] = False
        return trace


__all__ = [
    "LifecycleClinicalToolSandbox",
    "TOOL_CALL_TRACE_VERSION",
    "TOOL_MAP_VERSION",
    "TOOL_NAMES",
    "load_lifecycle_tool_map",
]
