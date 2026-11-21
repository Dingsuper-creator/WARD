"""Clinical tool facade with state-based outcome routing."""

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
    LightweightClinicalToolSandbox,
    validate_tool_map as validate_v04_tool_map,
)

from .environment import (
    STATE_BASED_ADAPTER_VERSION,
    StateBasedPatientEHREnvironment,
)


TOOL_MAP_VERSION = "ward-patient-ehr-tool-map-0.5.0"


def load_state_based_tool_map(
    *,
    tool_map_path: Path,
    scenario_path: Path,
    freeze_manifest_path: Path,
    interface_path: Path,
    ontology_path: Path,
    graph_path: Path,
) -> tuple[dict[str, Any], str, Path]:
    tool_map = read_json(tool_map_path)
    checkable = copy.deepcopy(tool_map)
    checkable["tool_map_version"] = (
        "ward-patient-ehr-tool-map-0.4.2"
    )
    errors = validate_v04_tool_map(checkable)
    if tool_map.get("tool_map_version") != TOOL_MAP_VERSION:
        errors.append("wrong_tool_map_version")
    if tool_map.get("state_based_adapter_version") != (
        STATE_BASED_ADAPTER_VERSION
    ):
        errors.append("wrong_state_based_adapter_version")
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
    route_value = tool_map.get("state_routes_path")
    if not isinstance(route_value, str) or not route_value:
        errors.append("state_routes_path_missing")
        state_routes_path = Path()
    else:
        state_routes_path = Path(route_value)
        if not state_routes_path.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            state_routes_path = project_root / state_routes_path
        if not state_routes_path.is_file():
            errors.append("state_routes_path_not_found")
        elif tool_map.get("state_routes_sha256") != sha256_file(
            state_routes_path
        ):
            errors.append("state_routes_sha256_mismatch")
    if errors:
        raise InteractionContractError(
            "invalid v0.5 tool map: " + "; ".join(errors)
        )
    return tool_map, sha256_file(tool_map_path), state_routes_path


class StateBasedClinicalToolSandbox(LightweightClinicalToolSandbox):
    """v0.4 public tools backed by the v0.5 state-based resolver."""

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
        ) = load_state_based_tool_map(
            tool_map_path=tool_map_path,
            scenario_path=scenario_path,
            freeze_manifest_path=freeze_manifest_path,
            interface_path=interface_path,
            ontology_path=ontology_path,
            graph_path=graph_path,
        )
        self._env = StateBasedPatientEHREnvironment(
            state_routes_path=state_routes_path,
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

    def tool_call_to_action(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        action = super().tool_call_to_action(tool_call)
        action["parameters"]["public_tool"] = tool_call["tool"]
        return action

    def step_tool_call(
        self,
        tool_call: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        observation, outcome, done = super().step_tool_call(tool_call)
        env_entry = self._env._trace_entries[-1]
        route_audit = env_entry["action_assessment"].get(
            "state_based_resolution_audit",
            {},
        )
        public_audit = {
            "adapter_version": STATE_BASED_ADAPTER_VERSION,
            "routing_method": route_audit.get("routing_method"),
            "candidate_count": route_audit.get("candidate_count", 0),
            "available_candidate_count": route_audit.get(
                "available_candidate_count",
                0,
            ),
            "selected": route_audit.get("selected_transition_id") is not None,
            "selected_feature_hits": list(
                route_audit.get("selected_feature_hits", [])
            ),
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
        observation["state_based_adapter_version"] = (
            STATE_BASED_ADAPTER_VERSION
        )
        return observation, outcome, done

    def export_tool_trace(self) -> dict[str, Any]:
        trace = super().export_tool_trace()
        trace["state_based_adapter_version"] = STATE_BASED_ADAPTER_VERSION
        trace["state_based_outcome_validation"] = True
        trace["hidden_identifier_required_from_agent"] = False
        return trace


__all__ = [
    "STATE_BASED_ADAPTER_VERSION",
    "StateBasedClinicalToolSandbox",
    "TOOL_CALL_TRACE_VERSION",
    "TOOL_MAP_VERSION",
    "TOOL_NAMES",
    "load_state_based_tool_map",
]
