"""Frozen public tools with reviewed-ontology batch routing."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import world_model_patient_ehr_v053 as v053_package
import world_model_patient_ehr_v055 as v055_package
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
from world_model_patient_ehr_v055.tool_facade import (
    EventPhaseClinicalToolSandbox,
    _resolve,
)

from .environment import (
    ACTION_ENVELOPE_VERSION,
    EVENT_FRAME_VERSION,
    EVENT_PHASE_FRAME_VERSION,
    EVENT_RELATIVE_PHASE_POLICY_VERSION,
    HYBRID_ADAPTER_VERSION,
    HYBRID_ROUTING_METHOD,
    HybridConceptPatientEHREnvironment,
    LIFECYCLE_POLICY_VERSION,
)


TOOL_MAP_VERSION = "ward-patient-ehr-tool-map-0.5.6"


def load_hybrid_tool_map(
    *,
    tool_map_path: Path,
    scenario_path: Path,
    freeze_manifest_path: Path,
    interface_path: Path,
    ontology_path: Path,
    graph_path: Path,
) -> tuple[dict[str, Any], str, Path, Path, Path]:
    tool_map = read_json(tool_map_path)
    checkable = copy.deepcopy(tool_map)
    checkable["tool_map_version"] = (
        "ward-patient-ehr-tool-map-0.4.2"
    )
    errors = validate_v04_tool_map(checkable)
    expected_versions = {
        "tool_map_version": TOOL_MAP_VERSION,
        "state_based_adapter_version": HYBRID_ADAPTER_VERSION,
        "action_envelope_version": ACTION_ENVELOPE_VERSION,
        "event_frame_version": EVENT_FRAME_VERSION,
        "lifecycle_policy_version": LIFECYCLE_POLICY_VERSION,
        "event_phase_frame_version": EVENT_PHASE_FRAME_VERSION,
        "event_relative_phase_policy_version": (
            EVENT_RELATIVE_PHASE_POLICY_VERSION
        ),
    }
    for key, expected in expected_versions.items():
        if tool_map.get(key) != expected:
            errors.append(f"wrong_version:{key}")
    if credential_findings(tool_map):
        errors.append("credential_like_material_present")
    expected_hashes = {
        "scenario_sha256": sha256_file(scenario_path),
        "freeze_manifest_sha256": sha256_file(freeze_manifest_path),
        "action_interface_sha256": sha256_file(interface_path),
        "action_ontology_sha256": sha256_file(ontology_path),
        "action_graph_sha256": sha256_file(graph_path),
    }
    for key, expected in expected_hashes.items():
        if tool_map.get(key) != expected:
            errors.append(f"{key}_mismatch")
    project_root = Path(__file__).resolve().parents[2]
    v055_root = Path(v055_package.__file__).resolve().parents[2]
    v053_root = Path(v053_package.__file__).resolve().parents[2]
    route_path = _resolve(
        tool_map,
        project_root=project_root,
        path_key="state_routes_path",
        hash_key="state_routes_sha256",
        errors=errors,
        fallback_root=v055_root,
    )
    lifecycle_path = _resolve(
        tool_map,
        project_root=project_root,
        path_key="lifecycle_policy_path",
        hash_key="lifecycle_policy_sha256",
        errors=errors,
        fallback_root=v053_root,
    )
    phase_path = _resolve(
        tool_map,
        project_root=project_root,
        path_key="event_relative_phase_policy_path",
        hash_key="event_relative_phase_policy_sha256",
        errors=errors,
        fallback_root=v055_root,
    )
    if errors:
        raise InteractionContractError(
            "invalid v0.5.6 hybrid tool map: " + "; ".join(errors)
        )
    return (
        tool_map,
        sha256_file(tool_map_path),
        route_path,
        lifecycle_path,
        phase_path,
    )


class HybridClinicalToolSandbox(EventPhaseClinicalToolSandbox):
    """Existing native tool schema backed by hybrid concept routing."""

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
            route_path,
            lifecycle_path,
            phase_path,
        ) = load_hybrid_tool_map(
            tool_map_path=tool_map_path,
            scenario_path=scenario_path,
            freeze_manifest_path=freeze_manifest_path,
            interface_path=interface_path,
            ontology_path=ontology_path,
            graph_path=graph_path,
        )
        self._env = HybridConceptPatientEHREnvironment(
            state_routes_path=route_path,
            lifecycle_policy_path=lifecycle_path,
            event_phase_policy_path=phase_path,
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
        observation, outcome, done = super().step_tool_call(tool_call)
        env_audit = self._env._trace_entries[-1][
            "action_assessment"
        ].get("state_based_resolution_audit", {})
        entry = self._tool_entries[-1]
        entry.pop("tool_entry_sha256", None)
        public_audit = entry.get("state_based_resolution_audit", {})
        public_audit.update(
            {
                "adapter_version": HYBRID_ADAPTER_VERSION,
                "routing_method": HYBRID_ROUTING_METHOD,
                "reviewed_ontology_routing": True,
                "reviewed_ontology_match_count": env_audit.get(
                    "reviewed_ontology_match_count",
                    0,
                ),
                "reviewed_ontology_match_modes": list(
                    env_audit.get(
                        "reviewed_ontology_match_modes",
                        [],
                    )
                ),
            }
        )
        entry["state_based_resolution_audit"] = public_audit
        entry_hash = canonical_hash(entry)
        entry["tool_entry_sha256"] = entry_hash
        self._previous_tool_entry_hash = entry_hash
        observation["state_based_adapter_version"] = HYBRID_ADAPTER_VERSION
        return observation, outcome, done

    def export_tool_trace(self) -> dict[str, Any]:
        trace = super().export_tool_trace()
        trace["state_based_adapter_version"] = HYBRID_ADAPTER_VERSION
        trace["reviewed_ontology_routing"] = True
        trace["routing_method"] = HYBRID_ROUTING_METHOD
        return trace


__all__ = [
    "HybridClinicalToolSandbox",
    "TOOL_CALL_TRACE_VERSION",
    "TOOL_MAP_VERSION",
    "TOOL_NAMES",
    "load_hybrid_tool_map",
]
