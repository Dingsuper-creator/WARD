"""Unchanged public tools with event-relative phase separation."""

from __future__ import annotations

from runtime_assets import resolve_asset

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
import world_model_patient_ehr_v053 as v053_package

from .environment import (
    ACTION_ENVELOPE_VERSION,
    EVENT_FRAME_VERSION,
    EVENT_PHASE_ADAPTER_VERSION,
    EVENT_PHASE_FRAME_VERSION,
    EVENT_RELATIVE_PHASE_POLICY_VERSION,
    EventPhasePatientEHREnvironment,
    LIFECYCLE_POLICY_VERSION,
)


TOOL_MAP_VERSION = "ward-patient-ehr-tool-map-0.5.5"


def _resolve(
    tool_map: dict[str, Any],
    *,
    project_root: Path,
    path_key: str,
    hash_key: str,
    errors: list[str],
    fallback_root: Path | None = None,
) -> Path:
    value = tool_map.get(path_key)
    if not isinstance(value, str) or not value:
        errors.append(f"{path_key}_missing")
        return Path()
    path = Path(value)
    if not path.is_absolute():
        path = resolve_asset(value, project_root)
        if not path.is_file() and fallback_root is not None:
            fallback = fallback_root / value
            if fallback.is_file():
                path = fallback
    if not path.is_file():
        errors.append(f"{path_key}_not_found")
    elif tool_map.get(hash_key) != sha256_file(path):
        errors.append(f"{hash_key}_mismatch")
    return path


def load_event_phase_tool_map(
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
        "state_based_adapter_version": EVENT_PHASE_ADAPTER_VERSION,
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
    predecessor_root = Path(v053_package.__file__).resolve().parents[2]
    route_path = _resolve(
        tool_map,
        project_root=project_root,
        path_key="state_routes_path",
        hash_key="state_routes_sha256",
        errors=errors,
    )
    lifecycle_path = _resolve(
        tool_map,
        project_root=project_root,
        path_key="lifecycle_policy_path",
        hash_key="lifecycle_policy_sha256",
        errors=errors,
        fallback_root=predecessor_root,
    )
    phase_path = _resolve(
        tool_map,
        project_root=project_root,
        path_key="event_relative_phase_policy_path",
        hash_key="event_relative_phase_policy_sha256",
        errors=errors,
    )
    if errors:
        raise InteractionContractError(
            "invalid v0.5.5 event-phase tool map: "
            + "; ".join(errors)
        )
    return (
        tool_map,
        sha256_file(tool_map_path),
        route_path,
        lifecycle_path,
        phase_path,
    )


class EventPhaseClinicalToolSandbox(StateBasedClinicalToolSandbox):
    """Existing tool schema backed by corrected phase semantics."""

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
        ) = load_event_phase_tool_map(
            tool_map_path=tool_map_path,
            scenario_path=scenario_path,
            freeze_manifest_path=freeze_manifest_path,
            interface_path=interface_path,
            ontology_path=ontology_path,
            graph_path=graph_path,
        )
        self._env = EventPhasePatientEHREnvironment(
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
        observation, outcome, done = (
            super(StateBasedClinicalToolSandbox, self).step_tool_call(
                tool_call
            )
        )
        env_entry = self._env._trace_entries[-1]
        audit = env_entry["action_assessment"].get(
            "state_based_resolution_audit",
            {},
        )
        frame = copy.deepcopy(audit.get("event_frame", {}))
        public_audit = {
            "adapter_version": EVENT_PHASE_ADAPTER_VERSION,
            "action_envelope_version": ACTION_ENVELOPE_VERSION,
            "event_frame_version": EVENT_FRAME_VERSION,
            "event_phase_frame_version": EVENT_PHASE_FRAME_VERSION,
            "lifecycle_policy_version": LIFECYCLE_POLICY_VERSION,
            "lifecycle_policy_sha256": self._env._lifecycle_policy_sha,
            "event_relative_phase_policy_version": (
                EVENT_RELATIVE_PHASE_POLICY_VERSION
            ),
            "event_relative_phase_policy_sha256": (
                self._env._event_phase_policy_sha
            ),
            "routing_method": audit.get("routing_method"),
            "candidate_count": audit.get("candidate_count", 0),
            "available_candidate_count": audit.get(
                "available_candidate_count",
                0,
            ),
            "top_candidate_count": audit.get("top_candidate_count", 0),
            "ambiguity_detected": bool(
                audit.get("ambiguity_detected", False)
            ),
            "selected": audit.get("selected_transition_id") is not None,
            "selected_feature_hits": list(
                audit.get("selected_feature_hits", [])
            ),
            "command_features": list(audit.get("command_features", [])),
            "command_feature_sources": copy.deepcopy(
                audit.get("command_feature_sources", {})
            ),
            "background_features": list(
                audit.get("background_features", [])
            ),
            "event_frame": frame,
            "phase_conflict": bool(frame.get("phase_conflict", False)),
            "event_relative_phase_decoupled": True,
            "post_biopsy_is_postoperative": False,
            "background_fields_used_for_routing": False,
            "background_fields_used_for_event_inference": False,
            "background_fields_used_for_phase_inference": False,
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
        ] = EVENT_PHASE_ADAPTER_VERSION
        return observation, outcome, done

    def export_tool_trace(self) -> dict[str, Any]:
        trace = super(
            StateBasedClinicalToolSandbox,
            self,
        ).export_tool_trace()
        trace["state_based_adapter_version"] = EVENT_PHASE_ADAPTER_VERSION
        trace["action_envelope_version"] = ACTION_ENVELOPE_VERSION
        trace["event_frame_version"] = EVENT_FRAME_VERSION
        trace["event_phase_frame_version"] = EVENT_PHASE_FRAME_VERSION
        trace["lifecycle_policy_version"] = LIFECYCLE_POLICY_VERSION
        trace["lifecycle_policy_sha256"] = self._env._lifecycle_policy_sha
        trace[
            "event_relative_phase_policy_version"
        ] = EVENT_RELATIVE_PHASE_POLICY_VERSION
        trace[
            "event_relative_phase_policy_sha256"
        ] = self._env._event_phase_policy_sha
        trace["event_relative_phase_decoupled"] = True
        trace["post_biopsy_is_postoperative"] = False
        trace["state_based_outcome_validation"] = True
        trace["role_separated_routing"] = True
        trace["background_fields_used_for_routing"] = False
        trace["hidden_identifier_required_from_agent"] = False
        return trace


__all__ = [
    "EventPhaseClinicalToolSandbox",
    "TOOL_CALL_TRACE_VERSION",
    "TOOL_MAP_VERSION",
    "TOOL_NAMES",
    "load_event_phase_tool_map",
]
