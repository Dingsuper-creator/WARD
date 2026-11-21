"""Deterministic lifecycle routing with event-relative phase separation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from world_model_patient_ehr_v01.contracts import (
    InteractionContractError,
    sha256_file,
)
from world_model_patient_ehr_v052.environment import (
    ACTION_ENVELOPE_VERSION,
    ROUTING_METHOD,
    FlexiblePatientEHREnvironment,
)
from world_model_patient_ehr_v053.environment import (
    LifecyclePatientEHREnvironment,
    validate_lifecycle_state_routes,
)
from world_model_patient_ehr_v053.lifecycle import (
    EVENT_FRAME_VERSION,
    LIFECYCLE_POLICY_VERSION,
    load_lifecycle_policy,
)

from .event_phase import (
    EVENT_PHASE_FRAME_VERSION,
    EVENT_RELATIVE_PHASE_POLICY_VERSION,
    build_event_phase_action_envelope,
    load_event_phase_policy,
)


EVENT_PHASE_ADAPTER_VERSION = (
    "ward-patient-ehr-event-relative-phase-adapter-0.5.5"
)


def validate_event_phase_state_routes(
    routes: dict[str, Any],
    *,
    lifecycle_policy: dict[str, Any],
    lifecycle_policy_sha256: str,
    event_phase_policy_sha256: str,
    scenario_path: Path,
    freeze_manifest_path: Path,
    interface_path: Path,
    ontology_path: Path,
    graph_path: Path,
) -> list[str]:
    compatibility = copy.deepcopy(routes)
    compatibility["artifact_revision"] = "0.5.3"
    errors = validate_lifecycle_state_routes(
        compatibility,
        lifecycle_policy=lifecycle_policy,
        lifecycle_policy_sha256=lifecycle_policy_sha256,
        scenario_path=scenario_path,
        freeze_manifest_path=freeze_manifest_path,
        interface_path=interface_path,
        ontology_path=ontology_path,
        graph_path=graph_path,
    )
    if routes.get("artifact_revision") != "0.5.5":
        errors.append("wrong_artifact_revision")
    if routes.get("event_relative_phase_policy_sha256") != (
        event_phase_policy_sha256
    ):
        errors.append("event_phase_policy_sha256_mismatch")
    routing = routes.get("routing_policy", {})
    for key in (
        "event_relative_time_decoupled_from_surgical_phase",
        "surgery_anchor_required_for_derived_surgical_phase",
    ):
        if routing.get(key) is not True:
            errors.append(f"invalid_event_phase_routing_policy:{key}")
    if routing.get("post_biopsy_is_postoperative") is not False:
        errors.append("post_biopsy_must_not_be_postoperative")
    for transition_id in (
        "ward_007_s2_node_003",
        "ward_007_s2_node_008",
    ):
        route = routes.get("routes", {}).get(transition_id, {})
        if "postoperative" in route.get("required_features", []):
            errors.append(
                f"event_relative_route_requires_surgical_phase:{transition_id}"
            )
        if not route.get("event_constraints"):
            errors.append(f"event_constraints_missing:{transition_id}")
    return errors


class EventPhasePatientEHREnvironment(LifecyclePatientEHREnvironment):
    """v0.5.3 lifecycle state with corrected surgery-phase semantics."""

    def __init__(
        self,
        *,
        state_routes_path: Path,
        lifecycle_policy_path: Path,
        event_phase_policy_path: Path,
        scenario_path: Path,
        freeze_manifest_path: Path,
        interface_path: Path,
        ontology_path: Path,
        graph_path: Path,
        run_id: str,
        agent_id: str,
    ) -> None:
        lifecycle_policy, lifecycle_sha = load_lifecycle_policy(
            lifecycle_policy_path
        )
        event_phase_policy, event_phase_sha = load_event_phase_policy(
            event_phase_policy_path
        )
        FlexiblePatientEHREnvironment.__init__(
            self,
            state_routes_path=state_routes_path,
            scenario_path=scenario_path,
            freeze_manifest_path=freeze_manifest_path,
            interface_path=interface_path,
            ontology_path=ontology_path,
            graph_path=graph_path,
            run_id=run_id,
            agent_id=agent_id,
        )
        errors = validate_event_phase_state_routes(
            self._state_routes,
            lifecycle_policy=lifecycle_policy,
            lifecycle_policy_sha256=lifecycle_sha,
            event_phase_policy_sha256=event_phase_sha,
            scenario_path=scenario_path,
            freeze_manifest_path=freeze_manifest_path,
            interface_path=interface_path,
            ontology_path=ontology_path,
            graph_path=graph_path,
        )
        if set(self._state_routes.get("routes", {})) != set(
            self._transitions
        ):
            errors.append("route_transition_set_mismatch")
        if self._state_routes.get("case_id") != self.case_id:
            errors.append("route_case_id_mismatch")
        if errors:
            raise InteractionContractError(
                "invalid v0.5.5 event-phase routes: "
                + "; ".join(errors)
            )
        self._lifecycle_policy = lifecycle_policy
        self._lifecycle_policy_sha = lifecycle_sha
        self._event_phase_policy = event_phase_policy
        self._event_phase_policy_sha = event_phase_sha

    def _observation(
        self,
        newly_released: list[str],
        *,
        public_outcome: dict[str, Any] | None,
    ) -> dict[str, Any]:
        observation = super()._observation(
            newly_released,
            public_outcome=public_outcome,
        )
        observation[
            "state_based_adapter_version"
        ] = EVENT_PHASE_ADAPTER_VERSION
        observation["decision_space"][
            "event_relative_phase_decoupled"
        ] = True
        return observation

    def _route_matches(
        self,
        action: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        envelope = build_event_phase_action_envelope(
            action,
            self._lifecycle_policy,
            self._event_phase_policy,
        )
        parameters = action.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        module = str(parameters.get("module", "")).strip()
        public_tool = str(parameters.get("public_tool", "")).strip()
        features = set(envelope["command_features"])
        frame = envelope["event_frame"]
        relations = set(frame["generic_temporal_relations"])
        event_features = set(frame["referenced_event_features"])
        candidates: list[dict[str, Any]] = []
        for transition_id, route in self._state_routes["routes"].items():
            if action.get("action_type") not in route["action_types"]:
                continue
            public_tools = route.get("public_tools", [])
            if public_tool and public_tools and public_tool not in public_tools:
                continue
            modules = route.get("modules", [])
            if module and modules and module not in modules:
                continue
            if set(route.get("forbidden_features", [])) & features:
                continue
            route_features = set(route["features"])
            hits = sorted(route_features & features)
            required = set(route.get("required_features", []))
            if not required.issubset(features):
                continue
            if len(hits) < int(route["minimum_feature_matches"]):
                continue
            constraints = route.get("event_constraints", {})
            required_relations = set(
                constraints.get("required_temporal_relations", [])
            )
            required_event_features = set(
                constraints.get(
                    "required_referenced_event_features",
                    [],
                )
            )
            if not required_relations.issubset(relations):
                continue
            if not required_event_features.issubset(event_features):
                continue
            specificity = (
                len(hits) * 10
                + len(required)
                + len(required_relations)
                + len(required_event_features)
            )
            candidates.append(
                {
                    "transition_id": transition_id,
                    "feature_hits": hits,
                    "feature_match_count": len(hits),
                    "specificity_score": specificity,
                    "required_features_satisfied": True,
                    "event_constraints_satisfied": True,
                    "action_type_matched": True,
                    "public_tool_matched": (
                        not public_tool
                        or not public_tools
                        or public_tool in public_tools
                    ),
                    "module_matched": (
                        not module or not modules or module in modules
                    ),
                    "routing_method": ROUTING_METHOD,
                }
            )
        candidates.sort(
            key=lambda item: (
                item["specificity_score"],
                -self._transitions[item["transition_id"]]["sequence_index"],
            ),
            reverse=True,
        )
        return candidates, envelope

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        observation, assessment, done = super().step(action)
        if not self._trace_entries:
            return observation, assessment, done
        entry = self._trace_entries[-1]
        frame = copy.deepcopy(
            entry.get("action_envelope", {}).get("event_frame", {})
        )
        audit = assessment.get("state_based_resolution_audit")
        if isinstance(audit, dict):
            audit.update(
                {
                    "adapter_version": EVENT_PHASE_ADAPTER_VERSION,
                    "event_phase_frame_version": EVENT_PHASE_FRAME_VERSION,
                    "event_relative_phase_policy_version": (
                        EVENT_RELATIVE_PHASE_POLICY_VERSION
                    ),
                    "event_relative_phase_policy_sha256": (
                        self._event_phase_policy_sha
                    ),
                    "event_relative_phase_decoupled": True,
                    "post_biopsy_is_postoperative": False,
                    "event_frame": frame,
                    "phase_conflict": bool(
                        frame.get("phase_conflict", False)
                    ),
                }
            )
        assessment["state_based_resolution_audit"] = copy.deepcopy(audit)
        observation[
            "state_based_adapter_version"
        ] = EVENT_PHASE_ADAPTER_VERSION
        entry["action_assessment"] = copy.deepcopy(assessment)
        entry[
            "state_based_adapter_version"
        ] = EVENT_PHASE_ADAPTER_VERSION
        entry["event_relative_phase_decoupled"] = True
        self._rehash_latest_entry()
        return observation, assessment, done

    def export_trace(self) -> dict[str, Any]:
        trace = super().export_trace()
        trace["state_based_adapter_version"] = EVENT_PHASE_ADAPTER_VERSION
        trace["event_phase_frame_version"] = EVENT_PHASE_FRAME_VERSION
        trace[
            "event_relative_phase_policy_version"
        ] = EVENT_RELATIVE_PHASE_POLICY_VERSION
        trace[
            "event_relative_phase_policy_sha256"
        ] = self._event_phase_policy_sha
        trace["event_relative_phase_decoupled"] = True
        trace["post_biopsy_is_postoperative"] = False
        return trace


__all__ = [
    "ACTION_ENVELOPE_VERSION",
    "EVENT_FRAME_VERSION",
    "EVENT_PHASE_ADAPTER_VERSION",
    "EVENT_PHASE_FRAME_VERSION",
    "EVENT_RELATIVE_PHASE_POLICY_VERSION",
    "EventPhasePatientEHREnvironment",
    "LIFECYCLE_POLICY_VERSION",
    "validate_event_phase_state_routes",
]
