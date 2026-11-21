"""Deterministic Patient/EHR routing with a generic event lifecycle frame."""

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
    validate_flexible_state_routes,
)

from .lifecycle import (
    EVENT_FRAME_VERSION,
    LIFECYCLE_POLICY_VERSION,
    build_lifecycle_action_envelope,
    load_lifecycle_policy,
)


LIFECYCLE_ADAPTER_VERSION = (
    "ward-patient-ehr-clinical-event-lifecycle-adapter-0.5.3"
)


def validate_lifecycle_state_routes(
    routes: dict[str, Any],
    *,
    lifecycle_policy: dict[str, Any],
    lifecycle_policy_sha256: str,
    scenario_path: Path,
    freeze_manifest_path: Path,
    interface_path: Path,
    ontology_path: Path,
    graph_path: Path,
) -> list[str]:
    """Validate the append-only lifecycle extension to v0.5.2 routes."""

    errors = validate_flexible_state_routes(
        routes,
        scenario_path=scenario_path,
        freeze_manifest_path=freeze_manifest_path,
        interface_path=interface_path,
        ontology_path=ontology_path,
        graph_path=graph_path,
    )
    if routes.get("artifact_revision") != "0.5.3":
        errors.append("wrong_artifact_revision")
    if routes.get("lifecycle_policy_sha256") != lifecycle_policy_sha256:
        errors.append("lifecycle_policy_sha256_mismatch")
    routing_policy = routes.get("routing_policy", {})
    for key in (
        "clinical_event_lifecycle_enabled",
        "phase_conflicts_audited",
    ):
        if routing_policy.get(key) is not True:
            errors.append(f"invalid_lifecycle_routing_policy:{key}")
    if routing_policy.get(
        "phase_conflicts_automatically_override_structured_field"
    ) is not False:
        errors.append(
            "invalid_lifecycle_routing_policy:"
            "phase_conflicts_automatically_override_structured_field"
        )

    allowed_relations = set(lifecycle_policy["temporal_relations"])
    allowed_features = {
        feature
        for features in lifecycle_policy["event_feature_families"].values()
        for feature in features
    }
    for transition_id, route in routes.get("routes", {}).items():
        constraints = route.get("event_constraints")
        if constraints is None:
            continue
        if not isinstance(constraints, dict):
            errors.append(f"invalid_event_constraints:{transition_id}")
            continue
        relations = constraints.get("required_temporal_relations", [])
        features = constraints.get("required_referenced_event_features", [])
        if (
            not isinstance(relations, list)
            or not relations
            or set(relations) - allowed_relations
        ):
            errors.append(
                f"invalid_required_temporal_relations:{transition_id}"
            )
        if (
            not isinstance(features, list)
            or not features
            or set(features) - allowed_features
        ):
            errors.append(
                f"invalid_required_referenced_event_features:{transition_id}"
            )
    return errors


class LifecyclePatientEHREnvironment(FlexiblePatientEHREnvironment):
    """Case-independent event/status/time interpretation over frozen state."""

    def __init__(
        self,
        *,
        state_routes_path: Path,
        lifecycle_policy_path: Path,
        scenario_path: Path,
        freeze_manifest_path: Path,
        interface_path: Path,
        ontology_path: Path,
        graph_path: Path,
        run_id: str,
        agent_id: str,
    ) -> None:
        lifecycle_policy, lifecycle_policy_sha = load_lifecycle_policy(
            lifecycle_policy_path
        )
        super().__init__(
            state_routes_path=state_routes_path,
            scenario_path=scenario_path,
            freeze_manifest_path=freeze_manifest_path,
            interface_path=interface_path,
            ontology_path=ontology_path,
            graph_path=graph_path,
            run_id=run_id,
            agent_id=agent_id,
        )
        errors = validate_lifecycle_state_routes(
            self._state_routes,
            lifecycle_policy=lifecycle_policy,
            lifecycle_policy_sha256=lifecycle_policy_sha,
            scenario_path=scenario_path,
            freeze_manifest_path=freeze_manifest_path,
            interface_path=interface_path,
            ontology_path=ontology_path,
            graph_path=graph_path,
        )
        if errors:
            raise InteractionContractError(
                "invalid v0.5.3 lifecycle state routes: "
                + "; ".join(errors)
            )
        self._lifecycle_policy = lifecycle_policy
        self._lifecycle_policy_sha = lifecycle_policy_sha

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
        ] = LIFECYCLE_ADAPTER_VERSION
        observation["decision_space"]["clinical_event_lifecycle"] = True
        observation["decision_space"]["phase_conflicts_audited"] = True
        return observation

    def _route_matches(
        self,
        action: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        envelope = build_lifecycle_action_envelope(
            action,
            self._lifecycle_policy,
        )
        parameters = action.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        module = str(parameters.get("module", "")).strip()
        public_tool = str(parameters.get("public_tool", "")).strip()
        features = set(envelope["command_features"])
        event_frame = envelope["event_frame"]
        event_relations = set(event_frame["temporal_relations"])
        event_features = set(event_frame["referenced_event_features"])
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
            forbidden = set(route.get("forbidden_features", []))
            if forbidden & features:
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
            if not required_relations.issubset(event_relations):
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
        envelope = entry.get("action_envelope", {})
        event_frame = copy.deepcopy(envelope.get("event_frame", {}))
        audit = assessment.get("state_based_resolution_audit")
        if isinstance(audit, dict):
            audit.update(
                {
                    "adapter_version": LIFECYCLE_ADAPTER_VERSION,
                    "event_frame_version": EVENT_FRAME_VERSION,
                    "lifecycle_policy_version": LIFECYCLE_POLICY_VERSION,
                    "lifecycle_policy_sha256": self._lifecycle_policy_sha,
                    "event_frame": event_frame,
                    "phase_conflict": bool(
                        event_frame.get("phase_conflict", False)
                    ),
                    "background_fields_used_for_event_inference": False,
                }
            )
        observation[
            "state_based_adapter_version"
        ] = LIFECYCLE_ADAPTER_VERSION
        assessment[
            "state_based_resolution_audit"
        ] = copy.deepcopy(audit)
        entry["action_assessment"] = copy.deepcopy(assessment)
        entry[
            "state_based_adapter_version"
        ] = LIFECYCLE_ADAPTER_VERSION
        entry["clinical_event_lifecycle"] = True
        self._rehash_latest_entry()
        return observation, assessment, done

    def export_trace(self) -> dict[str, Any]:
        trace = super().export_trace()
        trace["state_based_adapter_version"] = LIFECYCLE_ADAPTER_VERSION
        trace["event_frame_version"] = EVENT_FRAME_VERSION
        trace["lifecycle_policy_version"] = LIFECYCLE_POLICY_VERSION
        trace["lifecycle_policy_sha256"] = self._lifecycle_policy_sha
        trace["clinical_event_lifecycle"] = True
        trace["phase_conflicts_audited"] = True
        trace[
            "phase_conflicts_automatically_override_structured_field"
        ] = False
        return trace


__all__ = [
    "ACTION_ENVELOPE_VERSION",
    "EVENT_FRAME_VERSION",
    "LIFECYCLE_ADAPTER_VERSION",
    "LIFECYCLE_POLICY_VERSION",
    "LifecyclePatientEHREnvironment",
    "validate_lifecycle_state_routes",
]
