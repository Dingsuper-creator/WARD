"""Reviewed-ontology hybrid routing for frozen Patient/EHR semantics."""

from __future__ import annotations

from runtime_assets import vocabulary

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
from world_model_patient_ehr_v02.ontology import resolve_all_concepts
from world_model_patient_ehr_v03.environment import (
    PartialOrderPatientEHREnvironment,
)
from world_model_patient_ehr_v052.environment import (
    ACTION_ENVELOPE_VERSION,
    FlexiblePatientEHREnvironment,
)
from world_model_patient_ehr_v053.lifecycle import (
    EVENT_FRAME_VERSION,
    LIFECYCLE_POLICY_VERSION,
    load_lifecycle_policy,
)
from world_model_patient_ehr_v055.environment import (
    EVENT_PHASE_FRAME_VERSION,
    EVENT_RELATIVE_PHASE_POLICY_VERSION,
    EventPhasePatientEHREnvironment,
)
from world_model_patient_ehr_v055.event_phase import (
    build_event_phase_action_envelope,
    load_event_phase_policy,
)


HYBRID_ADAPTER_VERSION = (
    "ward-patient-ehr-reviewed-ontology-adapter-0.5.6"
)
HYBRID_ROUTE_CONTRACT_VERSION = (
    "ward-patient-ehr-state-route-contract-0.5.6"
)
HYBRID_ROUTING_METHOD = "reviewed_ontology_with_feature_fallback"

LATERALITY_TERMS = {'left': ('left',), 'right': ('right',), 'bilateral': ('bilateral', 'both sides')}

LATERALITY_TERMS = vocabulary("world_model_patient_ehr_v056.environment.LATERALITY_TERMS", LATERALITY_TERMS)


def _laterality_values(value: str) -> set[str]:
    text = str(value).casefold()
    values: set[str] = set()
    for side, terms in LATERALITY_TERMS.items():
        for term in terms:
            if term.isascii():
                if re.search(rf"\b{re.escape(term)}\b", text):
                    values.add(side)
            elif term in text:
                values.add(side)
    if {"left", "right"}.issubset(values):
        values.add("bilateral")
    return values


def _laterality_compatible(
    expected: str,
    observed: set[str],
) -> bool:
    if not observed:
        return True
    if expected == "bilateral":
        return "bilateral" in observed
    return expected in observed and "bilateral" not in observed


def validate_hybrid_state_routes(
    routes: dict[str, Any],
    *,
    lifecycle_policy_sha256: str,
    event_phase_policy_sha256: str,
    scenario_path: Path,
    freeze_manifest_path: Path,
    interface_path: Path,
    ontology_path: Path,
    graph_path: Path,
) -> list[str]:
    errors: list[str] = []
    if routes.get("route_contract_version") != (
        HYBRID_ROUTE_CONTRACT_VERSION
    ):
        errors.append("wrong_hybrid_route_contract_version")
    if routes.get("artifact_revision") != "0.5.6":
        errors.append("wrong_artifact_revision")
    if credential_findings(routes):
        errors.append("credential_like_material_present")
    expected = {
        "scenario_sha256": sha256_file(scenario_path),
        "freeze_manifest_sha256": sha256_file(freeze_manifest_path),
        "action_interface_sha256": sha256_file(interface_path),
        "action_ontology_sha256": sha256_file(ontology_path),
        "action_graph_sha256": sha256_file(graph_path),
        "lifecycle_policy_sha256": lifecycle_policy_sha256,
        "event_relative_phase_policy_sha256": (
            event_phase_policy_sha256
        ),
    }
    for key, value in expected.items():
        if routes.get(key) != value:
            errors.append(f"{key}_mismatch")
    for key, expected_value in {
        "gold_action_visible": False,
        "dependency_graph_visible": False,
        "future_evidence_visible": False,
        "automatic_semantic_judge_scoring": False,
        "official_scoring_allowed": False,
        "single_total_score_absent": True,
    }.items():
        if routes.get(key) is not expected_value:
            errors.append(f"invalid_boundary:{key}")
    ontology = read_json(ontology_path)
    known_concepts = set(ontology.get("concepts", {}))
    route_values = routes.get("routes")
    if not isinstance(route_values, dict) or not route_values:
        return [*errors, "routes_must_be_nonempty_object"]
    for transition_id, route in route_values.items():
        if not isinstance(route, dict):
            errors.append(f"invalid_route:{transition_id}")
            continue
        action_types = route.get("action_types")
        concept_ids = route.get("concept_ids")
        features = route.get("features")
        minimum = route.get("minimum_feature_matches")
        if not isinstance(action_types, list) or not action_types:
            errors.append(f"invalid_action_types:{transition_id}")
        if (
            not isinstance(concept_ids, list)
            or not concept_ids
            or set(concept_ids) - known_concepts
        ):
            errors.append(f"invalid_concept_ids:{transition_id}")
        if not isinstance(features, list):
            errors.append(f"invalid_features:{transition_id}")
        if (
            not isinstance(minimum, int)
            or minimum < 0
            or minimum > len(features or [])
        ):
            errors.append(
                f"invalid_minimum_feature_matches:{transition_id}"
            )
        if not concept_ids and not features:
            errors.append(f"route_without_match_surface:{transition_id}")
        critical_slots = route.get("critical_slots", {})
        if not isinstance(critical_slots, dict):
            errors.append(f"invalid_critical_slots:{transition_id}")
        else:
            laterality = critical_slots.get("laterality")
            if laterality not in {None, "left", "right", "bilateral"}:
                errors.append(
                    f"invalid_critical_laterality:{transition_id}"
                )
    return errors


class HybridConceptPatientEHREnvironment(EventPhasePatientEHREnvironment):
    """Route command text through reviewed ontology, then safe features."""

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
        lifecycle, lifecycle_sha = load_lifecycle_policy(
            lifecycle_policy_path
        )
        phase, phase_sha = load_event_phase_policy(
            event_phase_policy_path
        )
        PartialOrderPatientEHREnvironment.__init__(
            self,
            scenario_path=scenario_path,
            freeze_manifest_path=freeze_manifest_path,
            interface_path=interface_path,
            ontology_path=ontology_path,
            graph_path=graph_path,
            run_id=run_id,
            agent_id=agent_id,
        )
        routes = read_json(state_routes_path)
        errors = validate_hybrid_state_routes(
            routes,
            lifecycle_policy_sha256=lifecycle_sha,
            event_phase_policy_sha256=phase_sha,
            scenario_path=scenario_path,
            freeze_manifest_path=freeze_manifest_path,
            interface_path=interface_path,
            ontology_path=ontology_path,
            graph_path=graph_path,
        )
        if set(routes.get("routes", {})) != set(self._transitions):
            errors.append("route_transition_set_mismatch")
        if routes.get("case_id") != self.case_id:
            errors.append("route_case_id_mismatch")
        if errors:
            raise InteractionContractError(
                "invalid v0.5.6 hybrid routes: " + "; ".join(errors)
            )
        self._state_routes = routes
        self._state_routes_sha = sha256_file(state_routes_path)
        self._lifecycle_policy = lifecycle
        self._lifecycle_policy_sha = lifecycle_sha
        self._event_phase_policy = phase
        self._event_phase_policy_sha = phase_sha

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
        command_values = envelope.get("command_values", {})
        command_text = " ".join(
            str(command_values.get(key, ""))
            for key in (
                "query",
                "resource_type",
                "clinical_phase",
                "anatomy",
                "laterality",
                "procedure_family",
                "result_family",
            )
            if str(command_values.get(key, "")).strip()
        ).strip()
        if not command_text:
            command_text = str(action.get("target", "")).strip()
        proxy = {
            "action_type": action.get("action_type"),
            "target": command_text,
        }
        ontology_matches = resolve_all_concepts(proxy, self._ontology)
        ontology_by_id = {
            item["concept_id"]: item for item in ontology_matches
        }
        envelope["reviewed_ontology_match_count"] = len(
            ontology_matches
        )
        envelope["reviewed_ontology_match_modes"] = sorted(
            {item["match_mode"] for item in ontology_matches}
        )
        module = str(parameters.get("module", "")).strip()
        public_tool = str(parameters.get("public_tool", "")).strip()
        features = set(envelope["command_features"])
        observed_laterality = _laterality_values(command_text)
        frame = envelope["event_frame"]
        relations = set(frame["generic_temporal_relations"])
        event_features = set(frame["referenced_event_features"])
        mode_score = {
            "controlled_phrase": 4,
            "semantic_anchor": 3,
            "lexical_similarity": 2,
            "action_type_only": 1,
        }
        candidates: list[dict[str, Any]] = []
        critical_slot_rejections = 0
        for transition_id, route in self._state_routes["routes"].items():
            if action.get("action_type") not in route["action_types"]:
                continue
            public_tools = route.get("public_tools", [])
            if public_tool and public_tools and public_tool not in public_tools:
                continue
            modules = route.get("modules", [])
            if module and modules and module not in modules:
                continue
            expected_laterality = route.get("critical_slots", {}).get(
                "laterality"
            )
            if (
                expected_laterality
                and not _laterality_compatible(
                    expected_laterality,
                    observed_laterality,
                )
            ):
                critical_slot_rejections += 1
                continue
            if set(route.get("forbidden_features", [])) & features:
                continue
            concept_hits = [
                ontology_by_id[concept_id]
                for concept_id in route.get("concept_ids", [])
                if concept_id in ontology_by_id
            ]
            route_features = set(route.get("features", []))
            feature_hits = sorted(route_features & features)
            required = set(route.get("required_features", []))
            feature_match = (
                bool(route.get("feature_fallback_allowed", False))
                and required.issubset(features)
                and len(feature_hits)
                >= int(route.get("minimum_feature_matches", 0))
            )
            if not concept_hits and not feature_match:
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
            best_concept_score = max(
                (
                    mode_score.get(item["match_mode"], 0)
                    for item in concept_hits
                ),
                default=0,
            )
            specificity = (
                best_concept_score * 1000
                + len(feature_hits) * 10
                + len(required)
                + len(required_relations)
                + len(required_event_features)
            )
            candidates.append(
                {
                    "transition_id": transition_id,
                    "feature_hits": feature_hits,
                    "feature_match_count": len(feature_hits),
                    "concept_match_count": len(concept_hits),
                    "concept_match_modes": sorted(
                        {item["match_mode"] for item in concept_hits}
                    ),
                    "specificity_score": specificity,
                    "required_features_satisfied": True,
                    "event_constraints_satisfied": True,
                    "action_type_matched": True,
                    "public_tool_matched": True,
                    "module_matched": True,
                    "critical_slots_satisfied": True,
                    "routing_method": HYBRID_ROUTING_METHOD,
                }
            )
        envelope["critical_slot_route_rejection_count"] = (
            critical_slot_rejections
        )
        candidates.sort(
            key=lambda item: (
                item["specificity_score"],
                -self._transitions[item["transition_id"]][
                    "sequence_index"
                ],
            ),
            reverse=True,
        )
        return candidates, envelope

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        observation, assessment, done = (
            FlexiblePatientEHREnvironment.step(self, action)
        )
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
                    "adapter_version": HYBRID_ADAPTER_VERSION,
                    "routing_method": HYBRID_ROUTING_METHOD,
                    "event_frame_version": EVENT_FRAME_VERSION,
                    "event_phase_frame_version": EVENT_PHASE_FRAME_VERSION,
                    "lifecycle_policy_version": LIFECYCLE_POLICY_VERSION,
                    "lifecycle_policy_sha256": self._lifecycle_policy_sha,
                    "event_relative_phase_policy_version": (
                        EVENT_RELATIVE_PHASE_POLICY_VERSION
                    ),
                    "event_relative_phase_policy_sha256": (
                        self._event_phase_policy_sha
                    ),
                    "event_relative_phase_decoupled": True,
                    "post_biopsy_is_postoperative": False,
                    "reviewed_ontology_routing": True,
                    "reviewed_ontology_match_count": entry.get(
                        "action_envelope", {}
                    ).get("reviewed_ontology_match_count", 0),
                    "reviewed_ontology_match_modes": entry.get(
                        "action_envelope", {}
                    ).get("reviewed_ontology_match_modes", []),
                    "critical_slot_route_rejection_count": entry.get(
                        "action_envelope", {}
                    ).get("critical_slot_route_rejection_count", 0),
                    "event_frame": frame,
                }
            )
        assessment["state_based_resolution_audit"] = copy.deepcopy(audit)
        observation["state_based_adapter_version"] = HYBRID_ADAPTER_VERSION
        entry["action_assessment"] = copy.deepcopy(assessment)
        entry["state_based_adapter_version"] = HYBRID_ADAPTER_VERSION
        entry["event_relative_phase_decoupled"] = True
        entry["reviewed_ontology_routing"] = True
        self._rehash_latest_entry()
        return observation, assessment, done

    def export_trace(self) -> dict[str, Any]:
        trace = FlexiblePatientEHREnvironment.export_trace(self)
        trace["state_based_adapter_version"] = HYBRID_ADAPTER_VERSION
        trace["event_frame_version"] = EVENT_FRAME_VERSION
        trace["event_phase_frame_version"] = EVENT_PHASE_FRAME_VERSION
        trace[
            "event_relative_phase_policy_version"
        ] = EVENT_RELATIVE_PHASE_POLICY_VERSION
        trace[
            "event_relative_phase_policy_sha256"
        ] = self._event_phase_policy_sha
        trace["event_relative_phase_decoupled"] = True
        trace["post_biopsy_is_postoperative"] = False
        trace["reviewed_ontology_routing"] = True
        return trace


__all__ = [
    "ACTION_ENVELOPE_VERSION",
    "EVENT_FRAME_VERSION",
    "EVENT_PHASE_FRAME_VERSION",
    "EVENT_RELATIVE_PHASE_POLICY_VERSION",
    "HYBRID_ADAPTER_VERSION",
    "HYBRID_ROUTE_CONTRACT_VERSION",
    "HYBRID_ROUTING_METHOD",
    "HybridConceptPatientEHREnvironment",
    "LIFECYCLE_POLICY_VERSION",
    "validate_hybrid_state_routes",
]
