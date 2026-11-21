"""State-based resolver layered over the frozen Patient/EHR replay.

The v0.3 environment remains authoritative for state transitions and evidence
release.  This adapter changes only how a public clinical action is routed to a
recorded transition: structured tool/resource fields and bilingual clinical
features are matched against a reviewed route contract.  Hidden transition
identifiers are never required from the Agent.
"""

from __future__ import annotations

from runtime_assets import vocabulary

import copy
import re
from pathlib import Path
from typing import Any

from world_model_patient_ehr_v01.contracts import (
    ACTION_TYPES,
    InteractionContractError,
    canonical_hash,
    credential_findings,
    read_json,
    sha256_file,
)
from world_model_patient_ehr_v02.ontology import normalize, resolve_all_concepts
from world_model_patient_ehr_v03.environment import (
    PartialOrderPatientEHREnvironment,
)


STATE_BASED_ADAPTER_VERSION = (
    "ward-patient-ehr-state-based-adapter-0.5.0"
)
STATE_ROUTE_CONTRACT_VERSION = (
    "ward-patient-ehr-state-route-contract-0.5.0"
)

CLINICAL_FEATURE_LEXICON = {'admission': ('admission', 'admit', 'inpatient intake'),
 'preoperative': ('preoperative', 'pre-op', 'preop', 'before surgery'),
 'postoperative': ('postoperative',
                   'post-op',
                   'postop',
                   'post-procedure',
                   'post procedure',
                   'post-biopsy',
                   'post biopsy',
                   'after surgery'),
 'consent': ('consent', 'informed consent'),
 'risk': ('risk', 'readiness', 'vte'),
 'history': ('history', 'symptom'),
 'physical_exam': ('physical exam', 'examine', 'examination', 'palpate'),
 'biopsy': ('core biopsy', 'needle biopsy', 'biopsy'),
 'pathology': ('pathology', 'histology', 'pathologic'),
 'consultation': ('consultation', 'consult', 'multidisciplinary'),
 'followup': ('follow-up', 'followup', 'follow up'),
 'imaging': ('imaging', 'image'),
 'xray': ('x-ray', 'xray', 'radiograph'),
 'mri': ('mri', 'magnetic resonance'),
 'ct': ('ct', 'computed tomography'),
 'surgery': ('surgery', 'operation', 'operative'),
 'breast': ('breast', 'mammary'),
 'conserving': ('breast conserving', 'conservation', 'lumpectomy'),
 'sentinel_node': ('sentinel node', 'sentinel lymph'),
 'fascial': ('fascial', 'fascia'),
 'discharge': ('discharge', 'disposition', 'go home', 'home')}

CLINICAL_FEATURE_LEXICON = vocabulary("world_model_patient_ehr_v05.environment.CLINICAL_FEATURE_LEXICON", CLINICAL_FEATURE_LEXICON)


def _term_present(text: str, term: str) -> bool:
    normalized_term = normalize(term)
    if not normalized_term:
        return False
    if normalized_term.isascii():
        return bool(
            re.search(
                rf"(?:^|\s){re.escape(normalized_term)}(?:$|\s)",
                text,
            )
        )
    return normalized_term in text


def extract_clinical_features(action: dict[str, Any]) -> set[str]:
    """Extract generic clinical features from public action content only."""

    parameters = action.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}
    text_parts = [
        str(action.get("target", "")),
        str(action.get("rationale", "")),
        str(parameters.get("query", "")),
        str(parameters.get("resource_type", "")),
        str(parameters.get("clinical_phase", "")),
        str(parameters.get("anatomy", "")),
        str(parameters.get("procedure_family", "")),
        str(parameters.get("result_family", "")),
    ]
    text = normalize(" ".join(text_parts))
    return {
        feature
        for feature, terms in CLINICAL_FEATURE_LEXICON.items()
        if any(_term_present(text, term) for term in terms)
    }


def validate_state_routes(
    routes: dict[str, Any],
    *,
    scenario_path: Path,
    freeze_manifest_path: Path,
    interface_path: Path,
    ontology_path: Path,
    graph_path: Path,
) -> list[str]:
    errors: list[str] = []
    if routes.get("route_contract_version") != STATE_ROUTE_CONTRACT_VERSION:
        errors.append("wrong_route_contract_version")
    if credential_findings(routes):
        errors.append("credential_like_material_present")
    expected = {
        "scenario_sha256": sha256_file(scenario_path),
        "freeze_manifest_sha256": sha256_file(freeze_manifest_path),
        "action_interface_sha256": sha256_file(interface_path),
        "action_ontology_sha256": sha256_file(ontology_path),
        "action_graph_sha256": sha256_file(graph_path),
    }
    for key, value in expected.items():
        if routes.get(key) != value:
            errors.append(f"{key}_mismatch")
    for key in (
        "gold_action_visible",
        "dependency_graph_visible",
        "future_evidence_visible",
        "official_scoring_allowed",
        "automatic_semantic_judge_scoring",
    ):
        if routes.get(key) is not False:
            errors.append(f"invalid_boundary:{key}")
    if routes.get("single_total_score_absent") is not True:
        errors.append("single_total_score_boundary_invalid")
    raw_routes = routes.get("routes")
    if not isinstance(raw_routes, dict) or not raw_routes:
        errors.append("routes_missing")
        return errors
    allowed_features = set(CLINICAL_FEATURE_LEXICON)
    for transition_id, route in raw_routes.items():
        if not isinstance(transition_id, str) or not isinstance(route, dict):
            errors.append("invalid_route")
            continue
        action_types = route.get("action_types")
        if not isinstance(action_types, list) or not action_types:
            errors.append(f"invalid_action_types:{transition_id}")
        features = route.get("features")
        if not isinstance(features, list) or not features:
            errors.append(f"invalid_features:{transition_id}")
            continue
        unknown = set(features) - allowed_features
        if unknown:
            errors.append(f"unknown_features:{transition_id}")
        minimum = route.get("minimum_feature_matches")
        if not isinstance(minimum, int) or minimum < 1 or minimum > len(features):
            errors.append(f"invalid_minimum_feature_matches:{transition_id}")
        required = route.get("required_features", [])
        forbidden = route.get("forbidden_features", [])
        if not isinstance(required, list) or not set(required).issubset(features):
            errors.append(f"invalid_required_features:{transition_id}")
        if not isinstance(forbidden, list) or not set(forbidden).issubset(
            allowed_features
        ):
            errors.append(f"invalid_forbidden_features:{transition_id}")
    return errors


class StateBasedPatientEHREnvironment(PartialOrderPatientEHREnvironment):
    """Route clinical tool actions by reviewed state/resource semantics."""

    def __init__(
        self,
        *,
        state_routes_path: Path,
        scenario_path: Path,
        freeze_manifest_path: Path,
        interface_path: Path,
        ontology_path: Path,
        graph_path: Path,
        run_id: str,
        agent_id: str,
    ) -> None:
        super().__init__(
            scenario_path=scenario_path,
            freeze_manifest_path=freeze_manifest_path,
            interface_path=interface_path,
            ontology_path=ontology_path,
            graph_path=graph_path,
            run_id=run_id,
            agent_id=agent_id,
        )
        routes = read_json(state_routes_path)
        errors = validate_state_routes(
            routes,
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
                "invalid v0.5 state routes: " + "; ".join(errors)
            )
        self._state_routes = routes
        self._state_routes_sha = sha256_file(state_routes_path)

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
        observation["state_based_adapter_version"] = (
            STATE_BASED_ADAPTER_VERSION
        )
        observation["decision_space"]["state_based_outcome_validation"] = True
        observation["decision_space"]["hidden_identifier_matching_required"] = (
            False
        )
        return observation

    def _route_matches(self, action: dict[str, Any]) -> list[dict[str, Any]]:
        parameters = action.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        module = str(parameters.get("module", "")).strip()
        public_tool = str(parameters.get("public_tool", "")).strip()
        features = extract_clinical_features(action)
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
            candidates.append(
                {
                    "transition_id": transition_id,
                    "feature_hits": hits,
                    "feature_match_count": len(hits),
                    "required_features_satisfied": True,
                    "action_type_matched": True,
                    "public_tool_matched": (
                        not public_tool
                        or not public_tools
                        or public_tool in public_tools
                    ),
                    "module_matched": (
                        not module or not modules or module in modules
                    ),
                    "routing_method": "state_resource_feature_contract",
                }
            )
        candidates.sort(
            key=lambda item: (
                item["feature_match_count"],
                -self._transitions[item["transition_id"]]["sequence_index"],
            ),
            reverse=True,
        )
        return candidates

    def _rehash_latest_entry(self) -> None:
        entry = self._trace_entries[-1]
        entry.pop("entry_sha256", None)
        entry_hash = canonical_hash(entry)
        entry["entry_sha256"] = entry_hash
        self._previous_entry_hash = entry_hash

    def _routed_action(
        self,
        original: dict[str, Any],
        transition_id: str,
    ) -> dict[str, Any]:
        transition = self._transitions[transition_id]
        concept_id = self._interface["transition_concepts"][transition_id]
        concept = self._ontology["concepts"][concept_id]
        routed = copy.deepcopy(original)
        routed["action_type"] = transition["gold_action"]["action_type"]
        routed["target"] = str(concept["phrases"][0])
        return routed

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        if not isinstance(action, dict):
            return super().step(action)
        if credential_findings(action):
            return super().step(action)
        if (
            action.get("action_type") not in ACTION_TYPES
            or not isinstance(action.get("target"), str)
            or not action["target"].strip()
        ):
            return super().step(action)

        original = copy.deepcopy(action)
        route_candidates = self._route_matches(original)
        available_ids = set(self._available_transition_ids())
        completed_ids = set(
            self._state["completed_transition_ids"] if self._state else []
        )
        available_candidates = [
            item
            for item in route_candidates
            if item["transition_id"] in available_ids
        ]
        selected = available_candidates[0] if available_candidates else None

        evidence_ids = original.get("evidence_ids", [])
        grounding_valid = isinstance(evidence_ids, list) and set(
            item for item in evidence_ids if isinstance(item, str)
        ).issubset(
            set(self._state["visible_evidence_ids"] if self._state else [])
        )

        if selected is not None and grounding_valid:
            delegated = self._routed_action(
                original,
                selected["transition_id"],
            )
        else:
            delegated = copy.deepcopy(original)
            # A reviewed route contract, not a legacy fuzzy phrase match,
            # controls evidence release for routed v0.5 cases.
            delegated["action_type"] = "monitor_or_defer"

        observation, base_assessment, done = super().step(delegated)
        ontology_resolutions = resolve_all_concepts(original, self._ontology)
        ontology_concepts = [
            item["concept_id"] for item in ontology_resolutions
        ]
        recognized_transition_ids = [
            item["transition_id"] for item in route_candidates
        ]

        if not grounding_valid:
            outcome_status = "invalid_evidence_grounding"
        elif selected is not None:
            outcome_status = "ehr_path_advanced"
        elif any(
            item["transition_id"] not in completed_ids
            for item in route_candidates
        ):
            outcome_status = "recognized_prerequisite_pending"
        elif route_candidates:
            outcome_status = "recognized_completed_action"
        else:
            outcome_status = "valid_exploration_no_recorded_outcome"

        state_advanced = outcome_status == "ehr_path_advanced"
        newly_released = (
            list(observation["last_action_outcome"][
                "newly_released_evidence_ids"
            ])
            if state_advanced
            else []
        )
        action_understood = bool(route_candidates or ontology_concepts)
        public_outcome = {
            "status": outcome_status,
            "action_understood": action_understood,
            "state_advanced": state_advanced,
            "newly_released_evidence_ids": newly_released,
            "message": {
                "ehr_path_advanced": (
                    "A recorded EHR action advanced and new evidence is available."
                ),
                "invalid_evidence_grounding": (
                    "The action cited evidence that is not currently visible."
                ),
                "recognized_completed_action": (
                    "That recorded clinical action has already been completed."
                ),
                "recognized_prerequisite_pending": (
                    "The action is understood, but its recorded prerequisites "
                    "are not yet complete."
                ),
                "valid_exploration_no_recorded_outcome": (
                    "This valid exploration has no currently releasable recorded "
                    "EHR outcome."
                ),
            }[outcome_status],
        }
        route_audit = {
            "adapter_version": STATE_BASED_ADAPTER_VERSION,
            "routing_method": "state_resource_feature_contract",
            "candidate_count": len(route_candidates),
            "available_candidate_count": len(available_candidates),
            "selected_transition_id": (
                selected["transition_id"] if selected else None
            ),
            "selected_feature_hits": (
                list(selected["feature_hits"]) if selected else []
            ),
            "hidden_identifier_required_from_agent": False,
            "state_based_outcome_validation": True,
        }
        assessment = copy.deepcopy(base_assessment)
        assessment.update(
            {
                "grounding_valid": grounding_valid,
                "outcome_status": outcome_status,
                "recognized_concept_ids": ontology_concepts,
                "recognized_transition_ids": recognized_transition_ids,
                "matched_transition_id": (
                    selected["transition_id"] if selected else None
                ),
                "matched_resolution": (
                    {
                        "equivalent": True,
                        "match_mode": "state_resource_feature_contract",
                        "feature_hits": list(selected["feature_hits"]),
                    }
                    if selected
                    else None
                ),
                "state_advanced": state_advanced,
                "released_evidence_ids": newly_released,
                "state_based_resolution_audit": route_audit,
                "single_total_score_absent": True,
            }
        )

        observation["last_action_outcome"] = copy.deepcopy(public_outcome)
        entry = self._trace_entries[-1]
        entry["agent_action"] = original
        entry["action_assessment"] = copy.deepcopy(assessment)
        entry["public_action_outcome"] = copy.deepcopy(public_outcome)
        entry["newly_released_evidence_ids"] = newly_released
        entry["state_based_adapter_version"] = STATE_BASED_ADAPTER_VERSION
        self._rehash_latest_entry()
        return observation, assessment, done

    def export_trace(self) -> dict[str, Any]:
        trace = super().export_trace()
        trace["state_based_adapter_version"] = STATE_BASED_ADAPTER_VERSION
        trace["state_routes_sha256"] = self._state_routes_sha
        trace["state_based_outcome_validation"] = True
        trace["hidden_identifier_required_from_agent"] = False
        return trace


__all__ = [
    "CLINICAL_FEATURE_LEXICON",
    "STATE_BASED_ADAPTER_VERSION",
    "STATE_ROUTE_CONTRACT_VERSION",
    "StateBasedPatientEHREnvironment",
    "extract_clinical_features",
    "validate_state_routes",
]
