"""Role-separated flexible resolver over the deterministic Patient/EHR state.

Natural-language input remains free form.  Only the command surface (query and
optional structured action slots) can select a recorded transition.  Intent,
rationale, and cited evidence are retained as auditable context but cannot
route or release evidence.
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
from world_model_patient_ehr_v05.environment import (
    CLINICAL_FEATURE_LEXICON,
    StateBasedPatientEHREnvironment,
)


FLEXIBLE_ADAPTER_VERSION = (
    "ward-patient-ehr-flexible-semantic-adapter-0.5.2"
)
ACTION_ENVELOPE_VERSION = (
    "ward-patient-ehr-role-separated-action-envelope-0.1.0"
)
ROUTING_METHOD = "role_separated_command_surface_contract"

COMMAND_FIELDS = (
    "query",
    "resource_type",
    "clinical_phase",
    "anatomy",
    "laterality",
    "procedure_family",
    "result_family",
)
BACKGROUND_FIELDS = ("target", "rationale")

# These phrases describe prior evidence rather than the operation currently
# requested.  Masking is deliberately narrow: "review biopsy pathology" still
# retains biopsy, while "biopsy-proven carcinoma" does not become a biopsy.
BACKGROUND_QUALIFIER_PATTERNS = ('\\bbiopsy[- ]proven\\b',
 '\\bbiopsy[- ]confirmed\\b',
 '\\bconfirmed by (?:a )?biopsy\\b',
 '\\bstatus post\\b',
 '\\bhistory of\\b')

BACKGROUND_QUALIFIER_PATTERNS = vocabulary("world_model_patient_ehr_v052.environment.BACKGROUND_QUALIFIER_PATTERNS", BACKGROUND_QUALIFIER_PATTERNS)

FLEXIBLE_FEATURE_LEXICON = {'admission': ('admission', 'admit', 'inpatient intake'),
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
 'surgery': ('surgery',
             'operation',
             'operative',
             'mastectomy',
             'lumpectomy',
             'resection',
             'definitive surgical',
             'definitive surgery'),
 'breast': ('breast', 'mammary'),
 'conserving': ('breast conserving',
                'conservation',
                'lumpectomy',
                'segmental resection',
                'wide local excision'),
 'sentinel_node': ('sentinel node', 'sentinel lymph'),
 'fascial': ('fascial', 'fascia'),
 'discharge': ('discharge', 'disposition', 'go home', 'home'),
 'mastectomy': ('mastectomy', 'modified radical', 'radical mastectomy'),
 'operative_record': ('operative record',
                      'operation record',
                      'surgical record',
                      'procedure performed',
                      'intraoperative'),
 'surgical_pathology': ('surgical pathology', 'resection pathology', 'final pathology')}

FLEXIBLE_FEATURE_LEXICON = vocabulary("world_model_patient_ehr_v052.environment.FLEXIBLE_FEATURE_LEXICON", FLEXIBLE_FEATURE_LEXICON)


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


def _mask_background_qualifiers(value: str) -> str:
    masked = value
    for pattern in BACKGROUND_QUALIFIER_PATTERNS:
        masked = re.sub(pattern, " prior-confirmed ", masked, flags=re.I)
    return masked


def _features(text: str) -> set[str]:
    normalized = normalize(text)
    return {
        feature
        for feature, terms in FLEXIBLE_FEATURE_LEXICON.items()
        if any(_term_present(normalized, term) for term in terms)
    }


def build_action_envelope(action: dict[str, Any]) -> dict[str, Any]:
    """Build an auditable role-separated envelope from one public action."""

    parameters = action.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}
    sources: dict[str, list[str]] = {}
    command_features: set[str] = set()
    command_values: dict[str, str] = {}
    for field in COMMAND_FIELDS:
        value = str(parameters.get(field, "")).strip()
        if not value:
            continue
        command_values[field] = value
        current = _features(_mask_background_qualifiers(value))
        for feature in current:
            sources.setdefault(feature, []).append(field)
        command_features.update(current)

    # Direct environment calls used by offline replay may not have query.
    # Public tool calls always have query, so target is only a compatibility
    # fallback and never combines with rationale.
    fallback_used = False
    if not command_values.get("query"):
        fallback = str(action.get("target", "")).strip()
        if fallback:
            fallback_used = True
            current = _features(_mask_background_qualifiers(fallback))
            for feature in current:
                sources.setdefault(feature, []).append("target_fallback")
            command_features.update(current)

    background_text = " ".join(
        str(action.get(field, "")) for field in BACKGROUND_FIELDS
    )
    background_features = _features(background_text)
    return {
        "action_envelope_version": ACTION_ENVELOPE_VERSION,
        "operation": str(action.get("action_type", "")),
        "public_tool": str(parameters.get("public_tool", "")).strip(),
        "module": str(parameters.get("module", "")).strip(),
        "command_values": command_values,
        "command_features": sorted(command_features),
        "command_feature_sources": {
            key: sorted(set(value)) for key, value in sorted(sources.items())
        },
        "background_features": sorted(background_features),
        "background_fields_used_for_routing": False,
        "target_fallback_used": fallback_used,
    }


def validate_flexible_state_routes(
    routes: dict[str, Any],
    *,
    scenario_path: Path,
    freeze_manifest_path: Path,
    interface_path: Path,
    ontology_path: Path,
    graph_path: Path,
) -> list[str]:
    """Validate an append-only route file against the shared flexible schema."""

    errors: list[str] = []
    if routes.get("route_contract_version") != (
        "ward-patient-ehr-state-route-contract-0.5.0"
    ):
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
    policy = routes.get("routing_policy", {})
    for key in (
        "query_and_structured_fields_are_command_surface",
        "intent_and_rationale_are_non_routing_context",
        "ambiguous_equal_candidates_do_not_advance",
    ):
        if policy.get(key) is not True:
            errors.append(f"invalid_flexible_policy:{key}")

    raw_routes = routes.get("routes")
    if not isinstance(raw_routes, dict) or not raw_routes:
        return [*errors, "routes_missing"]
    allowed_features = set(FLEXIBLE_FEATURE_LEXICON)
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
        if set(features) - allowed_features:
            errors.append(f"unknown_features:{transition_id}")
        minimum = route.get("minimum_feature_matches")
        if not isinstance(minimum, int) or minimum < 1 or minimum > len(
            features
        ):
            errors.append(f"invalid_minimum_feature_matches:{transition_id}")
        required = route.get("required_features", [])
        forbidden = route.get("forbidden_features", [])
        if not isinstance(required, list) or not set(required).issubset(
            features
        ):
            errors.append(f"invalid_required_features:{transition_id}")
        if not isinstance(forbidden, list) or not set(forbidden).issubset(
            allowed_features
        ):
            errors.append(f"invalid_forbidden_features:{transition_id}")
        if (
            "perform_procedure" in (action_types or [])
            and len(set(required)) < 2
        ):
            errors.append(
                f"procedure_route_needs_two_discriminators:{transition_id}"
            )
    return errors


class FlexiblePatientEHREnvironment(StateBasedPatientEHREnvironment):
    """Flexible language input with deterministic, role-separated execution."""

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
        errors = validate_flexible_state_routes(
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
                "invalid v0.5.2 flexible state routes: "
                + "; ".join(errors)
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
        observation["state_based_adapter_version"] = FLEXIBLE_ADAPTER_VERSION
        observation["decision_space"]["role_separated_routing"] = True
        observation["decision_space"]["background_can_route"] = False
        return observation

    def _route_matches(
        self,
        action: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        envelope = build_action_envelope(action)
        parameters = action.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        module = str(parameters.get("module", "")).strip()
        public_tool = str(parameters.get("public_tool", "")).strip()
        features = set(envelope["command_features"])
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
            specificity = len(hits) * 10 + len(required)
            candidates.append(
                {
                    "transition_id": transition_id,
                    "feature_hits": hits,
                    "feature_match_count": len(hits),
                    "specificity_score": specificity,
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
        if not isinstance(action, dict):
            return PartialOrderPatientEHREnvironment.step(self, action)
        if credential_findings(action):
            return PartialOrderPatientEHREnvironment.step(self, action)
        if (
            action.get("action_type") not in ACTION_TYPES
            or not isinstance(action.get("target"), str)
            or not action["target"].strip()
        ):
            return PartialOrderPatientEHREnvironment.step(self, action)

        original = copy.deepcopy(action)
        route_candidates, envelope = self._route_matches(original)
        available_ids = set(self._available_transition_ids())
        completed_ids = set(
            self._state["completed_transition_ids"] if self._state else []
        )
        available_candidates = [
            item
            for item in route_candidates
            if item["transition_id"] in available_ids
        ]
        top_candidates: list[dict[str, Any]] = []
        if available_candidates:
            top_score = available_candidates[0]["specificity_score"]
            top_candidates = [
                item
                for item in available_candidates
                if item["specificity_score"] == top_score
            ]
        ambiguity_detected = len(top_candidates) > 1
        selected = (
            top_candidates[0]
            if len(top_candidates) == 1
            else None
        )

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
            delegated["action_type"] = "monitor_or_defer"

        observation, base_assessment, done = (
            PartialOrderPatientEHREnvironment.step(self, delegated)
        )
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
        elif ambiguity_detected:
            outcome_status = "valid_exploration_no_recorded_outcome"
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
            list(
                observation["last_action_outcome"][
                    "newly_released_evidence_ids"
                ]
            )
            if state_advanced
            else []
        )
        action_understood = bool(route_candidates or ontology_concepts)
        messages = {
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
        }
        if ambiguity_detected:
            messages["valid_exploration_no_recorded_outcome"] = (
                "The current action has multiple equally supported recorded "
                "interpretations; clarification is required before state "
                "advancement."
            )
        public_outcome = {
            "status": outcome_status,
            "action_understood": action_understood,
            "state_advanced": state_advanced,
            "newly_released_evidence_ids": newly_released,
            "message": messages[outcome_status],
            "clarification_required": ambiguity_detected,
        }
        route_audit = {
            "adapter_version": FLEXIBLE_ADAPTER_VERSION,
            "action_envelope_version": ACTION_ENVELOPE_VERSION,
            "routing_method": ROUTING_METHOD,
            "candidate_count": len(route_candidates),
            "available_candidate_count": len(available_candidates),
            "top_candidate_count": len(top_candidates),
            "ambiguity_detected": ambiguity_detected,
            "selected_transition_id": (
                selected["transition_id"] if selected else None
            ),
            "selected_feature_hits": (
                list(selected["feature_hits"]) if selected else []
            ),
            "command_features": list(envelope["command_features"]),
            "command_feature_sources": copy.deepcopy(
                envelope["command_feature_sources"]
            ),
            "background_features": list(envelope["background_features"]),
            "background_fields_used_for_routing": False,
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
                        "match_mode": ROUTING_METHOD,
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
        observation["state_based_adapter_version"] = FLEXIBLE_ADAPTER_VERSION
        entry = self._trace_entries[-1]
        entry["agent_action"] = original
        entry["action_assessment"] = copy.deepcopy(assessment)
        entry["public_action_outcome"] = copy.deepcopy(public_outcome)
        entry["newly_released_evidence_ids"] = newly_released
        entry["state_based_adapter_version"] = FLEXIBLE_ADAPTER_VERSION
        entry["action_envelope"] = copy.deepcopy(envelope)
        self._rehash_latest_entry()
        return observation, assessment, done

    def export_trace(self) -> dict[str, Any]:
        trace = PartialOrderPatientEHREnvironment.export_trace(self)
        trace["state_based_adapter_version"] = FLEXIBLE_ADAPTER_VERSION
        trace["state_routes_sha256"] = self._state_routes_sha
        trace["state_based_outcome_validation"] = True
        trace["role_separated_routing"] = True
        trace["background_fields_used_for_routing"] = False
        trace["hidden_identifier_required_from_agent"] = False
        return trace


__all__ = [
    "ACTION_ENVELOPE_VERSION",
    "COMMAND_FIELDS",
    "FLEXIBLE_ADAPTER_VERSION",
    "FLEXIBLE_FEATURE_LEXICON",
    "FlexiblePatientEHREnvironment",
    "ROUTING_METHOD",
    "build_action_envelope",
    "validate_flexible_state_routes",
]
