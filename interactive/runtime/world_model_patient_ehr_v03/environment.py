"""Partial-order deterministic Patient/EHR environment v0.3."""

from __future__ import annotations

import copy
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from world_model_patient_ehr_v01.contracts import (
    ACTION_TYPES,
    InteractionContractError,
    canonical_hash,
    credential_findings,
    load_bound_scenario,
)
from world_model_patient_ehr_v02.contracts import load_bound_action_interface
from world_model_patient_ehr_v02.ontology import (
    RESOLVER_VERSION,
    resolve_against_concept,
    resolve_all_concepts,
)

from .contracts import (
    ENVIRONMENT_VERSION,
    TRACE_VERSION,
    load_bound_action_graph,
)


STAGE_RANK = {"S1": 1, "S2": 2, "S3": 3}
MATCH_RANK = {
    "controlled_phrase": 3,
    "semantic_anchor": 2,
    "lexical_similarity": 1,
    "action_type_only": 0,
    "unresolved": -1,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PartialOrderPatientEHREnvironment:
    """EHR replay in which eligible recorded actions form a dependency graph."""

    def __init__(
        self,
        *,
        scenario_path: Path,
        freeze_manifest_path: Path,
        interface_path: Path,
        ontology_path: Path,
        graph_path: Path,
        run_id: str,
        agent_id: str,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,119}", run_id):
            raise InteractionContractError("invalid run_id")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{1,119}", agent_id):
            raise InteractionContractError("invalid agent_id")
        scenario, scenario_sha, freeze_sha = load_bound_scenario(
            scenario_path, freeze_manifest_path
        )
        interface, ontology, interface_sha, ontology_sha = (
            load_bound_action_interface(
                scenario_path=scenario_path,
                interface_path=interface_path,
                ontology_path=ontology_path,
            )
        )
        graph, graph_sha = load_bound_action_graph(
            graph_path=graph_path,
            scenario_path=scenario_path,
            interface_path=interface_path,
            ontology_path=ontology_path,
        )
        self._scenario = scenario
        self._scenario_sha = scenario_sha
        self._freeze_sha = freeze_sha
        self._interface = interface
        self._interface_sha = interface_sha
        self._ontology = ontology
        self._ontology_sha = ontology_sha
        self._graph = graph
        self._graph_sha = graph_sha
        self._transitions = {
            transition["transition_id"]: transition
            for transition in scenario["transitions"]
        }
        self._run_id = run_id
        self._agent_id = agent_id
        self._state: dict[str, Any] | None = None
        self._trace_entries: list[dict[str, Any]] = []
        self._previous_entry_hash: str | None = None
        self._created_at: str | None = None

    @property
    def case_id(self) -> str:
        return str(self._scenario["case_id"])

    def _state_hash(self) -> str:
        if self._state is None:
            raise InteractionContractError("environment has not been reset")
        return canonical_hash(self._state)

    def _available_transition_ids(self) -> list[str]:
        if self._state is None:
            raise InteractionContractError("environment has not been reset")
        completed = set(self._state["completed_transition_ids"])
        return [
            transition["transition_id"]
            for transition in self._scenario["transitions"]
            if transition["transition_id"] not in completed
            and set(
                self._graph["nodes"][transition["transition_id"]]["requires"]
            ).issubset(completed)
        ]

    def _observation(
        self,
        newly_released: list[str],
        *,
        public_outcome: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if self._state is None:
            raise InteractionContractError("environment has not been reset")
        store = self._scenario["evidence_store"]
        return {
            "environment_version": ENVIRONMENT_VERSION,
            "case_id": self.case_id,
            "run_id": self._run_id,
            "stage": self._state["stage"],
            "step_index": len(self._state["completed_transition_ids"]),
            "completed_recorded_action_count": len(
                self._state["completed_transition_ids"]
            ),
            "available_recorded_action_count": len(
                self._available_transition_ids()
            ),
            "environment_clock": self._state["clock"],
            "public_context": copy.deepcopy(self._scenario["public_context"]),
            "visible_evidence_ids": list(self._state["visible_evidence_ids"]),
            "new_evidence": [copy.deepcopy(store[item]) for item in newly_released],
            "last_action_outcome": copy.deepcopy(public_outcome),
            "state_fingerprint": self._state_hash(),
            "done": self._state["done"],
            "action_schema": {
                "required": ["action_type", "target"],
                "optional": ["parameters", "rationale", "evidence_ids"],
                "action_type_values": sorted(ACTION_TYPES),
                "target_contract": "free_text_clinical_intent",
                "exact_hidden_identifier_required": False,
            },
            "decision_space": {
                "public_capabilities": list(
                    self._interface["public_capabilities"]
                ),
                "free_text_targets_allowed": True,
                "multiple_recorded_actions_may_be_available": True,
                "prerequisites_may_apply": True,
                "schema_valid_exploration_allowed": True,
                "counterfactual_outcomes_limited_to_recorded_ehr": True,
            },
            "gold_action_hidden": True,
            "gold_sequence_hidden": True,
            "future_evidence_hidden": True,
            "single_total_score_absent": True,
        }

    def reset(self) -> dict[str, Any]:
        initial = self._scenario["initial_state"]
        self._created_at = now_iso()
        self._trace_entries = []
        self._previous_entry_hash = None
        self._state = {
            "case_id": self.case_id,
            "stage": initial["stage"],
            "clock": initial.get("clock"),
            "visible_evidence_ids": list(initial["visible_evidence_ids"]),
            "completed_transition_ids": [],
            "state_facts": copy.deepcopy(initial.get("state_facts", {})),
            "done": False,
        }
        return self._observation(
            list(initial["visible_evidence_ids"]),
            public_outcome=None,
        )

    def _append_entry(self, entry: dict[str, Any]) -> None:
        entry["previous_entry_sha256"] = self._previous_entry_hash
        entry_hash = canonical_hash(entry)
        entry["entry_sha256"] = entry_hash
        self._trace_entries.append(entry)
        self._previous_entry_hash = entry_hash

    def _best_available_match(
        self,
        action: dict[str, Any],
        available_ids: list[str],
    ) -> tuple[str, dict[str, Any]] | None:
        candidates: list[tuple[str, dict[str, Any]]] = []
        for transition_id in available_ids:
            concept_id = self._interface["transition_concepts"][transition_id]
            resolution = resolve_against_concept(
                action, concept_id, self._ontology
            )
            if resolution["equivalent"]:
                candidates.append((transition_id, resolution))
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                MATCH_RANK[item[1]["match_mode"]],
                len(item[1]["anchor_group_hits"]),
                item[1]["similarity"],
                -self._transitions[item[0]]["sequence_index"],
            ),
            reverse=True,
        )
        return candidates[0]

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        if self._state is None:
            raise InteractionContractError("call reset before step")
        if self._state["done"]:
            raise InteractionContractError("episode is already complete")
        if not isinstance(action, dict):
            raise InteractionContractError("action must be an object")
        if credential_findings(action):
            raise InteractionContractError("credential-like material is forbidden")
        if action.get("action_type") not in ACTION_TYPES:
            raise InteractionContractError("action_type is not allowed")
        if not isinstance(action.get("target"), str) or not action["target"].strip():
            raise InteractionContractError("target is required")
        evidence_ids = action.get("evidence_ids", [])
        if (
            not isinstance(evidence_ids, list)
            or any(not isinstance(item, str) or not item for item in evidence_ids)
            or len(evidence_ids) != len(set(evidence_ids))
        ):
            raise InteractionContractError(
                "evidence_ids must be a unique list of strings"
            )

        completed_before = set(self._state["completed_transition_ids"])
        available_before = self._available_transition_ids()
        grounding_valid = set(evidence_ids).issubset(
            set(self._state["visible_evidence_ids"])
        )
        best_match = self._best_available_match(action, available_before)
        all_resolutions = resolve_all_concepts(action, self._ontology)
        recognized_concept_ids = [
            item["concept_id"] for item in all_resolutions
        ]
        concept_to_transitions: dict[str, list[str]] = {}
        for transition_id, concept_id in self._interface[
            "transition_concepts"
        ].items():
            concept_to_transitions.setdefault(concept_id, []).append(transition_id)
        recognized_transition_ids = [
            transition_id
            for concept_id in recognized_concept_ids
            for transition_id in concept_to_transitions.get(concept_id, [])
        ]
        matched_transition_id: str | None = None
        matched_resolution: dict[str, Any] | None = None
        state_advanced = False
        newly_released: list[str] = []
        state_before_sha = self._state_hash()

        if best_match is not None and grounding_valid:
            matched_transition_id, matched_resolution = best_match
            transition = self._transitions[matched_transition_id]
            newly_released = list(transition["release_evidence_ids"])
            self._state["visible_evidence_ids"].extend(newly_released)
            self._state["visible_evidence_ids"] = sorted(
                set(self._state["visible_evidence_ids"])
            )
            self._state["completed_transition_ids"].append(
                matched_transition_id
            )
            self._state["completed_transition_ids"].sort(
                key=lambda item: self._transitions[item]["sequence_index"]
            )
            if STAGE_RANK[transition["stage_after"]] > STAGE_RANK[
                self._state["stage"]
            ]:
                self._state["stage"] = transition["stage_after"]
            next_clock = transition.get("clock_after")
            if next_clock is not None and (
                self._state["clock"] is None or next_clock > self._state["clock"]
            ):
                self._state["clock"] = next_clock
            self._state["state_facts"].update(
                copy.deepcopy(transition.get("state_updates", {}))
            )
            self._state["done"] = bool(transition["terminal"])
            state_advanced = True
            outcome_status = "ehr_path_advanced"
        elif not grounding_valid:
            outcome_status = "invalid_evidence_grounding"
        elif any(item in completed_before for item in recognized_transition_ids):
            outcome_status = "recognized_completed_action"
        elif recognized_transition_ids:
            outcome_status = "recognized_prerequisite_pending"
        else:
            outcome_status = "valid_exploration_no_recorded_outcome"

        state_after_sha = self._state_hash()
        public_outcome = {
            "status": outcome_status,
            "action_understood": bool(recognized_concept_ids),
            "state_advanced": state_advanced,
            "newly_released_evidence_ids": list(newly_released),
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
        assessment = {
            "interface_valid": True,
            "grounding_valid": grounding_valid,
            "outcome_status": outcome_status,
            "recognized_concept_ids": recognized_concept_ids,
            "recognized_transition_ids": recognized_transition_ids,
            "matched_transition_id": matched_transition_id,
            "matched_resolution": matched_resolution,
            "available_transition_ids_before": available_before,
            "state_advanced": state_advanced,
            "released_evidence_ids": list(newly_released),
            "single_total_score_absent": True,
        }
        entry = {
            "trace_version": TRACE_VERSION,
            "run_id": self._run_id,
            "case_id": self.case_id,
            "attempt_index": len(self._trace_entries),
            "completed_transition_count_before": len(completed_before),
            "available_transition_count_before": len(available_before),
            "stage_before": (
                self._transitions[matched_transition_id]["stage_before"]
                if matched_transition_id
                else self._state["stage"]
            ),
            "stage_after": self._state["stage"],
            "environment_clock_after": self._state["clock"],
            "state_before_sha256": state_before_sha,
            "state_after_sha256": state_after_sha,
            "agent_action": copy.deepcopy(action),
            "action_assessment": copy.deepcopy(assessment),
            "public_action_outcome": copy.deepcopy(public_outcome),
            "newly_released_evidence_ids": list(newly_released),
            "done": self._state["done"],
        }
        self._append_entry(entry)
        return (
            self._observation(
                newly_released,
                public_outcome=public_outcome,
            ),
            assessment,
            self._state["done"],
        )

    def export_trace(self) -> dict[str, Any]:
        if self._state is None:
            raise InteractionContractError("call reset before export")
        return {
            "trace_version": TRACE_VERSION,
            "environment_version": ENVIRONMENT_VERSION,
            "resolver_version": RESOLVER_VERSION,
            "status": "complete" if self._state["done"] else "incomplete",
            "created_at": self._created_at,
            "exported_at": now_iso(),
            "run_id": self._run_id,
            "agent_id": self._agent_id,
            "case_id": self.case_id,
            "scenario_sha256": self._scenario_sha,
            "action_interface_sha256": self._interface_sha,
            "action_ontology_sha256": self._ontology_sha,
            "action_graph_sha256": self._graph_sha,
            "scorer_freeze": {
                "freeze_id": self._scenario["scorer_freeze"]["freeze_id"],
                "manifest_sha256": self._freeze_sha,
            },
            "entries": copy.deepcopy(self._trace_entries),
            "final_state": copy.deepcopy(self._state),
            "final_trace_entry_sha256": self._previous_entry_hash,
            "automatic_semantic_judge_scoring": False,
            "official_scoring_allowed": False,
            "single_total_score_absent": True,
        }


def validate_trace(trace: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if trace.get("trace_version") != TRACE_VERSION:
        errors.append("wrong_trace_version")
    if trace.get("environment_version") != ENVIRONMENT_VERSION:
        errors.append("wrong_environment_version")
    if trace.get("resolver_version") != RESOLVER_VERSION:
        errors.append("wrong_resolver_version")
    if trace.get("official_scoring_allowed") is not False:
        errors.append("official_scoring_boundary_invalid")
    if trace.get("single_total_score_absent") is not True:
        errors.append("single_total_score_boundary_invalid")
    entries = trace.get("entries")
    if not isinstance(entries, list):
        return [*errors, "entries_not_list"]
    previous: str | None = None
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            errors.append(f"invalid_entry:{index}")
            continue
        entry = copy.deepcopy(raw_entry)
        observed_hash = entry.pop("entry_sha256", None)
        if entry.get("previous_entry_sha256") != previous:
            errors.append(f"broken_previous_hash:{index}")
        if observed_hash != canonical_hash(entry):
            errors.append(f"entry_hash_mismatch:{index}")
        previous = observed_hash
    if trace.get("final_trace_entry_sha256") != previous:
        errors.append("final_trace_hash_mismatch")
    final_state = trace.get("final_state")
    if not isinstance(final_state, dict):
        errors.append("missing_final_state")
    elif trace.get("status") == "complete" and final_state.get("done") is not True:
        errors.append("complete_trace_without_terminal_state")
    return errors

