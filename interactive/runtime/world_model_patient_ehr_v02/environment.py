"""Wide-action deterministic Patient/EHR environment v0.2."""

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

from .contracts import (
    ENVIRONMENT_VERSION,
    TRACE_VERSION,
    load_bound_action_interface,
)
from .ontology import RESOLVER_VERSION, resolve_against_concept, resolve_all_concepts


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WideActionPatientEHREnvironment:
    """Deterministic EHR replay with free-text clinical target resolution."""

    def __init__(
        self,
        *,
        scenario_path: Path,
        freeze_manifest_path: Path,
        interface_path: Path,
        ontology_path: Path,
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
        self._scenario = scenario
        self._scenario_sha = scenario_sha
        self._freeze_sha = freeze_sha
        self._interface = interface
        self._interface_sha = interface_sha
        self._ontology = ontology
        self._ontology_sha = ontology_sha
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
            "step_index": self._state["next_transition_index"],
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
            "next_transition_index": 0,
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

        transition = self._scenario["transitions"][
            self._state["next_transition_index"]
        ]
        expected_concept_id = self._interface["transition_concepts"][
            transition["transition_id"]
        ]
        resolution = resolve_against_concept(
            action, expected_concept_id, self._ontology
        )
        all_resolutions = resolve_all_concepts(action, self._ontology)
        recognized_other = [
            item["concept_id"]
            for item in all_resolutions
            if item["concept_id"] != expected_concept_id
        ]
        grounding_valid = set(evidence_ids).issubset(
            set(self._state["visible_evidence_ids"])
        )
        trajectory_match = bool(resolution["equivalent"] and grounding_valid)
        if trajectory_match:
            outcome_status = "ehr_path_advanced"
        elif not grounding_valid:
            outcome_status = "invalid_evidence_grounding"
        elif recognized_other:
            outcome_status = "recognized_non_current_action"
        else:
            outcome_status = "valid_exploration_no_recorded_outcome"

        state_before_sha = self._state_hash()
        newly_released: list[str] = []
        if trajectory_match:
            newly_released = list(transition["release_evidence_ids"])
            self._state["visible_evidence_ids"].extend(newly_released)
            self._state["visible_evidence_ids"] = sorted(
                set(self._state["visible_evidence_ids"])
            )
            self._state["completed_transition_ids"].append(
                transition["transition_id"]
            )
            self._state["stage"] = transition["stage_after"]
            if transition.get("clock_after") is not None:
                self._state["clock"] = transition["clock_after"]
            self._state["state_facts"].update(
                copy.deepcopy(transition.get("state_updates", {}))
            )
            self._state["next_transition_index"] += 1
            self._state["done"] = bool(transition["terminal"])
        state_after_sha = self._state_hash()

        public_outcome = {
            "status": outcome_status,
            "action_understood": bool(
                resolution["equivalent"] or recognized_other
            ),
            "state_advanced": trajectory_match,
            "newly_released_evidence_ids": list(newly_released),
            "message": (
                "Recorded EHR path advanced and new evidence is available."
                if trajectory_match
                else (
                    "The action is clinically interpretable, but this recorded EHR "
                    "replay has no currently releasable outcome for it."
                    if grounding_valid
                    else "The action cited evidence that is not currently visible."
                )
            ),
        }
        assessment = {
            "interface_valid": True,
            "grounding_valid": grounding_valid,
            "trajectory_match": trajectory_match,
            "outcome_status": outcome_status,
            "expected_concept_resolution": resolution,
            "recognized_non_current_concept_ids": recognized_other,
            "state_advanced": trajectory_match,
            "released_evidence_ids": list(newly_released),
            "single_total_score_absent": True,
        }
        entry = {
            "trace_version": TRACE_VERSION,
            "run_id": self._run_id,
            "case_id": self.case_id,
            "attempt_index": len(self._trace_entries),
            "expected_transition_index": transition["sequence_index"],
            "stage_before": transition["stage_before"],
            "stage_after": self._state["stage"],
            "environment_clock_before": transition.get("clock_before"),
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

