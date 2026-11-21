"""Deterministic Patient/EHR action, evidence, and state-transition environment."""

from __future__ import annotations

import copy
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import (
    ACTION_TYPES,
    ENVIRONMENT_VERSION,
    TRACE_VERSION,
    InteractionContractError,
    canonical_hash,
    credential_findings,
    load_bound_scenario,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_term(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).strip().lower()
    text = re.sub(r"[_/\\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


class PatientEHREnvironment:
    """One-case deterministic environment with strict next-EHR-action matching."""

    def __init__(
        self,
        scenario_path: Path,
        freeze_manifest_path: Path,
        *,
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
        self._scenario = scenario
        self._scenario_sha = scenario_sha
        self._freeze_sha = freeze_sha
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

    def _observation(self, newly_released: list[str]) -> dict[str, Any]:
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
            "state_fingerprint": self._state_hash(),
            "done": self._state["done"],
            "action_schema": {
                "required": ["action_type", "target"],
                "optional": ["parameters", "rationale", "evidence_ids"],
                "action_type_values": sorted(ACTION_TYPES),
            },
            "gold_action_hidden": True,
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
        return self._observation(list(initial["visible_evidence_ids"]))

    def _match_action(
        self, action: dict[str, Any], gold: dict[str, Any]
    ) -> tuple[bool, str, dict[str, str]]:
        raw_type = str(action.get("action_type", ""))
        raw_target = str(action.get("target", ""))
        type_norm = normalize_term(raw_type)
        target_norm = normalize_term(raw_target)
        canonical_type = normalize_term(str(gold["action_type"]))
        canonical_target = normalize_term(str(gold["target"]))
        type_aliases = {
            normalize_term(value)
            for value in gold.get("accepted_action_type_aliases", [])
        }
        target_aliases = {
            normalize_term(value) for value in gold.get("accepted_target_aliases", [])
        }
        type_match = type_norm == canonical_type or type_norm in type_aliases
        target_match = target_norm == canonical_target or target_norm in target_aliases
        exact = raw_type == gold["action_type"] and raw_target == gold["target"]
        match_status = "exact_gold" if exact else "normalized_gold" if (
            type_match and target_match
        ) else "off_trajectory"
        return type_match and target_match, match_status, {
            "action_type": type_norm,
            "target": target_norm,
        }

    def _append_entry(self, entry: dict[str, Any]) -> None:
        entry["previous_entry_sha256"] = self._previous_entry_hash
        entry_hash = canonical_hash(entry)
        entry["entry_sha256"] = entry_hash
        self._trace_entries.append(entry)
        self._previous_entry_hash = entry_hash

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], bool]:
        if self._state is None:
            raise InteractionContractError("call reset before step")
        if self._state["done"]:
            raise InteractionContractError("episode is already complete")
        if not isinstance(action, dict):
            raise InteractionContractError("action must be an object")
        if credential_findings(action):
            raise InteractionContractError("credential-like material is forbidden")
        if not isinstance(action.get("action_type"), str) or not action["action_type"]:
            raise InteractionContractError("action_type is required")
        if not isinstance(action.get("target"), str) or not action["target"]:
            raise InteractionContractError("target is required")
        evidence_ids = action.get("evidence_ids", [])
        if not isinstance(evidence_ids, list) or any(
            not isinstance(item, str) or not item for item in evidence_ids
        ):
            raise InteractionContractError("evidence_ids must be a list of strings")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise InteractionContractError("evidence_ids contains duplicates")

        transition = self._scenario["transitions"][
            self._state["next_transition_index"]
        ]
        gold = transition["gold_action"]
        state_before_sha = self._state_hash()
        grounding_valid = set(evidence_ids).issubset(
            set(self._state["visible_evidence_ids"])
        )
        action_match, match_status, normalized = self._match_action(action, gold)
        accepted = action_match and grounding_valid
        newly_released: list[str] = []
        if accepted:
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
        assessment = {
            "accepted": accepted,
            "match_status": match_status if grounding_valid else "invalid_evidence_grounding",
            "grounding_valid": grounding_valid,
            "gold_source_type": gold["gold_source_type"],
            "clinical_gold_action": (
                gold["action_type"]
                if gold["gold_source_type"]
                in {
                    "ehr_observed_action",
                    "ehr_observed_result_implies_order",
                    "ehr_derived_terminal_gold",
                }
                else None
            ),
            "access_policy_action": (
                gold["action_type"]
                if gold["gold_source_type"]
                in {"patient_question_exception", "ehr_evidence_access"}
                else None
            ),
            "off_trajectory": not accepted,
            "state_advanced": accepted,
            "released_evidence_ids": newly_released,
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
            "normalized_action": normalized,
            "action_assessment": copy.deepcopy(assessment),
            "newly_released_evidence_ids": newly_released,
            "done": self._state["done"],
        }
        self._append_entry(entry)
        return self._observation(newly_released), assessment, self._state["done"]

    def expected_gold_action_for_replay(self) -> dict[str, Any]:
        """Return the next Gold action for test/replay tooling, never for an Agent."""
        if self._state is None:
            raise InteractionContractError("call reset before replay")
        if self._state["done"]:
            raise InteractionContractError("episode is already complete")
        gold = self._scenario["transitions"][
            self._state["next_transition_index"]
        ]["gold_action"]
        return {
            "action_type": gold["action_type"],
            "target": gold["target"],
            "parameters": copy.deepcopy(gold.get("parameters", {})),
            "rationale": "deterministic Gold replay",
            "evidence_ids": [],
        }

    def export_trace(self) -> dict[str, Any]:
        if self._state is None:
            raise InteractionContractError("call reset before export")
        return {
            "trace_version": TRACE_VERSION,
            "environment_version": ENVIRONMENT_VERSION,
            "status": (
                "complete" if self._state["done"] else "incomplete"
            ),
            "created_at": self._created_at,
            "exported_at": now_iso(),
            "run_id": self._run_id,
            "agent_id": self._agent_id,
            "case_id": self.case_id,
            "scenario_sha256": self._scenario_sha,
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
        expected_hash = canonical_hash(entry)
        if observed_hash != expected_hash:
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
