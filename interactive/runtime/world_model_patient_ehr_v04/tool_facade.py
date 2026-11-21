"""Tool facade over the v0.3 partial-order Patient/EHR environment."""

from __future__ import annotations

from runtime_assets import vocabulary

import copy
import json
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
from world_model_patient_ehr_v03.environment import (
    PartialOrderPatientEHREnvironment,
    validate_trace,
)
from world_model_patient_ehr_v02.ontology import normalize, resolve_against_concept


TOOL_MAP_VERSION = "ward-patient-ehr-tool-map-0.4.2"
SUPPORTED_TOOL_MAP_VERSIONS = {
    "ward-patient-ehr-tool-map-0.4.0",
    "ward-patient-ehr-tool-map-0.4.1",
    TOOL_MAP_VERSION,
}
TOOL_CALL_TRACE_VERSION = "ward-patient-ehr-tool-call-trace-0.4.0"
TOOL_NAMES = {
    "patient.ask",
    "ehr.search",
    "ehr.read",
    "ehr.order",
    "procedure.perform",
    "treatment.start",
    "care.transition",
}
MAX_INTENT_CHARS = 240
MAX_ARGS_CHARS = 2000
MEMORY_LIMIT = 12
LATERALITY_TERMS = {'left': ('left',), 'right': ('right',), 'bilateral': ('bilateral', 'both sides')}

LATERALITY_TERMS = vocabulary("world_model_patient_ehr_v04.tool_facade.LATERALITY_TERMS", LATERALITY_TERMS)
SEMANTIC_TARGET_LEXICON = {'mri': ('mri', 'magnetic resonance'),
 'ct': ('ct', 'computed tomography'),
 'ultrasound': ('ultrasound', 'sonography'),
 'mammography': ('mammography',),
 'bone_scan': ('bone scan', 'bone scintigraphy'),
 'pathology': ('pathology', 'histology'),
 'biopsy': ('biopsy', 'aspiration'),
 'surgery': ('surgery', 'operation'),
 'sentinel_node': ('sentinel',),
 'breast': ('breast',),
 'supraclavicular': ('supraclavicular',),
 'axillary': ('axillary',),
 'lymph_node': ('lymph node', 'node'),
 'chest': ('chest', 'thoracic', 'lung'),
 'abdomen': ('abdomen', 'abdominal', 'liver'),
 'bone': ('bone', 'skeletal'),
 'cardiac': ('cardiac', 'heart'),
 'lower_extremity': ('lower extremity', 'leg'),
 'venous': ('venous', 'vein'),
 'discharge': ('discharge', 'home')}

SEMANTIC_TARGET_LEXICON = vocabulary("world_model_patient_ehr_v04.tool_facade.SEMANTIC_TARGET_LEXICON", SEMANTIC_TARGET_LEXICON)


def _content_text(content: Any) -> str:
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "").strip()


def parse_tool_call(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        value = copy.deepcopy(content)
    else:
        text = _content_text(content)
        if credential_findings(text):
            raise InteractionContractError("credential-like model output rejected")
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
            text = re.sub(r"\s*```$", "", text)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise InteractionContractError(
                    "model response does not contain a JSON object"
                ) from None
            try:
                value = json.loads(text[start : end + 1])
            except json.JSONDecodeError as exc:
                raise InteractionContractError(
                    "model response contains invalid JSON"
                ) from exc
    if not isinstance(value, dict):
        raise InteractionContractError("model response must be a JSON object")
    allowed = {"tool", "intent", "args", "rationale", "evidence_ids"}
    if set(value) - allowed:
        raise InteractionContractError("model tool call contains unsupported fields")
    if credential_findings(value):
        raise InteractionContractError("credential-like model tool call rejected")
    tool = value.get("tool")
    if tool not in TOOL_NAMES:
        raise InteractionContractError("model tool is not allowed")
    args = value.get("args", {})
    if not isinstance(args, dict):
        raise InteractionContractError("model args must be an object")
    if len(json.dumps(args, ensure_ascii=False)) > MAX_ARGS_CHARS:
        raise InteractionContractError("model args exceed size limit")
    intent = value.get("intent")
    if intent is None:
        query = args.get("query")
        if isinstance(query, str) and query.strip():
            intent = query.strip()[:MAX_INTENT_CHARS]
    if (
        not isinstance(intent, str)
        or not intent.strip()
        or len(intent) > MAX_INTENT_CHARS
    ):
        raise InteractionContractError(
            "model intent or args.query must be 1-240 characters"
        )
    rationale = value.get("rationale", "")
    if not isinstance(rationale, str) or len(rationale) > 1200:
        raise InteractionContractError("model rationale must be at most 1200 characters")
    evidence_ids = value.get("evidence_ids", [])
    if (
        not isinstance(evidence_ids, list)
        or any(not isinstance(item, str) or not item for item in evidence_ids)
        or len(evidence_ids) != len(set(evidence_ids))
    ):
        raise InteractionContractError(
            "model evidence_ids must be a unique list of strings"
        )
    return {
        "tool": tool,
        "intent": intent.strip(),
        "args": copy.deepcopy(args),
        "rationale": rationale.strip(),
        "evidence_ids": list(evidence_ids),
    }


def validate_tool_map(tool_map: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if tool_map.get("tool_map_version") not in SUPPORTED_TOOL_MAP_VERSIONS:
        errors.append("wrong_tool_map_version")
    if credential_findings(tool_map):
        errors.append("credential_like_material_present")
    for key in (
        "gold_action_visible",
        "dependency_graph_visible",
        "future_evidence_visible",
        "official_scoring_allowed",
        "automatic_semantic_judge_scoring",
    ):
        if tool_map.get(key) is not False:
            errors.append(f"invalid_boundary:{key}")
    if tool_map.get("single_total_score_absent") is not True:
        errors.append("single_total_score_boundary_invalid")
    public_tools = tool_map.get("public_tools")
    if not isinstance(public_tools, list) or set(public_tools) != TOOL_NAMES:
        errors.append("invalid_public_tools")
    modules = tool_map.get("public_ehr_modules")
    if not isinstance(modules, list) or not modules:
        errors.append("invalid_public_ehr_modules")
    defaults = tool_map.get("tool_action_defaults")
    if not isinstance(defaults, dict):
        errors.append("invalid_tool_action_defaults")
    else:
        for tool, action_type in defaults.items():
            if tool not in TOOL_NAMES or action_type not in ACTION_TYPES:
                errors.append(f"invalid_tool_action_default:{tool}")
    overrides = tool_map.get("module_action_overrides", {})
    if not isinstance(overrides, dict):
        errors.append("invalid_module_action_overrides")
    else:
        for module, values in overrides.items():
            if not isinstance(module, str) or not isinstance(values, dict):
                errors.append(f"invalid_module_override:{module}")
                continue
            for tool, action_type in values.items():
                if tool not in TOOL_NAMES or action_type not in ACTION_TYPES:
                    errors.append(f"invalid_module_action_override:{module}:{tool}")
    navigation_visible = tool_map.get("tool_design", {}).get(
        "coarse_available_action_families_visible"
    )
    if (
        tool_map.get("tool_map_version")
        == "ward-patient-ehr-tool-map-0.4.1"
        and navigation_visible is not True
    ):
        errors.append("coarse_available_action_families_boundary_invalid")
    if (
        tool_map.get("tool_map_version") == TOOL_MAP_VERSION
        and navigation_visible is not False
    ):
        errors.append("gold_derived_navigation_must_be_hidden")
    return errors


def load_tool_map(
    *,
    tool_map_path: Path,
    scenario_path: Path,
    interface_path: Path,
    ontology_path: Path,
    graph_path: Path,
) -> tuple[dict[str, Any], str]:
    tool_map = read_json(tool_map_path)
    errors = validate_tool_map(tool_map)
    expected = {
        "scenario_sha256": sha256_file(scenario_path),
        "action_interface_sha256": sha256_file(interface_path),
        "action_ontology_sha256": sha256_file(ontology_path),
        "action_graph_sha256": sha256_file(graph_path),
    }
    for key, value in expected.items():
        if tool_map.get(key) != value:
            errors.append(f"{key}_mismatch")
    if errors:
        raise InteractionContractError("invalid v0.4 tool map: " + "; ".join(errors))
    return tool_map, sha256_file(tool_map_path)


def concept_family(concept_ids: list[str]) -> str | None:
    if not concept_ids:
        return None
    concept = concept_ids[0]
    if concept.startswith("patient."):
        return "history"
    if concept.startswith("ehr."):
        return "documents"
    if concept.startswith("diagnostic."):
        return "imaging"
    if concept.startswith("procedure."):
        return "procedure"
    if concept.startswith("result."):
        return "results"
    if concept.startswith("disposition."):
        return "disposition"
    if concept.startswith("treatment."):
        return "treatment"
    return "clinical"


def tool_family(tool_call: dict[str, Any]) -> str:
    tool = tool_call["tool"]
    if tool == "patient.ask":
        return "history"
    if tool in {"ehr.search", "ehr.read"}:
        module = str(tool_call.get("args", {}).get("module", "documents"))
        if "imaging" in module:
            return "imaging"
        if "laboratory" in module or "pathology" in module or "result" in module:
            return "results"
        return "documents"
    if tool == "ehr.order":
        return "imaging"
    if tool == "procedure.perform":
        return "procedure"
    if tool == "treatment.start":
        return "treatment"
    if tool == "care.transition":
        return "disposition"
    return "clinical"


def _laterality_values(value: str) -> set[str]:
    target = normalize(value)
    values: set[str] = set()
    for side, terms in LATERALITY_TERMS.items():
        if any(normalize(term) in target for term in terms):
            values.add(side)
    if "bilateral" in values:
        values.update({"left", "right"})
    return values


def semantic_target_fingerprint(tool_call: dict[str, Any]) -> str:
    """Return a case-independent hash for soft duplicate detection.

    The fingerprint uses only public tool-call content and a small bilingual
    clinical lexicon. It is not a resolver result and does not expose a Gold
    concept or transition.
    """

    args = tool_call.get("args", {})
    module = normalize(str(args.get("module", "")))
    target = normalize(
        " ".join(
            [
                str(tool_call.get("intent", "")),
                str(args.get("query", "")),
            ]
        )
    )
    def term_present(term: str) -> bool:
        normalized_term = normalize(term)
        if not normalized_term:
            return False
        if normalized_term.isascii():
            return bool(
                re.search(
                    rf"(?:^|\s){re.escape(normalized_term)}(?:$|\s)",
                    target,
                )
            )
        return normalized_term in target

    features = [
        label
        for label, terms in SEMANTIC_TARGET_LEXICON.items()
        if any(term_present(term) for term in terms)
    ]
    laterality = sorted(_laterality_values(target))
    if features:
        semantic_target = "|".join(sorted(set(features)))
    else:
        semantic_target = target[:160]
    public_key = {
        "tool": str(tool_call.get("tool", "")),
        "module": module,
        "semantic_target": semantic_target,
        "laterality": laterality,
    }
    return canonical_hash(public_key)


class LightweightClinicalToolSandbox:
    """A reusable tool interface that delegates state to v0.3 EHR replay."""

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
        self._tool_map, self._tool_map_sha = load_tool_map(
            tool_map_path=tool_map_path,
            scenario_path=scenario_path,
            interface_path=interface_path,
            ontology_path=ontology_path,
            graph_path=graph_path,
        )
        self._env = PartialOrderPatientEHREnvironment(
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

    @property
    def tool_map_sha256(self) -> str:
        return self._tool_map_sha

    @property
    def tool_map_version(self) -> str:
        return str(self._tool_map["tool_map_version"])

    def _empty_memory(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "completed_tool_calls": [],
            "pending_tool_calls": [],
            "unrecorded_tool_calls": [],
            "invalid_tool_calls": [],
            "do_not_repeat_exact_targets": [],
            "do_not_repeat_semantic_targets": [],
            "semantic_target_history": [],
        }

    def _memory_public(self) -> dict[str, Any]:
        public_keys = (
            "completed_tool_calls",
            "pending_tool_calls",
            "unrecorded_tool_calls",
            "invalid_tool_calls",
        )
        value = {
            key: copy.deepcopy(items[-MEMORY_LIMIT:])
            for key, items in self._memory.items()
            if key in public_keys
        }
        value["do_not_repeat_semantic_targets"] = copy.deepcopy(
            [
                item
                for item in self._memory["do_not_repeat_semantic_targets"]
                if item.get("evidence_generation") == self._evidence_generation
            ][-MEMORY_LIMIT:]
        )
        value["do_not_repeat_exact_targets"] = copy.deepcopy(
            [
                item
                for item in self._memory["do_not_repeat_exact_targets"]
                if item.get("evidence_generation") == self._evidence_generation
            ][-MEMORY_LIMIT:]
        )
        value["retryable_pending_families"] = sorted(
            {
                item["tool_family"]
                for item in self._memory["pending_tool_calls"]
                if isinstance(item.get("tool_family"), str)
            }
        )
        return value

    def _available_action_family_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for transition_id in self._env._available_transition_ids():
            concept_id = self._env._interface["transition_concepts"][transition_id]
            family = concept_family([concept_id]) or "clinical"
            counts[family] = counts.get(family, 0) + 1
        return dict(sorted(counts.items()))

    def _public_observation(
        self,
        observation: dict[str, Any],
        *,
        last_tool_outcome: dict[str, Any] | None,
    ) -> dict[str, Any]:
        public = copy.deepcopy(observation)
        navigation_visible = bool(
            self._tool_map.get("tool_design", {}).get(
                "coarse_available_action_families_visible",
                False,
            )
        )
        public["tool_interface"] = {
            "tool_call_schema": {
                "required": ["tool"],
                "conditional_required": "provide intent or non-empty args.query",
                "optional": ["intent", "args", "rationale", "evidence_ids"],
                "tool_values": list(self._tool_map["public_tools"]),
                "args_contract": {
                    "module": "one of public_ehr_modules when using EHR tools",
                    "query": "free-text clinical query or target",
                },
            },
            "public_ehr_modules": list(self._tool_map["public_ehr_modules"]),
            "coarse_available_action_families_visible": navigation_visible,
            "unrecorded_actions_allowed": True,
            "counterfactual_results_invented": False,
        }
        if navigation_visible:
            family_counts = self._available_action_family_counts()
            public["available_action_families"] = list(family_counts)
            public["available_action_family_counts"] = family_counts
            public["available_family_tool_hints"] = {
                family: list(
                    self._tool_map.get("public_recovery_hints", {}).get(
                        family,
                        [],
                    )
                )
                for family in family_counts
            }
        else:
            public.pop("available_action_families", None)
            public.pop("available_action_family_counts", None)
            public.pop("available_family_tool_hints", None)
        public["agent_memory"] = self._memory_public()
        public["last_tool_outcome"] = copy.deepcopy(last_tool_outcome)
        return public

    def reset(self) -> dict[str, Any]:
        self._memory = self._empty_memory()
        self._tool_entries = []
        self._previous_tool_entry_hash = None
        self._evidence_generation = 0
        return self._public_observation(
            self._env.reset(),
            last_tool_outcome=None,
        )

    def _action_type_for(self, tool_call: dict[str, Any]) -> str:
        tool = tool_call["tool"]
        module = str(tool_call.get("args", {}).get("module", "")).strip()
        overrides = self._tool_map.get("module_action_overrides", {})
        if module in overrides and tool in overrides[module]:
            return str(overrides[module][tool])
        return str(self._tool_map["tool_action_defaults"][tool])

    def tool_call_to_action(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        action_type = self._action_type_for(tool_call)
        args = tool_call.get("args", {})
        module = str(args.get("module", "")).strip()
        query = str(args.get("query", "")).strip()
        target = tool_call["intent"].strip()
        if module or query:
            suffix = "; ".join(
                item for item in (f"module={module}" if module else "", query) if item
            )
            target = f"{target} ({suffix})"
        if len(target) > 240:
            target = target[:237].rstrip() + "..."
        return {
            "action_type": action_type,
            "target": target,
            "parameters": copy.deepcopy(args),
            "rationale": str(tool_call.get("rationale", "")).strip(),
            "evidence_ids": list(tool_call.get("evidence_ids", [])),
        }

    def _apply_critical_slot_guard(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        target_laterality = _laterality_values(str(action.get("target", "")))
        if not target_laterality:
            return action, False
        equivalent_candidates = 0
        nonconflicting_candidates = 0
        for transition_id in self._env._available_transition_ids():
            concept_id = self._env._interface["transition_concepts"][transition_id]
            resolution = resolve_against_concept(
                action,
                concept_id,
                self._env._ontology,
            )
            if not resolution["equivalent"]:
                continue
            equivalent_candidates += 1
            concept = self._env._ontology["concepts"][concept_id]
            concept_text = " ".join(
                [
                    *concept.get("phrases", []),
                    *[
                        term
                        for group in concept.get("anchor_groups", [])
                        for term in group
                    ],
                ]
            )
            concept_laterality = _laterality_values(concept_text)
            if not concept_laterality or target_laterality & concept_laterality:
                nonconflicting_candidates += 1
        if equivalent_candidates == 0 or nonconflicting_candidates > 0:
            return action, False
        guarded = copy.deepcopy(action)
        guarded["action_type"] = "monitor_or_defer"
        guarded["parameters"]["critical_slot_guard"] = "laterality_conflict"
        return guarded, True

    def _summarize_tool_call(
        self,
        tool_call: dict[str, Any],
        *,
        family: str,
        status: str,
        fingerprint: str,
        semantic_target_seen_before: bool,
        semantic_repeat_without_new_evidence: bool,
        semantic_revisit_after_new_evidence: bool,
    ) -> dict[str, Any]:
        return {
            "tool": tool_call["tool"],
            "intent": tool_call["intent"],
            "module": str(tool_call.get("args", {}).get("module", "")),
            "tool_family": family,
            "status": status,
            "semantic_target_fingerprint": fingerprint,
            "semantic_target_seen_before": semantic_target_seen_before,
            "semantic_repeat_without_new_evidence": (
                semantic_repeat_without_new_evidence
            ),
            "semantic_revisit_after_new_evidence": (
                semantic_revisit_after_new_evidence
            ),
            "evidence_generation": self._evidence_generation,
        }

    def _update_memory(
        self,
        *,
        tool_call: dict[str, Any],
        outcome: dict[str, Any],
        family: str,
        fingerprint: str,
        semantic_target_seen_before: bool,
        semantic_repeat_without_new_evidence: bool,
        semantic_revisit_after_new_evidence: bool,
    ) -> None:
        status = outcome["status"]
        summary = self._summarize_tool_call(
            tool_call,
            family=family,
            status=status,
            fingerprint=fingerprint,
            semantic_target_seen_before=semantic_target_seen_before,
            semantic_repeat_without_new_evidence=(
                semantic_repeat_without_new_evidence
            ),
            semantic_revisit_after_new_evidence=(
                semantic_revisit_after_new_evidence
            ),
        )
        self._memory["semantic_target_history"].append(summary)
        if status == "ehr_path_advanced":
            self._memory["completed_tool_calls"].append(summary)
        elif status == "recognized_prerequisite_pending":
            self._memory["pending_tool_calls"].append(summary)
        elif status == "valid_exploration_no_recorded_outcome":
            self._memory["unrecorded_tool_calls"].append(summary)
        elif status in {"recognized_completed_action", "invalid_evidence_grounding"}:
            self._memory["invalid_tool_calls"].append(summary)
        if status in {
            "recognized_completed_action",
            "recognized_prerequisite_pending",
            "valid_exploration_no_recorded_outcome",
        }:
            self._memory["do_not_repeat_exact_targets"].append(
                {
                    "tool": tool_call["tool"],
                    "intent": tool_call["intent"],
                    "status": status,
                    "evidence_generation": self._evidence_generation,
                }
            )
            self._memory["do_not_repeat_semantic_targets"].append(summary)

    def _append_tool_entry(self, entry: dict[str, Any]) -> None:
        entry["previous_tool_entry_sha256"] = self._previous_tool_entry_hash
        entry_hash = canonical_hash(entry)
        entry["tool_entry_sha256"] = entry_hash
        self._tool_entries.append(entry)
        self._previous_tool_entry_hash = entry_hash

    def step_tool_call(
        self, tool_call: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        parsed = parse_tool_call(tool_call)
        fingerprint = semantic_target_fingerprint(parsed)
        prior_same_target = [
            item
            for item in self._memory["semantic_target_history"]
            if item.get("semantic_target_fingerprint") == fingerprint
        ]
        semantic_target_seen_before = bool(prior_same_target)
        prior_generation = max(
            (
                int(item.get("evidence_generation", 0))
                for item in prior_same_target
            ),
            default=-1,
        )
        semantic_repeat_without_new_evidence = (
            semantic_target_seen_before
            and prior_generation == self._evidence_generation
        )
        semantic_revisit_after_new_evidence = (
            semantic_target_seen_before
            and prior_generation < self._evidence_generation
        )
        action = self.tool_call_to_action(parsed)
        action, critical_slot_guard_triggered = self._apply_critical_slot_guard(
            action
        )
        observation, assessment, done = self._env.step(action)
        outcome = copy.deepcopy(observation["last_action_outcome"])
        family = concept_family(assessment.get("recognized_concept_ids", []))
        if family is None:
            family = tool_family(parsed)
        hints = self._tool_map.get("public_recovery_hints", {})
        tool_outcome = {
            "status": outcome["status"],
            "action_understood": outcome["action_understood"],
            "state_advanced": outcome["state_advanced"],
            "newly_released_evidence_ids": list(
                outcome["newly_released_evidence_ids"]
            ),
            "message": outcome["message"],
            "tool_family": family,
            "suggested_next_tool_families": list(hints.get(family, [])),
            "do_not_immediately_retry_same_intent": outcome["status"]
            in {
                "recognized_completed_action",
                "recognized_prerequisite_pending",
                "valid_exploration_no_recorded_outcome",
            },
            "semantic_target_seen_before": semantic_target_seen_before,
            "semantic_repeat_without_new_evidence": (
                semantic_repeat_without_new_evidence
            ),
            "semantic_revisit_after_new_evidence": (
                semantic_revisit_after_new_evidence
            ),
            "semantic_target_fingerprint": fingerprint,
            "critical_slot_conflict_prevented": critical_slot_guard_triggered,
            "single_total_score_absent": True,
        }
        self._update_memory(
            tool_call=parsed,
            outcome=tool_outcome,
            family=family,
            fingerprint=fingerprint,
            semantic_target_seen_before=semantic_target_seen_before,
            semantic_repeat_without_new_evidence=(
                semantic_repeat_without_new_evidence
            ),
            semantic_revisit_after_new_evidence=(
                semantic_revisit_after_new_evidence
            ),
        )
        if tool_outcome["state_advanced"] is True:
            self._evidence_generation += 1
        entry = {
            "tool_trace_version": TOOL_CALL_TRACE_VERSION,
            "run_id": self._run_id,
            "agent_id": self._agent_id,
            "case_id": observation["case_id"],
            "tool_attempt_index": len(self._tool_entries),
            "tool_call": copy.deepcopy(parsed),
            "converted_action": copy.deepcopy(action),
            "critical_slot_guard_triggered": critical_slot_guard_triggered,
            "public_tool_outcome": copy.deepcopy(tool_outcome),
            "completed_recorded_action_count_after": observation[
                "completed_recorded_action_count"
            ],
            "available_recorded_action_count_after": observation[
                "available_recorded_action_count"
            ],
            "stage_after": observation["stage"],
            "done": observation["done"],
            "hidden_transition_ids_exposed": False,
            "single_total_score_absent": True,
        }
        self._append_tool_entry(entry)
        public_observation = self._public_observation(
            observation,
            last_tool_outcome=tool_outcome,
        )
        return public_observation, tool_outcome, done

    def export_tool_trace(self) -> dict[str, Any]:
        env_trace = self._env.export_trace()
        trace_errors = validate_trace(env_trace)
        if trace_errors:
            raise InteractionContractError(
                "underlying v0.3 trace invalid: " + "; ".join(trace_errors)
            )
        return {
            "tool_trace_version": TOOL_CALL_TRACE_VERSION,
            "status": env_trace["status"],
            "run_id": self._run_id,
            "agent_id": self._agent_id,
            "case_id": env_trace["case_id"],
            "tool_map_sha256": self._tool_map_sha,
            "environment_trace_final_entry_sha256": env_trace.get(
                "final_trace_entry_sha256"
            ),
            "entries": copy.deepcopy(self._tool_entries),
            "final_tool_entry_sha256": self._previous_tool_entry_hash,
            "final_public_memory": self._memory_public(),
            "final_state": copy.deepcopy(env_trace["final_state"]),
            "gold_action_visible": False,
            "dependency_graph_visible": False,
            "future_evidence_visible": False,
            "official_scoring_allowed": False,
            "single_total_score_absent": True,
        }


def validate_tool_trace(trace: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if trace.get("tool_trace_version") != TOOL_CALL_TRACE_VERSION:
        errors.append("wrong_tool_trace_version")
    if trace.get("official_scoring_allowed") is not False:
        errors.append("official_scoring_boundary_invalid")
    if trace.get("single_total_score_absent") is not True:
        errors.append("single_total_score_boundary_invalid")
    if trace.get("gold_action_visible") is not False:
        errors.append("gold_action_boundary_invalid")
    if trace.get("dependency_graph_visible") is not False:
        errors.append("dependency_graph_boundary_invalid")
    if trace.get("future_evidence_visible") is not False:
        errors.append("future_evidence_boundary_invalid")
    entries = trace.get("entries")
    if not isinstance(entries, list):
        return [*errors, "entries_not_list"]
    previous: str | None = None
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            errors.append(f"invalid_tool_entry:{index}")
            continue
        entry = copy.deepcopy(raw_entry)
        observed = entry.pop("tool_entry_sha256", None)
        if entry.get("previous_tool_entry_sha256") != previous:
            errors.append(f"broken_previous_tool_hash:{index}")
        if observed != canonical_hash(entry):
            errors.append(f"tool_entry_hash_mismatch:{index}")
        previous = observed
    if trace.get("final_tool_entry_sha256") != previous:
        errors.append("final_tool_trace_hash_mismatch")
    if credential_findings(trace):
        errors.append("credential_like_material_present")
    return errors


__all__ = [
    "LightweightClinicalToolSandbox",
    "TOOL_CALL_TRACE_VERSION",
    "TOOL_MAP_VERSION",
    "TOOL_NAMES",
    "load_tool_map",
    "parse_tool_call",
    "semantic_target_fingerprint",
    "validate_tool_map",
    "validate_tool_trace",
]
