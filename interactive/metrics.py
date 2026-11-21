"""Case-level process metrics derived from auditable execution traces."""
from __future__ import annotations

from collections import Counter
from statistics import mean


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def derive_metrics(scenario: dict, manifest: dict, tool_trace: dict, interaction_trace: dict) -> dict:
    entries = tool_trace["entries"]
    steps = interaction_trace["entries"]
    if len(entries) != len(steps):
        raise ValueError("tool and environment trace lengths differ")
    transitions = {t["transition_id"]: t for t in scenario["transitions"]}
    reference = {e for t in transitions.values() for e in t["release_evidence_ids"]}
    acquired, completed, pending = set(), set(), {}
    productive = []
    eligible_evidence = []
    advances = []
    outcomes = Counter()
    last_index = max(t["sequence_index"] for t in transitions.values())
    for index, (entry, step) in enumerate(zip(entries, steps)):
        outcome = entry["public_tool_outcome"]
        assessment = step.get("action_assessment", {})
        advanced = outcome.get("state_advanced") is True
        evidence = outcome.get("newly_released_evidence_ids", [])
        matched = assessment.get("matched_transition_id")
        outcomes[outcome["status"]] += 1
        acquired.update(evidence)
        productive.append(advanced or bool(evidence))
        if advanced and matched:
            completed.add(matched)
            advances.append((index, matched))
        transition = transitions.get(matched, {})
        if evidence and not transition.get("terminal") and transition.get("sequence_index") != last_index:
            eligible_evidence.append(index)
        if outcome["status"] == "recognized_prerequisite_pending":
            signature = tuple(sorted(assessment.get("available_transition_ids_before", [])))
            if signature:
                pending.setdefault(signature, index)
    recovered = sum(any(i > first and transition in signature for i, transition in advances)
                    for signature, first in pending.items())
    followed = sum(any(i > first for i, _ in advances) for first in eligible_evidence)
    streak = longest = 0
    for flag in productive:
        streak = 0 if flag else streak + 1
        longest = max(streak, longest)
    budget = manifest["agent"]["max_actions"]
    if budget < len(entries):
        raise ValueError("submitted actions exceed configured budget")
    if not completed.issubset(transitions):
        raise ValueError("trace contains an unknown transition")
    from world_model_patient_ehr_v04.trajectory_evaluator import _semantic_repeat_flags
    repeats = sum(_semantic_repeat_flags(entries)["without_new_evidence"])
    return {"run_id": manifest["run_id"], "model_id": manifest["agent"]["requested_model"], "case_id": manifest["case_id"],
            "action_count": len(entries), "action_budget": budget,
            "unused_action_budget": budget - len(entries),
            "outcome_counts": dict(sorted(outcomes.items())),
            "outcome_shares": {k: ratio(v, len(entries)) for k, v in sorted(outcomes.items())},
            "milestones_completed": len(completed), "milestones_total": len(transitions),
            "milestone_coverage": ratio(len(completed), len(transitions)),
            "productive_actions": sum(productive),
            "productive_budget_yield": ratio(sum(productive), budget),
            "reference_evidence_recall": ratio(len(acquired & reference), len(reference)),
            "reference_evidence_acquired": len(acquired & reference), "reference_evidence_total": len(reference),
            "pending_targets_encountered": len(pending), "pending_targets_recovered": recovered,
            "pending_recovery_rate": ratio(recovered, len(pending)),
            "post_evidence_advance_rate": ratio(followed, len(eligible_evidence)),
            "no_new_evidence_repeat_rate": ratio(repeats, len(entries)),
            "longest_nonproductive_run_fraction": ratio(longest, len(entries)),
            "environment_terminal": manifest["run_summary"]["environment_terminal"],
            "terminal_reason": manifest["run_summary"]["terminal_reason"],
            "clinical_quality": None, "clinical_quality_source": "not provided"}


def aggregate(rows: list[dict]) -> list[dict]:
    """Macro-average one selected trajectory per model and case."""
    keys = ("milestone_coverage", "productive_budget_yield", "reference_evidence_recall",
            "pending_recovery_rate", "post_evidence_advance_rate", "no_new_evidence_repeat_rate",
            "longest_nonproductive_run_fraction", "clinical_quality")
    identities = [(row["model_id"], row["case_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate model/case trajectories; select a single final continuation")
    result = []
    for model in sorted({row["model_id"] for row in rows}):
        selected = [row for row in rows if row["model_id"] == model]
        counts = Counter()
        for row in selected:
            counts.update(row["outcome_counts"])
        total = sum(counts.values())
        result.append({"model_id": model, "case_count": len(selected), "executed_actions": total,
                       "outcome_shares": {k: ratio(v, total) for k, v in sorted(counts.items())},
                       "case_macro": {key: mean(values) if (values := [row[key] for row in selected if row[key] is not None]) else None for key in keys},
                       "valid_case_counts": {key: sum(row[key] is not None for row in selected) for key in keys}})
    return result
