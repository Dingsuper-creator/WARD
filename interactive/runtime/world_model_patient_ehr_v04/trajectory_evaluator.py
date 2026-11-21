"""Deterministic, non-official evaluation of Patient/EHR Agent trajectories."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from world_model_patient_ehr_v01.contracts import (
    InteractionContractError,
    credential_findings,
    read_json,
    sha256_file,
    write_json,
)

from world_model_patient_ehr_v04.case_registry import (
    CURRENT_ROOT,
    DEFAULT_CASE_REGISTRY,
    load_case_bundle,
)
from world_model_patient_ehr_v04.tool_facade import semantic_target_fingerprint


EVALUATION_VERSION = "ward-patient-ehr-trajectory-evaluation-0.1.0"
REVIEW_QUEUE_VERSION = "ward-patient-ehr-human-review-queue-0.1.0"
NONADVANCING_COOLDOWN_STATUSES = {
    "recognized_completed_action",
    "recognized_prerequisite_pending",
    "valid_exploration_no_recorded_outcome",
}


def _ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": (
            round(numerator / denominator, 6)
            if denominator
            else None
        ),
    }


def _safe_divide(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _max_no_progress_streak(entries: list[dict[str, Any]]) -> int:
    maximum = 0
    current = 0
    for entry in entries:
        advanced = (
            entry.get("public_tool_outcome", {}).get("state_advanced") is True
        )
        if advanced:
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def _semantic_repeat_flags(
    entries: list[dict[str, Any]],
) -> dict[str, list[bool]]:
    last_generation: dict[str, int] = {}
    generation = 0
    strict_repeat_flags: list[bool] = []
    revisit_flags: list[bool] = []
    for entry in entries:
        tool_call = entry.get("tool_call")
        outcome = entry.get("public_tool_outcome", {})
        if not isinstance(tool_call, dict) or not isinstance(outcome, dict):
            strict_repeat_flags.append(False)
            revisit_flags.append(False)
            continue
        fingerprint = semantic_target_fingerprint(tool_call)
        previous = last_generation.get(fingerprint)
        stored_repeat = outcome.get("semantic_repeat_without_new_evidence")
        stored_revisit = outcome.get("semantic_revisit_after_new_evidence")
        repeated = (
            stored_repeat
            if isinstance(stored_repeat, bool)
            else previous == generation
        )
        revisited = (
            stored_revisit
            if isinstance(stored_revisit, bool)
            else previous is not None and previous < generation
        )
        strict_repeat_flags.append(repeated is True)
        revisit_flags.append(revisited is True)
        last_generation[fingerprint] = generation
        if outcome.get("state_advanced") is True:
            generation += 1
    return {
        "without_new_evidence": strict_repeat_flags,
        "after_new_evidence": revisit_flags,
    }


def _family_counts(entries: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for entry in entries:
        outcome = entry.get("public_tool_outcome", {})
        family = str(outcome.get("tool_family") or "unknown")
        status = str(outcome.get("status") or "unknown")
        bucket = result.setdefault(
            family,
            {"submitted": 0, "advanced": 0, "nonadvancing": 0},
        )
        bucket["submitted"] += 1
        if outcome.get("state_advanced") is True:
            bucket["advanced"] += 1
        else:
            bucket["nonadvancing"] += 1
        bucket[status] = bucket.get(status, 0) + 1
    return dict(sorted(result.items()))


def _review_queue(
    *,
    manifest: dict[str, Any],
    entries: list[dict[str, Any]],
    max_no_progress_streak: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    invalid_grounding = [
        entry.get("tool_attempt_index")
        for entry in entries
        if entry.get("public_tool_outcome", {}).get("status")
        == "invalid_evidence_grounding"
    ]
    if invalid_grounding:
        items.append(
            {
                "review_type": "evidence_grounding",
                "priority": "high",
                "tool_attempt_indices": invalid_grounding,
                "reason": "One or more tool calls cited evidence that was not visible.",
            }
        )
    critical_slot = [
        entry.get("tool_attempt_index")
        for entry in entries
        if entry.get("critical_slot_guard_triggered") is True
    ]
    if critical_slot:
        items.append(
            {
                "review_type": "critical_slot_conflict",
                "priority": "high",
                "tool_attempt_indices": critical_slot,
                "reason": "A deterministic critical-slot guard prevented a conflicting action.",
            }
        )
    summary = manifest.get("run_summary", {})
    if (
        summary.get("environment_terminal") is not True
        and max_no_progress_streak >= 3
    ):
        items.append(
            {
                "review_type": "environment_observability_or_agent_strategy",
                "priority": "medium",
                "tool_attempt_indices": [],
                "reason": (
                    "The incomplete episode contains at least three consecutive "
                    "non-advancing calls. Review rendered evidence conclusions "
                    "before attributing the failure to the Agent."
                ),
            }
        )
    if summary.get("terminal_reason") == "api_error":
        items.append(
            {
                "review_type": "transport_failure",
                "priority": "high",
                "tool_attempt_indices": [],
                "reason": "The episode ended because of an external API error.",
            }
        )
    return items


def evaluate_run(
    *,
    run_dir: Path,
    case_registry_path: Path = DEFAULT_CASE_REGISTRY,
    workspace_root: Path = CURRENT_ROOT,
    write_outputs: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = {
        "manifest": run_dir / "run_manifest.json",
        "interaction_trace": run_dir / "interaction_trace.json",
        "tool_trace": run_dir / "tool_trace.json",
        "call_audit": run_dir / "agent_call_audit.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise InteractionContractError(
            "run artifacts are missing: " + ", ".join(sorted(missing))
        )
    manifest = read_json(paths["manifest"])
    interaction_trace = read_json(paths["interaction_trace"])
    tool_trace = read_json(paths["tool_trace"])
    audit = read_json(paths["call_audit"])
    run_id = str(manifest.get("run_id") or "")
    case_id = str(manifest.get("case_id") or "")
    if not run_id or not case_id:
        raise InteractionContractError("run manifest lacks run_id or case_id")
    for name, artifact in (
        ("interaction_trace", interaction_trace),
        ("tool_trace", tool_trace),
        ("call_audit", audit),
    ):
        if artifact.get("run_id") != run_id or artifact.get("case_id") != case_id:
            raise InteractionContractError(f"{name} run or case binding mismatch")
    bundle = load_case_bundle(
        case_id=case_id,
        registry_path=case_registry_path,
        workspace_root=workspace_root,
    )
    scenario_path = bundle["artifacts"]["scenario"]
    scenario = read_json(scenario_path)
    if manifest.get("bindings", {}).get("scenario_sha256") != sha256_file(
        scenario_path
    ):
        raise InteractionContractError("scenario binding mismatch")

    entries = [
        copy.deepcopy(item)
        for item in tool_trace.get("entries", [])
        if isinstance(item, dict)
    ]
    calls = [
        copy.deepcopy(item)
        for item in audit.get("calls", [])
        if isinstance(item, dict)
    ]
    submitted = len(entries)
    invalid_contract = sum(
        1 for item in calls if item.get("status") == "invalid_tool_call_contract"
    )
    model_calls = int(audit.get("model_call_count") or len(calls))
    advanced = sum(
        1
        for item in entries
        if item.get("public_tool_outcome", {}).get("state_advanced") is True
    )
    outcome_counts: dict[str, int] = {}
    for entry in entries:
        status = str(entry.get("public_tool_outcome", {}).get("status") or "unknown")
        outcome_counts[status] = outcome_counts.get(status, 0) + 1
    semantic_flags = _semantic_repeat_flags(entries)
    semantic_repeat_count = sum(semantic_flags["without_new_evidence"])
    semantic_revisit_count = sum(semantic_flags["after_new_evidence"])
    max_no_progress_streak = _max_no_progress_streak(entries)
    usage = audit.get("usage_totals", {})
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
    completed_transitions = len(
        interaction_trace.get("final_state", {}).get("completed_transition_ids", [])
    )
    expected_transitions = len(scenario.get("transitions", []))
    critical_slot_conflicts = sum(
        1 for item in entries if item.get("critical_slot_guard_triggered") is True
    )
    invalid_grounding = outcome_counts.get("invalid_evidence_grounding", 0)

    evaluation = {
        "evaluation_version": EVALUATION_VERSION,
        "status": "diagnostic_non_official",
        "run_id": run_id,
        "case_id": case_id,
        "tool_execution": {
            "model_call_count": model_calls,
            "submitted_tool_call_count": submitted,
            "invalid_tool_contract_count": invalid_contract,
            "schema_valid_call_rate": _ratio(
                submitted,
                submitted + invalid_contract,
            ),
            "tool_submission_per_model_call": _ratio(submitted, model_calls),
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "tool_family_breakdown": _family_counts(entries),
        },
        "clinical_trajectory": {
            "final_stage": interaction_trace.get("final_state", {}).get("stage"),
            "environment_terminal": interaction_trace.get("final_state", {}).get(
                "done"
            ),
            "terminal_reason": manifest.get("run_summary", {}).get(
                "terminal_reason"
            ),
            "completed_transition_count": completed_transitions,
            "expected_recorded_transition_count": expected_transitions,
            "recorded_path_coverage": _ratio(
                completed_transitions,
                expected_transitions,
            ),
            "advanced_tool_call_count": advanced,
            "nonadvancing_tool_call_count": submitted - advanced,
            "semantic_repeat_without_new_evidence_count": semantic_repeat_count,
            "semantic_revisit_after_new_evidence_count": semantic_revisit_count,
            "max_consecutive_nonadvancing_tool_calls": max_no_progress_streak,
        },
        "efficiency": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "model_calls_per_completed_transition": _safe_divide(
                model_calls,
                completed_transitions,
            ),
            "submitted_tools_per_completed_transition": _safe_divide(
                submitted,
                completed_transitions,
            ),
            "tokens_per_completed_transition": _safe_divide(
                total_tokens,
                completed_transitions,
            ),
        },
        "safety_and_grounding": {
            "invalid_evidence_grounding_count": invalid_grounding,
            "critical_slot_conflict_count": critical_slot_conflicts,
            "hallucinated_result_assessment": "not_automatically_inferred",
            "critical_omission_assessment": "requires_triggered_review",
        },
        "environment_assessment": {
            "deterministic_patient_simulator": manifest.get(
                "patient_simulator", {}
            ).get("facts_deterministic"),
            "ehr_state_authoritative": manifest.get("patient_simulator", {}).get(
                "ehr_state_authoritative"
            ),
            "prompt_render_observability": (
                "review_required"
                if manifest.get("run_summary", {}).get("environment_terminal")
                is not True
                and max_no_progress_streak >= 3
                else "no_automatic_flag"
            ),
            "note": (
                "This evaluator does not infer clinical observability from hidden "
                "Gold transitions. Review deterministic rendered evidence when "
                "the observability flag is raised."
            ),
        },
        "bindings": {
            "run_manifest_sha256": sha256_file(paths["manifest"]),
            "interaction_trace_sha256": sha256_file(paths["interaction_trace"]),
            "tool_trace_sha256": sha256_file(paths["tool_trace"]),
            "agent_call_audit_sha256": sha256_file(paths["call_audit"]),
            "scenario_sha256": sha256_file(scenario_path),
            "case_registry_sha256": sha256_file(case_registry_path),
        },
        "automatic_semantic_judge_scoring": False,
        "official_scoring_allowed": False,
        "single_total_score_absent": True,
        "credentials_echoed": False,
    }
    queue_items = _review_queue(
        manifest=manifest,
        entries=entries,
        max_no_progress_streak=max_no_progress_streak,
    )
    review_queue = {
        "review_queue_version": REVIEW_QUEUE_VERSION,
        "status": "diagnostic_non_official",
        "run_id": run_id,
        "case_id": case_id,
        "review_item_count": len(queue_items),
        "items": queue_items,
        "routine_actions_require_review": False,
        "automatic_semantic_judge_scoring": False,
        "official_scoring_allowed": False,
        "single_total_score_absent": True,
        "credentials_echoed": False,
    }
    if credential_findings(evaluation) or credential_findings(review_queue):
        raise InteractionContractError(
            "trajectory evaluation contains credential-like material"
        )
    if write_outputs:
        write_json(run_dir / "trajectory_evaluation.json", evaluation)
        write_json(run_dir / "human_review_queue.json", review_queue)
    return evaluation, review_queue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--case-registry",
        type=Path,
        default=DEFAULT_CASE_REGISTRY,
    )
    args = parser.parse_args()
    evaluation, review_queue = evaluate_run(
        run_dir=args.run_dir,
        case_registry_path=args.case_registry,
    )
    print(
        json.dumps(
            {
            "status": evaluation["status"],
            "run_id": evaluation["run_id"],
            "case_id": evaluation["case_id"],
            "completed_transition_count": evaluation["clinical_trajectory"][
                "completed_transition_count"
            ],
            "final_stage": evaluation["clinical_trajectory"]["final_stage"],
            "review_item_count": review_queue["review_item_count"],
            "official_scoring_allowed": False,
            "single_total_score_absent": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
