"""Case-level longitudinal diagnostics from independent checkpoint predictions."""
from __future__ import annotations
from collections import defaultdict
from typing import Any
from .gold import GoldPackage
from .judge import FrozenJudge
from .longitudinal_atoms import build_gold_atom_index, build_model_observations

PRIMARY_METRICS = (
    "state_delta_immediate_response_rate", "linked_evidence_nonresponse_rate",
    "stopping_change_response_rate", "evidence_subsequent_match_rate",
    "evidence_regression_rate", "stopping_decision_regression_rate",
)

METRIC_META = {
    "state_delta_immediate_response_rate": {
        "direction": "higher",
        "readiness": "primary_ready",
        "scope": "checkpoint-local required State delta",
    },
    "linked_evidence_nonresponse_rate": {
        "direction": "lower",
        "readiness": "primary_ready",
        "scope": "same-checkpoint Gold-linked State/Critical Action",
    },
    "critical_action_checkpoint_response_rate": {
        "direction": "higher",
        "readiness": "primary_ready_checkpoint_not_longitudinal",
        "scope": "checkpoint-level critical-action Gold items",
    },
    "stopping_change_response_rate": {
        "direction": "higher",
        "readiness": "primary_ready",
        "scope": "Gold stopping-decision change segments",
    },
    "stopping_change_response_lag_mean": {
        "direction": "lower",
        "readiness": "primary_ready_conditional_on_response",
        "scope": "Gold stopping-decision change segments",
    },
    "evidence_subsequent_match_rate": {
        "direction": "higher",
        "readiness": "primary_ready",
        "scope": "stable Evidence ID + role",
    },
    "evidence_subsequent_match_lag_mean": {
        "direction": "lower",
        "readiness": "primary_ready_conditional_on_response",
        "scope": "stable Evidence ID + role",
    },
    "evidence_unresolved_after_followup_rate": {
        "direction": "lower",
        "readiness": "primary_ready_right_censored",
        "scope": "stable Evidence ID + role",
    },
    "evidence_regression_rate": {
        "direction": "lower",
        "readiness": "primary_ready",
        "scope": "stable Evidence ID + role",
    },
    "stopping_decision_regression_rate": {
        "direction": "lower",
        "readiness": "primary_ready",
        "scope": "stable stopping-decision category",
    },
    "claim_exact_text_subsequent_match_rate": {
        "direction": "higher",
        "readiness": "exploratory_exact_text_lower_bound",
        "scope": "Claim exact-text trackable subset",
    },
    "state_exact_text_subsequent_match_rate": {
        "direction": "higher",
        "readiness": "exploratory_exact_text_lower_bound",
        "scope": "State exact-text trackable subset",
    },
    "stopping_blocker_exact_text_subsequent_match_rate": {
        "direction": "higher",
        "readiness": "exploratory_exact_text_lower_bound",
        "scope": "Stopping-blocker exact-text trackable subset",
    },
    "claim_exact_text_regression_rate": {
        "direction": "lower",
        "readiness": "exploratory_exact_text_lower_bound",
        "scope": "Claim exact-text trackable subset",
    },
    "state_exact_text_regression_rate": {
        "direction": "lower",
        "readiness": "exploratory_exact_text_lower_bound",
        "scope": "State exact-text trackable subset",
    },
    "stopping_blocker_exact_text_regression_rate": {
        "direction": "lower",
        "readiness": "exploratory_exact_text_lower_bound",
        "scope": "Stopping-blocker exact-text trackable subset",
    },
}

def _derive_observed_issue_episodes(
    observations: list[dict[str, Any]],
    judge_policy: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[(row["model_id"], row["case_id"], row["atom_id"])].append(row)

    result: list[dict[str, Any]] = []
    for (model_id, case_id, item_id), rows in grouped.items():
        eligible = sorted(
            (row for row in rows if row["gold_state"] == "eligible"),
            key=lambda row: row["checkpoint_order"],
        )
        episode_number = 0
        open_episode: dict[str, Any] | None = None
        previous_eligible_order: int | None = None
        for eligible_rank, row in enumerate(eligible):
            if (
                open_episode is not None
                and previous_eligible_order is not None
                and row["checkpoint_order"] != previous_eligible_order + 1
            ):
                followup = open_episode["scorable_followup_count"]
                open_episode.update({
                    "subsequently_matched": 0,
                    "observed_recovery": 0,
                    "active_self_correction_inferred": 0,
                    "right_censored": 1,
                    "censor_reason": "gold_supersession",
                    "subsequent_match_checkpoint_id": "",
                    "subsequent_match_checkpoint_order": "",
                    "subsequent_match_lag_eligible_checkpoints": "",
                    "immediate_subsequent_match": 0,
                    "delayed_subsequent_match": 0,
                    "observed_followup_opportunity": int(followup > 0),
                    "unresolved_after_observed_followup": int(followup > 0),
                })
                result.append(open_episode)
                open_episode = None
            state = row["model_state"]
            if open_episode is not None and state in {"matched", "missed", "contradicted"}:
                open_episode["scorable_followup_count"] += 1
            if state in {"missed", "contradicted"} and open_episode is None:
                episode_number += 1
                open_episode = {
                    "issue_id": f"{judge_policy}|{model_id}|{case_id}|{item_id}|episode_{episode_number}",
                    "judge_policy": judge_policy,
                    "model_id": model_id,
                    "case_id": case_id,
                    "gold_item_id": item_id,
                    "dimension": row["dimension"],
                    "episode_number": episode_number,
                    "start_checkpoint_id": row["checkpoint_id"],
                    "start_checkpoint_order": row["checkpoint_order"],
                    "start_eligible_rank": eligible_rank,
                    "start_model_state": state,
                    "is_required_delta_miss": int(bool(row["is_first_required_delta"])),
                    "scorable_followup_count": 0,
                }
            elif state == "matched" and open_episode is not None:
                lag = eligible_rank - open_episode["start_eligible_rank"]
                open_episode.update({
                    "subsequently_matched": 1,
                    "observed_recovery": 1,
                    "active_self_correction_inferred": 0,
                    "right_censored": 0,
                    "censor_reason": "",
                    "subsequent_match_checkpoint_id": row["checkpoint_id"],
                    "subsequent_match_checkpoint_order": row["checkpoint_order"],
                    "subsequent_match_lag_eligible_checkpoints": lag,
                    "immediate_subsequent_match": int(lag == 1),
                    "delayed_subsequent_match": int(lag >= 2),
                    "observed_followup_opportunity": 1,
                    "unresolved_after_observed_followup": 0,
                })
                result.append(open_episode)
                open_episode = None
            previous_eligible_order = row["checkpoint_order"]
        if open_episode is not None:
            followup = open_episode["scorable_followup_count"]
            open_episode.update({
                "subsequently_matched": 0,
                "observed_recovery": 0,
                "active_self_correction_inferred": 0,
                "right_censored": 1,
                "censor_reason": "case_end_or_gold_supersession",
                "subsequent_match_checkpoint_id": "",
                "subsequent_match_checkpoint_order": "",
                "subsequent_match_lag_eligible_checkpoints": "",
                "immediate_subsequent_match": 0,
                "delayed_subsequent_match": 0,
                "observed_followup_opportunity": int(followup > 0),
                "unresolved_after_observed_followup": int(followup > 0),
            })
            result.append(open_episode)
    return sorted(
        result,
        key=lambda row: (
            row["judge_policy"], row["model_id"], row["case_id"],
            row["gold_item_id"], row["episode_number"],
        ),
    )

def _case_metric_rows(
    observations: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    judge_policy: str,
) -> list[dict[str, Any]]:
    obs_by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    episodes_by_case: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        obs_by_case[(row["model_id"], row["case_id"])].append(row)
    for row in episodes:
        episodes_by_case[(row["model_id"], row["case_id"])].append(row)

    result: list[dict[str, Any]] = []
    for (model_id, case_id), rows in sorted(obs_by_case.items()):
        case_episodes = episodes_by_case[(model_id, case_id)]
        state_delta = [
            row for row in rows
            if row["dimension"] == "state"
            and row["gold_state"] == "eligible"
            and row["is_first_required_delta"]
            and row["model_state"] != "unscorable"
        ]
        result.append(_metric_row(
            judge_policy, model_id, case_id,
            "state_delta_immediate_response_rate",
            sum(row["model_state"] == "matched" for row in state_delta),
            len(state_delta),
        ))

        evidence_targets = [
            row for row in rows
            if row["dimension"] in {"state", "critical_action"}
            and row["gold_state"] == "eligible"
            and row["evidence_response_opportunity"]
            and row["model_state"] != "unscorable"
        ]
        result.append(_metric_row(
            judge_policy, model_id, case_id,
            "linked_evidence_nonresponse_rate",
            sum(row["evidence_nonresponse"] for row in evidence_targets),
            len(evidence_targets),
        ))

        critical = [
            row for row in rows
            if row["dimension"] == "critical_action"
            and row["gold_state"] == "eligible"
            and row["model_state"] != "unscorable"
        ]
        result.append(_metric_row(
            judge_policy, model_id, case_id,
            "critical_action_checkpoint_response_rate",
            sum(row["model_state"] == "matched" for row in critical),
            len(critical),
        ))

        stopping = _stopping_change_metrics(rows)
        result.append(_metric_row(
            judge_policy, model_id, case_id,
            "stopping_change_response_rate",
            stopping["response_count"], stopping["scorable_change_segment_count"],
        ))
        result.append(_metric_row(
            judge_policy, model_id, case_id,
            "stopping_change_response_lag_mean",
            stopping["lag_sum"], stopping["response_count"],
        ))

        for dimension in ("evidence", "claim", "state", "stopping_blocker"):
            dimension_episodes = [row for row in case_episodes if row["dimension"] == dimension]
            opportunities = [
                row for row in dimension_episodes if row["observed_followup_opportunity"]
            ]
            prefix = "evidence" if dimension == "evidence" else f"{dimension}_exact_text"
            result.append(_metric_row(
                judge_policy, model_id, case_id,
                f"{prefix}_subsequent_match_rate",
                sum(row["subsequently_matched"] for row in opportunities),
                len(opportunities),
            ))
            if dimension == "evidence":
                matched_episodes = [row for row in opportunities if row["subsequently_matched"]]
                result.append(_metric_row(
                    judge_policy, model_id, case_id,
                    "evidence_subsequent_match_lag_mean",
                    sum(row["subsequent_match_lag_eligible_checkpoints"] for row in matched_episodes),
                    len(matched_episodes),
                ))
                result.append(_metric_row(
                    judge_policy, model_id, case_id,
                    "evidence_unresolved_after_followup_rate",
                    sum(row["unresolved_after_observed_followup"] for row in opportunities),
                    len(opportunities),
                ))

            regression_count, regression_denominator = _regression_counts(rows, dimension)
            result.append(_metric_row(
                judge_policy, model_id, case_id,
                f"{prefix}_regression_rate",
                regression_count, regression_denominator,
            ))

        stopping_regression_count, stopping_regression_denominator = _regression_counts(
            rows, "stopping_decision"
        )
        result.append(_metric_row(
            judge_policy, model_id, case_id,
            "stopping_decision_regression_rate",
            stopping_regression_count, stopping_regression_denominator,
        ))
    return [row for row in result if row["metric"] in METRIC_META]

def _metric_row(
    judge_policy: str,
    model_id: str,
    case_id: str,
    metric: str,
    numerator: int | float,
    denominator: int | float,
) -> dict[str, Any]:
    meta = METRIC_META[metric]
    return {
        "judge_policy": judge_policy,
        "model_id": model_id,
        "case_id": case_id,
        "metric": metric,
        "direction": meta["direction"],
        "readiness": meta["readiness"],
        "scope": meta["scope"],
        "numerator": numerator,
        "denominator": denominator,
        "value": "" if denominator == 0 else numerator / denominator,
    }

def _stopping_change_metrics(rows: list[dict[str, Any]]) -> dict[str, int]:
    by_checkpoint: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row["dimension"] == "stopping_decision" and row["gold_state"] == "eligible":
            by_checkpoint[row["checkpoint_order"]] = row
    ordered = [by_checkpoint[key] for key in sorted(by_checkpoint)]
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_item = None
    for row in ordered:
        if current_item is None or row["atom_id"] == current_item:
            current.append(row)
        else:
            segments.append(current)
            current = [row]
        current_item = row["atom_id"]
    if current:
        segments.append(current)

    scorable_segments = 0
    response_count = 0
    lag_sum = 0
    unscorable_only = 0
    for segment in segments[1:]:
        scorable = [row for row in segment if row["model_state"] != "unscorable"]
        if not scorable:
            unscorable_only += 1
            continue
        scorable_segments += 1
        first_match = next(
            (index for index, row in enumerate(segment) if row["model_state"] == "matched"),
            None,
        )
        if first_match is not None:
            response_count += 1
            lag_sum += first_match
    return {
        "change_segment_count": max(0, len(segments) - 1),
        "scorable_change_segment_count": scorable_segments,
        "unscorable_only_change_segment_count": unscorable_only,
        "response_count": response_count,
        "lag_sum": lag_sum,
    }

def _regression_counts(
    rows: list[dict[str, Any]],
    dimension: str,
) -> tuple[int, int]:
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["dimension"] == dimension and row["gold_state"] == "eligible":
            by_item[row["atom_id"]].append(row)
    numerator = 0
    denominator = 0
    for item_rows in by_item.values():
        item_rows.sort(key=lambda row: row["checkpoint_order"])
        for current, following in zip(item_rows, item_rows[1:]):
            if current["model_state"] != "matched":
                continue
            if following["model_state"] == "unscorable":
                continue
            denominator += 1
            numerator += int(following["model_state"] in {"missed", "contradicted"})
    return numerator, denominator

def evaluate_temporal(gold: GoldPackage, submissions: list[dict[str, Any]], checkpoint_scores: list[dict[str, Any]], *, judge: FrozenJudge, model_id: str = "model") -> dict[str, Any]:
    """Return six case-macro metrics and auditable case numerators/denominators.

    Invalid or missing outputs are unscorable here. Zero-denominator cases are
    excluded from each metric, while checkpoint evaluation scores them as empty.
    """
    index = build_gold_atom_index(gold)
    # The scorer has already resolved duplicates and malformed/missing records.
    by_id = {row.get("checkpoint_id"): row for row in submissions if isinstance(row, dict)}
    rows = [by_id.get(cp_id, {"checkpoint_id": cp_id}) for cp_id in gold.checkpoint_by_id]
    observations, errors, reconciliation = build_model_observations(
        gold, index, model_id=model_id, submissions=rows,
        checkpoint_scores=checkpoint_scores, judge=judge,
    )
    episodes = _derive_observed_issue_episodes(observations, "frozen_judge")
    cases = [row for row in _case_metric_rows(observations, episodes, "frozen_judge") if row["metric"] in PRIMARY_METRICS]
    summary = {}
    for metric in PRIMARY_METRICS:
        eligible = [row["value"] for row in cases if row["metric"] == metric and row["denominator"] > 0]
        summary[metric] = {
            "case_macro_value": sum(eligible) / len(eligible) if eligible else None,
            "contributing_case_count": len(eligible),
            "total_case_count": len(gold.cases),
            "direction": METRIC_META[metric]["direction"],
        }
    return {"schema_version": "ward-temporal-metrics-1.0", "model_id": model_id,
            "metrics": summary, "case_metrics": cases, "atom_audit": index.audit,
            "reconciliation": reconciliation, "prediction_error_count": len(errors)}
