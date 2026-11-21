from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from .contracts import SINGLE_STATE_BUCKETS, STATE_BUCKETS
from .gold import GoldPackage
from .judge import FrozenJudge
from .matcher import SemanticMatchResult, semantic_match
from .util import normalize_text


ATOM_DIMENSIONS = (
    "evidence",
    "claim",
    "state",
    "critical_action",
    "stopping_blocker",
    "stopping_decision",
)


@dataclass
class GoldAtomIndex:
    registry: list[dict[str, Any]]
    checkpoint_items: dict[str, list[dict[str, Any]]]
    source_to_atom: dict[tuple[str, str, str], str]
    audit: dict[str, Any]


def stable_atom_id(case_id: str, dimension: str, identity: str) -> str:
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"{case_id}|{dimension}|{digest}"


def build_gold_atom_index(gold: GoldPackage) -> GoldAtomIndex:
    """Create a deterministic, read-only sidecar over checkpoint-scoped Gold IDs."""
    checkpoint_items: dict[str, list[dict[str, Any]]] = {}
    source_to_atom: dict[tuple[str, str, str], str] = {}
    registry_acc: dict[str, dict[str, Any]] = {}
    collision_rows: list[dict[str, Any]] = []
    source_item_count = 0

    for case in gold.cases:
        case_id = case["case_id"]
        ordered = sorted(case["checkpoints"], key=lambda row: row["sequence_index"])
        state_collision_keys = _state_collision_keys(gold, ordered)
        for checkpoint_order, checkpoint in enumerate(ordered):
            checkpoint_id = checkpoint["checkpoint_id"]
            projection = gold.project_checkpoint_en(gold.checkpoint_by_id[checkpoint_id])
            raw_items = _project_atom_items(
                case_id,
                checkpoint_id,
                checkpoint_order,
                projection,
                state_collision_keys,
            )
            checkpoint_items[checkpoint_id] = raw_items
            seen_atom_ids: set[str] = set()
            for item in raw_items:
                source_item_count += 1
                atom_id = item["atom_id"]
                if atom_id in seen_atom_ids:
                    raise ValueError(f"duplicate atom at {checkpoint_id}: {atom_id}")
                seen_atom_ids.add(atom_id)
                source_to_atom[(checkpoint_id, item["dimension"], item["source_gold_id"])] = atom_id
                current = registry_acc.setdefault(atom_id, {
                    "atom_id": atom_id,
                    "case_id": case_id,
                    "dimension": item["dimension"],
                    "canonical_text": item["text"],
                    "normalized_identity": item["identity"],
                    "identity_basis": item["identity_basis"],
                    "first_eligible_order": checkpoint_order,
                    "last_eligible_order": checkpoint_order,
                    "eligible_checkpoint_orders": [],
                    "eligible_checkpoint_ids": [],
                    "source_gold_ids": [],
                    "state_buckets": [],
                    "linked_evidence_atom_ids": [],
                    "first_required_delta_order": None,
                    "is_critical_action": item["dimension"] == "critical_action",
                })
                current["first_eligible_order"] = min(current["first_eligible_order"], checkpoint_order)
                current["last_eligible_order"] = max(current["last_eligible_order"], checkpoint_order)
                current["eligible_checkpoint_orders"].append(checkpoint_order)
                current["eligible_checkpoint_ids"].append(checkpoint_id)
                current["source_gold_ids"].append(item["source_gold_id"])
                if item.get("state_bucket"):
                    current["state_buckets"].append(item["state_bucket"])
                if item.get("is_required_delta") and current["first_required_delta_order"] is None:
                    current["first_required_delta_order"] = checkpoint_order

        _attach_evidence_links(case_id, ordered, checkpoint_items, source_to_atom)

        for checkpoint in ordered:
            for item in checkpoint_items[checkpoint["checkpoint_id"]]:
                if item.get("collision_disambiguated"):
                    collision_rows.append({
                        "case_id": case_id,
                        "checkpoint_id": checkpoint["checkpoint_id"],
                        "source_gold_id": item["source_gold_id"],
                        "state_bucket": item.get("state_bucket"),
                        "normalized_text": normalize_text(item["text"]),
                        "atom_id": item["atom_id"],
                    })

    for checkpoint_id, items in checkpoint_items.items():
        for item in items:
            current = registry_acc[item["atom_id"]]
            current["linked_evidence_atom_ids"].extend(item.get("linked_evidence_atom_ids", []))

    registry: list[dict[str, Any]] = []
    for atom_id in sorted(registry_acc):
        row = registry_acc[atom_id]
        for field in (
            "eligible_checkpoint_orders",
            "eligible_checkpoint_ids",
            "source_gold_ids",
            "state_buckets",
            "linked_evidence_atom_ids",
        ):
            row[field] = sorted(set(row[field]))
        row["eligible_checkpoint_count"] = len(row["eligible_checkpoint_ids"])
        registry.append(row)

    return GoldAtomIndex(
        registry=registry,
        checkpoint_items=checkpoint_items,
        source_to_atom=source_to_atom,
        audit={
            "source_item_count": source_item_count,
            "stable_atom_count": len(registry),
            "checkpoint_count": len(checkpoint_items),
            "state_collision_source_item_count": len(collision_rows),
            "state_collision_rows": collision_rows,
            "identity_policy": {
                "evidence": "case_id + evidence_id + role",
                "text_atoms": "case_id + dimension + normalized canonical text",
                "state_collision": (
                    "when normalized state text is duplicated within any checkpoint in a case, "
                    "state bucket and within-bucket occurrence index are added"
                ),
                "gold_mutated": False,
            },
        },
    )


def build_model_observations(
    gold: GoldPackage,
    atom_index: GoldAtomIndex,
    *,
    model_id: str,
    submissions: list[dict[str, Any]],
    checkpoint_scores: list[dict[str, Any]],
    judge: FrozenJudge,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Expand one frozen run into atom observations, prediction errors, and reconciliation rows."""
    submission_by_checkpoint = {row["checkpoint_id"]: row for row in submissions}
    score_by_checkpoint = {row["checkpoint_id"]: row for row in checkpoint_scores}
    registry_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    registry_by_id = {row["atom_id"]: row for row in atom_index.registry}
    for row in atom_index.registry:
        registry_by_case[row["case_id"]].append(row)

    observations: list[dict[str, Any]] = []
    prediction_errors: list[dict[str, Any]] = []
    reconciliation: list[dict[str, Any]] = []

    for case in gold.cases:
        case_id = case["case_id"]
        ordered = sorted(case["checkpoints"], key=lambda row: row["sequence_index"])
        case_atoms = registry_by_case[case_id]
        for checkpoint_order, checkpoint in enumerate(ordered):
            checkpoint_id = checkpoint["checkpoint_id"]
            score = score_by_checkpoint[checkpoint_id]
            valid = bool(score["valid_submission"])
            record = submission_by_checkpoint[checkpoint_id]
            current_items = atom_index.checkpoint_items[checkpoint_id]
            current_by_atom = {item["atom_id"]: item for item in current_items}
            if valid:
                states, errors = _evaluate_valid_checkpoint(
                    gold,
                    checkpoint_id,
                    record,
                    atom_index,
                    judge,
                    model_id,
                    checkpoint_order,
                )
                prediction_errors.extend(errors)
            else:
                states = {item["atom_id"]: "unscorable" for item in current_items}

            cp_observations: list[dict[str, Any]] = []
            for registry_row in case_atoms:
                atom_id = registry_row["atom_id"]
                if atom_id in current_by_atom:
                    gold_state = "eligible"
                    item = current_by_atom[atom_id]
                    model_state = states[atom_id]
                    stale_carryover = 0
                elif checkpoint_order < registry_row["first_eligible_order"]:
                    gold_state = "not_yet_eligible"
                    item = None
                    model_state = None
                    stale_carryover = 0
                else:
                    gold_state = "superseded"
                    item = None
                    model_state = None
                    stale_carryover = int(
                        valid
                        and _is_stale_carryover(registry_row, record, current_items)
                    )
                observation = {
                    "model_id": model_id,
                    "case_id": case_id,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_order": checkpoint_order,
                    "atom_id": atom_id,
                    "dimension": registry_row["dimension"],
                    "gold_state": gold_state,
                    "model_state": model_state,
                    "is_required_delta": int(bool(item and item.get("is_required_delta"))),
                    "is_first_required_delta": int(
                        bool(
                            item
                            and item.get("is_required_delta")
                            and registry_row["first_required_delta_order"] == checkpoint_order
                        )
                    ),
                    "linked_evidence_atom_ids": (
                        item.get("linked_evidence_atom_ids", []) if item else []
                    ),
                    "evidence_response_opportunity": 0,
                    "evidence_nonresponse": 0,
                    "stale_carryover": stale_carryover,
                    "regression_event": 0,
                    "oscillation_atom": 0,
                    "valid_submission": valid,
                    "submission_status": score["submission_status"],
                }
                cp_observations.append(observation)
            _mark_evidence_nonresponse(cp_observations)
            observations.extend(cp_observations)
            reconciliation.extend(_reconcile_checkpoint(cp_observations, score))

    _annotate_transitions(observations)
    return observations, prediction_errors, reconciliation


def derive_issue_episodes(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[(row["model_id"], row["case_id"], row["atom_id"])].append(row)

    episodes: list[dict[str, Any]] = []
    for (model_id, case_id, atom_id), rows in grouped.items():
        rows.sort(key=lambda row: row["checkpoint_order"])
        eligible_indexes = [i for i, row in enumerate(rows) if row["gold_state"] == "eligible"]
        eligible_rank = {row_index: rank for rank, row_index in enumerate(eligible_indexes)}
        episode_number = 0
        open_episode: dict[str, Any] | None = None
        previous_gold_state: str | None = None
        for row_index, row in enumerate(rows):
            if (
                open_episode is not None
                and previous_gold_state == "eligible"
                and row["gold_state"] != "eligible"
            ):
                open_episode.update({
                    "corrected": 0,
                    "right_censored": 1,
                    "persistent_error": 0,
                    "censor_reason": "superseded",
                    "correction_checkpoint_id": None,
                    "correction_checkpoint_order": None,
                    "correction_lag": None,
                    "immediate_correction": 0,
                    "delayed_correction": 0,
                    "eligible_followup_count": (
                        open_episode["last_eligible_rank"] - open_episode["start_eligible_rank"]
                    ),
                    "correction_opportunity": int(
                        open_episode["last_eligible_rank"] > open_episode["start_eligible_rank"]
                    ),
                })
                episodes.append(open_episode)
                open_episode = None

            model_state = row["model_state"]
            if open_episode is not None and row["gold_state"] == "eligible":
                open_episode["last_eligible_rank"] = eligible_rank[row_index]
            is_error = row["gold_state"] == "eligible" and model_state in {"missed", "contradicted"}
            if is_error and open_episode is None:
                episode_number += 1
                open_episode = {
                    "issue_id": f"{model_id}|{case_id}|{atom_id}|episode_{episode_number}",
                    "model_id": model_id,
                    "case_id": case_id,
                    "atom_id": atom_id,
                    "dimension": row["dimension"],
                    "episode_number": episode_number,
                    "start_checkpoint_id": row["checkpoint_id"],
                    "start_checkpoint_order": row["checkpoint_order"],
                    "start_eligible_rank": eligible_rank[row_index],
                    "last_eligible_rank": eligible_rank[row_index],
                    "initial_deficit": int(eligible_rank[row_index] == 0),
                    "delta_update_miss": int(bool(row["is_first_required_delta"])),
                    "start_model_state": model_state,
                }
            elif model_state == "matched" and open_episode is not None:
                lag = eligible_rank[row_index] - open_episode["start_eligible_rank"]
                open_episode.update({
                    "corrected": 1,
                    "right_censored": 0,
                    "persistent_error": 0,
                    "censor_reason": None,
                    "correction_checkpoint_id": row["checkpoint_id"],
                    "correction_checkpoint_order": row["checkpoint_order"],
                    "correction_lag": lag,
                    "immediate_correction": int(lag == 1),
                    "delayed_correction": int(lag >= 2),
                    "eligible_followup_count": lag,
                    "correction_opportunity": 1,
                })
                episodes.append(open_episode)
                open_episode = None
            previous_gold_state = row["gold_state"]

        if open_episode is not None:
            persistent = int(rows[-1]["gold_state"] == "eligible")
            open_episode.update({
                "corrected": 0,
                "right_censored": 1,
                "persistent_error": persistent,
                "censor_reason": "case_end" if persistent else "superseded",
                "correction_checkpoint_id": None,
                "correction_checkpoint_order": None,
                "correction_lag": None,
                "immediate_correction": 0,
                "delayed_correction": 0,
                "eligible_followup_count": (
                    open_episode["last_eligible_rank"] - open_episode["start_eligible_rank"]
                ),
                "correction_opportunity": int(
                    open_episode["last_eligible_rank"] > open_episode["start_eligible_rank"]
                ),
            })
            episodes.append(open_episode)
    return sorted(episodes, key=lambda row: (row["model_id"], row["case_id"], row["atom_id"], row["episode_number"]))


def summarize_model_cases(
    observations: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    obs_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    episode_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        obs_groups[(row["model_id"], row["case_id"])].append(row)
    for row in episodes:
        episode_groups[(row["model_id"], row["case_id"])].append(row)

    summaries: list[dict[str, Any]] = []
    for key in sorted(obs_groups):
        model_id, case_id = key
        rows = obs_groups[key]
        issue_rows = episode_groups.get(key, [])
        correction_opportunities = [row for row in issue_rows if row["correction_opportunity"]]
        delta_rows = [row for row in rows if row["is_first_required_delta"]]
        regression_denominator = _regression_denominator(rows)
        adjustment = _adjustment_lags(rows)
        evidence_targets = [
            row for row in rows
            if row["gold_state"] == "eligible"
            and row["dimension"] in {"state", "critical_action"}
            and row["evidence_response_opportunity"]
            and row["model_state"] != "unscorable"
        ]
        final_by_atom: dict[str, dict[str, Any]] = {}
        for row in sorted(rows, key=lambda value: value["checkpoint_order"]):
            final_by_atom[row["atom_id"]] = row
        issue_atom_ids = {row["atom_id"] for row in issue_rows}
        persistent_denominator = sum(
            final_by_atom[atom_id]["gold_state"] == "eligible"
            for atom_id in issue_atom_ids
        )
        summary = {
            "model_id": model_id,
            "case_id": case_id,
            "checkpoint_count": len({row["checkpoint_id"] for row in rows}),
            "unscorable_checkpoint_count": len({
                row["checkpoint_id"] for row in rows if not row["valid_submission"]
            }),
            "delta_response_numerator": sum(row["model_state"] == "matched" for row in delta_rows),
            "delta_response_denominator": len(delta_rows),
            "issue_episode_count": len(issue_rows),
            "correction_opportunity_count": len(correction_opportunities),
            "correction_count": sum(row["corrected"] for row in issue_rows),
            "immediate_correction_count": sum(row["immediate_correction"] for row in issue_rows),
            "delayed_correction_count": sum(row["delayed_correction"] for row in issue_rows),
            "persistent_error_count": sum(row["persistent_error"] for row in issue_rows),
            "persistent_error_denominator": persistent_denominator,
            "right_censored_issue_count": sum(row["right_censored"] for row in issue_rows),
            "correction_lag_sum": sum(row["correction_lag"] for row in issue_rows if row["corrected"]),
            "correction_lag_denominator": sum(row["corrected"] for row in issue_rows),
            "regression_count": sum(row["regression_event"] for row in rows),
            "regression_denominator": regression_denominator,
            "oscillating_atom_count": len({row["atom_id"] for row in rows if row["oscillation_atom"]}),
            "stale_carryover_count": sum(row["stale_carryover"] for row in rows),
            "evidence_nonresponse_count": sum(row["evidence_nonresponse"] for row in evidence_targets),
            "evidence_nonresponse_denominator": len(evidence_targets),
            **adjustment,
        }
        for metric, numerator, denominator in (
            ("delta_response_rate", "delta_response_numerator", "delta_response_denominator"),
            ("correction_rate", "correction_count", "correction_opportunity_count"),
            ("immediate_correction_rate", "immediate_correction_count", "correction_opportunity_count"),
            ("delayed_correction_rate", "delayed_correction_count", "correction_opportunity_count"),
            ("persistent_error_rate", "persistent_error_count", "persistent_error_denominator"),
            ("regression_rate", "regression_count", "regression_denominator"),
            ("evidence_nonresponse_rate", "evidence_nonresponse_count", "evidence_nonresponse_denominator"),
        ):
            summary[metric] = _ratio(summary[numerator], summary[denominator])
        summary["correction_lag_mean"] = _ratio(
            summary["correction_lag_sum"], summary["correction_lag_denominator"]
        )
        for prefix in ("critical_action_adjustment", "stopping_adjustment"):
            summary[f"{prefix}_response_rate"] = _ratio(
                summary[f"{prefix}_lag_denominator"], summary[f"{prefix}_episode_count"]
            )
        summaries.append(summary)
    return summaries


def _project_atom_items(
    case_id: str,
    checkpoint_id: str,
    checkpoint_order: int,
    projection: dict[str, Any],
    state_collision_keys: set[str],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    def add(
        dimension: str,
        source_gold_id: str,
        text: str,
        aliases: list[str],
        identity: str,
        identity_basis: str,
        **extra: Any,
    ) -> None:
        items.append({
            "atom_id": stable_atom_id(case_id, dimension, identity),
            "case_id": case_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_order": checkpoint_order,
            "dimension": dimension,
            "source_gold_id": source_gold_id,
            "text": text,
            "aliases": aliases,
            "identity": identity,
            "identity_basis": identity_basis,
            **extra,
        })

    for item in projection["layer1"]:
        identity = f'{item["evidence_id"]}|{item["role"]}'
        add(
            "evidence",
            identity,
            item["evidence_id"],
            [],
            identity,
            "evidence_id+role",
            evidence_id=item["evidence_id"],
            evidence_role=item["role"],
        )
    for item in projection["layer3"]:
        normalized = normalize_text(item["text"])
        add(
            "claim", item["id"], item["text"], item.get("aliases", []), normalized,
            "normalized_canonical_en", evidence_ids=item.get("evidence_ids", []),
        )

    bucket_occurrence: Counter[tuple[str, str]] = Counter()
    for bucket in (*STATE_BUCKETS, *SINGLE_STATE_BUCKETS):
        values = projection["layer4"][bucket]
        values = values if isinstance(values, list) else ([values] if values else [])
        for item in values:
            normalized = normalize_text(item["text"])
            bucket_occurrence[(bucket, normalized)] += 1
            collision = normalized in state_collision_keys
            identity = normalized
            if collision:
                identity = f"{normalized}|bucket={bucket}|occurrence={bucket_occurrence[(bucket, normalized)]}"
            add(
                "state", item["id"], item["text"], item.get("aliases", []), identity,
                "normalized_canonical_en+collision_disambiguator" if collision else "normalized_canonical_en",
                state_bucket=bucket,
                is_required_delta=bucket == "required_delta_from_previous_checkpoint",
                collision_disambiguated=collision,
                evidence_ids=item.get("evidence_ids", []),
            )
    for item in projection["layer5"]:
        if not item["critical_omission_if_missing"]:
            continue
        normalized = normalize_text(item["text"])
        add(
            "critical_action", item["id"], item["text"], item.get("aliases", []), normalized,
            "normalized_canonical_action_en", evidence_ids=item.get("evidence_ids", []),
        )
    for index, text in enumerate(projection["stopping"]["remaining_blockers"]):
        normalized = normalize_text(text)
        add(
            "stopping_blocker", f"gold_blocker_{index}", text, [], normalized,
            "normalized_blocker_en",
        )
    decision = projection["stopping"]["decision"]
    add(
        "stopping_decision", f"stopping_decision:{decision}", decision, [], decision,
        "decision_category",
    )
    return items


def _state_collision_keys(gold: GoldPackage, ordered: list[dict[str, Any]]) -> set[str]:
    collisions: set[str] = set()
    for checkpoint in ordered:
        projection = gold.project_checkpoint_en(gold.checkpoint_by_id[checkpoint["checkpoint_id"]])
        normalized: list[str] = []
        for bucket in (*STATE_BUCKETS, *SINGLE_STATE_BUCKETS):
            values = projection["layer4"][bucket]
            values = values if isinstance(values, list) else ([values] if values else [])
            normalized.extend(normalize_text(item["text"]) for item in values)
        collisions.update(value for value, count in Counter(normalized).items() if count > 1)
    return collisions


def _attach_evidence_links(
    case_id: str,
    ordered: list[dict[str, Any]],
    checkpoint_items: dict[str, list[dict[str, Any]]],
    source_to_atom: dict[tuple[str, str, str], str],
) -> None:
    del case_id
    for checkpoint in ordered:
        checkpoint_id = checkpoint["checkpoint_id"]
        items = checkpoint_items[checkpoint_id]
        evidence_by_id: dict[str, list[str]] = defaultdict(list)
        for item in items:
            if item["dimension"] == "evidence":
                evidence_by_id[item["evidence_id"]].append(item["atom_id"])
        for item in items:
            linked: set[str] = set()
            for evidence_id in item.get("evidence_ids", []):
                linked.update(evidence_by_id.get(evidence_id, []))
            item["linked_evidence_atom_ids"] = sorted(linked)


def _evaluate_valid_checkpoint(
    gold: GoldPackage,
    checkpoint_id: str,
    record: dict[str, Any],
    atom_index: GoldAtomIndex,
    judge: FrozenJudge,
    model_id: str,
    checkpoint_order: int,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    projection = gold.project_checkpoint_en(gold.checkpoint_by_id[checkpoint_id])
    current_items = atom_index.checkpoint_items[checkpoint_id]
    states = {item["atom_id"]: "missed" for item in current_items}
    errors: list[dict[str, Any]] = []

    gold_evidence = {(item["evidence_id"], item["role"]) for item in projection["layer1"]}
    pred_evidence = {(item["evidence_id"], item["role"]) for item in record["key_evidence"]}
    for item in current_items:
        if item["dimension"] == "evidence" and (item["evidence_id"], item["evidence_role"]) in pred_evidence:
            states[item["atom_id"]] = "matched"
    for evidence_id, role in sorted(pred_evidence - gold_evidence):
        errors.append(_prediction_error(
            model_id, checkpoint_id, checkpoint_order, "evidence", f"{evidence_id}|{role}",
            evidence_id, "unmatched_prediction",
        ))

    claim_pred = [{"id": item["claim_id"], "text": item["text_en"]} for item in record["claims"]]
    claim_match = semantic_match(
        claim_pred, projection["layer3"], dimension="layer3_claims",
        checkpoint_id=checkpoint_id, judge=judge,
    )
    _apply_semantic_states(states, claim_match, checkpoint_id, "claim", atom_index)
    errors.extend(_semantic_prediction_errors(
        model_id, checkpoint_id, checkpoint_order, "claim", claim_pred, claim_match
    ))

    for bucket in (*STATE_BUCKETS, *SINGLE_STATE_BUCKETS):
        gold_values = projection["layer4"][bucket]
        gold_values = gold_values if isinstance(gold_values, list) else ([gold_values] if gold_values else [])
        pred_values = record["patient_state"][bucket]
        pred_values = pred_values if isinstance(pred_values, list) else ([pred_values] if pred_values else [])
        pred_semantic = [{"id": item["state_id"], "text": item["text_en"]} for item in pred_values]
        match = semantic_match(
            pred_semantic, gold_values, dimension=f"layer4_state.{bucket}",
            checkpoint_id=checkpoint_id, judge=judge,
        )
        _apply_semantic_states(states, match, checkpoint_id, "state", atom_index)
        errors.extend(_semantic_prediction_errors(
            model_id, checkpoint_id, checkpoint_order, f"state.{bucket}", pred_semantic, match
        ))

    action_pred = [{"id": item["action_id"], "text": item["text_en"]} for item in record["actions"]]
    action_match = semantic_match(
        action_pred, projection["layer5"], dimension="layer5_actions",
        checkpoint_id=checkpoint_id, judge=judge,
    )
    _apply_semantic_states(states, action_match, checkpoint_id, "critical_action", atom_index)
    errors.extend(_semantic_prediction_errors(
        model_id, checkpoint_id, checkpoint_order, "action", action_pred, action_match
    ))

    blocker_gold = [
        {"id": f"gold_blocker_{index}", "text": text, "aliases": []}
        for index, text in enumerate(projection["stopping"]["remaining_blockers"])
    ]
    blocker_pred = [
        {"id": f"pred_blocker_{index}", "text": text}
        for index, text in enumerate(record["stopping"]["remaining_blockers"])
    ]
    blocker_match = semantic_match(
        blocker_pred, blocker_gold, dimension="stopping_blockers",
        checkpoint_id=checkpoint_id, judge=judge,
    )
    _apply_semantic_states(states, blocker_match, checkpoint_id, "stopping_blocker", atom_index)
    errors.extend(_semantic_prediction_errors(
        model_id, checkpoint_id, checkpoint_order, "stopping_blocker", blocker_pred, blocker_match
    ))

    decision = record["stopping"]["decision"]
    gold_decision = projection["stopping"]["decision"]
    decision_atom = atom_index.source_to_atom[
        (checkpoint_id, "stopping_decision", f"stopping_decision:{gold_decision}")
    ]
    states[decision_atom] = "matched" if decision == gold_decision else "missed"
    if decision != gold_decision:
        errors.append(_prediction_error(
            model_id, checkpoint_id, checkpoint_order, "stopping_decision",
            f"pred_stopping:{decision}", decision, "decision_mismatch",
        ))
    return states, errors


def _apply_semantic_states(
    states: dict[str, str],
    match: SemanticMatchResult,
    checkpoint_id: str,
    dimension: str,
    atom_index: GoldAtomIndex,
) -> None:
    for gold_id in match.mapping.values():
        atom_id = atom_index.source_to_atom.get((checkpoint_id, dimension, gold_id))
        if atom_id is not None:
            states[atom_id] = "matched"
    for pair in match.contradiction_pairs:
        atom_id = atom_index.source_to_atom.get((checkpoint_id, dimension, pair["gold_id"]))
        if atom_id is not None and states[atom_id] != "matched":
            states[atom_id] = "contradicted"


def _semantic_prediction_errors(
    model_id: str,
    checkpoint_id: str,
    checkpoint_order: int,
    dimension: str,
    predictions: list[dict[str, Any]],
    match: SemanticMatchResult,
) -> list[dict[str, Any]]:
    matched = set(match.mapping)
    contradiction = {row["prediction_id"] for row in match.contradiction_pairs}
    partial = {row["prediction_id"] for row in match.partial_pairs}
    result = []
    for prediction in predictions:
        if prediction["id"] in matched:
            continue
        error_type = (
            "contradiction" if prediction["id"] in contradiction
            else "partial" if prediction["id"] in partial
            else "unmatched_prediction"
        )
        result.append(_prediction_error(
            model_id, checkpoint_id, checkpoint_order, dimension,
            prediction["id"], prediction["text"], error_type,
        ))
    return result


def _prediction_error(
    model_id: str,
    checkpoint_id: str,
    checkpoint_order: int,
    dimension: str,
    prediction_id: str,
    prediction_text: str,
    error_type: str,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "case_id": checkpoint_id.split("_cp_", 1)[0],
        "checkpoint_id": checkpoint_id,
        "checkpoint_order": checkpoint_order,
        "dimension": dimension,
        "prediction_id": prediction_id,
        "prediction_text": prediction_text,
        "error_type": error_type,
    }


def _is_stale_carryover(
    registry_row: dict[str, Any],
    record: dict[str, Any],
    current_items: list[dict[str, Any]],
) -> bool:
    dimension = registry_row["dimension"]
    accepted = {normalize_text(registry_row["canonical_text"])}
    if any(
        item["dimension"] == dimension
        and normalize_text(item["text"]) in accepted
        for item in current_items
    ):
        return False
    if dimension == "evidence":
        identity = registry_row["normalized_identity"]
        return identity in {
            f'{item["evidence_id"]}|{item["role"]}' for item in record["key_evidence"]
        }
    if dimension == "claim":
        predictions = [item["text_en"] for item in record["claims"]]
    elif dimension == "state":
        predictions = []
        for bucket in (*STATE_BUCKETS, *SINGLE_STATE_BUCKETS):
            values = record["patient_state"][bucket]
            values = values if isinstance(values, list) else ([values] if values else [])
            predictions.extend(item["text_en"] for item in values)
    elif dimension == "critical_action":
        predictions = [item["text_en"] for item in record["actions"]]
    elif dimension == "stopping_blocker":
        predictions = record["stopping"]["remaining_blockers"]
    elif dimension == "stopping_decision":
        predictions = [record["stopping"]["decision"]]
    else:
        return False
    return any(normalize_text(value) in accepted for value in predictions)


def _mark_evidence_nonresponse(rows: list[dict[str, Any]]) -> None:
    state_by_atom = {row["atom_id"]: row["model_state"] for row in rows}
    for row in rows:
        if row["dimension"] not in {"state", "critical_action"}:
            continue
        if row["gold_state"] != "eligible" or row["model_state"] == "unscorable":
            continue
        if any(state_by_atom.get(atom_id) == "matched" for atom_id in row["linked_evidence_atom_ids"]):
            row["evidence_response_opportunity"] = 1
        if row["evidence_response_opportunity"] and row["model_state"] in {"missed", "contradicted"}:
            row["evidence_nonresponse"] = 1


def _reconcile_checkpoint(
    rows: list[dict[str, Any]], score: dict[str, Any]
) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row["gold_state"] == "eligible"]
    observed: dict[str, tuple[int, int]] = {}
    for dimension in ATOM_DIMENSIONS:
        values = [row for row in eligible if row["dimension"] == dimension]
        observed[dimension] = (
            sum(row["model_state"] == "matched" for row in values),
            len(values),
        )
    expected = {
        "evidence": (
            score["dimensions"]["layer1_evidence"]["counts"]["tp"],
            score["dimensions"]["layer1_evidence"]["gold_item_count"],
        ),
        "claim": (
            score["dimensions"]["layer3_claims"]["counts"]["tp"],
            score["dimensions"]["layer3_claims"]["gold_item_count"],
        ),
        "state": (
            score["dimensions"]["layer4_state"]["counts"]["tp"],
            score["dimensions"]["layer4_state"]["gold_item_count"],
        ),
        "critical_action": (
            score["critical_action_recall"]["matched_gold_critical_action_count"],
            score["critical_action_recall"]["gold_critical_action_count"],
        ),
        "stopping_blocker": (
            score["dimensions"]["stopping_blockers"]["counts"]["tp"],
            score["dimensions"]["stopping_blockers"]["gold_item_count"],
        ),
        "stopping_decision": (
            score["stopping_decision"]["correct"],
            score["stopping_decision"]["eligible"],
        ),
    }
    result = []
    for dimension in ATOM_DIMENSIONS:
        observed_tp, observed_gold = observed[dimension]
        expected_tp, expected_gold = expected[dimension]
        result.append({
            "model_id": rows[0]["model_id"],
            "case_id": rows[0]["case_id"],
            "checkpoint_id": rows[0]["checkpoint_id"],
            "checkpoint_order": rows[0]["checkpoint_order"],
            "dimension": dimension,
            "observed_tp": observed_tp,
            "expected_frozen_tp": expected_tp,
            "observed_gold_count": observed_gold,
            "expected_frozen_gold_count": expected_gold,
            "tp_aligned": observed_tp == expected_tp,
            "gold_count_aligned": observed_gold == expected_gold,
        })
    return result


def _annotate_transitions(observations: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        grouped[(row["model_id"], row["case_id"], row["atom_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["checkpoint_order"])
        direction_changes = 0
        previous: dict[str, Any] | None = None
        for row in rows:
            if row["gold_state"] != "eligible":
                previous = None
                continue
            if previous is not None:
                prev_binary = _binary(previous["model_state"])
                current_binary = _binary(row["model_state"])
                if prev_binary is not None and current_binary is not None and prev_binary != current_binary:
                    direction_changes += 1
                    if prev_binary == 1 and current_binary == 0:
                        row["regression_event"] = 1
            previous = row
        if direction_changes >= 2:
            for row in rows:
                row["oscillation_atom"] = 1


def _regression_denominator(rows: list[dict[str, Any]]) -> int:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["atom_id"]].append(row)
    denominator = 0
    for atom_rows in grouped.values():
        atom_rows.sort(key=lambda row: row["checkpoint_order"])
        for current, following in zip(atom_rows, atom_rows[1:]):
            if (
                current["gold_state"] == "eligible"
                and following["gold_state"] == "eligible"
                and current["model_state"] == "matched"
            ):
                denominator += 1
    return denominator


def _adjustment_lags(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dimension, prefix in (
        ("critical_action", "critical_action_adjustment"),
        ("stopping_decision", "stopping_adjustment"),
    ):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["dimension"] == dimension:
                grouped[row["atom_id"]].append(row)
        lags: list[int] = []
        episode_count = 0
        censored = 0
        for atom_rows in grouped.values():
            atom_rows.sort(key=lambda row: row["checkpoint_order"])
            current_episode: list[dict[str, Any]] = []
            for row in [*atom_rows, {"gold_state": "sentinel"}]:
                if row["gold_state"] == "eligible":
                    current_episode.append(row)
                    continue
                if not current_episode:
                    continue
                episode_count += 1
                matched_index = next(
                    (index for index, value in enumerate(current_episode) if value["model_state"] == "matched"),
                    None,
                )
                if matched_index is None:
                    censored += 1
                else:
                    lags.append(matched_index)
                current_episode = []
        result[f"{prefix}_lag_sum"] = sum(lags)
        result[f"{prefix}_lag_denominator"] = len(lags)
        result[f"{prefix}_episode_count"] = episode_count
        result[f"{prefix}_censored_count"] = censored
        result[f"{prefix}_lag_mean"] = _ratio(sum(lags), len(lags))
    return result


def synthetic_transition_labels(sequence: Iterable[int | None]) -> dict[str, Any]:
    """Small public helper used by tests to exercise transition semantics."""
    rows = []
    for order, value in enumerate(sequence):
        rows.append({
            "model_id": "synthetic",
            "case_id": "case",
            "checkpoint_id": f"cp_{order}",
            "checkpoint_order": order,
            "atom_id": "atom",
            "dimension": "state",
            "gold_state": "eligible",
            "model_state": None if value is None else ("matched" if value else "missed"),
            "is_first_required_delta": 0,
            "regression_event": 0,
            "oscillation_atom": 0,
        })
    _annotate_transitions(rows)
    episodes = derive_issue_episodes(rows)
    return {
        "episodes": episodes,
        "regression_count": sum(row["regression_event"] for row in rows),
        "oscillation": any(row["oscillation_atom"] for row in rows),
    }


def _binary(model_state: str | None) -> int | None:
    if model_state == "matched":
        return 1
    if model_state in {"missed", "contradicted"}:
        return 0
    return None


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
