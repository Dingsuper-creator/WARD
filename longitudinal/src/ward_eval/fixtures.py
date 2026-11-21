from __future__ import annotations

import copy
from typing import Any

from .contracts import SINGLE_STATE_BUCKETS, STATE_BUCKETS
from .gold import GoldPackage


def make_gold_submission_records(gold: GoldPackage) -> list[dict[str, Any]]:
    return [
        projection_to_submission(gold.project_checkpoint_en(gold_cp))
        for gold_cp in gold.checkpoint_by_id.values()
    ]


def make_model_preannotation_submission_records(gold: GoldPackage) -> list[dict[str, Any]]:
    return [
        projection_to_submission(gold.project_model_preannotation_en(gold_cp))
        for gold_cp in gold.checkpoint_by_id.values()
    ]


def projection_to_submission(projection: dict[str, Any]) -> dict[str, Any]:
    claim_map = {item["id"]: f"pred_claim_{i:03d}" for i, item in enumerate(projection["layer3"], start=1)}
    state_map: dict[str, str] = {}
    state_output: dict[str, Any] = {}
    state_index = 1
    for bucket in STATE_BUCKETS:
        state_output[bucket] = []
        for item in projection["layer4"][bucket]:
            pred_id = f"pred_state_{state_index:03d}"
            state_index += 1
            state_map[item["id"]] = pred_id
            state_output[bucket].append(_state_submission(item, pred_id))
    for bucket in SINGLE_STATE_BUCKETS:
        item = projection["layer4"][bucket]
        if item is None:
            state_output[bucket] = None
        else:
            pred_id = f"pred_state_{state_index:03d}"
            state_index += 1
            state_map[item["id"]] = pred_id
            state_output[bucket] = _state_submission(item, pred_id)

    action_map = {item["id"]: f"pred_action_{i:03d}" for i, item in enumerate(projection["layer5"], start=1)}
    return {
        "schema_version": "ward-checkpoint-submission-1.1",
        "case_id": projection["case_id"],
        "checkpoint_id": projection["checkpoint_id"],
        "public_snapshot_sha256": projection["snapshot_sha256"],
        "language": "en",
        "submission_status": "complete",
        "key_evidence": [
            {
                "evidence_id": item["evidence_id"],
                "role": item["role"],
                "interpretation_en": item.get("interpretation_en"),
            }
            for item in projection["layer1"]
        ],
        "utilization_links": [
            {
                "evidence_id": link["evidence_id"],
                "claim_ids": [claim_map[value] for value in (link.get("claim_ids") or []) if value in claim_map],
                "state_ids": [state_map[value] for value in (link.get("state_ids") or []) if value in state_map],
                "action_ids": [action_map[value] for value in (link.get("action_ids") or []) if value in action_map],
                "expected_outcome_en": link.get("expected_outcome"),
            }
            for link in projection["layer2"]
        ],
        "claims": [
            {
                "claim_id": claim_map[item["id"]],
                "text_en": item["text"],
                "status": item["status"],
                "evidence_ids": item["evidence_ids"],
            }
            for item in projection["layer3"]
        ],
        "patient_state": state_output,
        "actions": [
            {
                "action_id": action_map[item["id"]],
                "text_en": item["text"],
                "category": item["category"],
                "action_scope": item["action_scope"],
                "evidence_ids": item["evidence_ids"],
                "critical_omission_if_missing": item["critical_omission_if_missing"],
                "prohibited_action_en": item.get("prohibited_action_en"),
            }
            for item in projection["layer5"]
        ],
        "safety": copy.deepcopy(projection["safety"]),
        "stopping": copy.deepcopy(projection["stopping"]),
    }


def make_empty_submission_records(gold: GoldPackage) -> list[dict[str, Any]]:
    rows = []
    for gold_cp in gold.checkpoint_by_id.values():
        projection = gold.project_checkpoint_en(gold_cp)
        rows.append({
            "schema_version": "ward-checkpoint-submission-1.1",
            "case_id": gold_cp.case_id,
            "checkpoint_id": gold_cp.checkpoint_id,
            "public_snapshot_sha256": gold_cp.snapshot_sha256,
            "language": "en",
            "submission_status": "complete",
            "key_evidence": [],
            "utilization_links": [],
            "claims": [],
            "patient_state": {
                **{bucket: [] for bucket in STATE_BUCKETS},
                **{bucket: None for bucket in SINGLE_STATE_BUCKETS},
            },
            "actions": [],
            "safety": {
                "hard_failure_rules": [],
                "critical_omission_rules": [],
                "preliminary_risk_triggers": [],
            },
            "stopping": {
                "decision": projection["stopping"]["decision"],
                "remaining_blockers": [],
                "rationale_en": "No clinical content was submitted.",
            },
        })
    return rows


def make_mutations(
    gold_records: list[dict[str, Any]],
    gold: GoldPackage | None = None,
) -> dict[str, list[dict[str, Any]]]:
    mutations: dict[str, list[dict[str, Any]]] = {}

    def mutate(name: str, fn) -> None:
        rows = copy.deepcopy(gold_records)
        target = next((row for row in rows if _has_mutatable_content(row, name)), rows[0])
        fn(target)
        mutations[name] = rows

    mutate("delete_evidence", lambda row: row["key_evidence"].pop() if row["key_evidence"] else None)
    future_by_checkpoint = {
        checkpoint_id: sorted(checkpoint.future_evidence_ids)
        for checkpoint_id, checkpoint in (gold.checkpoint_by_id.items() if gold is not None else [])
    }
    mutate("add_unavailable_evidence", lambda row: row["key_evidence"].append({
        "evidence_id": (future_by_checkpoint.get(row["checkpoint_id"]) or ["future:forbidden:synthetic"])[0],
        "role": "critical",
        "interpretation_en": "Synthetic unavailable evidence.",
    }))
    mutate("delete_claim", lambda row: row["claims"].pop() if row["claims"] else None)
    mutate("delete_state", _delete_state)
    mutate("delete_action", lambda row: row["actions"].pop() if row["actions"] else None)
    mutate("harmful_action", _make_harmful_recommendation)
    mutate("flip_stopping", lambda row: row["stopping"].update({
        "decision": (
            "safe_to_discharge_or_follow_up"
            if row["stopping"]["decision"] != "safe_to_discharge_or_follow_up"
            else "continue_investigation"
        )
    }))
    mutate("delete_blocker", lambda row: row["stopping"]["remaining_blockers"].pop() if row["stopping"]["remaining_blockers"] else None)
    mutate("negation_error", lambda row: _mutate_claim_text(row, "No evidence supports: "))
    mutate("laterality_error", lambda row: _mutate_claim_text(row, "Left-sided instead of right-sided: "))
    mutate("temporal_error", lambda row: _mutate_claim_text(row, "Occurred in 1900: "))
    mutate("unit_error", lambda row: _mutate_claim_text(row, "Measured as 999 mg: "))
    mutate("claim_status_error", _mutate_claim_status)
    mutate("active_resolved_confusion", _move_active_to_resolved)
    mutate("snapshot_hash_mismatch", lambda row: row.update({"public_snapshot_sha256": "0" * 64}))
    mutate("schema_invalid", lambda row: row.pop("claims", None))
    mutate("refusal", lambda row: row.update({"submission_status": "refusal", "error_code": "target_refusal"}))
    mutate("truncated", lambda row: row.update({"submission_status": "truncated", "error_code": "finish_length"}))
    mutate("missing_output_status", lambda row: row.update({"submission_status": "missing_output", "error_code": "no_visible_result"}))
    mutate("unsupported_format", lambda row: row.update({"submission_status": "unsupported_format", "error_code": "prose_only"}))
    mutate("transport_error", lambda row: row.update({"submission_status": "transport_error", "error_code": "provider_transport"}))
    mutate("translation_failed_status", lambda row: row.update({"submission_status": "translation_failed", "error_code": "literal_translation_cache_incomplete"}))
    mutate("non_target_language", _make_non_target_language)

    duplicated = copy.deepcopy(gold_records)
    duplicated.insert(1, copy.deepcopy(duplicated[0]))
    mutations["duplicate_checkpoint"] = duplicated
    missing = copy.deepcopy(gold_records)
    missing.pop(0)
    mutations["missing_checkpoint"] = missing
    wrong_order = copy.deepcopy(gold_records)
    first_case_indexes = [i for i, row in enumerate(wrong_order) if row["case_id"] == wrong_order[0]["case_id"]]
    if len(first_case_indexes) >= 2:
        left, right = first_case_indexes[:2]
        wrong_order[left], wrong_order[right] = wrong_order[right], wrong_order[left]
    mutations["checkpoint_wrong_order"] = wrong_order
    return mutations


def _state_submission(item: dict[str, Any], pred_id: str) -> dict[str, Any]:
    return {
        "state_id": pred_id,
        "text_en": item["text"],
        "status": item["status"],
        "evidence_ids": item["evidence_ids"],
    }


def _delete_state(row: dict[str, Any]) -> None:
    for bucket in STATE_BUCKETS:
        if row["patient_state"][bucket]:
            row["patient_state"][bucket].pop()
            return
    for bucket in SINGLE_STATE_BUCKETS:
        if row["patient_state"][bucket] is not None:
            row["patient_state"][bucket] = None
            return


def _has_mutatable_content(row: dict[str, Any], name: str) -> bool:
    if name == "delete_evidence":
        return bool(row["key_evidence"])
    if name == "delete_claim":
        return bool(row["claims"])
    if name == "delete_state":
        return any(row["patient_state"][bucket] for bucket in (*STATE_BUCKETS, *SINGLE_STATE_BUCKETS))
    if name == "delete_action":
        return bool(row["actions"])
    if name == "harmful_action":
        return any(
            item.get("prohibited_action_en")
            and item["text_en"] != item["prohibited_action_en"]
            for item in row["actions"]
        )
    if name == "delete_blocker":
        return bool(row["stopping"]["remaining_blockers"])
    return True


def _mutate_claim_text(row: dict[str, Any], prefix: str) -> None:
    if row["claims"]:
        row["claims"][0]["text_en"] = prefix + row["claims"][0]["text_en"]


def _mutate_claim_status(row: dict[str, Any]) -> None:
    if row["claims"]:
        row["claims"][0]["status"] = (
            "negative" if row["claims"][0]["status"] != "negative" else "confirmed"
        )


def _move_active_to_resolved(row: dict[str, Any]) -> None:
    active = row["patient_state"]["active_problems"]
    if active:
        row["patient_state"]["resolved_problems"].append(active.pop(0))


def _make_non_target_language(row: dict[str, Any]) -> None:
    row["language"] = "zh"
    if row["claims"]:
        row["claims"][0]["text_en"] = "Synthetic output with an invalid language declaration."


def _make_harmful_recommendation(row: dict[str, Any]) -> None:
    item = next(
        action for action in row["actions"]
        if action.get("prohibited_action_en")
        and action["text_en"] != action["prohibited_action_en"]
    )
    item["text_en"] = item["prohibited_action_en"]
    # Deliberately falsify the model-controlled labels. The independent evaluator
    # must still recognize the Gold-prohibited action by semantic mapping.
    item["category"] = "necessary"
    item["action_scope"] = "prospective"
