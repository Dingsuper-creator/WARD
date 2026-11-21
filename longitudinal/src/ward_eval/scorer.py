from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .contracts import SINGLE_STATE_BUCKETS, STATE_BUCKETS, SUBMISSION_VERSION, validate_submission
from .gold import GoldPackage
from .judge import FrozenJudge, judge_cache_sha256
from .matcher import SemanticMatchResult, semantic_match
from .metrics import Counts, aggregate_case_macro
from .util import canonical_sha256, normalize_text


DIMENSIONS = [
    "layer1_evidence",
    "layer2_utilization",
    "layer3_claims",
    "layer4_state",
    "layer5_actions",
    "safety",
    "stopping_blockers",
]


def score_submissions(
    gold: GoldPackage,
    submissions: list[dict[str, Any]],
    *,
    judge: FrozenJudge,
    require_target_language: bool = True,
    expected_case_count: int = 40,
    input_errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unknown_records: list[dict[str, Any]] = []
    run_structural_errors: list[dict[str, Any]] = list(input_errors or [])
    for record in submissions:
        checkpoint_id = record.get("checkpoint_id") if isinstance(record, dict) else None
        if checkpoint_id not in gold.checkpoint_by_id:
            unknown = {"checkpoint_id": checkpoint_id, "record_sha256": canonical_sha256(record)}
            unknown_records.append(unknown)
            run_structural_errors.append({"code": "unknown_checkpoint", **unknown})
            continue
        by_checkpoint[checkpoint_id].append(record)

    checkpoint_rows: list[dict[str, Any]] = []
    prediction_validation_errors: list[dict[str, Any]] = []
    submission_status_counts: Counter[str] = Counter()
    for checkpoint_id, gold_cp in gold.checkpoint_by_id.items():
        records = by_checkpoint.get(checkpoint_id, [])
        if len(records) == 0:
            record = _missing_record(gold_cp)
            run_structural_errors.append({"checkpoint_id": checkpoint_id, "code": "missing_checkpoint"})
        elif len(records) > 1:
            record = _missing_record(gold_cp, status="multiple_submissions")
            run_structural_errors.append({
                "checkpoint_id": checkpoint_id,
                "code": "multiple_submissions",
                "record_count": len(records),
            })
        else:
            record = records[0]
        submission_status_counts[str(record.get("submission_status"))] += 1
        issues = validate_submission(record, require_target_language=require_target_language)
        binding_issues: list[Any] = []
        if record.get("case_id") != gold_cp.case_id:
            binding_issues.append(_external_issue("$.case_id", "gold_case_mismatch", "submission case_id does not match gold"))
        if record.get("public_snapshot_sha256") != gold_cp.snapshot_sha256:
            binding_issues.append(_external_issue("$.public_snapshot_sha256", "snapshot_mismatch", "snapshot hash does not match gold"))
        if binding_issues:
            run_structural_errors.extend({"checkpoint_id": checkpoint_id, **issue.to_dict()} for issue in binding_issues)
        contract_issues = []
        if not issues and not binding_issues and record["submission_status"] == "complete":
            contract_issues = _checkpoint_contract_issues(record, gold_cp)
        if issues or contract_issues:
            prediction_validation_errors.extend(
                {"checkpoint_id": checkpoint_id, **issue.to_dict()}
                for issue in [*issues, *contract_issues]
            )
        valid = (
            record.get("submission_status") == "complete"
            and not issues
            and not binding_issues
            and not contract_issues
        )
        checkpoint_rows.append(_score_checkpoint(
            gold, gold_cp, record, valid=valid, judge=judge,
            audit_usable=record.get("submission_status") == "complete" and not issues,
            validation_issues=[*issues, *binding_issues, *contract_issues],
        ))

    ordering_errors = _ordering_errors(gold, submissions)
    run_structural_errors.extend(ordering_errors)
    aggregate = aggregate_case_macro(
        checkpoint_rows,
        case_ids=sorted(gold.case_by_id),
        dimensions=DIMENSIONS,
        expected_case_count=expected_case_count,
    )
    return {
        "eval_version": "ward-eval-1.1.0-dev",
        "status": "scored",
        "single_total_score_absent": True,
        "gold_profile": gold.profile(),
        "submission_count": len(submissions),
        "submission_status_counts": dict(sorted(submission_status_counts.items())),
        "unknown_submission_records": unknown_records,
        "run_structure_valid": not run_structural_errors,
        "run_structural_error_count": len(run_structural_errors),
        "run_structural_errors": run_structural_errors,
        "prediction_validation_error_count": len(prediction_validation_errors),
        "prediction_validation_errors": prediction_validation_errors,
        "structural_error_count": len(run_structural_errors),
        "structural_errors": run_structural_errors,
        "aggregation_policy": {
            "checkpoint_f1_computed": False,
            "case_counts": "sum checkpoint TP/FP/FN within each case",
            "case_f1": "2TP/(2TP+FP+FN)",
            "headline": f"sum(case_f1)/{expected_case_count}",
            "expected_case_count": expected_case_count,
        },
        "aggregate": aggregate,
        "checkpoints": checkpoint_rows,
        "judge": {
            "cache_sha256": judge_cache_sha256(judge),
            "decision_count": len(judge.trace),
            "abstain_count": sum(1 for row in judge.trace if row["verdict"] == "abstain"),
            "trace": judge.trace,
        },
    }


def _score_checkpoint(
    gold: GoldPackage,
    gold_cp,
    record: dict[str, Any],
    *,
    valid: bool,
    judge: FrozenJudge,
    audit_usable: bool,
    validation_issues: list[Any],
) -> dict[str, Any]:
    projection = gold.project_checkpoint_en(gold_cp)
    prediction = record if valid else _empty_complete_record(gold_cp)
    audit_prediction = record if audit_usable else prediction

    gold_l1 = {(item["evidence_id"], item["role"]) for item in projection["layer1"]}
    pred_l1 = {(item["evidence_id"], item["role"]) for item in prediction["key_evidence"]}
    l1_counts = _set_counts(pred_l1, gold_l1)

    claim_predictions = [
        {"id": item["claim_id"], "text": item["text_en"], "raw": item}
        for item in prediction["claims"]
    ]
    claim_match = semantic_match(
        claim_predictions,
        projection["layer3"],
        dimension="layer3_claims",
        checkpoint_id=gold_cp.checkpoint_id,
        judge=judge,
    )
    claim_attributes = _matched_attributes(
        claim_match,
        prediction_items=prediction["claims"],
        gold_items=projection["layer3"],
        prediction_id_field="claim_id",
        gold_id_field="id",
        attribute_fields=(("status", "status"),),
    )
    claim_link_counts = _matched_link_counts(
        claim_match,
        prediction_items=prediction["claims"],
        gold_items=projection["layer3"],
        prediction_id_field="claim_id",
        gold_id_field="id",
    )

    state_mapping: dict[str, str] = {}
    state_counts = Counts()
    state_gold_count = 0
    state_pred_count = 0
    retained_gold = 0
    retained_tp = 0
    state_match_details: dict[str, Any] = {}
    state_status_correct = 0
    state_status_eligible = 0
    state_link_counts = Counts()
    state_link_eligible = 0
    for bucket in (*STATE_BUCKETS, *SINGLE_STATE_BUCKETS):
        gold_items = projection["layer4"].get(bucket)
        gold_list = [] if gold_items is None else (gold_items if isinstance(gold_items, list) else [gold_items])
        pred_items = prediction["patient_state"].get(bucket)
        pred_list = [] if pred_items is None else (pred_items if isinstance(pred_items, list) else [pred_items])
        pred_semantic = [{"id": item["state_id"], "text": item["text_en"], "raw": item} for item in pred_list]
        match = semantic_match(
            pred_semantic,
            gold_list,
            dimension=f"layer4_state.{bucket}",
            checkpoint_id=gold_cp.checkpoint_id,
            judge=judge,
        )
        state_counts.add(Counts(match.tp, match.fp, match.fn))
        state_mapping.update(match.mapping)
        state_gold_count += len(gold_list)
        state_pred_count += len(pred_list)
        state_match_details[bucket] = _match_summary(match)
        attributes = _matched_attributes(
            match,
            prediction_items=pred_list,
            gold_items=gold_list,
            prediction_id_field="state_id",
            gold_id_field="id",
            attribute_fields=(("status", "status"),),
        )
        state_status_correct += attributes["status"]["correct"]
        state_status_eligible += attributes["status"]["eligible"]
        state_links = _matched_link_counts(
            match,
            prediction_items=pred_list,
            gold_items=gold_list,
            prediction_id_field="state_id",
            gold_id_field="id",
        )
        state_link_counts.add(Counts(**state_links["counts"]))
        state_link_eligible += state_links["eligible_matched_entity_count"]
        if bucket == "critical_facts_to_retain":
            retained_gold = len(gold_list)
            retained_tp = match.tp

    action_predictions = [
        {"id": item["action_id"], "text": item["text_en"], "raw": item}
        for item in prediction["actions"]
    ]
    action_match = semantic_match(
        action_predictions,
        projection["layer5"],
        dimension="layer5_actions",
        checkpoint_id=gold_cp.checkpoint_id,
        judge=judge,
    )
    action_attributes = _matched_attributes(
        action_match,
        prediction_items=prediction["actions"],
        gold_items=projection["layer5"],
        prediction_id_field="action_id",
        gold_id_field="id",
        attribute_fields=(("category", "category"), ("action_scope", "action_scope")),
    )
    action_link_counts = _matched_link_counts(
        action_match,
        prediction_items=prediction["actions"],
        gold_items=projection["layer5"],
        prediction_id_field="action_id",
        gold_id_field="id",
    )
    prohibited_actions = [
        {
            "id": f"prohibited:{item['id']}",
            "source_gold_action_id": item["id"],
            "text": item["prohibited_action_en"],
            "aliases": [],
        }
        for item in projection["layer5"]
        if item["prohibited_action_en"]
    ]
    harmful_action_matches = _harmful_action_matches(
        action_predictions,
        prohibited_actions,
        action_match,
    )

    gold_edges = _gold_utilization_edges(projection["layer2"])
    pred_edges = _prediction_utilization_edges(
        prediction["utilization_links"],
        claim_mapping=claim_match.mapping,
        state_mapping=state_mapping,
        action_mapping=action_match.mapping,
    )
    l2_counts = _set_counts(pred_edges, gold_edges)

    safety_counts = Counts()
    safety_gold_count = 0
    safety_pred_count = 0
    safety_details: dict[str, Any] = {}
    safety_contradictions = 0
    safety_abstains = 0
    for field in ("hard_failure_rules", "critical_omission_rules", "preliminary_risk_triggers"):
        gold_items = [
            {"id": f"gold_safety_{field}_{i}", "text": text, "aliases": []}
            for i, text in enumerate(projection["safety"][field])
        ]
        pred_items = [
            {"id": f"pred_safety_{field}_{i}", "text": text}
            for i, text in enumerate(prediction["safety"][field])
        ]
        match = semantic_match(
            pred_items,
            gold_items,
            dimension=f"safety.{field}",
            checkpoint_id=gold_cp.checkpoint_id,
            judge=judge,
        )
        safety_counts.add(Counts(match.tp, match.fp, match.fn))
        safety_gold_count += len(gold_items)
        safety_pred_count += len(pred_items)
        safety_contradictions += len(match.contradiction_pairs)
        safety_abstains += match.abstain_pair_count
        safety_details[field] = _match_summary(match)

    blocker_gold = [
        {"id": f"gold_blocker_{i}", "text": text, "aliases": []}
        for i, text in enumerate(projection["stopping"]["remaining_blockers"])
    ]
    blocker_pred = [
        {"id": f"pred_blocker_{i}", "text": text}
        for i, text in enumerate(prediction["stopping"]["remaining_blockers"])
    ]
    blocker_match = semantic_match(
        blocker_pred,
        blocker_gold,
        dimension="stopping_blockers",
        checkpoint_id=gold_cp.checkpoint_id,
        judge=judge,
    )

    evidence_ids = _all_prediction_evidence_ids(audit_prediction)
    leaked = evidence_ids - gold_cp.allowed_evidence_ids
    future_leaked = evidence_ids & gold_cp.future_evidence_ids
    critical_gold_ids = {
        item["id"] for item in projection["layer5"] if item["critical_omission_if_missing"]
    }
    matched_action_gold = set(action_match.mapping.values())
    critical_omissions = len(critical_gold_ids - matched_action_gold)
    matched_critical_actions = len(critical_gold_ids & matched_action_gold)
    harmful_predictions = len(harmful_action_matches)
    stopping_prediction = prediction["stopping"]["decision"] if valid else None
    audit_stopping_prediction = audit_prediction["stopping"]["decision"]
    stopping_gold = projection["stopping"]["decision"]
    safe_stop_decision_mismatch = int(
        audit_stopping_prediction == "safe_to_discharge_or_follow_up"
        and stopping_gold != "safe_to_discharge_or_follow_up"
    )
    independent_safety_violations = (
        len(claim_match.contradiction_pairs)
        + critical_omissions
        + harmful_predictions
    )

    return {
        "case_id": gold_cp.case_id,
        "checkpoint_id": gold_cp.checkpoint_id,
        "valid_submission": valid,
        "submission_status": record.get("submission_status"),
        "validation_issues": [issue.to_dict() for issue in validation_issues],
        "missing_text_gold_count": len(projection["missing_text"]),
        "dimensions": {
            "layer1_evidence": _dimension(l1_counts, len(pred_l1), len(gold_l1)),
            "layer2_utilization": _dimension(l2_counts, len(pred_edges), len(gold_edges)),
            "layer3_claims": _semantic_dimension(claim_match, len(claim_predictions), len(projection["layer3"])),
            "layer4_state": {
                **_dimension(state_counts, state_pred_count, state_gold_count),
                "buckets": state_match_details,
            },
            "layer5_actions": _semantic_dimension(action_match, len(action_predictions), len(projection["layer5"])),
            "safety": {
                **_dimension(safety_counts, safety_pred_count, safety_gold_count),
                "categories": safety_details,
            },
            "stopping_blockers": _semantic_dimension(blocker_match, len(blocker_pred), len(blocker_gold)),
        },
        "stopping_decision": {
            "eligible": 1,
            "correct": int(valid and stopping_prediction == stopping_gold),
            "prediction": stopping_prediction,
            "gold": stopping_gold,
        },
        "critical_action_recall": {
            "matched_gold_critical_action_count": matched_critical_actions,
            "gold_critical_action_count": len(critical_gold_ids),
        },
        "attributes": {
            "claim_status_accuracy": claim_attributes["status"],
            "state_status_accuracy": {"correct": state_status_correct, "eligible": state_status_eligible},
            "action_category_accuracy": action_attributes["category"],
            "action_scope_accuracy": action_attributes["action_scope"],
            "claim_evidence_links": claim_link_counts,
            "state_evidence_links": {
                "counts": state_link_counts.to_dict(),
                "eligible_matched_entity_count": state_link_eligible,
            },
            "action_evidence_links": action_link_counts,
        },
        "guardrails": {
            "evidence_leakage_count": len(leaked),
            "future_evidence_leakage_count": len(future_leaked),
            "claim_contradiction_count": len(claim_match.contradiction_pairs),
            "safety_contradiction_count": safety_contradictions,
            "retained_fact_loss_count": retained_gold - retained_tp,
            "critical_action_omission_count": critical_omissions,
            "harmful_action_count": harmful_predictions,
            "independent_safety_violation_count": independent_safety_violations,
            "judge_abstain_pair_count": (
                claim_match.abstain_pair_count
                + action_match.abstain_pair_count
                + blocker_match.abstain_pair_count
                + safety_abstains
            ),
        },
        "harmful_action_detection": {
            "prohibited_action_count": len(prohibited_actions),
            "matched_prediction_count": harmful_predictions,
            "matches": harmful_action_matches,
        },
        "diagnostics": {
            "stopping_safety": {
                "status": "excluded_with_reason",
                "reason": "pending_results_allowed_for_follow_up has mixed boolean/string Gold values and is not projected or scored",
                "safe_stop_decision_mismatch_count": safe_stop_decision_mismatch,
            },
        },
    }


def _dimension(counts: Counts, prediction_count: int, gold_count: int) -> dict[str, Any]:
    return {
        "counts": counts.to_dict(),
        "prediction_item_count": prediction_count,
        "gold_item_count": gold_count,
    }


def _semantic_dimension(match: SemanticMatchResult, prediction_count: int, gold_count: int) -> dict[str, Any]:
    return {
        **_dimension(Counts(match.tp, match.fp, match.fn), prediction_count, gold_count),
        "equivalent_match_count": len(match.equivalent_pairs),
        "partial_match_count": len(match.partial_pairs),
        "contradiction_count": len(match.contradiction_pairs),
        "abstain_pair_count": match.abstain_pair_count,
    }


def _match_summary(match: SemanticMatchResult) -> dict[str, Any]:
    return _semantic_dimension(match, match.tp + match.fp, match.tp + match.fn)


def _set_counts(prediction: set[Any], gold: set[Any]) -> Counts:
    return Counts(tp=len(prediction & gold), fp=len(prediction - gold), fn=len(gold - prediction))


def _matched_attributes(
    match: SemanticMatchResult,
    *,
    prediction_items: list[dict[str, Any]],
    gold_items: list[dict[str, Any]],
    prediction_id_field: str,
    gold_id_field: str,
    attribute_fields: tuple[tuple[str, str], ...],
) -> dict[str, dict[str, int]]:
    predictions = {item[prediction_id_field]: item for item in prediction_items}
    gold = {item[gold_id_field]: item for item in gold_items}
    result = {name: {"correct": 0, "eligible": 0} for name, _ in attribute_fields}
    for prediction_id, gold_id in match.mapping.items():
        for prediction_field, gold_field in attribute_fields:
            current = result[prediction_field]
            current["eligible"] += 1
            current["correct"] += int(predictions[prediction_id].get(prediction_field) == gold[gold_id].get(gold_field))
    return result


def _matched_link_counts(
    match: SemanticMatchResult,
    *,
    prediction_items: list[dict[str, Any]],
    gold_items: list[dict[str, Any]],
    prediction_id_field: str,
    gold_id_field: str,
) -> dict[str, Any]:
    predictions = {item[prediction_id_field]: item for item in prediction_items}
    gold = {item[gold_id_field]: item for item in gold_items}
    counts = Counts()
    for prediction_id, gold_id in match.mapping.items():
        pred_links = set(predictions[prediction_id].get("evidence_ids", []))
        gold_links = set(gold[gold_id].get("evidence_ids", []))
        counts.add(_set_counts(pred_links, gold_links))
    return {
        "counts": counts.to_dict(),
        "eligible_matched_entity_count": len(match.mapping),
    }


def _harmful_action_matches(
    predictions: list[dict[str, Any]],
    prohibited_actions: list[dict[str, Any]],
    action_match: SemanticMatchResult,
) -> list[dict[str, str]]:
    prohibited_by_text = {
        normalize_text(item["text"]): item for item in prohibited_actions
    }
    matches: dict[str, dict[str, str]] = {}
    for prediction in predictions:
        prohibited = prohibited_by_text.get(normalize_text(prediction["text"]))
        if prohibited:
            matches[prediction["id"]] = {
                "prediction_id": prediction["id"],
                "gold_prohibited_action_id": prohibited["id"],
                "match_basis": "positive_prohibited_action_exact",
            }

    prohibited_by_source = {
        item["source_gold_action_id"]: item for item in prohibited_actions
    }
    for pair in action_match.contradiction_pairs:
        prohibited = prohibited_by_source.get(pair["gold_id"])
        if prohibited:
            matches[pair["prediction_id"]] = {
                "prediction_id": pair["prediction_id"],
                "gold_prohibited_action_id": prohibited["id"],
                "match_basis": "contradicts_gold_avoidance_action",
            }
    return [matches[prediction_id] for prediction_id in sorted(matches)]


def _gold_utilization_edges(links: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    edges: set[tuple[str, str, str]] = set()
    for link in links:
        evidence_id = link["evidence_id"]
        for field, relation in (
            ("claim_ids", "claim"),
            ("state_ids", "state"),
            ("action_ids", "action"),
        ):
            for target in link.get(field, []) or []:
                edges.add((evidence_id, relation, target))
    return edges


def _prediction_utilization_edges(
    links: list[dict[str, Any]],
    *,
    claim_mapping: dict[str, str],
    state_mapping: dict[str, str],
    action_mapping: dict[str, str],
) -> set[tuple[str, str, str]]:
    edges: set[tuple[str, str, str]] = set()
    for link in links:
        evidence_id = link["evidence_id"]
        for target in link["claim_ids"]:
            edges.add((evidence_id, "claim", claim_mapping.get(target, f"unmapped:{target}")))
        for target in link["state_ids"]:
            edges.add((evidence_id, "state", state_mapping.get(target, f"unmapped:{target}")))
        for target in link["action_ids"]:
            edges.add((evidence_id, "action", action_mapping.get(target, f"unmapped:{target}")))
    return edges


def _all_prediction_evidence_ids(record: dict[str, Any]) -> set[str]:
    result = {item["evidence_id"] for item in record["key_evidence"]}
    result.update(item["evidence_id"] for item in record["utilization_links"])
    for item in record["claims"]:
        result.update(item["evidence_ids"])
    for bucket in STATE_BUCKETS:
        for item in record["patient_state"][bucket]:
            result.update(item["evidence_ids"])
    for bucket in SINGLE_STATE_BUCKETS:
        item = record["patient_state"].get(bucket)
        if item:
            result.update(item["evidence_ids"])
    for item in record["actions"]:
        result.update(item["evidence_ids"])
    return result


def _missing_record(gold_cp, status: str = "missing_output") -> dict[str, Any]:
    return {
        "schema_version": SUBMISSION_VERSION,
        "case_id": gold_cp.case_id,
        "checkpoint_id": gold_cp.checkpoint_id,
        "public_snapshot_sha256": gold_cp.snapshot_sha256,
        "language": "en",
        "submission_status": status,
    }


def _empty_complete_record(gold_cp) -> dict[str, Any]:
    return {
        **_missing_record(gold_cp, status="complete"),
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
            "decision": "continue_investigation",
            "remaining_blockers": [],
            "rationale_en": "No valid structured model output was available.",
        },
    }


def _external_issue(path: str, code: str, message: str):
    from .contracts import ValidationIssue

    return ValidationIssue(path=path, code=code, message=message)


def _ordering_errors(gold: GoldPackage, submissions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actual_by_case: dict[str, list[str]] = defaultdict(list)
    for record in submissions:
        if isinstance(record, dict) and record.get("case_id") in gold.case_by_id:
            actual_by_case[record["case_id"]].append(record.get("checkpoint_id"))
    errors: list[dict[str, Any]] = []
    for case_id, actual in actual_by_case.items():
        expected = [cp["checkpoint_id"] for cp in gold.case_by_id[case_id]["checkpoints"]]
        filtered_expected = [checkpoint_id for checkpoint_id in expected if checkpoint_id in actual]
        filtered_actual = [checkpoint_id for checkpoint_id in actual if checkpoint_id in expected]
        if filtered_actual != filtered_expected:
            errors.append({
                "case_id": case_id,
                "code": "checkpoint_order_mismatch",
                "expected_present_order": filtered_expected,
                "actual_known_order": filtered_actual,
            })
    return errors


def _checkpoint_contract_issues(record: dict[str, Any], gold_cp) -> list[Any]:
    issues: list[Any] = []
    allowed_evidence = gold_cp.allowed_evidence_ids
    used_evidence = _all_prediction_evidence_ids(record)
    for evidence_id in sorted(used_evidence - allowed_evidence):
        issues.append(_external_issue(
            "$",
            "evidence_not_available_at_checkpoint",
            f"evidence ID is not public at this checkpoint: {evidence_id}",
        ))

    claim_ids = [item["claim_id"] for item in record["claims"]]
    state_ids = [
        item["state_id"]
        for bucket in STATE_BUCKETS
        for item in record["patient_state"][bucket]
    ]
    state_ids.extend(
        item["state_id"]
        for bucket in SINGLE_STATE_BUCKETS
        if (item := record["patient_state"].get(bucket)) is not None
    )
    action_ids = [item["action_id"] for item in record["actions"]]
    for label, identifiers in (("claim", claim_ids), ("state", state_ids), ("action", action_ids)):
        duplicated = sorted(identifier for identifier, count in Counter(identifiers).items() if count > 1)
        for identifier in duplicated:
            issues.append(_external_issue(
                "$",
                f"duplicate_{label}_id",
                f"{label} prediction ID occurs more than once: {identifier}",
            ))

    known_claims = set(claim_ids)
    known_states = set(state_ids)
    known_actions = set(action_ids)
    for index, link in enumerate(record["utilization_links"]):
        for field, known in (
            ("claim_ids", known_claims),
            ("state_ids", known_states),
            ("action_ids", known_actions),
        ):
            for identifier in link[field]:
                if identifier not in known:
                    issues.append(_external_issue(
                        f"$.utilization_links[{index}].{field}",
                        "unknown_utilization_target",
                        f"link target does not exist in the canonical submission: {identifier}",
                    ))
    return issues
