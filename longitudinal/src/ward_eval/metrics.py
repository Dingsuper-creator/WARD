from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def f1_from_case_counts(counts: Counts, *, explicitly_empty: bool, all_submissions_valid: bool) -> float:
    denominator = 2 * counts.tp + counts.fp + counts.fn
    if denominator:
        return 2 * counts.tp / denominator
    if explicitly_empty and all_submissions_valid:
        return 1.0
    return 0.0


def precision(counts: Counts) -> float | None:
    denominator = counts.tp + counts.fp
    return counts.tp / denominator if denominator else None


def recall(counts: Counts) -> float | None:
    denominator = counts.tp + counts.fn
    return counts.tp / denominator if denominator else None


def aggregate_case_macro(
    checkpoint_rows: list[dict[str, Any]],
    *,
    case_ids: list[str],
    dimensions: list[str],
    expected_case_count: int = 40,
) -> dict[str, Any]:
    case_rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        rows = [row for row in checkpoint_rows if row["case_id"] == case_id]
        dimension_results: dict[str, Any] = {}
        for dimension in dimensions:
            counts = Counts()
            all_valid = True
            all_gold_empty = True
            all_prediction_empty = True
            for row in rows:
                current = row["dimensions"][dimension]
                counts.add(Counts(**current["counts"]))
                all_valid = all_valid and row["valid_submission"]
                all_gold_empty = all_gold_empty and current["gold_item_count"] == 0
                all_prediction_empty = all_prediction_empty and current["prediction_item_count"] == 0
            explicitly_empty = all_gold_empty and all_prediction_empty
            dimension_results[dimension] = {
                "counts": counts.to_dict(),
                "precision": precision(counts),
                "recall": recall(counts),
                "f1": f1_from_case_counts(
                    counts,
                    explicitly_empty=explicitly_empty,
                    all_submissions_valid=all_valid,
                ),
                "explicitly_empty": explicitly_empty,
                "all_submissions_valid": all_valid,
            }
        stopping_eligible = sum(row["stopping_decision"]["eligible"] for row in rows)
        stopping_correct = sum(row["stopping_decision"]["correct"] for row in rows)
        critical_matched = sum(row["critical_action_recall"]["matched_gold_critical_action_count"] for row in rows)
        critical_gold = sum(row["critical_action_recall"]["gold_critical_action_count"] for row in rows)
        case_rows.append({
            "case_id": case_id,
            "checkpoint_count": len(rows),
            "dimensions": dimension_results,
            "stopping_decision": {
                "correct": stopping_correct,
                "eligible": stopping_eligible,
                "accuracy": stopping_correct / stopping_eligible if stopping_eligible else None,
            },
            "critical_action_recall": {
                "matched_gold_critical_action_count": critical_matched,
                "gold_critical_action_count": critical_gold,
                "recall": critical_matched / critical_gold if critical_gold else None,
            },
            "attributes": _aggregate_case_attributes(rows),
            "guardrails": _sum_guardrails(rows),
        })

    case_count_valid = len(case_rows) == expected_case_count
    headlines: dict[str, Any] = {}
    for dimension in dimensions:
        case_f1 = [row["dimensions"][dimension]["f1"] for row in case_rows]
        micro = Counts()
        for row in case_rows:
            micro.add(Counts(**row["dimensions"][dimension]["counts"]))
        checkpoint_dimension_rows = [row["dimensions"][dimension] for row in checkpoint_rows]
        partial_count = sum(row.get("partial_match_count", 0) for row in checkpoint_dimension_rows)
        contradiction_count = sum(row.get("contradiction_count", 0) for row in checkpoint_dimension_rows)
        equivalent_count = sum(row.get("equivalent_match_count", 0) for row in checkpoint_dimension_rows)
        abstain_pair_count = sum(row.get("abstain_pair_count", 0) for row in checkpoint_dimension_rows)
        semantic_pair_total = equivalent_count + partial_count + contradiction_count
        all_explicitly_empty = all(
            row["dimensions"][dimension]["explicitly_empty"] for row in case_rows
        )
        all_valid = all(
            row["dimensions"][dimension]["all_submissions_valid"] for row in case_rows
        )
        headlines[dimension] = {
            "case_macro_f1": sum(case_f1) / expected_case_count if case_count_valid else None,
            "case_f1_sum": sum(case_f1),
            "case_f1_denominator": expected_case_count,
            "micro_counts": micro.to_dict(),
            "micro_f1": f1_from_case_counts(
                micro,
                explicitly_empty=all_explicitly_empty,
                all_submissions_valid=all_valid,
            ),
            "equivalent_match_count": equivalent_count,
            "partial_match_count": partial_count,
            "partial_match_rate": partial_count / semantic_pair_total if semantic_pair_total else None,
            "contradiction_count": contradiction_count,
            "judge_abstain_pair_count": abstain_pair_count,
        }
    stopping_accuracies = [row["stopping_decision"]["accuracy"] for row in case_rows]
    headlines["stopping_decision"] = {
        "case_macro_accuracy": (
            sum(value for value in stopping_accuracies if value is not None) / expected_case_count
            if case_count_valid and all(value is not None for value in stopping_accuracies)
            else None
        ),
        "case_accuracy_denominator": expected_case_count,
    }
    critical_case_recalls = [
        row["critical_action_recall"]["recall"]
        for row in case_rows
        if row["critical_action_recall"]["recall"] is not None
    ]
    critical_micro_numerator = sum(
        row["critical_action_recall"]["matched_gold_critical_action_count"] for row in case_rows
    )
    critical_micro_denominator = sum(
        row["critical_action_recall"]["gold_critical_action_count"] for row in case_rows
    )
    headlines["critical_action_recall"] = {
        "case_macro_recall": (
            sum(critical_case_recalls) / len(critical_case_recalls)
            if case_count_valid and critical_case_recalls
            else None
        ),
        "eligible_case_count": len(critical_case_recalls),
        "micro_numerator": critical_micro_numerator,
        "micro_denominator": critical_micro_denominator,
        "micro_recall": (
            critical_micro_numerator / critical_micro_denominator
            if critical_micro_denominator
            else None
        ),
    }
    attribute_names = sorted({name for row in case_rows for name in row["attributes"]})
    attribute_headlines: dict[str, Any] = {}
    for name in attribute_names:
        values = [row["attributes"][name].get("accuracy", row["attributes"][name].get("f1")) for row in case_rows]
        applicable = [value for value in values if value is not None]
        attribute_headlines[name] = {
            "case_macro_value": sum(applicable) / len(applicable) if applicable else None,
            "applicable_case_count": len(applicable),
        }
    return {
        "expected_case_count": expected_case_count,
        "observed_case_count": len(case_rows),
        "official_case_count_valid": case_count_valid,
        "headline": headlines,
        "attribute_headline": attribute_headlines,
        "cases": case_rows,
    }


def _sum_guardrails(rows: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        for key, value in row["guardrails"].items():
            result[key] = result.get(key, 0) + int(value)
    return result


def _aggregate_case_attributes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({name for row in rows for name in row.get("attributes", {})})
    result: dict[str, Any] = {}
    for name in names:
        entries = [row["attributes"][name] for row in rows]
        if all("correct" in entry for entry in entries):
            correct = sum(entry["correct"] for entry in entries)
            eligible = sum(entry["eligible"] for entry in entries)
            result[name] = {
                "correct": correct,
                "eligible": eligible,
                "accuracy": correct / eligible if eligible else None,
            }
        else:
            counts = Counts()
            for entry in entries:
                counts.add(Counts(**entry["counts"]))
            eligible = sum(entry["eligible_matched_entity_count"] for entry in entries)
            denominator = 2 * counts.tp + counts.fp + counts.fn
            result[name] = {
                "counts": counts.to_dict(),
                "eligible_matched_entity_count": eligible,
                "f1": 2 * counts.tp / denominator if denominator else (1.0 if eligible else None),
            }
    return result
