from __future__ import annotations

import unittest

from ward_eval.metrics import Counts, aggregate_case_macro, f1_from_case_counts


def _row(case_id: str, checkpoint_id: str, counts: Counts, *, valid: bool = True, gold: int = 1, pred: int = 1):
    return {
        "case_id": case_id,
        "checkpoint_id": checkpoint_id,
        "valid_submission": valid,
        "dimensions": {"dimension": {
            "counts": counts.to_dict(),
            "gold_item_count": gold,
            "prediction_item_count": pred,
        }},
        "stopping_decision": {"eligible": 0, "correct": 0},
        "critical_action_recall": {
            "matched_gold_critical_action_count": counts.tp,
            "gold_critical_action_count": counts.tp + counts.fn,
        },
        "attributes": {},
        "guardrails": {},
    }


class CaseMacroTests(unittest.TestCase):
    def test_formula_is_counts_then_case_f1_then_divide_by_40(self):
        case_ids = [f"ward_{index:03d}" for index in range(1, 41)]
        rows = [
            _row(case_ids[0], "cp_1", Counts(tp=1)),
            _row(case_ids[0], "cp_2", Counts(fn=9), pred=0, gold=9),
        ]
        rows.extend(
            _row(case_id, "cp_1", Counts(tp=1))
            for case_id in case_ids[1:]
        )
        result = aggregate_case_macro(rows, case_ids=case_ids, dimensions=["dimension"])
        expected_first_case = 2 / 11
        expected_headline = (39 + expected_first_case) / 40
        self.assertAlmostEqual(result["cases"][0]["dimensions"]["dimension"]["f1"], expected_first_case)
        self.assertAlmostEqual(result["headline"]["dimension"]["case_macro_f1"], expected_headline)
        self.assertNotAlmostEqual(result["headline"]["dimension"]["case_macro_f1"], (1 + 0 + 39) / 41)

    def test_explicit_empty_valid_is_one(self):
        self.assertEqual(f1_from_case_counts(Counts(), explicitly_empty=True, all_submissions_valid=True), 1.0)

    def test_invalid_or_missing_empty_is_zero(self):
        self.assertEqual(f1_from_case_counts(Counts(), explicitly_empty=True, all_submissions_valid=False), 0.0)
        self.assertEqual(f1_from_case_counts(Counts(), explicitly_empty=False, all_submissions_valid=True), 0.0)

    def test_evidence_link_submetric_is_na_without_matched_entities(self):
        row = _row("ward_001", "cp_1", Counts())
        row["attributes"] = {
            "links": {
                "counts": Counts().to_dict(),
                "eligible_matched_entity_count": 0,
            },
        }
        result = aggregate_case_macro(
            [row], case_ids=["ward_001"], dimensions=["dimension"], expected_case_count=1
        )
        self.assertIsNone(result["cases"][0]["attributes"]["links"]["f1"])

    def test_evidence_link_submetric_scores_explicit_empty_and_missing_gold_link(self):
        empty = _row("ward_001", "cp_1", Counts())
        empty["attributes"] = {
            "links": {
                "counts": Counts().to_dict(),
                "eligible_matched_entity_count": 1,
            },
        }
        empty_result = aggregate_case_macro(
            [empty], case_ids=["ward_001"], dimensions=["dimension"], expected_case_count=1
        )
        self.assertEqual(empty_result["cases"][0]["attributes"]["links"]["f1"], 1.0)

        missing = _row("ward_001", "cp_1", Counts())
        missing["attributes"] = {
            "links": {
                "counts": Counts(fn=1).to_dict(),
                "eligible_matched_entity_count": 1,
            },
        }
        missing_result = aggregate_case_macro(
            [missing], case_ids=["ward_001"], dimensions=["dimension"], expected_case_count=1
        )
        self.assertEqual(missing_result["cases"][0]["attributes"]["links"]["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
