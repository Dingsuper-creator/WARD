"""Focused regression tests for privacy-preserving import and integrity checks."""
import json
import tempfile
import unittest
from pathlib import Path

from prepare import Transform, runtime_answer, shift_structured_times
from verify import inspect


class ImportTests(unittest.TestCase):
    def test_hash_strings_are_never_redacted_as_identifiers(self):
        transform = Transform({"replacements": [{"match": "private institution", "replace": "Institution A"}], "patterns": [{"pattern": r"1[3-9][0-9]{9}", "replace": "[telephone]"}]}, {})
        digest = "a" * 10 + "13812345678" + "b" * 43
        self.assertEqual(transform.text(digest), digest)

    def test_date_shift_is_limited_to_structured_date_fields(self):
        source = {"clinical_time": "2024-08-01T12:00:00", "clock_before": "2024-08-01T08:00:00", "dose": "2024 mg", "content": "2024-08-01"}
        output = shift_structured_times(source)
        self.assertEqual(output["clinical_time"], "2124-08-01T12:00:00")
        self.assertEqual(output["clock_before"], "2124-08-01T08:00:00")
        self.assertEqual(output["dose"], source["dose"])
        self.assertEqual(output["content"], source["content"])

    def test_reference_fallback_fields_are_preserved(self):
        answer = {key: [] for key in ("layer1_evidence", "layer2_utilization_links", "layer3_claims", "layer5_actions")}
        answer.update(layer4_state={}, safety={"preliminary_risk_triggers": ["Pending result"]}, stopping={"decision": "continue", "remaining_blockers": ["Pending result"]})
        projected = runtime_answer(answer)
        self.assertEqual(projected["safety"]["preliminary_risk_triggers_en"], ["Pending result"])
        self.assertEqual(projected["stopping"]["remaining_blockers_en"], ["Pending result"])

    def test_verifier_rejects_a_broken_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "sample.json").write_text("{}")
            manifest = {"files": [{"path": "sample.json", "sha256": "0" * 64, "size_bytes": 2}]}
            (root / "DATASET_MANIFEST.json").write_text(json.dumps(manifest))
            result = inspect(root, [])
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["findings"][0]["category"], "manifest_binding_mismatch")


if __name__ == "__main__":
    unittest.main()
