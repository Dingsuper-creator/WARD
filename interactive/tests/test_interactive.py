"""Offline integration tests for the anonymous interaction release."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "runtime"))
from fixture import build_fixture
from metrics import aggregate
from ratings import apply_ratings


class InteractiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.case = build_fixture(self.root / "data")
        self.addCleanup(self.temp.cleanup)

    def command(self, script, *args, success=True):
        result = subprocess.run([sys.executable, str(HERE / script), *map(str, args)],
                                cwd=self.root, capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        return result

    def run_episode(self, actions="actions.json", output="run", *extra):
        return self.command("run.py", "--case", "ward_001", "--case-config", self.case,
                            "--scripted-actions", self.case.parent / actions,
                            "--output", self.root / output, "--run-id", "synthetic-" + output, *extra)

    def test_seven_tools_pending_recovery_metrics_and_tamper(self):
        dry = self.command("run.py", "--case", "ward_001", "--case-config", self.case, "--dry-run")
        self.assertEqual(dry["tool_count"], 7)
        result = self.run_episode()
        self.assertTrue(result["environment_terminal"])
        self.assertEqual(result["submitted_tool_call_count"], 3)
        score = self.command("eval.py", "--run", self.root / "run", "--case-config", self.case)
        row = score["runs"][0]
        self.assertEqual((row["milestone_coverage"], row["reference_evidence_recall"], row["pending_recovery_rate"]), (1.0, 1.0, 1.0))
        self.assertEqual(row["productive_budget_yield"], 0.5)
        self.assertIsNone(row["clinical_quality"])
        self.assertIsNone(score["models"][0]["case_macro"]["clinical_quality"])
        path = self.root / "run" / "tool_trace.json"
        data = json.loads(path.read_text())
        data["entries"][0]["public_tool_outcome"]["state_advanced"] = True
        path.write_text(json.dumps(data))
        rejected = self.command("eval.py", "--run", self.root / "run", "--case-config", self.case, success=False)
        self.assertIn("sha256_mismatch", rejected.stderr)

    def test_continuation_replays_prefix_without_repeating_calls(self):
        initial = self.run_episode("interrupted_actions.json", "prefix")
        self.assertEqual(initial["terminal_reason"], "api_error")
        final = self.run_episode("continuation_actions.json", "continued", "--resume-from", self.root / "prefix")
        self.assertTrue(final["environment_terminal"])
        self.assertEqual(final["submitted_tool_call_count"], 2)
        self.assertEqual(final["provider_http_attempt_count"], 1)
        score = self.command("eval.py", "--run", self.root / "continued", "--case-config", self.case)
        self.assertEqual(score["runs"][0]["milestones_completed"], 2)

    def test_dispatcher_roundtrip_and_invalid_tool(self):
        from dispatcher import build_dispatcher_tool, adapt_dispatcher_message
        from world_model_patient_ehr_v04.agent_protocol import build_native_tool_schemas, extract_openai_native_tool_call
        from world_model_patient_ehr_v04.tool_facade import TOOL_NAMES
        tools = build_native_tool_schemas(public_tools=sorted(TOOL_NAMES), public_ehr_modules=["history"])
        dispatcher = build_dispatcher_tool(tools)
        self.assertEqual(len(dispatcher["function"]["parameters"]["properties"]["tool"]["enum"]), 7)
        message = {"tool_calls": [{"type": "function", "function": {"name": "clinical_action",
                   "arguments": json.dumps({"tool": "patient_ask", "query": "synthetic history"})}}]}
        result = adapt_dispatcher_message(message, extract_openai_native_tool_call)
        self.assertEqual(result["tool"], "patient.ask")
        message["tool_calls"][0]["function"]["arguments"] = json.dumps({"tool": "unknown", "query": "test"})
        with self.assertRaises(ValueError):
            adapt_dispatcher_message(message, extract_openai_native_tool_call)

    def test_clinical_ratings_validation(self):
        self.run_episode()
        rating_file = self.root / "ratings.csv"
        rating_file.write_text("model_id,case_id,clinical_quality\nscripted-test,ward_001,5\n")
        scored = self.command("eval.py", "--run", self.root / "run", "--case-config", self.case, "--clinical-ratings", rating_file)
        self.assertEqual(scored["models"][0]["case_macro"]["clinical_quality"], 5.0)
        template = scored["runs"]
        for ratings in ([{"run_id": "missing", "clinical_quality": 5}],
                        [{"run_id": "synthetic-run", "clinical_quality": 8}],
                        [{"run_id": "synthetic-run", "clinical_quality": float("nan")}],
                        [{"run_id": "synthetic-run", "clinical_quality": True}],
                        [{"clinical_quality": 5}],
                        [{"run_id": "synthetic-run", "clinical_quality": 5}] * 2):
            path = self.root / "ratings.json"
            path.write_text(json.dumps(ratings))
            with self.assertRaises(ValueError):
                apply_ratings(copy.deepcopy(template), path)
        with self.assertRaises(ValueError):
            aggregate(template * 2)


if __name__ == "__main__":
    unittest.main()
