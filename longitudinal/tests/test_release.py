from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from synthetic import make_gold
from ward_eval.contracts import CHECKPOINT_SUBMISSION_SCHEMA, validate_submission
from ward_eval.entrypoints import eval_main
from ward_eval.fixtures import make_gold_submission_records
from ward_eval.judge import FrozenJudge
from ward_eval.judge_jobs import prepare_judge_job
from ward_eval.prompt_audit import audit_prompt_manifests
from ward_eval.runner import prepare_model_preflight, run_model_preflight, _non_retryable_provider_error
from ward_eval.scorer import DIMENSIONS, score_submissions
from ward_eval.temporal import evaluate_temporal, PRIMARY_METRICS
from ward_eval.util import read_json, read_jsonl, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.gold = make_gold(self.root / "gold")
        self.rows = make_gold_submission_records(self.gold)

    def test_gold_self_score_and_six_temporal_metrics(self):
        self.assertTrue(self.gold.text_audit()["formal_text_ready"])
        self.assertTrue(all(not validate_submission(row) for row in self.rows))
        judge = FrozenJudge()
        result = score_submissions(self.gold, self.rows, judge=judge, expected_case_count=1)
        for dim in DIMENSIONS:
            self.assertEqual(result["aggregate"]["headline"][dim]["case_macro_f1"], 1.0)
        temporal = evaluate_temporal(self.gold, self.rows, result["checkpoints"], judge=judge)
        self.assertEqual(set(temporal["metrics"]), set(PRIMARY_METRICS))
        metrics = temporal["metrics"]
        self.assertEqual(metrics["state_delta_immediate_response_rate"]["case_macro_value"], 1.0)
        self.assertEqual(metrics["linked_evidence_nonresponse_rate"]["case_macro_value"], 0.0)
        self.assertEqual(metrics["stopping_change_response_rate"]["case_macro_value"], 1.0)
        self.assertIsNone(metrics["evidence_subsequent_match_rate"]["case_macro_value"])
        self.assertEqual(metrics["evidence_regression_rate"]["case_macro_value"], 0.0)
        self.assertEqual(metrics["stopping_decision_regression_rate"]["case_macro_value"], 0.0)

    def test_later_match_then_regression_and_invalid_exclusion(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["key_evidence"] = []
        rows[2]["key_evidence"] = []
        result = score_submissions(self.gold, rows, judge=FrozenJudge(), expected_case_count=1)
        temporal = evaluate_temporal(self.gold, rows, result["checkpoints"], judge=FrozenJudge())
        self.assertEqual(temporal["metrics"]["evidence_subsequent_match_rate"]["case_macro_value"], 1.0)
        self.assertEqual(temporal["metrics"]["evidence_regression_rate"]["case_macro_value"], 1.0)
        rows[2]["submission_status"] = "schema_invalid"
        result = score_submissions(self.gold, rows, judge=FrozenJudge(), expected_case_count=1)
        temporal = evaluate_temporal(self.gold, rows, result["checkpoints"], judge=FrozenJudge())
        self.assertFalse(result["checkpoints"][2]["valid_submission"])
        self.assertIsNone(temporal["metrics"]["evidence_regression_rate"]["case_macro_value"])

    def test_missing_and_non_object_outputs_do_not_crash_temporal(self):
        result = score_submissions(self.gold, [None, self.rows[0]], judge=FrozenJudge(), expected_case_count=1)
        temporal = evaluate_temporal(self.gold, [None, self.rows[0]], result["checkpoints"], judge=FrozenJudge())
        self.assertEqual(len(temporal["case_metrics"]), 6)
        self.assertFalse(result["run_structure_valid"])

    def test_runner_prepare_execute_and_prompt_audit_with_mocked_transport(self):
        roster = read_json(ROOT / "config/model_roster.example.json")
        roster["models"][0]["model_id"] = "synthetic-model"
        path = self.root / "roster.json"
        write_json(path, roster)
        prepare_model_preflight(self.gold, path, self.root / "job")
        manifests = read_jsonl(self.root / "job/prompt_request_manifests.jsonl")
        self.assertEqual(audit_prompt_manifests(manifests, self.gold)["status"], "passed")
        by_id = {row["checkpoint_id"]: row for row in self.rows}
        def response(request, *_):
            return {"choices": [{"message": {"content": json.dumps(by_id[request["checkpoint_id"]])}, "finish_reason": "stop"}]}, 200
        with patch.dict("os.environ", {"WARD_BASE_URL": "https://example.invalid/v1", "WARD_API_KEY": "synthetic-placeholder"}), patch("ward_eval.runner._post", side_effect=response):
            run_model_preflight(self.root / "job", path, self.root / "run")
        outputs = read_jsonl(self.root / "run/raw_model_submissions.jsonl")
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(row["submission_status"] == "complete" for row in outputs))
        self.assertEqual(eval_main(["--gold", str(self.root / "gold"), "--submissions", str(self.root / "run/raw_model_submissions.jsonl"), "--prompt-manifests", str(self.root / "job/prompt_request_manifests.jsonl"), "--expected-cases", "1", "--output", str(self.root / "scores")]), 0)

    def test_unresolved_semantics_are_provisional_and_judge_job_enumerates(self):
        self.rows[0]["claims"][0]["text_en"] = "A different synthetic assertion."
        source = self.root / "submissions.jsonl"
        write_jsonl(source, self.rows)
        result = prepare_judge_job(self.gold, source, self.root / "judge-job")
        self.assertGreater(result["pair_count"], 0)
        status = eval_main(["--gold", str(self.root / "gold"), "--submissions", str(source), "--expected-cases", "1", "--output", str(self.root / "scores")])
        self.assertEqual(status, 2)
        self.assertIn("semantic_cache_misses", read_json(self.root / "scores/eval_result.json")["release_evaluation"]["blockers"])
        self.assertFalse(read_json(self.root / "scores/eval_result.json")["formal_result"])
        saved = read_json(self.root / "scores/eval_result.json")
        self.assertEqual(saved["judge"]["decision_count"], len(saved["judge"]["trace"]))
        allowed = eval_main(["--gold", str(self.root / "gold"), "--submissions", str(source), "--expected-cases", "1", "--allow-cache-misses", "--output", str(self.root / "diagnostic")])
        self.assertEqual(allowed, 0)
        self.assertFalse(read_json(self.root / "diagnostic/eval_result.json")["formal_result"])

    def test_schema_and_entrypoints_work_from_another_directory(self):
        self.assertIn("properties", CHECKPOINT_SUBMISSION_SCHEMA)
        for script in ("run.py", "eval.py"):
            process = subprocess.run([sys.executable, str(ROOT / script), "--help"], cwd=self.root, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)

    def test_provider_error_codes_and_optional_compatibility_terms(self):
        self.assertTrue(_non_retryable_provider_error({"error": {"code": "channel_not_found"}}))
        self.assertTrue(_non_retryable_provider_error("deployment offline", ["deployment offline"]))
        self.assertFalse(_non_retryable_provider_error({"error": {"code": "rate_limit_exceeded"}}))

    def test_judge_execution_and_finalization_round_trip(self):
        from jsonschema import Draft202012Validator
        from ward_eval.judge_jobs import run_judge_adapter, finalize_judge_job
        from types import SimpleNamespace
        def validate_artifact(name, value):
            schema = read_json(ROOT / "src/ward_eval/schemas" / f"{name}.v1.schema.json")
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(schema).validate(value)

        def validate_run(directory):
            manifest = read_json(directory / "judge_run_manifest.json")
            validate_artifact("judge_run_manifest", manifest)
            self.assertIsNone(manifest["sampling_parameters"]["seed"])
            for row in read_jsonl(directory / "canonical_responses.jsonl"):
                validate_artifact("judge_response", row)

        def validate_finalization(directory):
            cache = read_json(directory / "private/judge_cache.v1.json")
            validate_artifact("judge_cache", cache)
            for row in read_jsonl(directory / "private/judge_consensus.jsonl"):
                validate_artifact("judge_consensus", row)
            return cache

        self.rows[0]["claims"][0]["text_en"] = "The synthetic measurement has become available."
        source = self.root / "submissions.jsonl"
        write_jsonl(source, self.rows)
        prepare_judge_job(self.gold, source, self.root / "judge-job")
        requests = list((self.root / "judge-job").rglob("batch_*.json"))
        self.assertTrue(requests)
        for request in requests:
            validate_artifact("judge_request", read_json(request))
        response_verdict = "equivalent"
        def execute(_args, **kwargs):
            envelope = json.loads(kwargs["input"])
            request, model = envelope["request"], envelope["model"]
            response = {"schema_version": "ward-judge-response-1.0", "request_id": request["request_id"], "judge_id": request["judge_id"], "results": [{"item_id": item["item_id"], "verdict": response_verdict, "confidence": 1.0, "reason_codes": ["ambiguous" if response_verdict == "abstain" else "same_meaning"]} for item in request["items"]]}
            metadata = {"provider": model["provider"], "requested_model": model["requested_model"], "effective_model": model["requested_model"], "response_id": "synthetic-response", "sampling_parameters": {}, "token_usage": {}, "finish_reason": "stop", "refusal": None, "latency_ms": 0}
            return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({"judge_response": response, "provider_metadata": metadata}))
        for judge_id in ("judge_a", "judge_b"):
            manifest = self.root / f"{judge_id}.json"
            write_json(manifest, {"provider": "synthetic", "requested_model": judge_id})
            with patch("ward_eval.judge_jobs.subprocess.run", side_effect=execute):
                result = run_judge_adapter(self.root / "judge-job", self.root / judge_id,
                    judge_id=judge_id, adapter=Path(sys.executable), adapter_args=[], model_manifest_path=manifest)
            self.assertEqual(result["status"], "complete")
            validate_run(self.root / judge_id)
        result = finalize_judge_job(self.root / "judge-job", self.root / "judge_a", self.root / "judge_b", self.root / "final", decision_policy="primary_judge_b")
        self.assertEqual(result["status"], "passed")
        validate_finalization(self.root / "final")
        judge = FrozenJudge(self.root / "final/private/judge_cache.v1.json")
        scores = score_submissions(self.gold, self.rows, judge=judge, expected_case_count=1)
        self.assertEqual(scores["aggregate"]["headline"]["layer3_claims"]["case_macro_f1"], 1.0)
        self.assertFalse(any(row["source"] == "cache_miss" for row in judge.trace))

        response_verdict = "abstain"
        with patch("ward_eval.judge_jobs.subprocess.run", side_effect=execute):
            result = run_judge_adapter(self.root / "judge-job", self.root / "judge_b_abstain",
                judge_id="judge_b", adapter=Path(sys.executable), adapter_args=[],
                model_manifest_path=self.root / "judge_b.json")
        self.assertEqual(result["status"], "complete")
        validate_run(self.root / "judge_b_abstain")
        result = finalize_judge_job(self.root / "judge-job", self.root / "judge_a",
            self.root / "judge_b_abstain", self.root / "blocked", decision_policy="primary_judge_b")
        self.assertEqual(result["status"], "blocked")
        self.assertIn("human_resolutions_missing", {row["code"] for row in result["hard_gate_blockers"]})
        self.assertFalse((self.root / "blocked/private/judge_cache.v1.json").exists())
        resolutions = read_jsonl(self.root / "blocked/private/human_resolutions.required.jsonl")
        self.assertTrue(resolutions)
        for resolution in resolutions:
            resolution.update(annotator_id="synthetic-reviewer", verdict="equivalent",
                confidence="high", reason_codes=["same_meaning"], reviewed_at="2026-01-01T00:00:00Z")
            validate_artifact("human_judge_label", resolution)
        resolution_path = self.root / "resolutions.jsonl"
        write_jsonl(resolution_path, resolutions)
        result = finalize_judge_job(self.root / "judge-job", self.root / "judge_a",
            self.root / "judge_b_abstain", self.root / "resolved",
            human_resolutions=resolution_path, decision_policy="primary_judge_b")
        self.assertEqual(result["status"], "passed")
        cache = validate_finalization(self.root / "resolved")
        self.assertTrue(all(row["source"] == "human_adjudication" for row in cache["decisions"].values()))

    def test_anthropic_parameters_and_unknown_protocol(self):
        from ward_eval.runner import _post
        from io import BytesIO
        request = {"protocol": "anthropic_messages", "model_id": "synthetic-model",
            "system_prompt": "Synthetic task.", "user_prompt": "Synthetic evidence.",
            "max_tokens": 64, "stream": False, "temperature": 0}
        response = BytesIO(b'{"content":[]}')
        response.status = 200
        with patch("urllib.request.urlopen", return_value=response) as transport:
            _post(request, "https://gateway.example", "synthetic-placeholder", 1)
        payload = json.loads(transport.call_args.args[0].data)
        self.assertEqual(payload["temperature"], 0)
        request["protocol"] = "unsupported"
        with self.assertRaises(ValueError):
            _post(request, "https://gateway.example", "synthetic-placeholder", 1)
