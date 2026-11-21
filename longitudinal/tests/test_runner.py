import json
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from unittest.mock import patch

from ward_eval.runner import (
    SYSTEM_PROMPT,
    _model_visible_payload,
    _invalid_visible_evidence_ids,
    _non_retryable_provider_error,
    _post,
    _preflight_report,
    _provider_outcome,
    _remove_outer_markdown_fence,
    _read_sse,
    prepare_model_preflight,
)
from ward_eval.util import canonical_sha256, read_jsonl


def _request():
    return {
        "request_id": "r1",
        "request_sha256": "a" * 64,
        "model_id": "test-model",
        "protocol": "openai_chat",
        "case_id": "ward_001",
        "checkpoint_id": "ward_001_cp_000",
        "public_snapshot_sha256": "b" * 64,
        "visible_evidence": [],
    }


def _checkpoint(visible_ids, evidence, future_ids=()):
    return SimpleNamespace(
        case_id="ward_001",
        checkpoint_id="ward_001_cp_001",
        snapshot_sha256="b" * 64,
        future_evidence_ids=set(future_ids),
        case={"normalized_evidence": evidence},
        checkpoint={
            "question": {
                "public_snapshot": {
                    "public_time": "2124-01-01T00:00:00",
                    "public_context": {"setting": "inpatient"},
                    "visible_evidence_ids": visible_ids,
                    "environment_state_facts": {"diagnosis_confirmed": True},
                    "stage": "S2",
                }
            }
        },
    )


def _evidence(evidence_id="doc:001", content="Clinical evidence."):
    return {
        "evidence_id": evidence_id,
        "record_kind": "document",
        "title": "Clinical note",
        "clinical_time": "2124-01-01T00:00:00",
        "available_time": "2124-01-01T00:01:00",
        "raw_content": content,
    }


def _complete_submission(*, include_snapshot=True):
    submission = {
        "schema_version": "ward-checkpoint-submission-1.1",
        "case_id": "ward_001",
        "checkpoint_id": "ward_001_cp_000",
        "language": "en",
        "submission_status": "complete",
        "key_evidence": [],
        "utilization_links": [],
        "claims": [],
        "patient_state": {
            "diagnoses": [], "active_problems": [], "resolved_problems": [],
            "blockers": [], "critical_facts_to_retain": [],
            "facts_to_retract_or_correct": [], "required_delta_from_previous_checkpoint": [],
            "treatment_readiness": None, "management_state": None,
        },
        "actions": [],
        "safety": {"hard_failure_rules": [], "critical_omission_rules": [], "preliminary_risk_triggers": []},
        "stopping": {"decision": "continue_investigation", "remaining_blockers": [], "rationale_en": "More evidence is needed."},
    }
    if include_snapshot:
        submission["public_snapshot_sha256"] = "b" * 64
    return submission


class RunnerTests(unittest.TestCase):
    def test_runner_bound_snapshot_is_removed_from_model_contract(self):
        checkpoint = _checkpoint(["doc:001"], [_evidence()])
        gold = SimpleNamespace(checkpoint_by_id={checkpoint.checkpoint_id: checkpoint})
        roster = {
            "execution": {"max_tokens": 100, "stream": False},
            "checkpoint_ids": [checkpoint.checkpoint_id],
            "models": [{
                "model_id": "test-model",
                "protocol": "openai_chat",
                "runner_supplied_public_snapshot_sha256": True,
            }],
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            roster_path = root / "roster.json"
            roster_path.write_text(json.dumps(roster), encoding="utf-8")
            output_dir = root / "job"
            prepare_model_preflight(gold, roster_path, output_dir)
            request = read_jsonl(output_dir / "requests.jsonl")[0]
            manifest = read_jsonl(output_dir / "prompt_request_manifests.jsonl")[0]

        self.assertTrue(request["runner_supplied_public_snapshot_sha256"])
        self.assertNotIn("public_snapshot_sha256:", request["user_prompt"])
        self.assertNotIn("public_snapshot_sha256", manifest["request_payload"]["tool_schema"]["required"])
        self.assertNotIn("public_snapshot_sha256", manifest["request_payload"]["tool_schema"]["properties"])

    def test_runner_bound_snapshot_is_injected_before_validation(self):
        request = _request() | {"runner_supplied_public_snapshot_sha256": True}
        response = {
            "id": "resp",
            "model": "effective",
            "choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps(_complete_submission(include_snapshot=False)),
            }}],
        }
        outcome = _provider_outcome(request, response, 0)
        self.assertEqual(outcome["output_status"], "complete")
        self.assertEqual(outcome["submission"]["public_snapshot_sha256"], "b" * 64)
        self.assertEqual(outcome["runner_binding"], {
            "method": "request_envelope",
            "supplied_fields": ["public_snapshot_sha256"],
        })

    def test_missing_snapshot_remains_invalid_in_strict_protocol(self):
        response = {
            "id": "resp",
            "model": "effective",
            "choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps(_complete_submission(include_snapshot=False)),
            }}],
        }
        outcome = _provider_outcome(_request(), response, 0)
        self.assertEqual(outcome["output_status"], "schema_invalid")

    def test_prepared_request_contains_visible_evidence_and_payload_hash(self):
        checkpoint = _checkpoint(["doc:001"], [_evidence()])
        gold = SimpleNamespace(checkpoint_by_id={checkpoint.checkpoint_id: checkpoint})
        roster = {
            "execution": {"max_tokens": 100, "stream": False},
            "checkpoint_ids": [checkpoint.checkpoint_id],
            "models": [{
                "model_id": "test-model",
                "provider_model_id": "provider/test-model",
                "enable_thinking": False,
                "protocol": "openai_chat",
            }],
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            roster_path = root / "roster.json"
            roster_path.write_text(json.dumps(roster), encoding="utf-8")
            output_dir = root / "job"
            prepare_model_preflight(gold, roster_path, output_dir)
            request = read_jsonl(output_dir / "requests.jsonl")[0]
            manifest = read_jsonl(output_dir / "prompt_request_manifests.jsonl")[0]

        payload_text = request["user_prompt"].split(
            "<UNTRUSTED_CLINICAL_DATA>\n", 1
        )[1].split("\n</UNTRUSTED_CLINICAL_DATA>", 1)[0]
        payload = json.loads(payload_text)
        self.assertEqual(payload["visible_evidence"][0]["content"], "Clinical evidence.")
        self.assertEqual(request["provider_model_id"], "provider/test-model")
        self.assertIs(request["enable_thinking"], False)
        self.assertIs(
            manifest["request_payload"]["sampling_parameters"]["enable_thinking"],
            False,
        )
        self.assertEqual(request["model_visible_payload_sha256"], canonical_sha256(payload))
        self.assertEqual(
            manifest["visible_evidence_sha256"], canonical_sha256(["doc:001"])
        )
        self.assertEqual(manifest["request_payload"]["model_visible_payload"], payload)

    def test_visible_payload_projects_raw_evidence_without_environment_state(self):
        payload = _model_visible_payload(_checkpoint(["doc:001"], [_evidence()]))
        self.assertEqual(payload["visible_evidence"][0]["content"], "Clinical evidence.")
        self.assertNotIn("environment_state_facts", payload)
        self.assertNotIn("stage", payload)

    def test_empty_visible_evidence_is_a_valid_checkpoint(self):
        payload = _model_visible_payload(_checkpoint([], []))
        self.assertEqual(payload["visible_evidence"], [])
        self.assertIn("empty visible_evidence array is a valid initial checkpoint", SYSTEM_PROMPT)

    def test_missing_visible_evidence_fails_at_job_boundary(self):
        with self.assertRaisesRegex(ValueError, "visible evidence not found"):
            _model_visible_payload(_checkpoint(["doc:missing"], []))

    def test_empty_visible_evidence_content_fails_at_job_boundary(self):
        with self.assertRaisesRegex(ValueError, "visible evidence has empty content"):
            _model_visible_payload(_checkpoint(["doc:001"], [_evidence(content="")]))

    def test_future_evidence_fails_at_job_boundary(self):
        with self.assertRaisesRegex(ValueError, "future evidence exposed"):
            _model_visible_payload(_checkpoint(["doc:001"], [_evidence()], future_ids=["doc:001"]))

    def test_provider_model_alias_is_sent_without_changing_model_id(self):
        request = _request() | {
            "provider_model_id": "provider/test-model",
            "enable_thinking": False,
            "system_prompt": "system",
            "user_prompt": "user",
            "max_tokens": 10,
            "stream": False,
        }
        response = BytesIO(b'{"choices": []}')
        response.status = 200
        with patch("ward_eval.runner.urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = response
            _post(request, "https://gateway.example", "key", 30)
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["model"], "provider/test-model")
        self.assertIs(payload["enable_thinking"], False)
        self.assertEqual(request["model_id"], "test-model")

    def test_explicit_openai_endpoint_and_temperature_are_sent(self):
        request = _request() | {
            "provider_model_id": "med-palm-2",
            "endpoint_url": "https://gateway.example/api/v1/medical/chat/completions",
            "user_agent": "curl/8.7.1",
            "temperature": 0,
            "system_prompt": "system",
            "user_prompt": "user",
            "max_tokens": 10,
            "stream": False,
        }
        response = BytesIO(b'{"choices": []}')
        response.status = 200
        with patch("ward_eval.runner.urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = response
            _post(request, "https://unused.example", "key", 30)
        http_request = urlopen.call_args.args[0]
        self.assertEqual(http_request.full_url, request["endpoint_url"])
        self.assertEqual(http_request.get_header("User-agent"), "curl/8.7.1")
        payload = json.loads(http_request.data)
        self.assertEqual(payload["model"], "med-palm-2")
        self.assertEqual(payload["temperature"], 0)

    def test_fenced_json_is_not_repaired(self):
        fenced_submission = {
            "schema_version": "ward-checkpoint-submission-1.1",
            "case_id": "ward_001",
            "checkpoint_id": "ward_001_cp_000",
            "public_snapshot_sha256": "b" * 64,
            "language": "en",
            "submission_status": "schema_invalid",
            "raw_output_sha256": None,
            "error_code": "model_wrapped_json_in_markdown",
        }
        response = {
            "id": "resp",
            "model": "effective",
            "choices": [{"finish_reason": "stop", "message": {
                "content": f"```json\n{json.dumps(fenced_submission)}\n```",
            }}],
        }
        outcome = _provider_outcome(_request(), response, 0)
        self.assertEqual(outcome["output_status"], "schema_invalid")
        self.assertEqual(outcome["submission"]["submission_status"], "schema_invalid")
        self.assertTrue(outcome["diagnostic_recovery"]["diagnostically_recoverable"])
        self.assertTrue(outcome["diagnostic_recovery"]["official_output_status_unchanged"])

    def test_outer_markdown_fence_removal_requires_a_complete_wrapper(self):
        self.assertEqual(
            _remove_outer_markdown_fence("```json\n{\"status\":\"ok\"}\n```"),
            '{"status":"ok"}',
        )
        self.assertIsNone(_remove_outer_markdown_fence("```json\n{}"))

    def test_unclosed_json_fence_is_diagnostic_only(self):
        fenced_submission = {
            "schema_version": "ward-checkpoint-submission-1.1",
            "case_id": "ward_001",
            "checkpoint_id": "ward_001_cp_000",
            "public_snapshot_sha256": "b" * 64,
            "language": "en",
            "submission_status": "schema_invalid",
            "raw_output_sha256": None,
            "error_code": "model_started_markdown_fence",
        }
        response = {
            "id": "resp",
            "model": "effective",
            "choices": [{"finish_reason": "stop", "message": {
                "content": f"```json\n{json.dumps(fenced_submission)}",
            }}],
        }
        outcome = _provider_outcome(_request(), response, 0)
        self.assertEqual(outcome["output_status"], "schema_invalid")
        self.assertTrue(outcome["diagnostic_recovery"]["diagnostically_recoverable"])
        self.assertFalse(outcome["diagnostic_recovery"]["closing_fence_present"])

    def test_length_finish_is_truncated_without_parse(self):
        response = {
            "id": "resp",
            "model": "effective",
            "choices": [{"finish_reason": "length", "message": {"content": json.dumps({"partial": True})}}],
        }
        outcome = _provider_outcome(_request(), response, 0)
        self.assertEqual(outcome["output_status"], "truncated")
        self.assertEqual(outcome["submission"]["submission_status"], "truncated")

    def test_invisible_evidence_id_is_schema_invalid(self):
        submission = {
            "schema_version": "ward-checkpoint-submission-1.1",
            "case_id": "ward_001",
            "checkpoint_id": "ward_001_cp_000",
            "public_snapshot_sha256": "b" * 64,
            "language": "en",
            "submission_status": "complete",
            "key_evidence": [{"evidence_id": "doc:hidden", "role": "critical"}],
            "utilization_links": [],
            "claims": [],
            "patient_state": {
                "diagnoses": [], "active_problems": [], "resolved_problems": [],
                "blockers": [], "critical_facts_to_retain": [],
                "facts_to_retract_or_correct": [], "required_delta_from_previous_checkpoint": [],
                "treatment_readiness": None, "management_state": None,
            },
            "actions": [],
            "safety": {"hard_failure_rules": [], "critical_omission_rules": [], "preliminary_risk_triggers": []},
            "stopping": {"decision": "continue_investigation", "remaining_blockers": [], "rationale_en": "More evidence is needed."},
        }
        response = {
            "id": "resp",
            "model": "effective",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(submission)}}],
        }
        outcome = _provider_outcome(_request(), response, 0)
        self.assertEqual(outcome["output_status"], "schema_invalid")
        self.assertEqual(outcome["validation_issues"][-1], {
            "code": "invisible_evidence_id",
            "evidence_ids": ["doc:hidden"],
        })

    def test_visible_evidence_id_helper_checks_nested_references(self):
        submission = {"claims": [{"evidence_ids": ["doc:visible", "doc:hidden"]}]}
        self.assertEqual(
            _invalid_visible_evidence_ids(submission, ["doc:visible"]),
            ["doc:hidden"],
        )

    def test_missing_model_channel_is_not_retryable(self):
        self.assertTrue(_non_retryable_provider_error({
            "error": {"code": "model_not_found", "message": "No available channel for model x"}
        }))

    def test_transport_error_blocks_formal_run_readiness(self):
        request = _request()
        outcome = {
            "request_id": request["request_id"],
            "case_id": request["case_id"],
            "checkpoint_id": request["checkpoint_id"],
            "public_snapshot_sha256": request["public_snapshot_sha256"],
            "output_status": "transport_error",
            "effective_model": None,
            "submission": {
                "schema_version": "ward-checkpoint-submission-1.1",
                "case_id": request["case_id"],
                "checkpoint_id": request["checkpoint_id"],
                "public_snapshot_sha256": request["public_snapshot_sha256"],
                "language": "en",
                "submission_status": "transport_error",
            },
        }
        report = _preflight_report([request], [outcome])
        self.assertEqual(report["artifact_validation_status"], "passed")
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["formal_run_ready"])

    def test_truncated_output_is_a_formal_model_result(self):
        request = _request()
        outcome = {
            "request_id": request["request_id"],
            "case_id": request["case_id"],
            "checkpoint_id": request["checkpoint_id"],
            "public_snapshot_sha256": request["public_snapshot_sha256"],
            "output_status": "truncated",
            "effective_model": "effective",
            "submission": {
                "schema_version": "ward-checkpoint-submission-1.1",
                "case_id": request["case_id"],
                "checkpoint_id": request["checkpoint_id"],
                "public_snapshot_sha256": request["public_snapshot_sha256"],
                "language": "en",
                "submission_status": "truncated",
                "raw_output_sha256": "c" * 64,
                "error_code": "truncated",
            },
        }
        report = _preflight_report([request], [outcome])
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["formal_run_ready"])

    def test_openai_sse_is_combined_without_text_repair(self):
        stream = BytesIO(
            b'data: {"id":"r","model":"m","choices":[{"delta":{"content":"{\\"ok\\":"},"finish_reason":null}]}\n\n'
            b'data: {"id":"r","model":"m","choices":[{"delta":{"content":"true}"},"finish_reason":"stop"}],"usage":{"completion_tokens":2}}\n\n'
            b'data: [DONE]\n\n'
        )
        response = _read_sse(stream, "openai_chat")
        self.assertEqual(response["choices"][0]["message"]["content"], '{"ok":true}')
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(response["usage"]["completion_tokens"], 2)
