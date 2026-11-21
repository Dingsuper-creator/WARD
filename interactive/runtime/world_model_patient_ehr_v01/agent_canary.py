"""Non-official real-Agent canary runner for the deterministic Patient/EHR environment."""

from __future__ import annotations

import copy
import json
import re
import ssl
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from .contracts import (
    ACTION_TYPES,
    ENVIRONMENT_VERSION,
    RUN_MANIFEST_VERSION,
    InteractionContractError,
    canonical_hash,
    credential_findings,
    read_json,
    sha256_file,
    validate_run_manifest,
    write_json,
)
from .environment import PatientEHREnvironment, validate_trace


AGENT_RUNNER_VERSION = "ward-patient-ehr-agent-canary-runner-0.1.0"
AGENT_PROMPT_VERSION = "ward-patient-ehr-agent-prompt-0.1.0"
AGENT_CALL_AUDIT_VERSION = "ward-patient-ehr-agent-call-audit-0.1.0"
DEFAULT_AGENT_MODEL = "gpt-5.6-sol"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 800
DEFAULT_MAX_ACTIONS = 36

CURRENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_CONTRACT = (
    CURRENT_ROOT
    / "outputs/world_model_patient_ehr_v01/agent_canary/PROMPT_CONTRACT.json"
)


class AgentClient(Protocol):
    """Minimal client boundary used by the canary loop and unit-test policies."""

    def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Return content, effective_model, usage, and optional response_id."""


def safe_name(value: str, *, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,119}", value):
        raise InteractionContractError(
            f"{label} must use 3-120 safe filename characters"
        )
    return value


def normalize_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    return normalized if normalized.endswith("/v1") else normalized + "/v1"


def is_loopback_url(value: str) -> bool:
    return urllib.parse.urlsplit(value).hostname in {"127.0.0.1", "localhost", "::1"}


def connection_configuration_errors(
    base_url: str,
    api_key: str,
    *,
    allow_insecure_http: bool = False,
) -> list[str]:
    errors: list[str] = []
    try:
        base_url.encode("ascii")
    except UnicodeEncodeError:
        errors.append("base_url_non_ascii")
        return errors
    parsed = urllib.parse.urlsplit(normalize_base_url(base_url))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        errors.append("base_url_invalid_scheme_or_host")
    if parsed.username is not None or parsed.password is not None:
        errors.append("base_url_embedded_credentials")
    if parsed.query or parsed.fragment:
        errors.append("base_url_query_or_fragment_present")
    if (
        parsed.scheme == "http"
        and not is_loopback_url(base_url)
        and not allow_insecure_http
    ):
        errors.append("base_url_non_loopback_http")
    if api_key:
        try:
            api_key.encode("ascii")
        except UnicodeEncodeError:
            errors.append("key_non_ascii")
        if any(character.isspace() for character in api_key):
            errors.append("key_contains_whitespace")
    return errors


def validate_api_connection_inputs(
    base_url: str,
    api_key: str,
    *,
    allow_insecure_http: bool = False,
) -> None:
    errors = connection_configuration_errors(
        base_url,
        api_key,
        allow_insecure_http=allow_insecure_http,
    )
    if errors:
        messages = {
            "base_url_non_ascii": "API base URL must contain ASCII only",
            "base_url_invalid_scheme_or_host": "invalid API base URL scheme or host",
            "base_url_embedded_credentials": (
                "API base URL must not contain embedded credentials"
            ),
            "base_url_query_or_fragment_present": (
                "API base URL must not contain a query string or fragment"
            ),
            "base_url_non_loopback_http": (
                "non-loopback API base URL must use HTTPS"
            ),
            "key_non_ascii": "API key must contain ASCII only",
            "key_contains_whitespace": "API key must not contain whitespace",
        }
        raise InteractionContractError(messages[errors[0]])


def readiness(
    *,
    base_url: str,
    api_key: str,
    requested_model: str,
    allow_insecure_http: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    configuration_error_codes: list[str] = []
    if not base_url:
        errors.append("agent_base_url_environment_missing")
    if not api_key and (not base_url or not is_loopback_url(base_url)):
        errors.append("agent_key_environment_missing")
    if not requested_model.strip():
        errors.append("agent_model_missing")
    if base_url:
        configuration_error_codes = connection_configuration_errors(
            base_url,
            api_key,
            allow_insecure_http=allow_insecure_http,
        )
        if configuration_error_codes:
            errors.append("agent_connection_configuration_invalid")
    return {
        "ready": not errors,
        "errors": errors,
        "configuration_error_codes": configuration_error_codes,
        "requested_model": requested_model,
        "base_url_environment_set": bool(base_url),
        "key_environment_set": bool(api_key),
        "insecure_http_explicit_opt_in": allow_insecure_http,
        "network_transport_encrypted": (
            urllib.parse.urlsplit(normalize_base_url(base_url)).scheme == "https"
            if base_url
            else None
        ),
        "credential_storage": "environment_only",
        "credentials_echoed": False,
        "official_scoring_allowed": False,
        "single_total_score_absent": True,
    }


def _content_text(content: Any) -> str:
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "").strip()


def parse_agent_action(content: Any) -> dict[str, Any]:
    text = _content_text(content)
    if credential_findings(text):
        raise InteractionContractError("credential-like model output rejected")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise InteractionContractError(
                "model response does not contain a JSON object"
            ) from None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise InteractionContractError(
                "model response contains invalid JSON"
            ) from exc
    if not isinstance(value, dict):
        raise InteractionContractError("model response must be a JSON object")
    allowed = {"action_type", "target", "parameters", "rationale", "evidence_ids"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise InteractionContractError("model action contains unsupported fields")
    if credential_findings(value):
        raise InteractionContractError("credential-like model action rejected")
    action_type = value.get("action_type")
    if action_type not in ACTION_TYPES:
        raise InteractionContractError("model action_type is not allowed")
    target = value.get("target")
    if not isinstance(target, str) or not target.strip() or len(target) > 240:
        raise InteractionContractError("model target must be 1-240 characters")
    parameters = value.get("parameters", {})
    if not isinstance(parameters, dict):
        raise InteractionContractError("model parameters must be an object")
    if len(json.dumps(parameters, ensure_ascii=False)) > 2000:
        raise InteractionContractError("model parameters exceed size limit")
    rationale = value.get("rationale", "")
    if not isinstance(rationale, str) or len(rationale) > 1200:
        raise InteractionContractError("model rationale must be at most 1200 characters")
    evidence_ids = value.get("evidence_ids", [])
    if (
        not isinstance(evidence_ids, list)
        or any(not isinstance(item, str) or not item for item in evidence_ids)
        or len(evidence_ids) != len(set(evidence_ids))
    ):
        raise InteractionContractError(
            "model evidence_ids must be a unique list of strings"
        )
    return {
        "action_type": action_type,
        "target": target.strip(),
        "parameters": copy.deepcopy(parameters),
        "rationale": rationale.strip(),
        "evidence_ids": list(evidence_ids),
    }


def load_prompt_contract(path: Path = DEFAULT_PROMPT_CONTRACT) -> dict[str, Any]:
    contract = read_json(path)
    if contract.get("prompt_version") != AGENT_PROMPT_VERSION:
        raise InteractionContractError("wrong Agent prompt contract version")
    if contract.get("gold_action_visible") is not False:
        raise InteractionContractError("Agent prompt cannot expose Gold action")
    if contract.get("future_evidence_visible") is not False:
        raise InteractionContractError("Agent prompt cannot expose future evidence")
    if credential_findings(contract):
        raise InteractionContractError("prompt contract contains credential-like material")
    return contract


def build_agent_messages(
    *,
    prompt_contract: dict[str, Any],
    observation: dict[str, Any],
    evidence_memory: dict[str, dict[str, Any]],
    prior_public_attempts: list[dict[str, Any]],
) -> list[dict[str, str]]:
    public_observation = {
        "case_id": observation["case_id"],
        "stage": observation["stage"],
        "step_index": observation["step_index"],
        "environment_clock": observation["environment_clock"],
        "public_context": copy.deepcopy(observation["public_context"]),
        "visible_evidence_ids": list(observation["visible_evidence_ids"]),
        "visible_evidence": [
            copy.deepcopy(evidence_memory[evidence_id])
            for evidence_id in observation["visible_evidence_ids"]
            if evidence_id in evidence_memory
        ],
        "done": observation["done"],
        "action_schema": copy.deepcopy(observation["action_schema"]),
    }
    system_payload = {
        "prompt_version": prompt_contract["prompt_version"],
        "role": prompt_contract["role"],
        "objective": prompt_contract["objective"],
        "rules": prompt_contract["rules"],
        "action_type_semantics": prompt_contract["action_type_semantics"],
        "output_contract": prompt_contract["output_contract"],
        "feedback_boundary": prompt_contract["feedback_boundary"],
    }
    user_payload = {
        "instruction": (
            "Select exactly one clinically appropriate next action using only the "
            "public observation and visible evidence. Return only the JSON action."
        ),
        "observation": public_observation,
        "prior_public_attempts": copy.deepcopy(prior_public_attempts),
    }
    messages = [
        {
            "role": "system",
            "content": json.dumps(system_payload, ensure_ascii=False, sort_keys=True),
        },
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False, sort_keys=True),
        },
    ]
    if credential_findings(messages):
        raise InteractionContractError("Agent messages contain credential-like material")
    return messages


class ChatCompletionsAgentClient:
    """Small OpenAI-compatible transport that never serializes credentials."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: int = 240,
        ca_bundle: Path | None = None,
        allow_insecure_http: bool = False,
    ) -> None:
        validate_api_connection_inputs(
            base_url,
            api_key,
            allow_insecure_http=allow_insecure_http,
        )
        if not api_key and not is_loopback_url(base_url):
            raise InteractionContractError("API key is required for non-loopback URL")
        if not model.strip():
            raise InteractionContractError("model is required")
        if temperature != DEFAULT_TEMPERATURE:
            raise InteractionContractError("Agent canary temperature is locked to 0")
        if max_tokens != DEFAULT_MAX_TOKENS:
            raise InteractionContractError("Agent canary max_tokens is locked to 800")
        if timeout < 1 or timeout > 600:
            raise InteractionContractError("timeout must be between 1 and 600 seconds")
        self._url = normalize_base_url(base_url) + "/chat/completions"
        self._api_key = api_key
        self._model = model.strip()
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._ca_bundle = ca_bundle
        self._allow_insecure_http = allow_insecure_http

    def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        context = (
            ssl.create_default_context(cafile=str(self._ca_bundle))
            if self._ca_bundle
            else ssl.create_default_context()
        )
        with urllib.request.urlopen(
            request, timeout=self._timeout, context=context
        ) as response:
            body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, dict):
            raise InteractionContractError("API response must be an object")
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InteractionContractError(
                "API response is missing assistant content"
            ) from exc
        usage = body.get("usage", {})
        if not isinstance(usage, dict):
            usage = {}
        clean_usage = {
            key: value
            for key, value in usage.items()
            if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
        }
        return {
            "content": content,
            "effective_model": str(body.get("model") or self._model),
            "usage": clean_usage,
            "response_id": (
                str(body["id"])[:200]
                if isinstance(body.get("id"), str)
                else None
            ),
        }


def _remember_evidence(
    observation: dict[str, Any],
    evidence_memory: dict[str, dict[str, Any]],
) -> None:
    for evidence in observation.get("new_evidence", []):
        if isinstance(evidence, dict) and isinstance(evidence.get("evidence_id"), str):
            evidence_memory[evidence["evidence_id"]] = copy.deepcopy(evidence)


def _clean_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        key: item
        for key, item in value.items()
        if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
        and isinstance(item, int)
        and not isinstance(item, bool)
        and item >= 0
    }


def validate_agent_call_audit(audit: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if audit.get("audit_version") != AGENT_CALL_AUDIT_VERSION:
        errors.append("wrong_agent_call_audit_version")
    if credential_findings(audit):
        errors.append("credential_like_material_present")
    if audit.get("gold_action_exposed_to_agent") is not False:
        errors.append("gold_exposure_boundary_invalid")
    if audit.get("future_evidence_exposed_to_agent") is not False:
        errors.append("future_evidence_boundary_invalid")
    if audit.get("internal_assessment_exposed_to_agent") is not False:
        errors.append("internal_assessment_boundary_invalid")
    transport_mode = audit.get("transport_security_mode")
    if transport_mode not in {
        "https",
        "loopback_http",
        "insecure_http_explicit_opt_in",
        "test_stub_no_network",
    }:
        errors.append("invalid_transport_security_mode")
    if (
        transport_mode == "insecure_http_explicit_opt_in"
        and audit.get("network_transport_encrypted") is not False
    ):
        errors.append("insecure_http_encryption_record_invalid")
    calls = audit.get("calls")
    if not isinstance(calls, list):
        errors.append("calls_not_list")
    else:
        for index, call in enumerate(calls):
            if not isinstance(call, dict):
                errors.append(f"invalid_call:{index}")
                continue
            for key in ("request_sha256", "assistant_content_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", str(call.get(key, ""))):
                    errors.append(f"invalid_{key}:{index}")
            if "assistant_content" in call:
                errors.append(f"raw_assistant_content_present:{index}")
    if audit.get("official_scoring_allowed") is not False:
        errors.append("official_scoring_boundary_invalid")
    if audit.get("single_total_score_absent") is not True:
        errors.append("single_total_score_boundary_invalid")
    return errors


def validate_agent_canary_manifest(
    manifest: dict[str, Any],
    *,
    trace_path: Path,
    call_audit_path: Path,
    scenario_path: Path,
    freeze_manifest_path: Path,
    prompt_contract_path: Path,
) -> list[str]:
    errors = validate_run_manifest(
        manifest,
        trace_path=trace_path,
        scenario_path=scenario_path,
        freeze_manifest_path=freeze_manifest_path,
    )
    if manifest.get("runner_version") != AGENT_RUNNER_VERSION:
        errors.append("wrong_agent_runner_version")
    agent = manifest.get("agent", {})
    if not isinstance(agent, dict):
        errors.append("missing_agent")
    else:
        if agent.get("deterministic_gold_replay") is not False:
            errors.append("deterministic_gold_replay_boundary_invalid")
        if agent.get("gold_accessor_used") is not False:
            errors.append("gold_accessor_boundary_invalid")
    transport = manifest.get("transport_security")
    if not isinstance(transport, dict):
        errors.append("missing_transport_security")
    else:
        mode = transport.get("mode")
        audit_mode = read_json(call_audit_path).get("transport_security_mode")
        if mode != audit_mode:
            errors.append("transport_security_audit_mismatch")
        if mode == "insecure_http_explicit_opt_in":
            if transport.get("insecure_http_explicit_opt_in") is not True:
                errors.append("missing_insecure_http_explicit_opt_in")
            if transport.get("network_transport_encrypted") is not False:
                errors.append("insecure_http_encryption_record_invalid")
    bindings = manifest.get("bindings", {})
    expected = {
        "agent_call_audit_sha256": sha256_file(call_audit_path),
        "agent_prompt_contract_sha256": sha256_file(prompt_contract_path),
    }
    for key, value in expected.items():
        if bindings.get(key) != value:
            errors.append(f"{key}_mismatch")
    audit_errors = validate_agent_call_audit(read_json(call_audit_path))
    errors.extend(audit_errors)
    return errors


def run_agent_canary(
    *,
    client: AgentClient,
    scenario_path: Path,
    freeze_manifest_path: Path,
    prompt_contract_path: Path,
    output_dir: Path,
    run_id: str,
    agent_id: str,
    requested_model: str,
    max_actions: int = DEFAULT_MAX_ACTIONS,
    transport_security_mode: str = "test_stub_no_network",
) -> dict[str, Any]:
    safe_name(run_id, label="run_id")
    safe_name(agent_id, label="agent_id")
    if max_actions < 1 or max_actions > 120:
        raise InteractionContractError("max_actions must be between 1 and 120")
    if transport_security_mode not in {
        "https",
        "loopback_http",
        "insecure_http_explicit_opt_in",
        "test_stub_no_network",
    }:
        raise InteractionContractError("invalid transport_security_mode")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise InteractionContractError("output directory already contains artifacts")
    prompt_contract = load_prompt_contract(prompt_contract_path)
    prompt_sha = sha256_file(prompt_contract_path)
    environment = PatientEHREnvironment(
        scenario_path,
        freeze_manifest_path,
        run_id=run_id,
        agent_id=agent_id,
    )
    observation = environment.reset()
    evidence_memory: dict[str, dict[str, Any]] = {}
    _remember_evidence(observation, evidence_memory)
    public_attempts: list[dict[str, Any]] = []
    call_records: list[dict[str, Any]] = []
    effective_models: list[str] = []
    terminal_reason = "action_budget_exhausted"
    api_error_type: str | None = None

    for call_index in range(max_actions):
        if observation["done"]:
            terminal_reason = "environment_terminal"
            break
        messages = build_agent_messages(
            prompt_contract=prompt_contract,
            observation=observation,
            evidence_memory=evidence_memory,
            prior_public_attempts=public_attempts,
        )
        request_sha = canonical_hash(
            {
                "prompt_contract_sha256": prompt_sha,
                "messages": messages,
                "requested_model": requested_model,
                "temperature": DEFAULT_TEMPERATURE,
                "max_tokens": DEFAULT_MAX_TOKENS,
            }
        )
        try:
            response = client.complete(messages)
            raw_content = _content_text(response.get("content"))
            content_sha = canonical_hash(raw_content)
            effective_model = str(response.get("effective_model") or requested_model)
            effective_models.append(effective_model)
            record = {
                "call_index": call_index,
                "request_sha256": request_sha,
                "assistant_content_sha256": content_sha,
                "response_id": (
                    response.get("response_id")
                    if isinstance(response.get("response_id"), str)
                    else None
                ),
                "effective_model": effective_model,
                "usage": _clean_usage(response.get("usage")),
                "raw_assistant_content_stored": False,
            }
            try:
                action = parse_agent_action(raw_content)
            except InteractionContractError as exc:
                record["status"] = "invalid_action_contract"
                record["error_code"] = str(exc)
                call_records.append(record)
                public_attempts.append(
                    {
                        "call_index": call_index,
                        "outcome": "invalid_action_contract",
                        "state_advanced": False,
                    }
                )
                continue
            observation, assessment, done = environment.step(action)
            _remember_evidence(observation, evidence_memory)
            record.update(
                {
                    "status": "action_submitted",
                    "action_sha256": canonical_hash(action),
                    "accepted": assessment["accepted"],
                    "state_advanced": assessment["state_advanced"],
                    "newly_released_evidence_ids": list(
                        observation["new_evidence"][index]["evidence_id"]
                        for index in range(len(observation["new_evidence"]))
                    ),
                }
            )
            call_records.append(record)
            public_attempts.append(
                {
                    "call_index": call_index,
                    "action": copy.deepcopy(action),
                    "outcome": (
                        "accepted" if assessment["accepted"] else "not_accepted"
                    ),
                    "state_advanced": assessment["state_advanced"],
                    "newly_released_evidence_ids": list(
                        record["newly_released_evidence_ids"]
                    ),
                }
            )
            if done:
                terminal_reason = "environment_terminal"
                break
        except Exception as exc:  # transport/provider failure; never persist its text
            api_error_type = type(exc).__name__
            terminal_reason = "api_error"
            break

    trace = environment.export_trace()
    trace_errors = validate_trace(trace)
    if trace_errors:
        raise InteractionContractError(
            "generated Agent trace is invalid: " + "; ".join(trace_errors)
        )
    accepted_action_count = sum(
        1
        for call in call_records
        if call.get("status") == "action_submitted" and call.get("accepted") is True
    )
    submitted_action_count = sum(
        1 for call in call_records if call.get("status") == "action_submitted"
    )
    call_audit = {
        "audit_version": AGENT_CALL_AUDIT_VERSION,
        "runner_version": AGENT_RUNNER_VERSION,
        "prompt_version": AGENT_PROMPT_VERSION,
        "run_id": run_id,
        "case_id": trace["case_id"],
        "requested_model": requested_model,
        "effective_models_observed": sorted(set(effective_models)),
        "call_count": len(call_records),
        "submitted_action_count": submitted_action_count,
        "accepted_action_count": accepted_action_count,
        "terminal_reason": terminal_reason,
        "api_error_type": api_error_type,
        "calls": call_records,
        "gold_action_exposed_to_agent": False,
        "future_evidence_exposed_to_agent": False,
        "internal_assessment_exposed_to_agent": False,
        "raw_assistant_content_stored": False,
        "credential_storage": "environment_only",
        "transport_security_mode": transport_security_mode,
        "network_transport_encrypted": transport_security_mode == "https",
        "credentials_echoed": False,
        "official_scoring_allowed": False,
        "single_total_score_absent": True,
    }
    audit_errors = validate_agent_call_audit(call_audit)
    if audit_errors:
        raise InteractionContractError(
            "generated Agent call audit is invalid: " + "; ".join(audit_errors)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "interaction_trace.json"
    call_audit_path = output_dir / "agent_call_audit.json"
    run_manifest_path = output_dir / "run_manifest.json"
    write_json(trace_path, trace)
    write_json(call_audit_path, call_audit)
    run_manifest = {
        "manifest_version": RUN_MANIFEST_VERSION,
        "runner_version": AGENT_RUNNER_VERSION,
        "status": (
            "complete_non_official_agent_canary"
            if trace["status"] == "complete"
            else "incomplete_non_official_agent_canary"
        ),
        "run_id": run_id,
        "case_id": trace["case_id"],
        "environment_version": ENVIRONMENT_VERSION,
        "agent": {
            "agent_id": agent_id,
            "requested_model": requested_model,
            "effective_models_observed": sorted(set(effective_models)),
            "prompt_version": AGENT_PROMPT_VERSION,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "deterministic_gold_replay": False,
            "gold_accessor_used": False,
        },
        "patient_simulator": {
            "llm_used": False,
            "facts_deterministic": True,
            "ehr_state_authoritative": True,
        },
        "run_summary": {
            "terminal_reason": terminal_reason,
            "call_count": len(call_records),
            "submitted_action_count": submitted_action_count,
            "accepted_action_count": accepted_action_count,
            "completed_transition_count": len(
                trace["final_state"]["completed_transition_ids"]
            ),
            "final_stage": trace["final_state"]["stage"],
            "environment_terminal": trace["final_state"]["done"],
        },
        "bindings": {
            "scorer_freeze_id": trace["scorer_freeze"]["freeze_id"],
            "scorer_freeze_manifest_sha256": sha256_file(freeze_manifest_path),
            "scenario_sha256": sha256_file(scenario_path),
            "trace_sha256": sha256_file(trace_path),
            "agent_prompt_contract_sha256": prompt_sha,
            "agent_call_audit_sha256": sha256_file(call_audit_path),
        },
        "credential_storage": "environment_only",
        "transport_security": {
            "mode": transport_security_mode,
            "network_transport_encrypted": transport_security_mode == "https",
            "insecure_http_explicit_opt_in": (
                transport_security_mode == "insecure_http_explicit_opt_in"
            ),
        },
        "credentials_echoed": False,
        "automatic_semantic_judge_scoring": False,
        "official_scoring_allowed": False,
        "single_total_score_absent": True,
    }
    write_json(run_manifest_path, run_manifest)
    manifest_errors = validate_agent_canary_manifest(
        run_manifest,
        trace_path=trace_path,
        call_audit_path=call_audit_path,
        scenario_path=scenario_path,
        freeze_manifest_path=freeze_manifest_path,
        prompt_contract_path=prompt_contract_path,
    )
    if manifest_errors:
        raise InteractionContractError(
            "generated Agent run manifest is invalid: "
            + "; ".join(manifest_errors)
        )
    return {
        "status": run_manifest["status"],
        "run_id": run_id,
        "case_id": trace["case_id"],
        "terminal_reason": terminal_reason,
        "call_count": len(call_records),
        "submitted_action_count": submitted_action_count,
        "accepted_action_count": accepted_action_count,
        "completed_transition_count": len(
            trace["final_state"]["completed_transition_ids"]
        ),
        "final_stage": trace["final_state"]["stage"],
        "environment_terminal": trace["final_state"]["done"],
        "output_dir": str(output_dir.resolve()),
        "transport_security_mode": transport_security_mode,
        "credentials_echoed": False,
        "official_scoring_allowed": False,
        "single_total_score_absent": True,
    }
