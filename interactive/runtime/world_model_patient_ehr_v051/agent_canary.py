"""Auditable bounded retry wrapper for transient provider HTTP failures."""

from __future__ import annotations

import copy
import hashlib
import time
import urllib.error
from pathlib import Path
from typing import Any

from world_model_patient_ehr_v01.contracts import (
    InteractionContractError,
    credential_findings,
    read_json,
    sha256_file,
    write_json,
)
from world_model_patient_ehr_v04.agent_canary import (
    validate_call_audit,
    validate_run_manifest,
)
from world_model_patient_ehr_v04.run_data_contract import (
    validate_model_output_trace,
)
from world_model_patient_ehr_v05 import agent_canary as v05_canary


AGENT_RUNNER_VERSION = "ward-patient-ehr-tool-agent-runner-0.5.1"
TRANSIENT_RETRY_POLICY_VERSION = (
    "ward-patient-ehr-transient-http-retry-0.1.0"
)
DEFAULT_RETRYABLE_HTTP_STATUSES = (502, 503, 504)
DEFAULT_MAX_PROVIDER_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = (1.0, 2.0)


def _message_hash(exc: BaseException) -> str:
    return hashlib.sha256(str(exc).encode("utf-8")).hexdigest()


class RetryingAgentClient:
    """Retry transient HTTP responses while retaining a separate audit."""

    def __init__(
        self,
        client: Any,
        *,
        max_provider_attempts: int = DEFAULT_MAX_PROVIDER_ATTEMPTS,
        retryable_http_statuses: tuple[int, ...] = (
            DEFAULT_RETRYABLE_HTTP_STATUSES
        ),
        backoff_seconds: tuple[float, ...] = DEFAULT_BACKOFF_SECONDS,
    ) -> None:
        if max_provider_attempts < 1 or max_provider_attempts > 5:
            raise InteractionContractError(
                "max_provider_attempts must be between 1 and 5"
            )
        self._client = client
        self.max_provider_attempts = max_provider_attempts
        self.retryable_http_statuses = tuple(retryable_http_statuses)
        self.backoff_seconds = tuple(backoff_seconds)
        self.logical_call_count = 0
        self.provider_http_attempt_count = 0
        self.retry_count = 0
        self.recovered_logical_call_count = 0
        self.exhausted_logical_call_count = 0
        self.events: list[dict[str, Any]] = []

    def _complete_inner(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if tools is None:
            return self._client.complete(messages)
        return self._client.complete(messages, tools=tools)

    def complete(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        logical_call_index = self.logical_call_count
        self.logical_call_count += 1
        attempts_for_call = 0
        for provider_attempt_index in range(self.max_provider_attempts):
            attempts_for_call += 1
            self.provider_http_attempt_count += 1
            try:
                result = self._complete_inner(messages, tools)
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                retryable = status in self.retryable_http_statuses
                will_retry = (
                    retryable
                    and provider_attempt_index + 1
                    < self.max_provider_attempts
                )
                event = {
                    "logical_call_index": logical_call_index,
                    "provider_attempt_index": provider_attempt_index,
                    "http_status": status,
                    "http_reason": str(exc.reason)[:120],
                    "message_sha256": _message_hash(exc),
                    "retryable": retryable,
                    "will_retry": will_retry,
                    "raw_error_body_stored": False,
                    "credentials_stored": False,
                }
                self.events.append(event)
                if not will_retry:
                    if retryable:
                        self.exhausted_logical_call_count += 1
                    raise
                self.retry_count += 1
                delay_index = min(
                    provider_attempt_index,
                    len(self.backoff_seconds) - 1,
                )
                delay = (
                    self.backoff_seconds[delay_index]
                    if self.backoff_seconds
                    else 0.0
                )
                event["backoff_seconds"] = delay
                if delay > 0:
                    time.sleep(delay)
                continue
            if attempts_for_call > 1:
                self.recovered_logical_call_count += 1
                self.events.append(
                    {
                        "logical_call_index": logical_call_index,
                        "provider_attempt_index": provider_attempt_index,
                        "event": "logical_call_recovered",
                        "provider_attempts_for_call": attempts_for_call,
                        "credentials_stored": False,
                    }
                )
            return result
        raise RuntimeError("unreachable retry loop")

    def export_audit(self) -> dict[str, Any]:
        audit = {
            "retry_policy_version": TRANSIENT_RETRY_POLICY_VERSION,
            "status": "complete_non_official",
            "max_provider_attempts_per_logical_call": (
                self.max_provider_attempts
            ),
            "retryable_http_statuses": list(self.retryable_http_statuses),
            "backoff_seconds": list(self.backoff_seconds),
            "logical_model_call_count": self.logical_call_count,
            "provider_http_attempt_count": self.provider_http_attempt_count,
            "transport_retry_count": self.retry_count,
            "recovered_logical_call_count": (
                self.recovered_logical_call_count
            ),
            "exhausted_logical_call_count": (
                self.exhausted_logical_call_count
            ),
            "events": copy.deepcopy(self.events),
            "http_transport_user_accepted": True,
            "http_transport_is_not_a_defect_or_blocker": True,
            "raw_provider_response_stored": False,
            "raw_error_body_stored": False,
            "credentials_stored": False,
            "credentials_echoed": False,
            "official_scoring_allowed": False,
            "single_total_score_absent": True,
        }
        if credential_findings(audit):
            raise InteractionContractError(
                "credential-like material found in retry audit"
            )
        return audit


def run_tool_agent_canary(
    *,
    client: Any,
    max_provider_attempts: int = DEFAULT_MAX_PROVIDER_ATTEMPTS,
    **kwargs: Any,
) -> dict[str, Any]:
    retrying_client = RetryingAgentClient(
        client,
        max_provider_attempts=max_provider_attempts,
    )
    original_runner_version = v05_canary.AGENT_RUNNER_VERSION
    v05_canary.AGENT_RUNNER_VERSION = AGENT_RUNNER_VERSION
    try:
        result = v05_canary.run_tool_agent_canary(
            client=retrying_client,
            **kwargs,
        )
    finally:
        v05_canary.AGENT_RUNNER_VERSION = original_runner_version

    output_dir = Path(kwargs["output_dir"])
    retry_audit_path = output_dir / "transport_retry_audit.json"
    call_audit_path = output_dir / "agent_call_audit.json"
    model_output_path = output_dir / "model_output_trace.json"
    manifest_path = output_dir / "run_manifest.json"
    retry_audit = retrying_client.export_audit()
    retry_audit["run_id"] = kwargs["run_id"]
    retry_audit["case_id"] = result["case_id"]
    write_json(retry_audit_path, retry_audit)

    call_audit = read_json(call_audit_path)
    call_audit["transport_retry_policy_version"] = (
        TRANSIENT_RETRY_POLICY_VERSION
    )
    call_audit["provider_http_attempt_count"] = (
        retrying_client.provider_http_attempt_count
    )
    call_audit["transport_retry_count"] = retrying_client.retry_count
    call_audit["transport_retry_recovered_logical_call_count"] = (
        retrying_client.recovered_logical_call_count
    )
    call_audit["transport_retry_exhausted_logical_call_count"] = (
        retrying_client.exhausted_logical_call_count
    )
    call_audit["transport_retry_audit_sha256"] = sha256_file(
        retry_audit_path
    )
    call_audit_errors = validate_call_audit(call_audit)
    if call_audit_errors:
        raise InteractionContractError(
            "post-retry call audit invalid: " + "; ".join(call_audit_errors)
        )
    write_json(call_audit_path, call_audit)

    model_output = read_json(model_output_path)
    model_output["transport_retry_summary"] = {
        "retry_policy_version": TRANSIENT_RETRY_POLICY_VERSION,
        "provider_http_attempt_count": (
            retrying_client.provider_http_attempt_count
        ),
        "transport_retry_count": retrying_client.retry_count,
        "recovered_logical_call_count": (
            retrying_client.recovered_logical_call_count
        ),
        "exhausted_logical_call_count": (
            retrying_client.exhausted_logical_call_count
        ),
        "raw_provider_response_stored": False,
    }
    model_output_errors = validate_model_output_trace(model_output)
    if model_output_errors:
        raise InteractionContractError(
            "post-retry model-output trace invalid: "
            + "; ".join(model_output_errors)
        )
    write_json(model_output_path, model_output)

    manifest = read_json(manifest_path)
    manifest["transport_retry"] = {
        "retry_policy_version": TRANSIENT_RETRY_POLICY_VERSION,
        "max_provider_attempts_per_logical_call": (
            retrying_client.max_provider_attempts
        ),
        "retryable_http_statuses": list(
            retrying_client.retryable_http_statuses
        ),
        "http_transport_user_accepted": True,
    }
    manifest["bindings"]["agent_call_audit_sha256"] = sha256_file(
        call_audit_path
    )
    manifest["bindings"]["model_output_trace_sha256"] = sha256_file(
        model_output_path
    )
    manifest["bindings"]["transport_retry_audit_sha256"] = sha256_file(
        retry_audit_path
    )
    write_json(manifest_path, manifest)

    manifest_errors = validate_run_manifest(
        manifest,
        interaction_trace_path=output_dir / "interaction_trace.json",
        tool_trace_path=output_dir / "tool_trace.json",
        call_audit_path=call_audit_path,
        scenario_path=Path(kwargs["scenario_path"]),
        freeze_manifest_path=Path(kwargs["freeze_manifest_path"]),
        interface_path=Path(kwargs["interface_path"]),
        ontology_path=Path(kwargs["ontology_path"]),
        graph_path=Path(kwargs["graph_path"]),
        tool_map_path=Path(kwargs["tool_map_path"]),
        prompt_contract_path=Path(kwargs["prompt_contract_path"]),
        case_registry_path=kwargs.get("case_registry_path"),
    )
    if manifest_errors:
        raise InteractionContractError(
            "post-retry run manifest invalid: "
            + "; ".join(manifest_errors)
        )

    result.update(
        {
            "transport_retry_policy_version": (
                TRANSIENT_RETRY_POLICY_VERSION
            ),
            "provider_http_attempt_count": (
                retrying_client.provider_http_attempt_count
            ),
            "transport_retry_count": retrying_client.retry_count,
            "transport_retry_recovered_logical_call_count": (
                retrying_client.recovered_logical_call_count
            ),
            "transport_retry_exhausted_logical_call_count": (
                retrying_client.exhausted_logical_call_count
            ),
            "transport_retry_audit_path": str(retry_audit_path),
        }
    )
    return result


__all__ = [
    "AGENT_RUNNER_VERSION",
    "DEFAULT_MAX_PROVIDER_ATTEMPTS",
    "RetryingAgentClient",
    "TRANSIENT_RETRY_POLICY_VERSION",
    "run_tool_agent_canary",
]
