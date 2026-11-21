"""Normalized model-output collection for Patient/EHR benchmark runs."""

from __future__ import annotations

import copy
import re
from typing import Any

from world_model_patient_ehr_v01.contracts import (
    canonical_hash,
    credential_findings,
)


MODEL_OUTPUT_TRACE_VERSION = (
    "ward-patient-ehr-model-output-trace-0.1.0"
)
RUN_DATA_CONTRACT_VERSION = "ward-patient-ehr-run-data-contract-0.1.0"

SUBMITTED_STATUSES = {
    "tool_call_submitted",
    "repair_tool_call_submitted",
}


def build_model_output_trace(
    *,
    run_id: str,
    case_id: str,
    agent_id: str,
    requested_model: str,
    effective_models_observed: list[str],
    terminal_reason: str,
    calls: list[dict[str, Any]],
    tool_trace: dict[str, Any],
    interaction_trace: dict[str, Any],
) -> dict[str, Any]:
    """Create a compact run-owned trace without copying source EHR records."""

    tool_entries = list(tool_trace.get("entries") or [])
    environment_entries = list(interaction_trace.get("entries") or [])
    submitted_index = 0
    entries: list[dict[str, Any]] = []

    for call in calls:
        submitted = call.get("status") in SUBMITTED_STATUSES
        normalized_tool_call = (
            copy.deepcopy(call.get("normalized_tool_call"))
            if submitted
            else None
        )
        receipt = None
        if submitted:
            tool_entry = (
                tool_entries[submitted_index]
                if submitted_index < len(tool_entries)
                else {}
            )
            environment_entry = (
                environment_entries[submitted_index]
                if submitted_index < len(environment_entries)
                else {}
            )
            receipt = {
                "outcome_status": call.get("outcome_status"),
                "action_understood": call.get("action_understood"),
                "state_advanced": call.get("state_advanced"),
                "newly_released_evidence_ids": list(
                    call.get("newly_released_evidence_ids") or []
                ),
                "stage_after": tool_entry.get("stage_after"),
                "done": tool_entry.get("done"),
                "state_before_sha256": environment_entry.get(
                    "state_before_sha256"
                ),
                "state_after_sha256": environment_entry.get(
                    "state_after_sha256"
                ),
                "environment_entry_sha256": environment_entry.get(
                    "entry_sha256"
                ),
                "tool_entry_sha256": tool_entry.get("tool_entry_sha256"),
            }
            submitted_index += 1

        entry = {
            "call_index": call.get("call_index"),
            "tool_turn_index": call.get("tool_turn_index"),
            "repair_call": bool(call.get("repair_call", False)),
            "status": call.get("status"),
            "response_mode": call.get("response_mode"),
            "tool_call_source": call.get("tool_call_source"),
            "assistant_output_sha256": call.get("assistant_content_sha256"),
            "normalized_tool_call": normalized_tool_call,
            "tool_call_sha256": call.get("tool_call_sha256"),
            "error_code": call.get("error_code"),
            "api_error_type": call.get("api_error_type"),
            "effective_model": call.get("effective_model"),
            "response_id": call.get("response_id"),
            "usage": copy.deepcopy(call.get("usage") or {}),
            "environment_receipt": receipt,
        }
        entries.append(entry)

    trace = {
        "trace_version": MODEL_OUTPUT_TRACE_VERSION,
        "run_data_contract_version": RUN_DATA_CONTRACT_VERSION,
        "run_id": run_id,
        "case_id": case_id,
        "agent_id": agent_id,
        "requested_model": requested_model,
        "effective_models_observed": list(effective_models_observed),
        "terminal_reason": terminal_reason,
        "collection_scope": {
            "case_artifacts_referenced_by_hash": True,
            "full_raw_ehr_copied_into_run": False,
            "raw_provider_response_stored": False,
            "raw_assistant_text_stored": False,
            "normalized_model_tool_calls_stored": True,
            "minimal_environment_receipts_stored": True,
            "credentials_stored": False,
        },
        "model_call_count": len(calls),
        "submitted_tool_call_count": submitted_index,
        "entries": entries,
        "credentials_echoed": False,
        "official_scoring_allowed": False,
        "single_total_score_absent": True,
    }
    errors = validate_model_output_trace(trace)
    if errors:
        raise ValueError(
            "generated model output trace is invalid: " + "; ".join(errors)
        )
    return trace


def validate_model_output_trace(trace: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if trace.get("trace_version") != MODEL_OUTPUT_TRACE_VERSION:
        errors.append("wrong_model_output_trace_version")
    if trace.get("run_data_contract_version") != RUN_DATA_CONTRACT_VERSION:
        errors.append("wrong_run_data_contract_version")
    if credential_findings(trace):
        errors.append("credential_like_material_present")
    if trace.get("credentials_echoed") is not False:
        errors.append("credentials_echoed_boundary_invalid")
    if trace.get("official_scoring_allowed") is not False:
        errors.append("official_scoring_boundary_invalid")
    if trace.get("single_total_score_absent") is not True:
        errors.append("single_total_score_boundary_invalid")

    scope = trace.get("collection_scope")
    expected_scope = {
        "case_artifacts_referenced_by_hash": True,
        "full_raw_ehr_copied_into_run": False,
        "raw_provider_response_stored": False,
        "raw_assistant_text_stored": False,
        "normalized_model_tool_calls_stored": True,
        "minimal_environment_receipts_stored": True,
        "credentials_stored": False,
    }
    if not isinstance(scope, dict):
        errors.append("collection_scope_missing")
    else:
        for key, expected in expected_scope.items():
            if scope.get(key) is not expected:
                errors.append(f"collection_scope_invalid:{key}")

    entries = trace.get("entries")
    if not isinstance(entries, list):
        errors.append("entries_not_list")
        return errors
    if trace.get("model_call_count") != len(entries):
        errors.append("model_call_count_mismatch")

    submitted_count = 0
    previous_environment_hash: str | None = None
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"invalid_entry:{index}")
            continue
        output_hash = str(entry.get("assistant_output_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", output_hash):
            errors.append(f"invalid_assistant_output_sha256:{index}")
        status = entry.get("status")
        tool_call = entry.get("normalized_tool_call")
        receipt = entry.get("environment_receipt")
        if status in SUBMITTED_STATUSES:
            submitted_count += 1
            if not isinstance(tool_call, dict):
                errors.append(f"submitted_tool_call_missing:{index}")
            else:
                tool_hash = str(entry.get("tool_call_sha256") or "")
                if tool_hash != canonical_hash(tool_call):
                    errors.append(f"tool_call_sha256_mismatch:{index}")
            if not isinstance(receipt, dict):
                errors.append(f"environment_receipt_missing:{index}")
            else:
                for key in (
                    "state_before_sha256",
                    "state_after_sha256",
                    "environment_entry_sha256",
                    "tool_entry_sha256",
                ):
                    if not re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(receipt.get(key) or ""),
                    ):
                        errors.append(f"invalid_{key}:{index}")
                state_before = receipt.get("state_before_sha256")
                if (
                    previous_environment_hash is not None
                    and state_before != previous_environment_hash
                ):
                    errors.append(f"state_hash_chain_mismatch:{index}")
                previous_environment_hash = receipt.get("state_after_sha256")
        else:
            if tool_call is not None:
                errors.append(f"unsubmitted_tool_call_present:{index}")
            if receipt is not None:
                errors.append(f"unsubmitted_environment_receipt_present:{index}")

    if trace.get("submitted_tool_call_count") != submitted_count:
        errors.append("submitted_tool_call_count_mismatch")
    return errors


__all__ = [
    "MODEL_OUTPUT_TRACE_VERSION",
    "RUN_DATA_CONTRACT_VERSION",
    "build_model_output_trace",
    "validate_model_output_trace",
]
