"""Reusable wide-action Agent canary for Patient/EHR v0.2."""

from __future__ import annotations

import copy
import json
import re
import urllib.parse
from pathlib import Path
from typing import Any, Protocol

from world_model_patient_ehr_v01.agent_canary import (
    ChatCompletionsAgentClient,
    connection_configuration_errors,
    parse_agent_action,
)
from world_model_patient_ehr_v01.contracts import (
    InteractionContractError,
    canonical_hash,
    credential_findings,
    read_json,
    sha256_file,
    write_json,
)

from .contracts import (
    ENVIRONMENT_VERSION,
    RUN_MANIFEST_VERSION,
)
from .environment import WideActionPatientEHREnvironment, validate_trace


AGENT_RUNNER_VERSION = "ward-patient-ehr-wide-agent-runner-0.2.0"
AGENT_PROMPT_VERSION = "ward-patient-ehr-agent-prompt-0.2.0"
AGENT_CALL_AUDIT_VERSION = "ward-patient-ehr-agent-call-audit-0.2.0"
DEFAULT_AGENT_MODEL = "gpt-5.6-sol"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 800
DEFAULT_MAX_ACTIONS = 48

CURRENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO = (
    CURRENT_ROOT
    / "outputs/world_model_patient_ehr_v01/cases/ward_001.patient_ehr.json"
)
DEFAULT_FREEZE_MANIFEST = (
    CURRENT_ROOT
    / "outputs/world_model_interaction_scorer_freeze_v01/FREEZE_MANIFEST.json"
)
DEFAULT_INTERFACE = (
    CURRENT_ROOT
    / "outputs/world_model_patient_ehr_v02/cases/"
    "ward_001.action_interface.v0.2.json"
)
DEFAULT_ONTOLOGY = (
    CURRENT_ROOT / "outputs/world_model_patient_ehr_v02/action_ontology.v0.2.json"
)
DEFAULT_PROMPT_CONTRACT = (
    CURRENT_ROOT
    / "outputs/world_model_patient_ehr_v02/agent_canary/"
    "PROMPT_CONTRACT.v0.2.json"
)


class AgentClient(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Return content, effective_model, usage, and optional response_id."""


def safe_name(value: str, *, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,119}", value):
        raise InteractionContractError(
            f"{label} must use 3-120 safe filename characters"
        )
    return value


def transport_security_mode(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme == "https":
        return "https"
    if parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
        return "loopback_http"
    return "http_unencrypted"


def readiness(
    *,
    base_url: str,
    api_key: str,
    requested_model: str,
) -> dict[str, Any]:
    errors: list[str] = []
    configuration_error_codes: list[str] = []
    if not base_url:
        errors.append("agent_base_url_environment_missing")
    if not api_key and (
        not base_url
        or urllib.parse.urlsplit(base_url).hostname
        not in {"127.0.0.1", "localhost", "::1"}
    ):
        errors.append("agent_key_environment_missing")
    if not requested_model.strip():
        errors.append("agent_model_missing")
    if base_url:
        configuration_error_codes = connection_configuration_errors(
            base_url,
            api_key,
            allow_insecure_http=True,
        )
        if configuration_error_codes:
            errors.append("agent_connection_configuration_invalid")
    mode = transport_security_mode(base_url) if base_url else None
    return {
        "ready": not errors,
        "errors": errors,
        "configuration_error_codes": configuration_error_codes,
        "requested_model": requested_model,
        "base_url_environment_set": bool(base_url),
        "key_environment_set": bool(api_key),
        "transport_security_mode": mode,
        "network_transport_encrypted": mode == "https" if mode else None,
        "http_supported_without_extra_gate": True,
        "credential_storage": "environment_only",
        "credentials_echoed": False,
        "official_scoring_allowed": False,
        "single_total_score_absent": True,
    }


def load_prompt_contract(path: Path = DEFAULT_PROMPT_CONTRACT) -> dict[str, Any]:
    contract = read_json(path)
    if contract.get("prompt_version") != AGENT_PROMPT_VERSION:
        raise InteractionContractError("wrong v0.2 Agent prompt version")
    for key in (
        "gold_action_visible",
        "gold_sequence_visible",
        "future_evidence_visible",
    ):
        if contract.get(key) is not False:
            raise InteractionContractError(f"invalid prompt boundary: {key}")
    if credential_findings(contract):
        raise InteractionContractError("prompt contains credential-like material")
    return contract


def _remember_evidence(
    observation: dict[str, Any],
    memory: dict[str, dict[str, Any]],
) -> None:
    for evidence in observation.get("new_evidence", []):
        if isinstance(evidence, dict) and isinstance(evidence.get("evidence_id"), str):
            memory[evidence["evidence_id"]] = copy.deepcopy(evidence)


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
        "last_action_outcome": copy.deepcopy(
            observation.get("last_action_outcome")
        ),
        "action_schema": copy.deepcopy(observation["action_schema"]),
        "decision_space": copy.deepcopy(observation["decision_space"]),
        "done": observation["done"],
    }
    system = {
        "prompt_version": prompt_contract["prompt_version"],
        "role": prompt_contract["role"],
        "objective": prompt_contract["objective"],
        "rules": prompt_contract["rules"],
        "public_capability_meanings": prompt_contract[
            "public_capability_meanings"
        ],
        "output_contract": prompt_contract["output_contract"],
        "feedback_boundary": prompt_contract["feedback_boundary"],
    }
    user = {
        "instruction": (
            "Choose one clinically appropriate next action. Free-text clinical "
            "targets are supported; do not guess internal identifiers. Return only JSON."
        ),
        "observation": public_observation,
        "prior_public_attempts": copy.deepcopy(prior_public_attempts[-20:]),
    }
    messages = [
        {
            "role": "system",
            "content": json.dumps(system, ensure_ascii=False, sort_keys=True),
        },
        {
            "role": "user",
            "content": json.dumps(user, ensure_ascii=False, sort_keys=True),
        },
    ]
    if credential_findings(messages):
        raise InteractionContractError("Agent messages contain credential-like material")
    return messages


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
    for key in (
        "gold_action_exposed_to_agent",
        "gold_sequence_exposed_to_agent",
        "future_evidence_exposed_to_agent",
        "resolver_internals_exposed_to_agent",
    ):
        if audit.get(key) is not False:
            errors.append(f"invalid_boundary:{key}")
    if audit.get("raw_assistant_content_stored") is not False:
        errors.append("raw_assistant_content_boundary_invalid")
    if audit.get("official_scoring_allowed") is not False:
        errors.append("official_scoring_boundary_invalid")
    if audit.get("single_total_score_absent") is not True:
        errors.append("single_total_score_boundary_invalid")
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
    return errors


def validate_run_manifest(
    manifest: dict[str, Any],
    *,
    trace_path: Path,
    call_audit_path: Path,
    scenario_path: Path,
    freeze_manifest_path: Path,
    interface_path: Path,
    ontology_path: Path,
    prompt_contract_path: Path,
) -> list[str]:
    errors: list[str] = []
    if manifest.get("manifest_version") != RUN_MANIFEST_VERSION:
        errors.append("wrong_run_manifest_version")
    if manifest.get("runner_version") != AGENT_RUNNER_VERSION:
        errors.append("wrong_runner_version")
    if credential_findings(manifest):
        errors.append("credential_like_material_present")
    if manifest.get("official_scoring_allowed") is not False:
        errors.append("official_scoring_boundary_invalid")
    if manifest.get("single_total_score_absent") is not True:
        errors.append("single_total_score_boundary_invalid")
    bindings = manifest.get("bindings")
    expected = {
        "trace_sha256": sha256_file(trace_path),
        "agent_call_audit_sha256": sha256_file(call_audit_path),
        "scenario_sha256": sha256_file(scenario_path),
        "scorer_freeze_manifest_sha256": sha256_file(freeze_manifest_path),
        "action_interface_sha256": sha256_file(interface_path),
        "action_ontology_sha256": sha256_file(ontology_path),
        "agent_prompt_contract_sha256": sha256_file(prompt_contract_path),
    }
    if not isinstance(bindings, dict):
        errors.append("missing_bindings")
    else:
        for key, value in expected.items():
            if bindings.get(key) != value:
                errors.append(f"{key}_mismatch")
    trace = read_json(trace_path)
    if trace.get("run_id") != manifest.get("run_id"):
        errors.append("trace_run_id_mismatch")
    if trace.get("case_id") != manifest.get("case_id"):
        errors.append("trace_case_id_mismatch")
    errors.extend(validate_trace(trace))
    errors.extend(validate_agent_call_audit(read_json(call_audit_path)))
    return errors


def run_agent_canary(
    *,
    client: AgentClient,
    scenario_path: Path,
    freeze_manifest_path: Path,
    interface_path: Path,
    ontology_path: Path,
    prompt_contract_path: Path,
    output_dir: Path,
    run_id: str,
    agent_id: str,
    requested_model: str,
    transport_mode: str,
    max_actions: int = DEFAULT_MAX_ACTIONS,
) -> dict[str, Any]:
    safe_name(run_id, label="run_id")
    safe_name(agent_id, label="agent_id")
    if max_actions < 1 or max_actions > 160:
        raise InteractionContractError("max_actions must be between 1 and 160")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise InteractionContractError("output directory already contains artifacts")
    if transport_mode not in {"https", "loopback_http", "http_unencrypted"}:
        raise InteractionContractError("invalid transport mode")

    prompt = load_prompt_contract(prompt_contract_path)
    prompt_sha = sha256_file(prompt_contract_path)
    environment = WideActionPatientEHREnvironment(
        scenario_path=scenario_path,
        freeze_manifest_path=freeze_manifest_path,
        interface_path=interface_path,
        ontology_path=ontology_path,
        run_id=run_id,
        agent_id=agent_id,
    )
    observation = environment.reset()
    evidence_memory: dict[str, dict[str, Any]] = {}
    _remember_evidence(observation, evidence_memory)
    public_attempts: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    effective_models: list[str] = []
    terminal_reason = "action_budget_exhausted"
    api_error_type: str | None = None

    for call_index in range(max_actions):
        if observation["done"]:
            terminal_reason = "environment_terminal"
            break
        messages = build_agent_messages(
            prompt_contract=prompt,
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
            raw_content = str(response.get("content") or "").strip()
            content_sha = canonical_hash(raw_content)
            effective_model = str(
                response.get("effective_model") or requested_model
            )
            effective_models.append(effective_model)
            record = {
                "call_index": call_index,
                "request_sha256": request_sha,
                "assistant_content_sha256": content_sha,
                "effective_model": effective_model,
                "response_id": (
                    response.get("response_id")
                    if isinstance(response.get("response_id"), str)
                    else None
                ),
                "usage": _clean_usage(response.get("usage")),
                "raw_assistant_content_stored": False,
            }
            try:
                action = parse_agent_action(raw_content)
            except InteractionContractError as exc:
                record["status"] = "invalid_action_contract"
                record["error_code"] = str(exc)
                calls.append(record)
                public_attempts.append(
                    {
                        "call_index": call_index,
                        "outcome": {
                            "status": "invalid_action_contract",
                            "state_advanced": False,
                        },
                    }
                )
                continue
            observation, assessment, done = environment.step(action)
            _remember_evidence(observation, evidence_memory)
            public_outcome = copy.deepcopy(observation["last_action_outcome"])
            record.update(
                {
                    "status": "action_submitted",
                    "action_sha256": canonical_hash(action),
                    "outcome_status": public_outcome["status"],
                    "action_understood": public_outcome["action_understood"],
                    "state_advanced": public_outcome["state_advanced"],
                    "newly_released_evidence_ids": list(
                        public_outcome["newly_released_evidence_ids"]
                    ),
                }
            )
            calls.append(record)
            public_attempts.append(
                {
                    "call_index": call_index,
                    "action": copy.deepcopy(action),
                    "outcome": public_outcome,
                }
            )
            if done:
                terminal_reason = "environment_terminal"
                break
        except Exception as exc:  # provider/transport details are not persisted
            api_error_type = type(exc).__name__
            terminal_reason = "api_error"
            break

    trace = environment.export_trace()
    trace_errors = validate_trace(trace)
    if trace_errors:
        raise InteractionContractError(
            "generated v0.2 trace is invalid: " + "; ".join(trace_errors)
        )
    submitted = sum(1 for item in calls if item["status"] == "action_submitted")
    advanced = sum(1 for item in calls if item.get("state_advanced") is True)
    exploration = sum(
        1
        for item in calls
        if item.get("outcome_status")
        in {
            "recognized_non_current_action",
            "valid_exploration_no_recorded_outcome",
        }
    )
    call_audit = {
        "audit_version": AGENT_CALL_AUDIT_VERSION,
        "runner_version": AGENT_RUNNER_VERSION,
        "prompt_version": AGENT_PROMPT_VERSION,
        "run_id": run_id,
        "case_id": trace["case_id"],
        "requested_model": requested_model,
        "effective_models_observed": sorted(set(effective_models)),
        "call_count": len(calls),
        "submitted_action_count": submitted,
        "advanced_action_count": advanced,
        "exploration_action_count": exploration,
        "terminal_reason": terminal_reason,
        "api_error_type": api_error_type,
        "calls": calls,
        "gold_action_exposed_to_agent": False,
        "gold_sequence_exposed_to_agent": False,
        "future_evidence_exposed_to_agent": False,
        "resolver_internals_exposed_to_agent": False,
        "raw_assistant_content_stored": False,
        "credential_storage": "environment_only",
        "transport_security_mode": transport_mode,
        "network_transport_encrypted": transport_mode == "https",
        "credentials_echoed": False,
        "official_scoring_allowed": False,
        "single_total_score_absent": True,
    }
    audit_errors = validate_agent_call_audit(call_audit)
    if audit_errors:
        raise InteractionContractError(
            "generated v0.2 call audit is invalid: " + "; ".join(audit_errors)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "interaction_trace.json"
    audit_path = output_dir / "agent_call_audit.json"
    manifest_path = output_dir / "run_manifest.json"
    write_json(trace_path, trace)
    write_json(audit_path, call_audit)
    manifest = {
        "manifest_version": RUN_MANIFEST_VERSION,
        "runner_version": AGENT_RUNNER_VERSION,
        "status": (
            "complete_non_official_wide_action_canary"
            if trace["status"] == "complete"
            else "incomplete_non_official_wide_action_canary"
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
            "gold_accessor_used": False,
        },
        "patient_simulator": {
            "llm_used": False,
            "facts_deterministic": True,
            "ehr_state_authoritative": True,
        },
        "action_interface": {
            "free_text_targets_allowed": True,
            "exact_hidden_identifier_required": False,
            "deterministic_semantic_resolver": True,
            "schema_valid_exploration_allowed": True,
        },
        "run_summary": {
            "terminal_reason": terminal_reason,
            "call_count": len(calls),
            "submitted_action_count": submitted,
            "advanced_action_count": advanced,
            "exploration_action_count": exploration,
            "completed_transition_count": len(
                trace["final_state"]["completed_transition_ids"]
            ),
            "final_stage": trace["final_state"]["stage"],
            "environment_terminal": trace["final_state"]["done"],
        },
        "transport_security": {
            "mode": transport_mode,
            "network_transport_encrypted": transport_mode == "https",
            "http_supported_without_extra_gate": True,
        },
        "bindings": {
            "scorer_freeze_id": trace["scorer_freeze"]["freeze_id"],
            "scorer_freeze_manifest_sha256": sha256_file(freeze_manifest_path),
            "scenario_sha256": sha256_file(scenario_path),
            "action_interface_sha256": sha256_file(interface_path),
            "action_ontology_sha256": sha256_file(ontology_path),
            "agent_prompt_contract_sha256": prompt_sha,
            "trace_sha256": sha256_file(trace_path),
            "agent_call_audit_sha256": sha256_file(audit_path),
        },
        "credential_storage": "environment_only",
        "credentials_echoed": False,
        "automatic_semantic_judge_scoring": False,
        "official_scoring_allowed": False,
        "single_total_score_absent": True,
    }
    write_json(manifest_path, manifest)
    manifest_errors = validate_run_manifest(
        manifest,
        trace_path=trace_path,
        call_audit_path=audit_path,
        scenario_path=scenario_path,
        freeze_manifest_path=freeze_manifest_path,
        interface_path=interface_path,
        ontology_path=ontology_path,
        prompt_contract_path=prompt_contract_path,
    )
    if manifest_errors:
        raise InteractionContractError(
            "generated v0.2 run manifest is invalid: "
            + "; ".join(manifest_errors)
        )
    return {
        "status": manifest["status"],
        "run_id": run_id,
        "case_id": trace["case_id"],
        "terminal_reason": terminal_reason,
        "call_count": len(calls),
        "submitted_action_count": submitted,
        "advanced_action_count": advanced,
        "exploration_action_count": exploration,
        "completed_transition_count": len(
            trace["final_state"]["completed_transition_ids"]
        ),
        "final_stage": trace["final_state"]["stage"],
        "environment_terminal": trace["final_state"]["done"],
        "transport_security_mode": transport_mode,
        "output_dir": str(output_dir.resolve()),
        "credentials_echoed": False,
        "official_scoring_allowed": False,
        "single_total_score_absent": True,
    }


def make_live_client(
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    ca_bundle: Path | None,
) -> ChatCompletionsAgentClient:
    """Construct the shared transport; v0.2 intentionally supports HTTP."""
    return ChatCompletionsAgentClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        ca_bundle=ca_bundle,
        allow_insecure_http=True,
    )

