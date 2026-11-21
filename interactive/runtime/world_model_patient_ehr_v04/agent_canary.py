"""Lightweight tool-calling Patient/EHR Agent canary v0.4."""

from __future__ import annotations

from runtime_assets import vocabulary

import copy
import json
import re
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from world_model_patient_ehr_v01.agent_canary import (
    normalize_base_url,
    validate_api_connection_inputs,
)
from world_model_patient_ehr_v01.contracts import (
    InteractionContractError,
    canonical_hash,
    credential_findings,
    read_json,
    sha256_file,
    write_json,
)
from world_model_patient_ehr_v02.agent_canary import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_FREEZE_MANIFEST,
    DEFAULT_INTERFACE,
    DEFAULT_ONTOLOGY,
    DEFAULT_SCENARIO,
    readiness,
    safe_name,
    transport_security_mode,
)
from world_model_patient_ehr_v03.agent_canary import DEFAULT_GRAPH
from world_model_patient_ehr_v03.contracts import ENVIRONMENT_VERSION
from world_model_patient_ehr_v03.environment import validate_trace

from .agent_protocol import (
    AGENT_PROTOCOL_VERSION,
    NATIVE_TOOL_SCHEMA_VERSION,
    build_native_tool_schemas,
    extract_openai_native_tool_call,
    validate_interaction_mode,
)
from .run_data_contract import (
    MODEL_OUTPUT_TRACE_VERSION,
    RUN_DATA_CONTRACT_VERSION,
    build_model_output_trace,
    validate_model_output_trace,
)
from .tool_facade import (
    LightweightClinicalToolSandbox,
    TOOL_CALL_TRACE_VERSION,
    TOOL_MAP_VERSION,
    parse_tool_call,
    validate_tool_trace,
)


AGENT_RUNNER_VERSION = "ward-patient-ehr-tool-agent-runner-0.4.6"
SUPPORTED_AGENT_RUNNER_VERSIONS = {
    "ward-patient-ehr-tool-agent-runner-0.4.2",
    "ward-patient-ehr-tool-agent-runner-0.4.3",
    "ward-patient-ehr-tool-agent-runner-0.4.4",
    "ward-patient-ehr-tool-agent-runner-0.4.5",
    AGENT_RUNNER_VERSION,
}
AGENT_PROMPT_VERSION = "ward-patient-ehr-agent-prompt-0.4.4"
SUPPORTED_AGENT_PROMPT_VERSIONS = {
    "ward-patient-ehr-agent-prompt-0.4.0",
    "ward-patient-ehr-agent-prompt-0.4.1",
    "ward-patient-ehr-agent-prompt-0.4.2",
    "ward-patient-ehr-agent-prompt-0.4.3",
    AGENT_PROMPT_VERSION,
}
AGENT_CALL_AUDIT_VERSION = "ward-patient-ehr-tool-agent-call-audit-0.4.2"
SUPPORTED_AGENT_CALL_AUDIT_VERSIONS = {
    "ward-patient-ehr-tool-agent-call-audit-0.4.0",
    "ward-patient-ehr-tool-agent-call-audit-0.4.1",
    AGENT_CALL_AUDIT_VERSION,
}
RUN_MANIFEST_VERSION = "ward-patient-ehr-tool-run-manifest-0.5.0"
SUPPORTED_RUN_MANIFEST_VERSIONS = {
    "ward-patient-ehr-tool-run-manifest-0.4.0",
    RUN_MANIFEST_VERSION,
}
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 640
DEFAULT_MAX_ACTIONS = 20
DEFAULT_MAX_MODEL_CALLS = 24
DEFAULT_REPEAT_INVALID_LIMIT = 2
EVIDENCE_HEAD_CHARS = 160
EVIDENCE_TAIL_CHARS = 180
EVIDENCE_CONCLUSION_CHARS = 320
RECENT_ATTEMPT_LIMIT = 6
MEMORY_ITEMS_PER_BUCKET = 4

CURRENT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOOL_MAP = (
    CURRENT_ROOT
    / "outputs/world_model_patient_ehr_v04/cases/"
    "ward_001.tool_map.v0.4.2.json"
)
DEFAULT_PROMPT_CONTRACT = (
    CURRENT_ROOT
    / "outputs/world_model_patient_ehr_v04/agent_canary/"
    "PROMPT_CONTRACT.v0.4.4.json"
)
DEFAULT_RUN_DATA_CONTRACT = Path(__file__).resolve().parents[1] / "assets" / "run_data_contract.json"


class AgentClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return normalized content/tool_call, model, usage, and response ID."""


class ToolChatCompletionsAgentClient:
    """OpenAI-compatible transport supporting native tools and text JSON."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: int,
        ca_bundle: Path | None,
        interaction_mode: str,
    ) -> None:
        validate_api_connection_inputs(
            base_url,
            api_key,
            allow_insecure_http=True,
        )
        if not model.strip():
            raise InteractionContractError("model is required")
        if timeout < 1 or timeout > 600:
            raise InteractionContractError("timeout must be between 1 and 600 seconds")
        self._url = normalize_base_url(base_url) + "/chat/completions"
        self._api_key = api_key
        self._model = model.strip()
        self._timeout = timeout
        self._ca_bundle = ca_bundle
        self._interaction_mode = validate_interaction_mode(interaction_mode)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
        if self._interaction_mode in {"native_tool_call", "auto"}:
            if not tools:
                raise InteractionContractError(
                    "native interaction mode requires tool schemas"
                )
            payload["tools"] = copy.deepcopy(tools)
            payload["tool_choice"] = "required"
            payload["parallel_tool_calls"] = False
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
            request,
            timeout=self._timeout,
            context=context,
        ) as response:
            body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, dict):
            raise InteractionContractError("API response must be an object")
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InteractionContractError(
                "API response is missing assistant message"
            ) from exc
        if not isinstance(message, dict):
            raise InteractionContractError(
                "API response assistant message must be an object"
            )
        content = message.get("content")
        normalized_tool_call: dict[str, Any] | None = None
        native_error: str | None = None
        response_mode = "text_json"
        if self._interaction_mode in {"native_tool_call", "auto"}:
            try:
                normalized_tool_call = extract_openai_native_tool_call(message)
                response_mode = "native_tool_call"
            except InteractionContractError as exc:
                native_error = str(exc)
                if self._interaction_mode == "native_tool_call":
                    response_mode = "native_tool_call_invalid"
        return {
            "content": content,
            "tool_call": normalized_tool_call,
            "native_tool_call_error": native_error,
            "response_mode": response_mode,
            "effective_model": str(body.get("model") or self._model),
            "usage": _clean_usage(body.get("usage")),
            "response_id": (
                str(body["id"])[:200]
                if isinstance(body.get("id"), str)
                else None
            ),
        }


def make_live_client(
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int,
    ca_bundle: Path | None,
    interaction_mode: str = "native_tool_call",
) -> ToolChatCompletionsAgentClient:
    return ToolChatCompletionsAgentClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        ca_bundle=ca_bundle,
        interaction_mode=interaction_mode,
    )


def _remember_evidence(
    observation: dict[str, Any],
    memory: dict[str, dict[str, Any]],
) -> None:
    for evidence in observation.get("new_evidence", []):
        if isinstance(evidence, dict) and isinstance(evidence.get("evidence_id"), str):
            memory[evidence["evidence_id"]] = copy.deepcopy(evidence)


def load_prompt_contract(path: Path = DEFAULT_PROMPT_CONTRACT) -> dict[str, Any]:
    contract = read_json(path)
    if contract.get("prompt_version") not in SUPPORTED_AGENT_PROMPT_VERSIONS:
        raise InteractionContractError("wrong v0.4 Agent prompt version")
    for key in (
        "gold_action_visible",
        "dependency_graph_visible",
        "future_evidence_visible",
    ):
        if contract.get(key) is not False:
            raise InteractionContractError(f"invalid prompt boundary: {key}")
    if contract.get("official_scoring_allowed") is not False:
        raise InteractionContractError("prompt cannot authorize official scoring")
    if contract.get("single_total_score_absent") is not True:
        raise InteractionContractError("prompt cannot contain single total scoring")
    if (
        contract.get("prompt_version") == AGENT_PROMPT_VERSION
        and contract.get("coarse_available_action_families_visible") is not False
    ):
        raise InteractionContractError(
            "neutral prompt cannot expose Gold-derived navigation hints"
        )
    if credential_findings(contract):
        raise InteractionContractError("prompt contains credential-like material")
    return contract


def _compact_prior_attempts(
    prior_public_attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in prior_public_attempts[-RECENT_ATTEMPT_LIMIT:]:
        tool_call = item.get("tool_call")
        outcome = item.get("outcome", {})
        value: dict[str, Any] = {
            "tool_turn_index": item.get("tool_turn_index"),
            "outcome": {
                key: copy.deepcopy(outcome.get(key))
                for key in (
                    "status",
                    "state_advanced",
                    "tool_family",
                    "do_not_immediately_retry_same_intent",
                    "semantic_repeat_without_new_evidence",
                )
                if key in outcome
            },
        }
        if isinstance(tool_call, dict):
            value["tool_call"] = {
                "tool": tool_call.get("tool"),
                "intent": str(tool_call.get("intent") or "")[:120],
                "module": tool_call.get("args", {}).get("module"),
            }
        compact.append(value)
    return compact


def _extract_conclusion(evidence: dict[str, Any]) -> str:
    for key in ("conclusion", "impression", "summary"):
        value = evidence.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:EVIDENCE_CONCLUSION_CHARS]
    content = evidence.get("content")
    if not isinstance(content, str) or not content.strip():
        return ""
    markers = (
        "examination conclusion",
        "pathology conclusion",
        "diagnostic opinion",
        "discharge diagnosis",
        "imaging conclusion",
        "impression",
        "conclusion",
    )
    markers = vocabulary("evidence.conclusion_markers", markers)
    for marker in markers:
        match = re.search(rf"{re.escape(marker)}\s*[::]\s*", content, flags=re.I)
        if not match:
            continue
        conclusion = content[match.end() :]
        boundary = re.search(
            vocabulary("evidence.conclusion_boundary", '\\n\\s*(?:clinical diagnosis|findings|pathology findings|notes|recommendations|clinical diagnosis|findings|recommendation)\\s*[::]'),
            conclusion,
            flags=re.I,
        )
        if boundary:
            conclusion = conclusion[: boundary.start()]
        conclusion = conclusion.strip()
        if conclusion:
            return conclusion[:EVIDENCE_CONCLUSION_CHARS]
    return ""


def _compact_visible_evidence(
    *,
    visible_evidence_ids: list[str],
    evidence_memory: dict[str, dict[str, Any]],
    new_evidence_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    new_ids = [
        evidence_id
        for evidence_id in (new_evidence_ids or [])
        if evidence_id in visible_evidence_ids
    ]
    ordered_ids = [*new_ids, *[item for item in visible_evidence_ids if item not in new_ids]]
    for evidence_id in ordered_ids:
        evidence = evidence_memory.get(evidence_id)
        if not isinstance(evidence, dict):
            continue
        item = {
            key: copy.deepcopy(evidence.get(key))
            for key in (
                "evidence_id",
                "record_kind",
                "title",
                "clinical_time",
                "available_time",
            )
            if key in evidence
        }
        item["newly_released"] = evidence_id in new_ids
        conclusion = _extract_conclusion(evidence)
        if conclusion:
            item["conclusion"] = conclusion
        content = evidence.get("content")
        if isinstance(content, str):
            item["content_head"] = content[:EVIDENCE_HEAD_CHARS]
            item["content_tail"] = (
                content[-EVIDENCE_TAIL_CHARS:]
                if len(content) > EVIDENCE_HEAD_CHARS
                else ""
            )
            item["content_truncated"] = len(content) > (
                EVIDENCE_HEAD_CHARS + EVIDENCE_TAIL_CHARS
            )
        compact.append(item)
    return compact


def _compact_agent_memory(memory: Any) -> dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in (
        "completed_tool_calls",
        "pending_tool_calls",
        "unrecorded_tool_calls",
        "invalid_tool_calls",
    ):
        values = memory.get(key, [])
        if not isinstance(values, list):
            continue
        compact[key] = [
            {
                field: (
                    str(item.get(field) or "")[:120]
                    if field == "intent"
                    else copy.deepcopy(item.get(field))
                )
                for field in (
                    "tool",
                    "intent",
                    "module",
                    "tool_family",
                    "status",
                    "semantic_target_fingerprint",
                    "semantic_target_seen_before",
                    "semantic_repeat_without_new_evidence",
                    "semantic_revisit_after_new_evidence",
                )
                if field in item
            }
            for item in values[-MEMORY_ITEMS_PER_BUCKET:]
            if isinstance(item, dict)
        ]
    semantic_targets = memory.get("do_not_repeat_semantic_targets", [])
    if isinstance(semantic_targets, list):
        compact["do_not_repeat_semantic_targets"] = [
            {
                field: (
                    str(item.get(field) or "")[:120]
                    if field == "intent"
                    else copy.deepcopy(item.get(field))
                )
                for field in (
                    "tool",
                    "intent",
                    "module",
                    "tool_family",
                    "status",
                    "semantic_target_fingerprint",
                    "semantic_target_seen_before",
                    "semantic_repeat_without_new_evidence",
                    "semantic_revisit_after_new_evidence",
                )
                if field in item
            }
            for item in semantic_targets[-MEMORY_ITEMS_PER_BUCKET:]
            if isinstance(item, dict)
        ]
    do_not_repeat = memory.get("do_not_repeat_exact_targets", [])
    if isinstance(do_not_repeat, list):
        compact["do_not_repeat_exact_targets"] = [
            {
                "tool": item.get("tool"),
                "intent": str(item.get("intent") or "")[:120],
                "status": item.get("status"),
            }
            for item in do_not_repeat[-MEMORY_ITEMS_PER_BUCKET:]
            if isinstance(item, dict)
        ]
    retryable = memory.get("retryable_pending_families", [])
    if isinstance(retryable, list):
        compact["retryable_pending_families"] = list(retryable)
    return compact


def _progress_guidance(
    *,
    prompt_contract: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    memory = observation.get("agent_memory", {})
    last_outcome = observation.get("last_tool_outcome") or {}
    navigation_visible = bool(
        observation.get("tool_interface", {}).get(
            "coarse_available_action_families_visible",
            False,
        )
    )
    guidance = {
        "stage_workflow_checklist": list(
            prompt_contract.get("workflow_playbook", {}).get(
                observation["stage"],
                [],
            )
        ),
        "retryable_pending_families": list(
            memory.get("retryable_pending_families", [])
        ),
        "last_outcome_status": last_outcome.get("status"),
        "last_outcome_family": last_outcome.get("tool_family"),
        "last_outcome_semantic_repeat": last_outcome.get(
            "semantic_repeat_without_new_evidence"
        ),
        "semantic_cooldown_target_count": len(
            memory.get("do_not_repeat_semantic_targets", [])
        ),
        "selection_rule": (
            "Choose one clinically supported next action from visible evidence, "
            "the public inpatient workflow, and prior public outcomes. Treat "
            "newly released evidence and its conclusion as the primary update. "
            "Do not ask the EHR to disclose hidden workflow state."
        ),
    }
    if navigation_visible:
        guidance.update(
            {
                "mode": (
                    "advance_recorded_path"
                    if observation["available_recorded_action_count"] > 0
                    else "clinically_grounded_exploration"
                ),
                "currently_available_coarse_families": list(
                    observation.get("available_action_families", [])
                ),
                "currently_available_family_tool_hints": copy.deepcopy(
                    observation.get("available_family_tool_hints", {})
                ),
            }
        )
    return guidance


def _rules_for_interaction_mode(
    rules: list[Any],
    *,
    interaction_mode: str,
) -> list[Any]:
    if interaction_mode == "text_json":
        return copy.deepcopy(rules)
    text_json_fragments = (
        "output the shortest valid json",
        "use tool plus args.module",
        "use only tool, intent, args",
    )
    return [
        copy.deepcopy(rule)
        for rule in rules
        if not (
            isinstance(rule, str)
            and any(
                fragment in rule.lower()
                for fragment in text_json_fragments
            )
        )
    ]


def build_tool_agent_messages(
    *,
    prompt_contract: dict[str, Any],
    observation: dict[str, Any],
    evidence_memory: dict[str, dict[str, Any]],
    prior_public_attempts: list[dict[str, Any]],
    interaction_mode: str = "text_json",
) -> list[dict[str, str]]:
    interaction_mode = validate_interaction_mode(interaction_mode)
    navigation_visible = bool(
        observation.get("tool_interface", {}).get(
            "coarse_available_action_families_visible",
            False,
        )
    )
    public_observation = {
        "case_id": observation["case_id"],
        "stage": observation["stage"],
        "environment_clock": observation["environment_clock"],
        "public_context": copy.deepcopy(observation["public_context"]),
        "visible_evidence_ids": list(observation["visible_evidence_ids"]),
        "new_evidence_ids": [
            item["evidence_id"]
            for item in observation.get("new_evidence", [])
            if isinstance(item, dict) and isinstance(item.get("evidence_id"), str)
        ],
        "visible_evidence": _compact_visible_evidence(
            visible_evidence_ids=list(observation["visible_evidence_ids"]),
            evidence_memory=evidence_memory,
            new_evidence_ids=[
                item["evidence_id"]
                for item in observation.get("new_evidence", [])
                if isinstance(item, dict)
                and isinstance(item.get("evidence_id"), str)
            ],
        ),
        "last_tool_outcome": copy.deepcopy(observation.get("last_tool_outcome")),
        "tool_interface": copy.deepcopy(observation["tool_interface"]),
        "agent_memory": _compact_agent_memory(observation["agent_memory"]),
        "done": observation["done"],
    }
    if navigation_visible:
        public_observation.update(
            {
                "completed_recorded_action_count": observation[
                    "completed_recorded_action_count"
                ],
                "available_recorded_action_count": observation[
                    "available_recorded_action_count"
                ],
                "available_action_families": list(
                    observation.get("available_action_families", [])
                ),
                "available_action_family_counts": copy.deepcopy(
                    observation.get("available_action_family_counts", {})
                ),
                "available_family_tool_hints": copy.deepcopy(
                    observation.get("available_family_tool_hints", {})
                ),
            }
        )
    system = {
        "prompt_version": prompt_contract["prompt_version"],
        "role": prompt_contract["role"],
        "objective": prompt_contract["objective"],
        "rules": _rules_for_interaction_mode(
            prompt_contract["rules"],
            interaction_mode=interaction_mode,
        ),
        "public_outcome_meanings": prompt_contract["public_outcome_meanings"],
        "feedback_boundary": prompt_contract["feedback_boundary"],
        "agent_protocol_version": AGENT_PROTOCOL_VERSION,
        "interaction_mode": interaction_mode,
    }
    if interaction_mode == "text_json":
        system["output_contract"] = prompt_contract["output_contract"]
    user = {
        "instruction": (
            (
                "Return one tool call now. The first character must be { and the "
                "last character must be }. Use compact JSON only, with no prose "
                "before or after."
            )
            if interaction_mode == "text_json"
            else (
                "Call exactly one provided clinical function now. Use the native "
                "tool-call channel and do not answer in prose."
            )
        ),
        "progress_guidance": _progress_guidance(
            prompt_contract=prompt_contract,
            observation=observation,
        ),
        "observation": public_observation,
        "recent_attempts": _compact_prior_attempts(prior_public_attempts),
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


def build_repair_messages(
    *,
    prompt_contract: dict[str, Any],
    original_messages: list[dict[str, str]],
    raw_content: str,
    interaction_mode: str = "text_json",
    native_error: str | None = None,
) -> list[dict[str, str]]:
    interaction_mode = validate_interaction_mode(interaction_mode)
    if credential_findings(raw_content):
        raise InteractionContractError("cannot repair credential-like output")
    if interaction_mode == "text_json":
        repair_context = {
            "instruction": prompt_contract["repair_contract"][
                "repair_instruction"
            ],
            "invalid_output_to_convert": raw_content[:2000],
            "required_schema": prompt_contract["output_contract"],
            "do_not_reconsider_clinical_plan": True,
        }
        system_instruction = (
            "Convert the supplied attempted action into one compact JSON tool "
            "call. Do not add analysis or choose a different clinical action."
        )
    else:
        repair_context = {
            "instruction": (
                "Retry the same intended clinical action using exactly one "
                "provided native function call."
            ),
            "native_tool_call_error": str(native_error or "")[:500],
            "do_not_answer_in_prose": True,
            "do_not_reconsider_clinical_plan": True,
        }
        system_instruction = (
            "Return exactly one valid native clinical function call. Do not "
            "answer in prose."
        )
    messages = [
        {
            "role": "system",
            "content": system_instruction,
        },
        {
            "role": "user",
            "content": json.dumps(
                repair_context,
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]
    if credential_findings(messages):
        raise InteractionContractError("repair messages contain credential-like material")
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


def _usage_totals(calls: list[dict[str, Any]]) -> dict[str, int]:
    prompt_tokens = sum(
        int(item.get("usage", {}).get("prompt_tokens", 0))
        for item in calls
    )
    completion_tokens = sum(
        int(item.get("usage", {}).get("completion_tokens", 0))
        for item in calls
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _emit_progress(enabled: bool, event: dict[str, Any]) -> None:
    if not enabled:
        return
    safe_event = {
        "patient_ehr_progress": True,
        **copy.deepcopy(event),
    }
    if credential_findings(safe_event):
        raise InteractionContractError("progress event contains credential-like material")
    print(
        json.dumps(safe_event, ensure_ascii=False, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def _safe_api_error_summary(exc: Exception) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "type": type(exc).__name__,
        "message_sha256": canonical_hash(str(exc)),
    }
    if isinstance(exc, urllib.error.HTTPError):
        summary["http_status"] = int(exc.code)
        summary["http_reason"] = str(exc.reason)[:120]
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        if body:
            summary["body_sha256"] = canonical_hash(body)
            if not credential_findings(body):
                summary["body_excerpt"] = body[:300]
            else:
                summary["body_excerpt_redacted"] = True
    return summary


def validate_call_audit(audit: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if audit.get("audit_version") not in SUPPORTED_AGENT_CALL_AUDIT_VERSIONS:
        errors.append("wrong_call_audit_version")
    if credential_findings(audit):
        errors.append("credential_like_material_present")
    for key in (
        "gold_action_exposed_to_agent",
        "dependency_graph_exposed_to_agent",
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
    if audit.get("audit_version") == AGENT_CALL_AUDIT_VERSION:
        try:
            validate_interaction_mode(str(audit.get("interaction_mode") or ""))
        except InteractionContractError:
            errors.append("invalid_interaction_mode")
        native_count = audit.get("native_tool_call_count")
        text_count = audit.get("text_json_tool_call_count")
        submitted = audit.get("submitted_tool_call_count")
        if (
            not isinstance(native_count, int)
            or isinstance(native_count, bool)
            or native_count < 0
        ):
            errors.append("invalid_native_tool_call_count")
        if (
            not isinstance(text_count, int)
            or isinstance(text_count, bool)
            or text_count < 0
        ):
            errors.append("invalid_text_json_tool_call_count")
        if (
            isinstance(native_count, int)
            and isinstance(text_count, int)
            and isinstance(submitted, int)
            and native_count + text_count != submitted
        ):
            errors.append("interaction_mode_submission_count_mismatch")
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
            if (
                audit.get("audit_version") == AGENT_CALL_AUDIT_VERSION
                and call.get("status")
                in {"tool_call_submitted", "repair_tool_call_submitted"}
            ):
                normalized_tool_call = call.get("normalized_tool_call")
                if not isinstance(normalized_tool_call, dict):
                    errors.append(f"normalized_tool_call_missing:{index}")
                elif call.get("tool_call_sha256") != canonical_hash(
                    normalized_tool_call
                ):
                    errors.append(f"normalized_tool_call_hash_mismatch:{index}")
    return errors


def validate_run_manifest(
    manifest: dict[str, Any],
    *,
    interaction_trace_path: Path,
    tool_trace_path: Path,
    call_audit_path: Path,
    scenario_path: Path,
    freeze_manifest_path: Path,
    interface_path: Path,
    ontology_path: Path,
    graph_path: Path,
    tool_map_path: Path,
    prompt_contract_path: Path,
    case_registry_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    manifest_version = manifest.get("manifest_version")
    if manifest_version not in SUPPORTED_RUN_MANIFEST_VERSIONS:
        errors.append("wrong_run_manifest_version")
    if manifest.get("runner_version") not in SUPPORTED_AGENT_RUNNER_VERSIONS:
        errors.append("wrong_runner_version")
    if credential_findings(manifest):
        errors.append("credential_like_material_present")
    if manifest.get("official_scoring_allowed") is not False:
        errors.append("official_scoring_boundary_invalid")
    if manifest.get("single_total_score_absent") is not True:
        errors.append("single_total_score_boundary_invalid")
    expected = {
        "interaction_trace_sha256": sha256_file(interaction_trace_path),
        "tool_trace_sha256": sha256_file(tool_trace_path),
        "agent_call_audit_sha256": sha256_file(call_audit_path),
        "scenario_sha256": sha256_file(scenario_path),
        "scorer_freeze_manifest_sha256": sha256_file(freeze_manifest_path),
        "action_interface_sha256": sha256_file(interface_path),
        "action_ontology_sha256": sha256_file(ontology_path),
        "action_graph_sha256": sha256_file(graph_path),
        "tool_map_sha256": sha256_file(tool_map_path),
        "agent_prompt_contract_sha256": sha256_file(prompt_contract_path),
    }
    if case_registry_path is not None:
        expected["case_registry_sha256"] = sha256_file(case_registry_path)
    model_output_trace_path = call_audit_path.parent / "model_output_trace.json"
    if manifest_version == RUN_MANIFEST_VERSION:
        if model_output_trace_path.is_file():
            expected["model_output_trace_sha256"] = sha256_file(
                model_output_trace_path
            )
        else:
            errors.append("model_output_trace_missing")
        if DEFAULT_RUN_DATA_CONTRACT.is_file():
            expected["run_data_contract_sha256"] = sha256_file(
                DEFAULT_RUN_DATA_CONTRACT
            )
        else:
            errors.append("run_data_contract_missing")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, dict):
        errors.append("missing_bindings")
    else:
        for key, value in expected.items():
            if bindings.get(key) != value:
                errors.append(f"{key}_mismatch")
    interaction_trace = read_json(interaction_trace_path)
    tool_trace = read_json(tool_trace_path)
    if interaction_trace.get("run_id") != manifest.get("run_id"):
        errors.append("interaction_trace_run_id_mismatch")
    if tool_trace.get("run_id") != manifest.get("run_id"):
        errors.append("tool_trace_run_id_mismatch")
    if interaction_trace.get("case_id") != manifest.get("case_id"):
        errors.append("interaction_trace_case_id_mismatch")
    if tool_trace.get("case_id") != manifest.get("case_id"):
        errors.append("tool_trace_case_id_mismatch")
    errors.extend(validate_trace(interaction_trace))
    errors.extend(validate_tool_trace(tool_trace))
    errors.extend(validate_call_audit(read_json(call_audit_path)))
    if manifest_version == RUN_MANIFEST_VERSION and model_output_trace_path.is_file():
        model_output_trace = read_json(model_output_trace_path)
        if model_output_trace.get("run_id") != manifest.get("run_id"):
            errors.append("model_output_trace_run_id_mismatch")
        if model_output_trace.get("case_id") != manifest.get("case_id"):
            errors.append("model_output_trace_case_id_mismatch")
        errors.extend(validate_model_output_trace(model_output_trace))
    return errors


def _model_call_record(
    *,
    call_index: int,
    tool_turn_index: int,
    request_sha: str,
    response: dict[str, Any],
    requested_model: str,
    repair: bool,
) -> tuple[dict[str, Any], str]:
    raw_content = str(response.get("content") or "").strip()
    native_tool_call = response.get("tool_call")
    assistant_output = (
        {"tool_call": native_tool_call}
        if isinstance(native_tool_call, dict)
        else {"content": raw_content}
    )
    record = {
        "call_index": call_index,
        "tool_turn_index": tool_turn_index,
        "request_sha256": request_sha,
        "assistant_content_sha256": canonical_hash(assistant_output),
        "effective_model": str(response.get("effective_model") or requested_model),
        "response_id": (
            response.get("response_id")
            if isinstance(response.get("response_id"), str)
            else None
        ),
        "usage": _clean_usage(response.get("usage")),
        "repair_call": repair,
        "raw_assistant_content_stored": False,
        "response_mode": str(response.get("response_mode") or "text_json"),
        "tool_call_source": (
            "native_tool_call"
            if isinstance(native_tool_call, dict)
            else "text_json"
        ),
    }
    return record, raw_content


def _tool_call_from_response(
    *,
    response: dict[str, Any],
    raw_content: str,
    interaction_mode: str,
) -> dict[str, Any]:
    native_tool_call = response.get("tool_call")
    if isinstance(native_tool_call, dict):
        return parse_tool_call(native_tool_call)
    native_error = response.get("native_tool_call_error")
    if interaction_mode == "native_tool_call":
        raise InteractionContractError(
            str(native_error or "native tool call required but missing")
        )
    return parse_tool_call(raw_content)


def _complete_agent(
    *,
    client: AgentClient,
    messages: list[dict[str, str]],
    interaction_mode: str,
    native_tools: list[dict[str, Any]],
) -> dict[str, Any]:
    if interaction_mode == "text_json":
        return client.complete(messages)
    return client.complete(messages, tools=native_tools)


def run_tool_agent_canary(
    *,
    client: AgentClient,
    scenario_path: Path,
    freeze_manifest_path: Path,
    interface_path: Path,
    ontology_path: Path,
    graph_path: Path,
    tool_map_path: Path,
    prompt_contract_path: Path,
    output_dir: Path,
    run_id: str,
    agent_id: str,
    requested_model: str,
    transport_mode: str,
    interaction_mode: str = "text_json",
    max_actions: int = DEFAULT_MAX_ACTIONS,
    max_model_calls: int = DEFAULT_MAX_MODEL_CALLS,
    repeat_invalid_limit: int = DEFAULT_REPEAT_INVALID_LIMIT,
    max_consecutive_no_advance: int | None = None,
    progress: bool = False,
    case_registry_path: Path | None = None,
    resume_from_output_dir: Path | None = None,
) -> dict[str, Any]:
    safe_name(run_id, label="run_id")
    safe_name(agent_id, label="agent_id")
    interaction_mode = validate_interaction_mode(interaction_mode)
    if max_actions < 1 or max_actions > 200:
        raise InteractionContractError("max_actions must be between 1 and 200")
    if max_model_calls < 1 or max_model_calls > 400:
        raise InteractionContractError("max_model_calls must be between 1 and 400")
    if repeat_invalid_limit < 1 or repeat_invalid_limit > 20:
        raise InteractionContractError(
            "repeat_invalid_limit must be between 1 and 20"
        )
    if (
        max_consecutive_no_advance is not None
        and (
            max_consecutive_no_advance < 1
            or max_consecutive_no_advance > 200
        )
    ):
        raise InteractionContractError(
            "max_consecutive_no_advance must be between 1 and 200"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise InteractionContractError("output directory already contains artifacts")
    prompt = load_prompt_contract(prompt_contract_path)
    prompt_version = str(prompt["prompt_version"])
    prompt_sha = sha256_file(prompt_contract_path)
    sandbox = LightweightClinicalToolSandbox(
        scenario_path=scenario_path,
        freeze_manifest_path=freeze_manifest_path,
        interface_path=interface_path,
        ontology_path=ontology_path,
        graph_path=graph_path,
        tool_map_path=tool_map_path,
        run_id=run_id,
        agent_id=agent_id,
    )
    observation = sandbox.reset()
    native_tools = build_native_tool_schemas(
        public_tools=list(
            observation["tool_interface"]["tool_call_schema"]["tool_values"]
        ),
        public_ehr_modules=list(
            observation["tool_interface"]["public_ehr_modules"]
        ),
    )
    tool_map_version = sandbox.tool_map_version
    evidence_memory: dict[str, dict[str, Any]] = {}
    _remember_evidence(observation, evidence_memory)
    public_attempts: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    effective_models: list[str] = []
    terminal_reason = "action_budget_exhausted"
    api_error_type: str | None = None
    api_error_summary: dict[str, Any] | None = None
    attempted_model_call_count = 0
    repair_success_count = 0
    invalid_output_hash_counts: dict[str, int] = {}
    repeated_invalid_output_sha256: str | None = None
    repeated_invalid_output_count = 0
    consecutive_no_advance = 0
    maximum_consecutive_no_advance_observed = 0

    resumed_tool_turn_count = 0
    if resume_from_output_dir is not None:
        source_audit = read_json(
            resume_from_output_dir / "agent_call_audit.json"
        )
        source_trace = read_json(
            resume_from_output_dir / "model_output_trace.json"
        )
        if source_audit.get("case_id") != observation["case_id"]:
            raise InteractionContractError("resume source case_id mismatch")
        if source_audit.get("requested_model") != requested_model:
            raise InteractionContractError("resume source model mismatch")
        calls = copy.deepcopy(source_audit["calls"])
        effective_models = list(source_audit["effective_models_observed"])
        attempted_model_call_count = int(
            source_audit["attempted_model_call_count"]
        )
        repair_success_count = int(source_audit["repair_success_count"])
        repeated_invalid_output_sha256 = source_audit[
            "repeated_invalid_output_sha256"
        ]
        repeated_invalid_output_count = int(
            source_audit["repeated_invalid_output_count"]
        )
        maximum_consecutive_no_advance_observed = int(
            source_audit["maximum_consecutive_no_advance_observed"]
        )
        for call in calls:
            if call.get("status") == "invalid_tool_call_contract":
                output_hash = str(call["assistant_content_sha256"])
                invalid_output_hash_counts[output_hash] = (
                    invalid_output_hash_counts.get(output_hash, 0) + 1
                )
        submitted_entries = [
            entry
            for entry in source_trace["entries"]
            if entry.get("status")
            in {"tool_call_submitted", "repair_tool_call_submitted"}
        ]
        for entry in submitted_entries:
            observation, outcome, done = sandbox.step_tool_call(
                entry["normalized_tool_call"]
            )
            _remember_evidence(observation, evidence_memory)
            public_attempts.append(
                {
                    "tool_turn_index": entry["tool_turn_index"],
                    "tool_call": copy.deepcopy(entry["normalized_tool_call"]),
                    "outcome": copy.deepcopy(outcome),
                }
            )
            if outcome["state_advanced"]:
                consecutive_no_advance = 0
            else:
                consecutive_no_advance += 1
        resumed_tool_turn_count = len(submitted_entries)

    _emit_progress(
        progress,
        {
            "event": "run_started",
            "run_id": run_id,
            "case_id": observation["case_id"],
            "max_actions": max_actions,
            "max_model_calls": max_model_calls,
            "interaction_mode": interaction_mode,
        },
    )

    for tool_turn_index in range(resumed_tool_turn_count, max_actions):
        if observation["done"]:
            terminal_reason = "environment_terminal"
            break
        if attempted_model_call_count >= max_model_calls:
            terminal_reason = "model_call_budget_exhausted"
            break
        messages = build_tool_agent_messages(
            prompt_contract=prompt,
            observation=observation,
            evidence_memory=evidence_memory,
            prior_public_attempts=public_attempts,
            interaction_mode=interaction_mode,
        )
        request_sha = canonical_hash(
            {
                "prompt_contract_sha256": prompt_sha,
                "messages": messages,
                "requested_model": requested_model,
                "temperature": DEFAULT_TEMPERATURE,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "interaction_mode": interaction_mode,
                "native_tool_schema_version": (
                    NATIVE_TOOL_SCHEMA_VERSION
                    if interaction_mode != "text_json"
                    else None
                ),
                "tools": native_tools if interaction_mode != "text_json" else [],
            }
        )
        try:
            attempted_model_call_count += 1
            response = _complete_agent(
                client=client,
                messages=messages,
                interaction_mode=interaction_mode,
                native_tools=native_tools,
            )
            record, raw_content = _model_call_record(
                call_index=len(calls),
                tool_turn_index=tool_turn_index,
                request_sha=request_sha,
                response=response,
                requested_model=requested_model,
                repair=False,
            )
            effective_models.append(record["effective_model"])
            try:
                tool_call = _tool_call_from_response(
                    response=response,
                    raw_content=raw_content,
                    interaction_mode=interaction_mode,
                )
            except InteractionContractError as exc:
                record["status"] = "invalid_tool_call_contract"
                record["error_code"] = str(exc)
                record["repair_attempted"] = False
                invalid_hash = str(record["assistant_content_sha256"])
                invalid_output_hash_counts[invalid_hash] = (
                    invalid_output_hash_counts.get(invalid_hash, 0) + 1
                )
                calls.append(record)
                _emit_progress(
                    progress,
                    {
                        "event": "invalid_tool_call",
                        "tool_turn_index": tool_turn_index,
                        "model_call_count": attempted_model_call_count,
                        "error_code": str(exc),
                        "same_output_count": invalid_output_hash_counts[invalid_hash],
                    },
                )
                if invalid_output_hash_counts[invalid_hash] >= repeat_invalid_limit:
                    repeated_invalid_output_sha256 = invalid_hash
                    repeated_invalid_output_count = invalid_output_hash_counts[
                        invalid_hash
                    ]
                    public_attempts.append(
                        {
                            "tool_turn_index": tool_turn_index,
                            "outcome": {
                                "status": "invalid_tool_contract",
                                "state_advanced": False,
                            },
                        }
                    )
                    terminal_reason = "repeated_invalid_tool_output"
                    break
                if (
                    prompt.get("repair_contract", {}).get("enabled") is True
                    and not credential_findings(raw_content)
                    and attempted_model_call_count < max_model_calls
                ):
                    record["repair_attempted"] = True
                    repair_messages = build_repair_messages(
                        prompt_contract=prompt,
                        original_messages=messages,
                        raw_content=raw_content,
                        interaction_mode=interaction_mode,
                        native_error=(
                            str(response.get("native_tool_call_error") or "")
                            if interaction_mode != "text_json"
                            else None
                        ),
                    )
                    repair_request_sha = canonical_hash(
                        {
                            "prompt_contract_sha256": prompt_sha,
                            "messages": repair_messages,
                            "requested_model": requested_model,
                            "temperature": DEFAULT_TEMPERATURE,
                            "max_tokens": DEFAULT_MAX_TOKENS,
                            "repair": True,
                            "interaction_mode": interaction_mode,
                            "native_tool_schema_version": (
                                NATIVE_TOOL_SCHEMA_VERSION
                                if interaction_mode != "text_json"
                                else None
                            ),
                            "tools": (
                                native_tools
                                if interaction_mode != "text_json"
                                else []
                            ),
                        }
                    )
                    attempted_model_call_count += 1
                    repair_response = _complete_agent(
                        client=client,
                        messages=repair_messages,
                        interaction_mode=interaction_mode,
                        native_tools=native_tools,
                    )
                    repair_record, repair_raw = _model_call_record(
                        call_index=len(calls),
                        tool_turn_index=tool_turn_index,
                        request_sha=repair_request_sha,
                        response=repair_response,
                        requested_model=requested_model,
                        repair=True,
                    )
                    effective_models.append(repair_record["effective_model"])
                    try:
                        tool_call = _tool_call_from_response(
                            response=repair_response,
                            raw_content=repair_raw,
                            interaction_mode=interaction_mode,
                        )
                        repair_record["status"] = "repair_tool_call_valid"
                        repair_success_count += 1
                    except InteractionContractError as repair_exc:
                        repair_record["status"] = "invalid_tool_call_contract"
                        repair_record["error_code"] = str(repair_exc)
                        calls.append(repair_record)
                        repair_invalid_hash = str(
                            repair_record["assistant_content_sha256"]
                        )
                        invalid_output_hash_counts[repair_invalid_hash] = (
                            invalid_output_hash_counts.get(repair_invalid_hash, 0)
                            + 1
                        )
                        public_attempts.append(
                            {
                                "tool_turn_index": tool_turn_index,
                                "outcome": {
                                    "status": "invalid_tool_contract",
                                    "state_advanced": False,
                                },
                            }
                        )
                        _emit_progress(
                            progress,
                            {
                                "event": "invalid_repair_tool_call",
                                "tool_turn_index": tool_turn_index,
                                "model_call_count": attempted_model_call_count,
                                "error_code": str(repair_exc),
                                "same_output_count": invalid_output_hash_counts[
                                    repair_invalid_hash
                                ],
                            },
                        )
                        if (
                            invalid_output_hash_counts[repair_invalid_hash]
                            >= repeat_invalid_limit
                        ):
                            repeated_invalid_output_sha256 = repair_invalid_hash
                            repeated_invalid_output_count = (
                                invalid_output_hash_counts[repair_invalid_hash]
                            )
                            terminal_reason = "repeated_invalid_tool_output"
                            break
                        continue
                    calls.append(repair_record)
                    record = repair_record
                else:
                    public_attempts.append(
                        {
                            "tool_turn_index": tool_turn_index,
                            "outcome": {
                                "status": "invalid_tool_contract",
                                "state_advanced": False,
                            },
                        }
                    )
                    if attempted_model_call_count >= max_model_calls:
                        terminal_reason = "model_call_budget_exhausted"
                        break
                    continue
            else:
                calls.append(record)

            record["normalized_tool_call"] = copy.deepcopy(tool_call)
            observation, outcome, done = sandbox.step_tool_call(tool_call)
            _remember_evidence(observation, evidence_memory)
            record.update(
                {
                    "status": (
                        "repair_tool_call_submitted"
                        if record.get("repair_call")
                        else "tool_call_submitted"
                    ),
                    "tool_call_sha256": canonical_hash(tool_call),
                    "outcome_status": outcome["status"],
                    "action_understood": outcome["action_understood"],
                    "state_advanced": outcome["state_advanced"],
                    "tool_family": outcome["tool_family"],
                    "semantic_target_seen_before": outcome[
                        "semantic_target_seen_before"
                    ],
                    "semantic_repeat_without_new_evidence": outcome[
                        "semantic_repeat_without_new_evidence"
                    ],
                    "semantic_revisit_after_new_evidence": outcome[
                        "semantic_revisit_after_new_evidence"
                    ],
                    "newly_released_evidence_ids": list(
                        outcome["newly_released_evidence_ids"]
                    ),
                }
            )
            public_attempts.append(
                {
                    "tool_turn_index": tool_turn_index,
                    "tool_call": copy.deepcopy(tool_call),
                    "outcome": copy.deepcopy(outcome),
                }
            )
            if outcome["state_advanced"]:
                consecutive_no_advance = 0
            else:
                consecutive_no_advance += 1
                maximum_consecutive_no_advance_observed = max(
                    maximum_consecutive_no_advance_observed,
                    consecutive_no_advance,
                )
            _emit_progress(
                progress,
                {
                    "event": "tool_outcome",
                    "tool_turn_index": tool_turn_index,
                    "model_call_count": attempted_model_call_count,
                    "tool": tool_call["tool"],
                    "tool_family": outcome["tool_family"],
                    "outcome_status": outcome["status"],
                    "state_advanced": outcome["state_advanced"],
                    "semantic_target_seen_before": outcome[
                        "semantic_target_seen_before"
                    ],
                    "semantic_repeat_without_new_evidence": outcome[
                        "semantic_repeat_without_new_evidence"
                    ],
                    "semantic_revisit_after_new_evidence": outcome[
                        "semantic_revisit_after_new_evidence"
                    ],
                    "completed_recorded_action_count": observation[
                        "completed_recorded_action_count"
                    ],
                    "stage": observation["stage"],
                },
            )
            if done:
                terminal_reason = "environment_terminal"
                break
            if (
                max_consecutive_no_advance is not None
                and consecutive_no_advance
                >= max_consecutive_no_advance
            ):
                terminal_reason = "stagnation_limit_reached"
                break
        except Exception as exc:
            api_error_type = type(exc).__name__
            api_error_summary = _safe_api_error_summary(exc)
            calls.append(
                {
                    "call_index": len(calls),
                    "tool_turn_index": tool_turn_index,
                    "request_sha256": request_sha,
                    "assistant_content_sha256": canonical_hash(""),
                    "effective_model": None,
                    "response_id": None,
                    "usage": {},
                    "repair_call": False,
                    "raw_assistant_content_stored": False,
                    "status": "api_error",
                    "api_error_type": api_error_type,
                    "api_error_summary": copy.deepcopy(api_error_summary),
                }
            )
            terminal_reason = "api_error"
            break

    interaction_trace = sandbox._env.export_trace()
    tool_trace = sandbox.export_tool_trace()
    submitted = sum(
        1
        for item in calls
        if item.get("status") in {"tool_call_submitted", "repair_tool_call_submitted"}
    )
    advanced = sum(1 for item in calls if item.get("state_advanced") is True)
    invalid_contract = sum(
        1 for item in calls if item.get("status") == "invalid_tool_call_contract"
    )
    outcome_counts: dict[str, int] = {}
    for item in calls:
        status = item.get("outcome_status")
        if status:
            outcome_counts[status] = outcome_counts.get(status, 0) + 1
    semantic_repeat_count = sum(
        1
        for item in calls
        if item.get("semantic_repeat_without_new_evidence") is True
    )
    semantic_revisit_count = sum(
        1
        for item in calls
        if item.get("semantic_revisit_after_new_evidence") is True
    )
    usage_totals = _usage_totals(calls)
    native_tool_call_count = sum(
        1
        for item in calls
        if item.get("tool_call_source") == "native_tool_call"
        and item.get("status")
        in {"tool_call_submitted", "repair_tool_call_submitted"}
    )
    text_json_tool_call_count = sum(
        1
        for item in calls
        if item.get("tool_call_source") == "text_json"
        and item.get("status")
        in {"tool_call_submitted", "repair_tool_call_submitted"}
    )
    text_json_fallback_count = (
        text_json_tool_call_count if interaction_mode == "auto" else 0
    )
    _emit_progress(
        progress,
        {
            "event": "run_finished",
            "terminal_reason": terminal_reason,
            "model_call_count": len(calls),
            "tool_turn_count": len(public_attempts),
            "advanced_action_count": advanced,
            "final_stage": interaction_trace["final_state"]["stage"],
            "usage": usage_totals,
        },
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    interaction_trace_path = output_dir / "interaction_trace.json"
    tool_trace_path = output_dir / "tool_trace.json"
    audit_path = output_dir / "agent_call_audit.json"
    model_output_trace_path = output_dir / "model_output_trace.json"
    manifest_path = output_dir / "run_manifest.json"
    write_json(interaction_trace_path, interaction_trace)
    write_json(tool_trace_path, tool_trace)

    model_output_trace = build_model_output_trace(
        run_id=run_id,
        case_id=interaction_trace["case_id"],
        agent_id=agent_id,
        requested_model=requested_model,
        effective_models_observed=sorted(set(effective_models)),
        terminal_reason=terminal_reason,
        calls=calls,
        tool_trace=tool_trace,
        interaction_trace=interaction_trace,
    )
    write_json(model_output_trace_path, model_output_trace)

    audit = {
        "audit_version": AGENT_CALL_AUDIT_VERSION,
        "runner_version": AGENT_RUNNER_VERSION,
        "prompt_version": prompt_version,
        "tool_map_version": tool_map_version,
        "tool_trace_version": TOOL_CALL_TRACE_VERSION,
        "agent_protocol_version": AGENT_PROTOCOL_VERSION,
        "interaction_mode": interaction_mode,
        "native_tool_schema_version": (
            NATIVE_TOOL_SCHEMA_VERSION
            if interaction_mode != "text_json"
            else None
        ),
        "run_id": run_id,
        "case_id": interaction_trace["case_id"],
        "requested_model": requested_model,
        "effective_models_observed": sorted(set(effective_models)),
        "attempted_model_call_count": attempted_model_call_count,
        "model_call_count": len(calls),
        "tool_turn_count": len(public_attempts),
        "submitted_tool_call_count": submitted,
        "native_tool_call_count": native_tool_call_count,
        "text_json_tool_call_count": text_json_tool_call_count,
        "text_json_fallback_count": text_json_fallback_count,
        "advanced_action_count": advanced,
        "invalid_tool_contract_count": invalid_contract,
        "repair_success_count": repair_success_count,
        "repeated_invalid_output_count": repeated_invalid_output_count,
        "repeated_invalid_output_sha256": repeated_invalid_output_sha256,
        "usage_totals": usage_totals,
        "outcome_counts": outcome_counts,
        "semantic_repeat_without_new_evidence_count": semantic_repeat_count,
        "semantic_revisit_after_new_evidence_count": semantic_revisit_count,
        "max_consecutive_no_advance": max_consecutive_no_advance,
        "maximum_consecutive_no_advance_observed": (
            maximum_consecutive_no_advance_observed
        ),
        "terminal_reason": terminal_reason,
        "api_error_type": api_error_type,
        "api_error_summary": copy.deepcopy(api_error_summary),
        "calls": calls,
        "gold_action_exposed_to_agent": False,
        "dependency_graph_exposed_to_agent": False,
        "future_evidence_exposed_to_agent": False,
        "resolver_internals_exposed_to_agent": False,
        "raw_assistant_content_stored": False,
        "normalized_model_output_trace_stored": True,
        "model_output_trace_version": MODEL_OUTPUT_TRACE_VERSION,
        "run_data_contract_version": RUN_DATA_CONTRACT_VERSION,
        "credential_storage": "environment_only",
        "transport_security_mode": transport_mode,
        "network_transport_encrypted": transport_mode == "https",
        "credentials_echoed": False,
        "official_scoring_allowed": False,
        "single_total_score_absent": True,
    }
    audit_errors = validate_call_audit(audit)
    if audit_errors:
        raise InteractionContractError(
            "generated v0.4 call audit is invalid: " + "; ".join(audit_errors)
        )
    write_json(audit_path, audit)

    manifest = {
        "manifest_version": RUN_MANIFEST_VERSION,
        "runner_version": AGENT_RUNNER_VERSION,
        "status": (
            "complete_non_official_tool_canary"
            if interaction_trace["status"] == "complete"
            else "incomplete_non_official_tool_canary"
        ),
        "run_id": run_id,
        "case_id": interaction_trace["case_id"],
        "environment_version": ENVIRONMENT_VERSION,
        "agent": {
            "agent_id": agent_id,
            "requested_model": requested_model,
            "effective_models_observed": sorted(set(effective_models)),
            "prompt_version": AGENT_PROMPT_VERSION,
            "temperature": DEFAULT_TEMPERATURE,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "max_actions": max_actions,
            "max_model_calls": max_model_calls,
            "max_consecutive_no_advance": max_consecutive_no_advance,
            "interaction_mode": interaction_mode,
            "gold_accessor_used": False,
        },
        "patient_simulator": {
            "llm_used": False,
            "facts_deterministic": True,
            "ehr_state_authoritative": True,
        },
        "tool_interface": {
            "tool_map_version": tool_map_version,
            "tool_facade_reusable_across_cases": True,
            "model_calls_tools": True,
            "agent_protocol_version": AGENT_PROTOCOL_VERSION,
            "interaction_mode": interaction_mode,
            "native_tool_calling_enabled": interaction_mode
            in {"native_tool_call", "auto"},
            "native_tool_schema_version": (
                NATIVE_TOOL_SCHEMA_VERSION
                if interaction_mode != "text_json"
                else None
            ),
            "text_json_fallback_enabled": interaction_mode == "auto",
            "unrecorded_clinically_valid_actions_allowed": True,
            "counterfactual_results_invented": False,
            "dependency_graph_exposed_to_agent": False,
            "available_transition_ids_exposed_to_agent": False,
            "gold_derived_navigation_hints_exposed_to_agent": bool(
                prompt.get("coarse_available_action_families_visible", False)
            ),
            "json_repair_enabled": True,
            "compact_repair_context": True,
            "repeated_invalid_output_limit": repeat_invalid_limit,
            "evidence_compaction": "title_conclusion_head_tail",
            "semantic_repeat_soft_memory_enabled": True,
            "semantic_repeat_blocks_tool_execution": False,
            "stagnation_limit_stops_live_spend": (
                max_consecutive_no_advance is not None
            ),
        },
        "run_data_collection": {
            "run_data_contract_version": RUN_DATA_CONTRACT_VERSION,
            "model_output_trace_version": MODEL_OUTPUT_TRACE_VERSION,
            "case_artifacts_referenced_by_hash": True,
            "full_raw_ehr_copied_into_run": False,
            "raw_provider_response_stored": False,
            "raw_assistant_text_stored": False,
            "normalized_model_tool_calls_stored": True,
            "minimal_environment_receipts_stored": True,
            "credentials_stored": False,
        },
        "run_summary": {
            "terminal_reason": terminal_reason,
            "attempted_model_call_count": attempted_model_call_count,
            "model_call_count": len(calls),
            "tool_turn_count": len(public_attempts),
            "submitted_tool_call_count": submitted,
            "native_tool_call_count": native_tool_call_count,
            "text_json_tool_call_count": text_json_tool_call_count,
            "text_json_fallback_count": text_json_fallback_count,
            "invalid_tool_contract_count": invalid_contract,
            "repair_success_count": repair_success_count,
            "repeated_invalid_output_count": repeated_invalid_output_count,
            "repeated_invalid_output_sha256": repeated_invalid_output_sha256,
            "usage_totals": usage_totals,
            "advanced_action_count": advanced,
            "outcome_counts": outcome_counts,
            "semantic_repeat_without_new_evidence_count": semantic_repeat_count,
            "semantic_revisit_after_new_evidence_count": semantic_revisit_count,
            "max_consecutive_no_advance": max_consecutive_no_advance,
            "maximum_consecutive_no_advance_observed": (
                maximum_consecutive_no_advance_observed
            ),
            "completed_transition_count": len(
                interaction_trace["final_state"]["completed_transition_ids"]
            ),
            "final_stage": interaction_trace["final_state"]["stage"],
            "environment_terminal": interaction_trace["final_state"]["done"],
        },
        "transport_security": {
            "mode": transport_mode,
            "network_transport_encrypted": transport_mode == "https",
            "http_supported_without_extra_gate": True,
        },
        "bindings": {
            "scorer_freeze_id": interaction_trace["scorer_freeze"]["freeze_id"],
            "scorer_freeze_manifest_sha256": sha256_file(freeze_manifest_path),
            "scenario_sha256": sha256_file(scenario_path),
            "action_interface_sha256": sha256_file(interface_path),
            "action_ontology_sha256": sha256_file(ontology_path),
            "action_graph_sha256": sha256_file(graph_path),
            "tool_map_sha256": sha256_file(tool_map_path),
            "agent_prompt_contract_sha256": prompt_sha,
            "interaction_trace_sha256": sha256_file(interaction_trace_path),
            "tool_trace_sha256": sha256_file(tool_trace_path),
            "agent_call_audit_sha256": sha256_file(audit_path),
            "model_output_trace_sha256": sha256_file(
                model_output_trace_path
            ),
            "run_data_contract_sha256": sha256_file(
                DEFAULT_RUN_DATA_CONTRACT
            ),
        },
        "credential_storage": "environment_only",
        "credentials_echoed": False,
        "automatic_semantic_judge_scoring": False,
        "official_scoring_allowed": False,
        "single_total_score_absent": True,
    }
    manifest["agent"]["prompt_version"] = prompt_version
    if case_registry_path is not None:
        manifest["bindings"]["case_registry_sha256"] = sha256_file(
            case_registry_path
        )
    if api_error_summary is not None:
        manifest["api_error_summary"] = copy.deepcopy(api_error_summary)
    write_json(manifest_path, manifest)
    errors = validate_run_manifest(
        manifest,
        interaction_trace_path=interaction_trace_path,
        tool_trace_path=tool_trace_path,
        call_audit_path=audit_path,
        scenario_path=scenario_path,
        freeze_manifest_path=freeze_manifest_path,
        interface_path=interface_path,
        ontology_path=ontology_path,
        graph_path=graph_path,
        tool_map_path=tool_map_path,
        prompt_contract_path=prompt_contract_path,
        case_registry_path=case_registry_path,
    )
    if errors:
        raise InteractionContractError(
            "generated v0.4 manifest is invalid: " + "; ".join(errors)
        )
    return {
        "status": manifest["status"],
        "run_id": run_id,
        "case_id": interaction_trace["case_id"],
        "terminal_reason": terminal_reason,
        "attempted_model_call_count": attempted_model_call_count,
        "model_call_count": len(calls),
        "tool_turn_count": len(public_attempts),
        "submitted_tool_call_count": submitted,
        "native_tool_call_count": native_tool_call_count,
        "text_json_tool_call_count": text_json_tool_call_count,
        "text_json_fallback_count": text_json_fallback_count,
        "advanced_action_count": advanced,
        "invalid_tool_contract_count": invalid_contract,
        "repair_success_count": repair_success_count,
        "repeated_invalid_output_count": repeated_invalid_output_count,
        "repeated_invalid_output_sha256": repeated_invalid_output_sha256,
        "usage_totals": usage_totals,
        "outcome_counts": outcome_counts,
        "semantic_repeat_without_new_evidence_count": semantic_repeat_count,
        "semantic_revisit_after_new_evidence_count": semantic_revisit_count,
        "max_consecutive_no_advance": max_consecutive_no_advance,
        "maximum_consecutive_no_advance_observed": (
            maximum_consecutive_no_advance_observed
        ),
        "completed_transition_count": len(
            interaction_trace["final_state"]["completed_transition_ids"]
        ),
        "final_stage": interaction_trace["final_state"]["stage"],
        "environment_terminal": interaction_trace["final_state"]["done"],
        "transport_security_mode": transport_mode,
        "interaction_mode": interaction_mode,
        "api_error_summary": copy.deepcopy(api_error_summary),
        "output_dir": str(output_dir.resolve()),
        "credentials_echoed": False,
        "official_scoring_allowed": False,
        "single_total_score_absent": True,
    }


__all__ = [
    "DEFAULT_AGENT_MODEL",
    "DEFAULT_FREEZE_MANIFEST",
    "DEFAULT_GRAPH",
    "DEFAULT_INTERFACE",
    "DEFAULT_MAX_ACTIONS",
    "DEFAULT_MAX_MODEL_CALLS",
    "DEFAULT_ONTOLOGY",
    "DEFAULT_PROMPT_CONTRACT",
    "DEFAULT_SCENARIO",
    "DEFAULT_TOOL_MAP",
    "build_tool_agent_messages",
    "load_prompt_contract",
    "make_live_client",
    "readiness",
    "run_tool_agent_canary",
    "safe_name",
    "transport_security_mode",
    "validate_run_manifest",
]
