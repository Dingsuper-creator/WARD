"""Canonical Agent interaction modes and native tool-call normalization."""

from __future__ import annotations

import copy
import json
from typing import Any

from world_model_patient_ehr_v01.contracts import (
    InteractionContractError,
    credential_findings,
)
from world_model_patient_ehr_v04.tool_facade import TOOL_NAMES


AGENT_PROTOCOL_VERSION = "ward-patient-ehr-agent-protocol-0.1.0"
NATIVE_TOOL_SCHEMA_VERSION = "ward-patient-ehr-native-tools-0.1.0"
INTERACTION_MODES = {
    "native_tool_call",
    "text_json",
    "auto",
}

CANONICAL_TO_NATIVE_NAME = {
    "patient.ask": "patient_ask",
    "ehr.search": "ehr_search",
    "ehr.read": "ehr_read",
    "ehr.order": "ehr_order",
    "procedure.perform": "procedure_perform",
    "treatment.start": "treatment_start",
    "care.transition": "care_transition",
}
NATIVE_TO_CANONICAL_NAME = {
    value: key for key, value in CANONICAL_TO_NATIVE_NAME.items()
}

TOOL_DESCRIPTIONS = {
    "patient.ask": (
        "Ask the patient for chart-grounded history or symptom information."
    ),
    "ehr.search": (
        "Search a public EHR module for currently available records."
    ),
    "ehr.read": (
        "Read or review currently available records in a public EHR module."
    ),
    "ehr.order": (
        "Order a clinically justified laboratory, imaging, pathology, or other "
        "EHR-supported investigation."
    ),
    "procedure.perform": (
        "Request or perform a clinically justified procedure represented by "
        "the Patient/EHR sandbox."
    ),
    "treatment.start": (
        "Start a clinically justified treatment represented by the "
        "Patient/EHR sandbox."
    ),
    "care.transition": (
        "Make a clinically justified care transition or disposition action."
    ),
}


def validate_interaction_mode(value: str) -> str:
    mode = str(value or "").strip()
    if mode not in INTERACTION_MODES:
        raise InteractionContractError(
            "interaction_mode must be native_tool_call, text_json, or auto"
        )
    return mode


def build_native_tool_schemas(
    *,
    public_tools: list[str],
    public_ehr_modules: list[str],
) -> list[dict[str, Any]]:
    if set(public_tools) != TOOL_NAMES:
        raise InteractionContractError("public tool set is not canonical")
    if (
        not public_ehr_modules
        or any(
            not isinstance(module, str) or not module
            for module in public_ehr_modules
        )
    ):
        raise InteractionContractError("public EHR modules are invalid")
    schemas: list[dict[str, Any]] = []
    for canonical_name in sorted(public_tools):
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": CANONICAL_TO_NATIVE_NAME[canonical_name],
                    "description": TOOL_DESCRIPTIONS[canonical_name],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 240,
                                "description": (
                                    "The concrete clinical question, target, "
                                    "order, procedure, treatment, or transition."
                                ),
                            },
                            "module": {
                                "type": "string",
                                "enum": list(public_ehr_modules),
                                "description": "Relevant public EHR module.",
                            },
                            "intent": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 240,
                            },
                            "rationale": {
                                "type": "string",
                                "maxLength": 1200,
                            },
                            "evidence_ids": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1},
                                "uniqueItems": True,
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    if credential_findings(schemas):
        raise InteractionContractError(
            "native tool schemas contain credential-like material"
        )
    return schemas


def native_function_call_to_tool_call(
    *,
    function_name: str,
    arguments: Any,
) -> dict[str, Any]:
    canonical_name = NATIVE_TO_CANONICAL_NAME.get(str(function_name or ""))
    if canonical_name is None:
        raise InteractionContractError("native function name is not allowed")
    if isinstance(arguments, str):
        try:
            value = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise InteractionContractError(
                "native function arguments contain invalid JSON"
            ) from exc
    elif isinstance(arguments, dict):
        value = copy.deepcopy(arguments)
    else:
        raise InteractionContractError(
            "native function arguments must be an object"
        )
    if not isinstance(value, dict):
        raise InteractionContractError(
            "native function arguments must decode to an object"
        )
    allowed = {"query", "module", "intent", "rationale", "evidence_ids"}
    if set(value) - allowed:
        raise InteractionContractError(
            "native function arguments contain unsupported fields"
        )
    query = value.get("query")
    if not isinstance(query, str) or not query.strip():
        raise InteractionContractError(
            "native function query must be a non-empty string"
        )
    args: dict[str, Any] = {"query": query.strip()}
    module = value.get("module")
    if module is not None:
        if not isinstance(module, str) or not module.strip():
            raise InteractionContractError(
                "native function module must be a non-empty string"
            )
        args["module"] = module.strip()
    tool_call = {
        "tool": canonical_name,
        "intent": (
            value["intent"].strip()
            if isinstance(value.get("intent"), str)
            and value["intent"].strip()
            else query.strip()
        ),
        "args": args,
        "rationale": (
            value["rationale"].strip()
            if isinstance(value.get("rationale"), str)
            else ""
        ),
        "evidence_ids": copy.deepcopy(value.get("evidence_ids", [])),
    }
    if credential_findings(tool_call):
        raise InteractionContractError(
            "native function call contains credential-like material"
        )
    return tool_call


def extract_openai_native_tool_call(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise InteractionContractError("API response message must be an object")
    tool_calls = message.get("tool_calls")
    if tool_calls is None and isinstance(message.get("function_call"), dict):
        tool_calls = [
            {
                "type": "function",
                "function": message["function_call"],
            }
        ]
    if not isinstance(tool_calls, list) or not tool_calls:
        raise InteractionContractError("API response is missing a native tool call")
    if len(tool_calls) != 1:
        raise InteractionContractError(
            "API response must contain exactly one native tool call"
        )
    item = tool_calls[0]
    if not isinstance(item, dict) or item.get("type", "function") != "function":
        raise InteractionContractError("API response tool call is not a function")
    function = item.get("function")
    if not isinstance(function, dict):
        raise InteractionContractError(
            "API response native tool call lacks function data"
        )
    return native_function_call_to_tool_call(
        function_name=str(function.get("name") or ""),
        arguments=function.get("arguments"),
    )


__all__ = [
    "AGENT_PROTOCOL_VERSION",
    "INTERACTION_MODES",
    "NATIVE_TOOL_SCHEMA_VERSION",
    "build_native_tool_schemas",
    "extract_openai_native_tool_call",
    "native_function_call_to_tool_call",
    "validate_interaction_mode",
]
