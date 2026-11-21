"""Native clinical tool schemas with optional structured routing fields."""

from __future__ import annotations

import copy
import json
from typing import Any

from world_model_patient_ehr_v01.contracts import (
    InteractionContractError,
    credential_findings,
)
from world_model_patient_ehr_v04.agent_protocol import (
    AGENT_PROTOCOL_VERSION,
    CANONICAL_TO_NATIVE_NAME,
    INTERACTION_MODES,
    NATIVE_TO_CANONICAL_NAME,
    TOOL_DESCRIPTIONS,
    validate_interaction_mode,
)
from world_model_patient_ehr_v04.tool_facade import TOOL_NAMES


NATIVE_TOOL_SCHEMA_VERSION = "ward-patient-ehr-native-tools-0.2.0"
STRUCTURED_CLINICAL_FIELDS = (
    "resource_type",
    "clinical_phase",
    "anatomy",
    "laterality",
    "procedure_family",
    "result_family",
)


def build_native_tool_schemas(
    *,
    public_tools: list[str],
    public_ehr_modules: list[str],
) -> list[dict[str, Any]]:
    if set(public_tools) != TOOL_NAMES:
        raise InteractionContractError("public tool set is not canonical")
    if not public_ehr_modules or any(
        not isinstance(module, str) or not module
        for module in public_ehr_modules
    ):
        raise InteractionContractError("public EHR modules are invalid")
    structured_descriptions = {
        "resource_type": (
            "Optional generic resource type, such as document, pathology, "
            "imaging, procedure, consultation, or discharge."
        ),
        "clinical_phase": (
            "Optional clinical phase, such as admission, preoperative, "
            "postoperative, follow-up, or discharge."
        ),
        "anatomy": "Optional anatomy stated in ordinary clinical language.",
        "laterality": "Optional laterality when clinically relevant.",
        "procedure_family": (
            "Optional generic procedure family, such as core biopsy or "
            "breast-conserving surgery."
        ),
        "result_family": (
            "Optional generic result family, such as pathology, X-ray, CT, "
            "or MRI."
        ),
    }
    schemas: list[dict[str, Any]] = []
    for canonical_name in sorted(public_tools):
        properties: dict[str, Any] = {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240,
                "description": (
                    "The concrete clinical question, target, order, "
                    "procedure, treatment, or transition."
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
            "rationale": {"type": "string", "maxLength": 1200},
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
            },
        }
        for field in STRUCTURED_CLINICAL_FIELDS:
            properties[field] = {
                "type": "string",
                "minLength": 1,
                "maxLength": 120,
                "description": structured_descriptions[field],
            }
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": CANONICAL_TO_NATIVE_NAME[canonical_name],
                    "description": TOOL_DESCRIPTIONS[canonical_name],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
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
    allowed = {
        "query",
        "module",
        "intent",
        "rationale",
        "evidence_ids",
        *STRUCTURED_CLINICAL_FIELDS,
    }
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
    for field in ("module", *STRUCTURED_CLINICAL_FIELDS):
        field_value = value.get(field)
        if field_value is None:
            continue
        if (
            not isinstance(field_value, str)
            or not field_value.strip()
            or len(field_value) > 120
        ):
            raise InteractionContractError(
                f"native function {field} must be a non-empty short string"
            )
        args[field] = field_value.strip()
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
            {"type": "function", "function": message["function_call"]}
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
    "STRUCTURED_CLINICAL_FIELDS",
    "build_native_tool_schemas",
    "extract_openai_native_tool_call",
    "native_function_call_to_tool_call",
    "validate_interaction_mode",
]
