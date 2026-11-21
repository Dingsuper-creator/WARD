"""Single-dispatcher adapter preserving the seven canonical clinical tools."""
from __future__ import annotations
import copy
import json
from typing import Any
from world_model_patient_ehr_v01.contracts import InteractionContractError
from world_model_patient_ehr_v04 import agent_canary as v04_canary
from world_model_patient_ehr_v04.agent_protocol import NATIVE_TO_CANONICAL_NAME
DISPATCHER_NAME = "clinical_action"
STRIPPED_WIRE_KEYWORDS = ("minLength", "maxLength", "uniqueItems")

def strip_wire_validation_keywords(value: Any) -> None:
    if isinstance(value, dict):
        for keyword in STRIPPED_WIRE_KEYWORDS:
            value.pop(keyword, None)
        for child in value.values():
            strip_wire_validation_keywords(child)
    elif isinstance(value, list):
        for child in value:
            strip_wire_validation_keywords(child)


def build_dispatcher_tool(tools: list[dict[str, Any]]) -> dict[str, Any]:
    function_names = [tool["function"]["name"] for tool in tools]
    if set(function_names) != set(NATIVE_TO_CANONICAL_NAME):
        raise InteractionContractError(
            "dispatcher requires the complete seven-tool v0.5 interface"
        )
    properties = copy.deepcopy(
        tools[0]["function"]["parameters"]["properties"]
    )
    strip_wire_validation_keywords(properties)
    properties = {
        "tool": {
            "type": "string",
            "enum": function_names,
            "description": "Select one frozen v0.5 clinical action.",
        },
        **properties,
    }
    return {
        "type": "function",
        "function": {
            "name": DISPATCHER_NAME,
            "description": (
                "Execute exactly one action from the frozen v0.5 clinical "
                "tool interface."
            ),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["tool", "query"],
                "additionalProperties": False,
            },
        },
    }


def adapt_dispatcher_message(
    message: Any,
    canonical_extractor: Any,
) -> dict[str, Any]:
    adapted = copy.deepcopy(message)
    try:
        function = adapted["tool_calls"][0]["function"]
    except (KeyError, IndexError, TypeError) as exc:
        raise InteractionContractError(
            "API response is missing dispatcher tool data"
        ) from exc
    if function.get("name") != DISPATCHER_NAME:
        raise InteractionContractError(
            "API response used a non-dispatcher function"
        )
    raw_arguments = function.get("arguments")
    if isinstance(raw_arguments, str):
        try:
            dispatcher_arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise InteractionContractError(
                "dispatcher arguments contain invalid JSON"
            ) from exc
    elif isinstance(raw_arguments, dict):
        dispatcher_arguments = copy.deepcopy(raw_arguments)
    else:
        raise InteractionContractError(
            "dispatcher arguments must be an object"
        )
    function_name = dispatcher_arguments.pop("tool", None)
    if function_name not in NATIVE_TO_CANONICAL_NAME:
        raise InteractionContractError(
            "dispatcher selected an unsupported frozen tool"
        )
    function["name"] = function_name
    function["arguments"] = json.dumps(
        dispatcher_arguments,
        ensure_ascii=False,
    )
    return canonical_extractor(adapted)


class KimiDispatcherClient:
    def __init__(self, client: Any) -> None:
        self._client = client

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not tools:
            raise InteractionContractError(
                "Kimi dispatcher requires canonical native tools"
            )
        dispatcher_tool = build_dispatcher_tool(tools)
        canonical_extractor = v04_canary.extract_openai_native_tool_call

        def adapted_extractor(message: Any) -> dict[str, Any]:
            return adapt_dispatcher_message(message, canonical_extractor)

        v04_canary.extract_openai_native_tool_call = adapted_extractor
        try:
            return self._client.complete(
                messages,
                tools=[dispatcher_tool],
            )
        finally:
            v04_canary.extract_openai_native_tool_call = canonical_extractor


