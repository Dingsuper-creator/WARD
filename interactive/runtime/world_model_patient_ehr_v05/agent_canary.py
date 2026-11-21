"""v0.5 state-based adapter for the validated v0.4.6 Agent runner."""

from __future__ import annotations

from typing import Any

from world_model_patient_ehr_v04 import agent_canary as v04_canary

from .agent_protocol import (
    NATIVE_TOOL_SCHEMA_VERSION,
    build_native_tool_schemas,
    extract_openai_native_tool_call,
)
from .tool_facade import StateBasedClinicalToolSandbox


AGENT_RUNNER_VERSION = "ward-patient-ehr-tool-agent-runner-0.5.0"


def run_tool_agent_canary(**kwargs: Any) -> dict[str, Any]:
    """Run the established canary with only the routing components replaced."""

    original = {
        "LightweightClinicalToolSandbox": (
            v04_canary.LightweightClinicalToolSandbox
        ),
        "build_native_tool_schemas": v04_canary.build_native_tool_schemas,
        "extract_openai_native_tool_call": (
            v04_canary.extract_openai_native_tool_call
        ),
        "NATIVE_TOOL_SCHEMA_VERSION": v04_canary.NATIVE_TOOL_SCHEMA_VERSION,
        "AGENT_RUNNER_VERSION": v04_canary.AGENT_RUNNER_VERSION,
    }
    try:
        v04_canary.LightweightClinicalToolSandbox = (
            StateBasedClinicalToolSandbox
        )
        v04_canary.build_native_tool_schemas = build_native_tool_schemas
        v04_canary.extract_openai_native_tool_call = (
            extract_openai_native_tool_call
        )
        v04_canary.NATIVE_TOOL_SCHEMA_VERSION = NATIVE_TOOL_SCHEMA_VERSION
        v04_canary.AGENT_RUNNER_VERSION = AGENT_RUNNER_VERSION
        v04_canary.SUPPORTED_AGENT_RUNNER_VERSIONS.add(
            AGENT_RUNNER_VERSION
        )
        return v04_canary.run_tool_agent_canary(**kwargs)
    finally:
        for name, value in original.items():
            setattr(v04_canary, name, value)


__all__ = [
    "AGENT_RUNNER_VERSION",
    "NATIVE_TOOL_SCHEMA_VERSION",
    "run_tool_agent_canary",
]
