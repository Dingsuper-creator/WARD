"""Benchmark protocol successor over the preserved v0.5.3 environment."""

from __future__ import annotations

from typing import Any

from world_model_patient_ehr_v04 import agent_canary as v04_canary
from world_model_patient_ehr_v053 import agent_canary as v053_canary

from .prompting import (
    AGENT_PROMPT_VERSION,
    build_benchmark_tool_agent_messages,
)


AGENT_RUNNER_VERSION = "ward-patient-ehr-tool-agent-runner-0.5.4"
TRANSIENT_RETRY_POLICY_VERSION = (
    v053_canary.TRANSIENT_RETRY_POLICY_VERSION
)
DEFAULT_MAX_PROVIDER_ATTEMPTS = (
    v053_canary.DEFAULT_MAX_PROVIDER_ATTEMPTS
)


def run_tool_agent_canary(**kwargs: Any) -> dict[str, Any]:
    """Run v0.5.4 while preserving all predecessor modules."""

    original_builder = v04_canary.build_tool_agent_messages
    original_prompt_versions = set(
        v04_canary.SUPPORTED_AGENT_PROMPT_VERSIONS
    )
    original_v053_runner = v053_canary.AGENT_RUNNER_VERSION
    try:
        v04_canary.build_tool_agent_messages = (
            build_benchmark_tool_agent_messages
        )
        v04_canary.SUPPORTED_AGENT_PROMPT_VERSIONS.add(
            AGENT_PROMPT_VERSION
        )
        v053_canary.AGENT_RUNNER_VERSION = AGENT_RUNNER_VERSION
        return v053_canary.run_tool_agent_canary(**kwargs)
    finally:
        v04_canary.build_tool_agent_messages = original_builder
        v04_canary.SUPPORTED_AGENT_PROMPT_VERSIONS.clear()
        v04_canary.SUPPORTED_AGENT_PROMPT_VERSIONS.update(
            original_prompt_versions
        )
        v053_canary.AGENT_RUNNER_VERSION = original_v053_runner


__all__ = [
    "AGENT_PROMPT_VERSION",
    "AGENT_RUNNER_VERSION",
    "DEFAULT_MAX_PROVIDER_ATTEMPTS",
    "TRANSIENT_RETRY_POLICY_VERSION",
    "run_tool_agent_canary",
]
