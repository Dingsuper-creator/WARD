"""Reviewed-ontology batch successor with the frozen benchmark prompt."""

from __future__ import annotations

from typing import Any

from world_model_patient_ehr_v053 import agent_canary as v053_canary
from world_model_patient_ehr_v054 import agent_canary as v054_canary

from .tool_facade import HybridClinicalToolSandbox


AGENT_RUNNER_VERSION = "ward-patient-ehr-tool-agent-runner-0.5.6"
TRANSIENT_RETRY_POLICY_VERSION = (
    v054_canary.TRANSIENT_RETRY_POLICY_VERSION
)
DEFAULT_MAX_PROVIDER_ATTEMPTS = (
    v054_canary.DEFAULT_MAX_PROVIDER_ATTEMPTS
)


def run_tool_agent_canary(**kwargs: Any) -> dict[str, Any]:
    """Run v0.5.6 without mutating any predecessor module on disk."""

    original_sandbox = v053_canary.LifecycleClinicalToolSandbox
    original_v054_runner = v054_canary.AGENT_RUNNER_VERSION
    try:
        v053_canary.LifecycleClinicalToolSandbox = (
            HybridClinicalToolSandbox
        )
        v054_canary.AGENT_RUNNER_VERSION = AGENT_RUNNER_VERSION
        return v054_canary.run_tool_agent_canary(**kwargs)
    finally:
        v053_canary.LifecycleClinicalToolSandbox = original_sandbox
        v054_canary.AGENT_RUNNER_VERSION = original_v054_runner


__all__ = [
    "AGENT_RUNNER_VERSION",
    "DEFAULT_MAX_PROVIDER_ATTEMPTS",
    "TRANSIENT_RETRY_POLICY_VERSION",
    "run_tool_agent_canary",
]
