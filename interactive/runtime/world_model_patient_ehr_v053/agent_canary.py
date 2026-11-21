"""Lifecycle successor with the existing auditable HTTP retry."""

from __future__ import annotations

from typing import Any

from world_model_patient_ehr_v05 import agent_canary as v05_canary
from world_model_patient_ehr_v051 import agent_canary as v051_canary

from .tool_facade import LifecycleClinicalToolSandbox


AGENT_RUNNER_VERSION = "ward-patient-ehr-tool-agent-runner-0.5.3"
TRANSIENT_RETRY_POLICY_VERSION = (
    v051_canary.TRANSIENT_RETRY_POLICY_VERSION
)
DEFAULT_MAX_PROVIDER_ATTEMPTS = (
    v051_canary.DEFAULT_MAX_PROVIDER_ATTEMPTS
)


def run_tool_agent_canary(**kwargs: Any) -> dict[str, Any]:
    """Run v0.5.3 without modifying preserved predecessor modules."""

    original_sandbox = v05_canary.StateBasedClinicalToolSandbox
    original_retry_runner = v051_canary.AGENT_RUNNER_VERSION
    try:
        v05_canary.StateBasedClinicalToolSandbox = (
            LifecycleClinicalToolSandbox
        )
        v051_canary.AGENT_RUNNER_VERSION = AGENT_RUNNER_VERSION
        return v051_canary.run_tool_agent_canary(**kwargs)
    finally:
        v05_canary.StateBasedClinicalToolSandbox = original_sandbox
        v051_canary.AGENT_RUNNER_VERSION = original_retry_runner


__all__ = [
    "AGENT_RUNNER_VERSION",
    "DEFAULT_MAX_PROVIDER_ATTEMPTS",
    "TRANSIENT_RETRY_POLICY_VERSION",
    "run_tool_agent_canary",
]
