"""Model-independent command expression and public recovery guidance."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from world_model_patient_ehr_v01.contracts import (
    InteractionContractError,
    credential_findings,
    read_json,
    sha256_file,
)
from world_model_patient_ehr_v04 import agent_canary as v04_canary


AGENT_PROMPT_VERSION = "ward-patient-ehr-agent-prompt-0.5.4"
BASE_BUILD_TOOL_AGENT_MESSAGES = v04_canary.build_tool_agent_messages
NO_ADVANCE_OUTCOMES = {
    "recognized_completed_action",
    "recognized_prerequisite_pending",
    "valid_exploration_no_recorded_outcome",
}


def validate_benchmark_prompt_contract(
    contract: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if contract.get("prompt_version") != AGENT_PROMPT_VERSION:
        errors.append("wrong_prompt_version")
    if credential_findings(contract):
        errors.append("credential_like_material_present")
    for key in (
        "gold_action_visible",
        "dependency_graph_visible",
        "future_evidence_visible",
        "coarse_available_action_families_visible",
        "automatic_semantic_judge_scoring",
        "official_scoring_allowed",
    ):
        if contract.get(key) is not False:
            errors.append(f"invalid_boundary:{key}")
    if contract.get("single_total_score_absent") is not True:
        errors.append("single_total_score_boundary_invalid")

    command = contract.get("command_expression_protocol", {})
    if command.get("executable_field") != "args.query":
        errors.append("wrong_executable_field")
    if command.get("intent_or_rationale_can_route") is not False:
        errors.append("background_routing_enabled")
    if command.get("hidden_identifier_required") is not False:
        errors.append("hidden_identifier_required")
    if set(command.get("temporal_relations", [])) != {
        "before",
        "during",
        "after",
    }:
        errors.append("invalid_temporal_relations")

    recovery = contract.get("recovery_protocol", {})
    if set(recovery.get("trigger_public_outcomes", [])) != (
        NO_ADVANCE_OUTCOMES
    ):
        errors.append("invalid_recovery_triggers")
    if recovery.get("maximum_reformulations_of_same_target") != 1:
        errors.append("invalid_reformulation_limit")
    for key in (
        "hidden_state_used",
        "gold_action_hint_used",
    ):
        if recovery.get(key) is not False:
            errors.append(f"invalid_recovery_boundary:{key}")
    if recovery.get("generic_missing_dimensions_only") is not True:
        errors.append("non_generic_recovery_enabled")

    composite = contract.get("composite_action_protocol", {})
    if composite.get(
        "complete_evidence_supported_operation_required"
    ) is not True:
        errors.append("complete_composite_operation_not_required")
    if composite.get("single_component_substitution_allowed") is not False:
        errors.append("single_component_substitution_allowed")
    if composite.get(
        "template_or_consent_proves_correct_operation"
    ) is not False:
        errors.append("template_treated_as_correct_operation")
    return errors


def load_benchmark_prompt_contract(
    path: Path,
) -> tuple[dict[str, Any], str]:
    contract = read_json(path)
    errors = validate_benchmark_prompt_contract(contract)
    if errors:
        raise InteractionContractError(
            "invalid v0.5.4 prompt contract: " + "; ".join(errors)
        )
    return contract, sha256_file(path)


def build_benchmark_tool_agent_messages(
    *,
    prompt_contract: dict[str, Any],
    observation: dict[str, Any],
    evidence_memory: dict[str, dict[str, Any]],
    prior_public_attempts: list[dict[str, Any]],
    interaction_mode: str = "text_json",
) -> list[dict[str, str]]:
    """Extend the preserved prompt builder using public state only."""

    errors = validate_benchmark_prompt_contract(prompt_contract)
    if errors:
        raise InteractionContractError(
            "invalid v0.5.4 prompt contract: " + "; ".join(errors)
        )
    messages = BASE_BUILD_TOOL_AGENT_MESSAGES(
        prompt_contract=prompt_contract,
        observation=observation,
        evidence_memory=evidence_memory,
        prior_public_attempts=prior_public_attempts,
        interaction_mode=interaction_mode,
    )
    system = json.loads(messages[0]["content"])
    user = json.loads(messages[1]["content"])
    system["command_expression_protocol"] = copy.deepcopy(
        prompt_contract["command_expression_protocol"]
    )
    system["recovery_protocol"] = copy.deepcopy(
        prompt_contract["recovery_protocol"]
    )
    system["composite_action_protocol"] = copy.deepcopy(
        prompt_contract["composite_action_protocol"]
    )
    system["cross_domain_non_gold_examples"] = copy.deepcopy(
        prompt_contract["cross_domain_non_gold_examples"]
    )

    last_outcome = observation.get("last_tool_outcome") or {}
    status = last_outcome.get("status")
    if status in NO_ADVANCE_OUTCOMES:
        user["recovery_guidance"] = {
            "triggered_by_public_outcome": status,
            "same_target_reformulation_limit": 1,
            "instruction": prompt_contract["recovery_protocol"][
                "instruction"
            ],
            "generic_dimensions": copy.deepcopy(
                prompt_contract["command_expression_protocol"][
                    "optional_clarifying_dimensions"
                ]
            ),
            "hidden_state_used": False,
            "gold_action_hint_used": False,
        }
    else:
        user["recovery_guidance"] = {
            "triggered_by_public_outcome": None,
            "instruction": (
                "Choose one evidence-supported action and state its complete "
                "executable meaning in args.query."
            ),
            "hidden_state_used": False,
            "gold_action_hint_used": False,
        }

    result = [
        {
            "role": messages[0]["role"],
            "content": json.dumps(
                system,
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
        {
            "role": messages[1]["role"],
            "content": json.dumps(
                user,
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]
    if credential_findings(result):
        raise InteractionContractError(
            "v0.5.4 Agent messages contain credential-like material"
        )
    return result


__all__ = [
    "AGENT_PROMPT_VERSION",
    "NO_ADVANCE_OUTCOMES",
    "build_benchmark_tool_agent_messages",
    "load_benchmark_prompt_contract",
    "validate_benchmark_prompt_contract",
]
