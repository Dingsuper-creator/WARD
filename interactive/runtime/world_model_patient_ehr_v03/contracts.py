"""Contracts for Patient/EHR partial-order action graphs v0.3."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from world_model_patient_ehr_v01.contracts import (
    InteractionContractError,
    credential_findings,
    read_json,
    sha256_file,
)


GRAPH_VERSION = "ward-patient-ehr-action-graph-0.3.0"
ENVIRONMENT_VERSION = "ward-patient-ehr-environment-0.3.0"
TRACE_VERSION = "ward-patient-ehr-interaction-trace-0.3.0"
RUN_MANIFEST_VERSION = "ward-patient-ehr-run-manifest-0.3.0"


def validate_action_graph(
    graph: dict[str, Any],
    *,
    transition_ids: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if graph.get("graph_version") != GRAPH_VERSION:
        errors.append("wrong_graph_version")
    if credential_findings(graph):
        errors.append("credential_like_material_present")
    if not isinstance(graph.get("case_id"), str) or not graph["case_id"]:
        errors.append("missing_case_id")
    for key in (
        "scenario_sha256",
        "action_interface_sha256",
        "action_ontology_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(graph.get(key, ""))):
            errors.append(f"invalid_{key}")
    nodes = graph.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        return [*errors, "nodes_must_be_nonempty_object"]
    if transition_ids is not None and set(nodes) != transition_ids:
        errors.append("graph_nodes_do_not_match_scenario")
    for node_id, node in nodes.items():
        if not isinstance(node, dict):
            errors.append(f"invalid_node:{node_id}")
            continue
        requires = node.get("requires")
        if (
            not isinstance(requires, list)
            or any(not isinstance(item, str) or not item for item in requires)
            or len(requires) != len(set(requires))
        ):
            errors.append(f"invalid_requires:{node_id}")
            continue
        for required in requires:
            if required not in nodes:
                errors.append(f"unknown_dependency:{node_id}:{required}")
            if required == node_id:
                errors.append(f"self_dependency:{node_id}")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            errors.append("dependency_cycle")
            return
        if node_id in visited:
            return
        visiting.add(node_id)
        node = nodes.get(node_id)
        if isinstance(node, dict):
            for required in node.get("requires", []):
                if required in nodes:
                    visit(required)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in nodes:
        visit(node_id)
    if graph.get("gold_action_visible") is not False:
        errors.append("gold_visibility_boundary_invalid")
    if graph.get("future_evidence_visible") is not False:
        errors.append("future_evidence_boundary_invalid")
    if graph.get("official_scoring_allowed") is not False:
        errors.append("official_scoring_boundary_invalid")
    if graph.get("single_total_score_absent") is not True:
        errors.append("single_total_score_boundary_invalid")
    return sorted(set(errors))


def load_bound_action_graph(
    *,
    graph_path: Path,
    scenario_path: Path,
    interface_path: Path,
    ontology_path: Path,
) -> tuple[dict[str, Any], str]:
    graph = read_json(graph_path)
    scenario = read_json(scenario_path)
    transition_ids = {
        transition["transition_id"] for transition in scenario["transitions"]
    }
    errors = validate_action_graph(graph, transition_ids=transition_ids)
    if errors:
        raise InteractionContractError(
            "invalid action graph: " + "; ".join(errors)
        )
    expected = {
        "scenario_sha256": sha256_file(scenario_path),
        "action_interface_sha256": sha256_file(interface_path),
        "action_ontology_sha256": sha256_file(ontology_path),
    }
    for key, value in expected.items():
        if graph.get(key) != value:
            raise InteractionContractError(f"action graph {key} binding mismatch")
    if graph["case_id"] != scenario.get("case_id"):
        raise InteractionContractError("action graph case_id mismatch")
    return graph, sha256_file(graph_path)

