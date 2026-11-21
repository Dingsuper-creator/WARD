"""Validate all case bindings and reference-graph reachability without models.

This privileged data check reads reference actions. It is not an evaluated agent,
not a public-tool policy, and does not establish clinical quality.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from _common import DEFAULT_DATA_ROOT, load_case, sandbox
from runtime_assets import configure


def validate_case(data_root: Path, case_id: str) -> dict:
    case = load_case(data_root, case_id)
    configure(case["root"])
    public_env = sandbox(case, "ward-binding-validation", "data-validator")
    initial = public_env.reset()
    from world_model_patient_ehr_v03.environment import PartialOrderPatientEHREnvironment
    paths = {name + "_path": case["paths"][name] for name in ("scenario", "freeze_manifest", "interface", "ontology", "graph")}
    env = PartialOrderPatientEHREnvironment(**paths, run_id="ward-reference-graph", agent_id="reference-validator")
    observation = env.reset()
    scenario = json.loads(case["paths"]["scenario"].read_text(encoding="utf-8"))
    graph = json.loads(case["paths"]["graph"].read_text(encoding="utf-8"))
    interface = json.loads(case["paths"]["interface"].read_text(encoding="utf-8"))
    ontology = json.loads(case["paths"]["ontology"].read_text(encoding="utf-8"))
    completed, released = set(), set()
    for transition in sorted(scenario["transitions"], key=lambda row: row["sequence_index"]):
        transition_id = transition["transition_id"]
        if not set(graph["nodes"][transition_id]["requires"]).issubset(completed):
            raise ValueError(f"reference order violates prerequisites: {case_id}/{transition_id}")
        concept = ontology["concepts"][interface["transition_concepts"][transition_id]]
        action = {"action_type": transition["gold_action"]["action_type"],
                  "target": concept["phrases"][0], "parameters": {}, "evidence_ids": []}
        observation, assessment, done = env.step(action)
        matched = assessment["matched_transition_id"]
        if not assessment["state_advanced"] or matched != transition_id:
            raise ValueError(f"reference concept does not reach its node: {case_id}/{transition_id}")
        actual_release = set(assessment["released_evidence_ids"])
        if actual_release != set(transition["release_evidence_ids"]):
            raise ValueError(f"reference release mismatch: {case_id}/{transition_id}")
        if released & actual_release:
            raise ValueError(f"duplicate evidence release: {case_id}/{transition_id}")
        released.update(actual_release)
        completed.add(transition_id)
    if completed != set(graph["nodes"]) or not observation["done"]:
        raise ValueError(f"reference graph did not complete: {case_id}")
    expected_visible = set(scenario["initial_state"]["visible_evidence_ids"]) | released
    if set(observation["visible_evidence_ids"]) != expected_visible:
        raise ValueError(f"final evidence visibility mismatch: {case_id}")
    return {"case_id": case_id, "public_tool_count": len(initial["tool_interface"]["tool_call_schema"]["tool_values"]),
            "reference_transitions_reached": len(completed), "reference_evidence_released": len(released),
            "evidence_store_items": len(scenario["evidence_store"]), "reference_graph_terminal": True,
            "validation_level": "input bindings and privileged graph execution; not public-tool reference policy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    index = json.loads((args.data_root / "cases.json").read_text(encoding="utf-8"))
    rows = [validate_case(args.data_root, item["case_id"]) for item in index["cases"]]
    print(json.dumps({"cases": rows, "case_count": len(rows),
                      "transitions_reached": sum(row["reference_transitions_reached"] for row in rows),
                      "evidence_items": sum(row["evidence_store_items"] for row in rows),
                      "network_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
