"""Build a two-transition synthetic episode with no patient-derived content."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

TEMPLATES = Path(__file__).resolve().parent / "templates"
BOUNDARIES = {"gold_action_visible": False, "dependency_graph_visible": False,
              "future_evidence_visible": False, "automatic_semantic_judge_scoring": False,
              "official_scoring_allowed": False, "single_total_score_absent": True}


def build_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        path = root / (name + ".json")
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = {}
    for name in ("lifecycle", "phase", "prompt"):
        hashes[name] = save(name, json.loads((TEMPLATES / (name + ".json")).read_text()))
    freeze = {"freeze_id": "ward-synthetic-contract-1.0", "official_scoring_allowed": False}
    hashes["freeze_manifest"] = save("freeze_manifest", freeze)
    transitions = []
    for index, (name, action_type, terminal) in enumerate([
        ("history", "request_patient_information", False), ("discharge", "discharge", True)
    ]):
        transitions.append({"transition_id": f"ward_001_{name}", "sequence_index": index,
                            "stage_before": "S1", "stage_after": "S3" if terminal else "S1",
                            "clock_before": None, "clock_after": None,
                            "gold_action": {"action_id": f"action_{index}", "action_type": action_type,
                                            "target": f"synthetic {name}", "parameters": {},
                                            "accepted_target_aliases": [], "accepted_action_type_aliases": [],
                                            "gold_source_type": "ehr_derived_terminal_gold" if terminal else "patient_question_exception"},
                            "release_evidence_ids": [f"evidence_{index}"], "state_updates": {}, "terminal": terminal})
    scenario = {"scenario_version": "ward-patient-ehr-case-0.1.0", "case_id": "ward_001",
                "scorer_freeze": {"freeze_id": freeze["freeze_id"], "manifest_sha256": hashes["freeze_manifest"]},
                "public_context": {"summary": "Synthetic workflow for software verification; no patient data."},
                "initial_state": {"stage": "S1", "clock": None, "visible_evidence_ids": [], "state_facts": {}},
                "evidence_store": {"evidence_0": {"evidence_id": "evidence_0", "content": "Synthetic history reviewed."},
                                   "evidence_1": {"evidence_id": "evidence_1", "content": "Synthetic episode completed."}},
                "transitions": transitions, **BOUNDARIES}
    hashes["scenario"] = save("scenario", scenario)
    concepts = {}
    for transition in transitions:
        gold = transition["gold_action"]
        concepts[gold["target"]] = {"action_types": [gold["action_type"]], "phrases": [gold["target"]],
                                    "anchor_groups": [[gold["target"]]], "minimum_anchor_groups": 1,
                                    "similarity_threshold": 1.0}
    hashes["ontology"] = save("ontology", {"ontology_version": "ward-patient-ehr-action-ontology-0.2.0", "concepts": concepts})
    hashes["interface"] = save("interface", {"interface_version": "ward-patient-ehr-action-interface-0.2.0",
        "case_id": "ward_001", "scenario_sha256": hashes["scenario"], "ontology_sha256": hashes["ontology"],
        "public_capabilities": ["Request history and perform recorded care actions."],
        "transition_concepts": {t["transition_id"]: t["gold_action"]["target"] for t in transitions}, **BOUNDARIES})
    bindings = {"scenario_sha256": hashes["scenario"], "action_interface_sha256": hashes["interface"], "action_ontology_sha256": hashes["ontology"]}
    hashes["graph"] = save("graph", {"graph_version": "ward-patient-ehr-action-graph-0.3.0", "case_id": "ward_001",
        "nodes": {"ward_001_history": {"requires": []}, "ward_001_discharge": {"requires": ["ward_001_history"]}},
        **bindings, **BOUNDARIES})
    bindings.update({"action_graph_sha256": hashes["graph"], "freeze_manifest_sha256": hashes["freeze_manifest"],
                     "lifecycle_policy_sha256": hashes["lifecycle"], "event_relative_phase_policy_sha256": hashes["phase"]})
    routes = {}
    for t in transitions:
        routes[t["transition_id"]] = {"action_types": [t["gold_action"]["action_type"]],
            "concept_ids": [t["gold_action"]["target"]], "features": [], "minimum_feature_matches": 0,
            "concept_match_preferred": True, "feature_fallback_allowed": False}
    hashes["routes"] = save("routes", {"route_contract_version": "ward-patient-ehr-state-route-contract-0.5.6",
        "artifact_revision": "0.5.6", "case_id": "ward_001", "routes": routes, **bindings, **BOUNDARIES})
    versions = {"tool_map_version": "ward-patient-ehr-tool-map-0.5.6",
        "state_based_adapter_version": "ward-patient-ehr-reviewed-ontology-adapter-0.5.6",
        "action_envelope_version": "ward-patient-ehr-role-separated-action-envelope-0.1.0",
        "event_frame_version": "ward-patient-ehr-clinical-event-frame-0.1.0",
        "lifecycle_policy_version": "ward-patient-ehr-clinical-event-lifecycle-policy-0.1.0",
        "event_phase_frame_version": "ward-patient-ehr-event-phase-frame-0.1.0",
        "event_relative_phase_policy_version": "ward-patient-ehr-event-relative-phase-policy-0.1.0"}
    defaults = {"patient.ask": "request_patient_information", "ehr.search": "review_document",
                "ehr.read": "review_document", "ehr.order": "order_imaging", "procedure.perform": "perform_procedure",
                "treatment.start": "start_treatment", "care.transition": "monitor_or_defer"}
    save("tool_map", {**versions, **bindings, **BOUNDARIES, "case_id": "ward_001",
        "public_tools": list(defaults), "public_ehr_modules": ["history", "discharge_documents"],
        "tool_action_defaults": defaults, "module_action_overrides": {"discharge_documents": {"care.transition": "discharge"}},
        "tool_design": {"coarse_available_action_families_visible": False},
        "state_routes_path": "routes.json", "state_routes_sha256": hashes["routes"],
        "lifecycle_policy_path": "lifecycle.json", "event_relative_phase_policy_path": "phase.json"})
    config = {"case_id": "ward_001", "adapter": "v056", "max_actions": 4, "max_model_calls": 6,
              "artifacts": {key: key + ".json" for key in ("scenario", "freeze_manifest", "interface", "ontology", "graph", "tool_map")}}
    config["artifacts"]["prompt_contract"] = "prompt.json"
    save("case", config)
    history = {"tool": "patient.ask", "args": {"query": "synthetic history", "module": "history"}}
    discharge = {"tool": "care.transition", "args": {"query": "synthetic discharge", "module": "discharge_documents"}}
    save("actions", [discharge, history, discharge])
    save("interrupted_actions", [history, {"simulate_transport_error": True}])
    save("continuation_actions", [discharge])
    return root / "case.json"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(build_fixture(args.output))
