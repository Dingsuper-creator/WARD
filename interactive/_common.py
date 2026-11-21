"""Shared portable loading for the interactive command-line tools."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))
DEFAULT_DATA_ROOT = HERE.parent / "data" / "interactive"


def load_case(data_root: Path, case_id: str, config: Path | None = None) -> dict:
    data_root = data_root.resolve()
    if not re.fullmatch(r"ward_\d{3}", case_id):
        raise ValueError("case ID must have the form ward_001")
    if config is not None:
        settings = json.loads(config.read_text(encoding="utf-8"))
        root = (config.resolve().parent / settings.get("data_root", ".")).resolve()
        if settings.get("case_id", case_id) != case_id:
            raise ValueError("case configuration ID mismatch")
        raw_paths = settings.get("artifacts", settings)
    elif (data_root / "cases.json").is_file():
        index = json.loads((data_root / "cases.json").read_text(encoding="utf-8"))
        if index.get("schema_version") != "ward-interactive-index-1.0":
            raise ValueError("unsupported interactive case index")
        matches = [item for item in index.get("cases", []) if item.get("case_id") == case_id]
        if len(matches) != 1:
            raise ValueError("case index must contain exactly one matching case")
        settings = matches[0]
        raw_paths = settings.get("artifacts", settings)
        root = data_root
    else:
        settings = {}
        root = data_root
        version = "0.5.5" if case_id == "ward_007" else "0.5.6"
        base = "outputs/world_model_patient_ehr_"
        ontology = f"{base}v02/action_ontology.{case_id}.v0.2.json"
        if not (root / ontology).is_file():
            ontology = f"{base}v02/action_ontology.v0.2.json"
        raw_paths = {
            "scenario": f"{base}v01/cases/{case_id}.patient_ehr.json",
            "freeze_manifest": "outputs/world_model_interaction_scorer_freeze_v01/FREEZE_MANIFEST.json",
            "interface": f"{base}v02/cases/{case_id}.action_interface.v0.2.json",
            "ontology": ontology,
            "graph": f"{base}v03/cases/{case_id}.action_graph.v0.3.json",
            "tool_map": f"{base}v05/cases/{case_id}.tool_map.v{version}.json",
            "prompt_contract": f"{base}v05/PROMPT_CONTRACT.v0.5.4.json",
        }
    names = ("scenario", "freeze_manifest", "interface", "ontology", "graph", "tool_map", "prompt_contract")
    paths = {name: (root / raw_paths[name]).resolve() for name in names}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"missing {name} artifact: {path}")
    scenario = json.loads(paths["scenario"].read_text(encoding="utf-8"))
    if scenario.get("case_id") != case_id:
        raise ValueError("scenario case ID mismatch")
    adapter = settings.get("adapter", "v055" if case_id == "ward_007" else "v056")
    if adapter not in ("v055", "v056"):
        raise ValueError("adapter must be v055 or v056")
    return {"paths": paths, "root": root, "adapter": adapter,
            "max_actions": settings.get("max_actions", 20 if case_id == "ward_007" else 40),
            "max_model_calls": settings.get("max_model_calls", 24 if case_id == "ward_007" else 48)}


def runtime_paths(case: dict) -> dict:
    return {name + "_path": path for name, path in case["paths"].items()}


def sandbox(case: dict, run_id: str, agent_id: str):
    if case["adapter"] == "v055":
        from world_model_patient_ehr_v055.tool_facade import EventPhaseClinicalToolSandbox as Sandbox
    else:
        from world_model_patient_ehr_v056.tool_facade import HybridClinicalToolSandbox as Sandbox
    paths = runtime_paths(case)
    paths.pop("prompt_contract_path")
    return Sandbox(**paths, run_id=run_id, agent_id=agent_id)
