#!/usr/bin/env python3
"""Validate and score selected interaction runs without model or judge calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import DEFAULT_DATA_ROOT, load_case, runtime_paths, sandbox
from metrics import aggregate, derive_metrics


def evaluate(run_dir: Path, case: dict) -> dict:
    from world_model_patient_ehr_v01.contracts import read_json
    from world_model_patient_ehr_v04 import agent_canary
    from world_model_patient_ehr_v054.prompting import AGENT_PROMPT_VERSION, load_benchmark_prompt_contract
    manifest = read_json(run_dir / "run_manifest.json")
    load_benchmark_prompt_contract(case["paths"]["prompt_contract"])
    sandbox(case, "ward-evaluation-validation", "ward-evaluator").reset()
    versions = {f"ward-patient-ehr-tool-agent-runner-0.5.{i}" for i in range(7)}
    # Validation uses the same successor versions accepted by the runner.
    old_runners = set(agent_canary.SUPPORTED_AGENT_RUNNER_VERSIONS)
    old_prompts = set(agent_canary.SUPPORTED_AGENT_PROMPT_VERSIONS)
    try:
        agent_canary.SUPPORTED_AGENT_RUNNER_VERSIONS.update(versions)
        agent_canary.SUPPORTED_AGENT_PROMPT_VERSIONS.add(AGENT_PROMPT_VERSION)
        errors = agent_canary.validate_run_manifest(
            manifest, **runtime_paths(case),
            interaction_trace_path=run_dir / "interaction_trace.json",
            tool_trace_path=run_dir / "tool_trace.json",
            call_audit_path=run_dir / "agent_call_audit.json",
        )
    finally:
        agent_canary.SUPPORTED_AGENT_RUNNER_VERSIONS.clear()
        agent_canary.SUPPORTED_AGENT_RUNNER_VERSIONS.update(old_runners)
        agent_canary.SUPPORTED_AGENT_PROMPT_VERSIONS.clear()
        agent_canary.SUPPORTED_AGENT_PROMPT_VERSIONS.update(old_prompts)
    if errors:
        raise ValueError("run validation failed: " + "; ".join(errors))
    return derive_metrics(read_json(case["paths"]["scenario"]), manifest,
                          read_json(run_dir / "tool_trace.json"),
                          read_json(run_dir / "interaction_trace.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="Selected final run directory; repeat to aggregate")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--case-config", type=Path, help="Single-case fixture configuration")
    parser.add_argument("--lexicon", type=Path)
    parser.add_argument("--clinical-ratings", type=Path,
                        help="Independent CSV/JSON ratings: run_id or model_id+case_id, clinical_quality (1-7)")
    parser.add_argument("--output", type=Path, help="Optional JSON output; otherwise print only")
    args = parser.parse_args()
    try:
        from runtime_assets import configure
        rows = []
        for run_dir in args.run:
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            case = load_case(args.data_root, manifest["case_id"], args.case_config)
            configure(case["root"], args.lexicon)
            rows.append(evaluate(run_dir, case))
        if args.clinical_ratings:
            from ratings import apply_ratings
            apply_ratings(rows, args.clinical_ratings)
        result = {"schema_version": "ward-interaction-evaluation-1.0", "runs": rows,
                  "models": aggregate(rows), "network_calls": 0,
                  "clinical_ratings_computed": False,
                  "clinical_ratings_provided": args.clinical_ratings is not None}
        text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"Interaction evaluation failed: {exc}") from None


if __name__ == "__main__":
    main()
