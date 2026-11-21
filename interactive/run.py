#!/usr/bin/env python3
"""Run or validate one deterministic clinical interaction episode."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from _common import DEFAULT_DATA_ROOT, load_case, runtime_paths, sandbox


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--case", required=True, help="Case ID, for example ward_001")
    parser.add_argument("--case-config", type=Path, help="Optional portable artifact-path configuration")
    parser.add_argument("--lexicon", type=Path, help="Optional clinical matching vocabulary asset")
    parser.add_argument("--output", type=Path, default=Path("runs/interaction"))
    parser.add_argument("--run-id", default="ward-interaction-run")
    parser.add_argument("--agent-id", default="ward-agent")
    parser.add_argument("--model", default=os.environ.get("WARD_AGENT_MODEL", ""))
    parser.add_argument("--protocol", choices=("native", "dispatcher"), default="native")
    parser.add_argument("--max-actions", type=int)
    parser.add_argument("--max-model-calls", type=int)
    parser.add_argument("--max-provider-attempts", type=int, default=3)
    parser.add_argument("--resume-from", type=Path, help="Replay a compatible interrupted prefix without repeating its model calls")
    parser.add_argument("--scripted-actions", type=Path, help="Offline list of canonical tool calls; intended for tests")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--ca-bundle", type=Path)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate artifacts and reset the environment without network calls")
    return parser.parse_args()


class ScriptedClient:
    """Deterministic local transport stub; never claims to be a model result."""
    def __init__(self, actions: list[dict]) -> None:
        self.actions = iter(actions)

    def complete(self, messages, *, tools=None):
        try:
            action = next(self.actions)
        except StopIteration:
            raise ConnectionError("scripted action list exhausted") from None
        if action == {"simulate_transport_error": True}:
            raise ConnectionError("scripted transport interruption")
        return {"tool_call": action, "content": "", "effective_model": "scripted-test",
                "response_mode": "native_tool_call", "usage": {}}


def execute(args: argparse.Namespace) -> dict:
    from runtime_assets import configure
    case = load_case(args.data_root, args.case, args.case_config)
    configure(case["root"], args.lexicon)
    # Imports follow configuration so the optional data vocabulary is applied.
    from world_model_patient_ehr_v04.agent_canary import make_live_client, readiness, transport_security_mode
    from world_model_patient_ehr_v054.prompting import load_benchmark_prompt_contract
    load_benchmark_prompt_contract(case["paths"]["prompt_contract"])
    max_actions = args.max_actions if args.max_actions is not None else case["max_actions"]
    max_calls = args.max_model_calls if args.max_model_calls is not None else case["max_model_calls"]
    if not 1 <= max_actions <= 200 or not 1 <= max_calls <= 400:
        raise ValueError("action budget must be 1-200 and model-call budget 1-400")
    if args.dry_run:
        env = sandbox(case, args.run_id, args.agent_id)
        observation = env.reset()
        return {"status": "validated", "case_id": args.case, "adapter": case["adapter"],
                "tool_count": len(observation["tool_interface"]["tool_call_schema"]["tool_values"]),
                "max_actions": max_actions,
                "max_model_calls": max_calls, "network_calls": 0}
    if args.scripted_actions:
        if args.protocol != "native":
            raise ValueError("scripted actions use the canonical native protocol")
        actions = json.loads(args.scripted_actions.read_text(encoding="utf-8"))
        if not isinstance(actions, list):
            raise ValueError("scripted actions must be a list")
        client = ScriptedClient(actions)
        model, transport = "scripted-test", "test_stub_no_network"
    else:
        model = args.model.strip()
        base_url = os.environ.get("WARD_API_BASE_URL", "").strip()
        api_key = os.environ.get("WARD_API_KEY", "")
        status = readiness(base_url=base_url, api_key=api_key, requested_model=model)
        if not status["ready"]:
            raise ValueError("set WARD_API_BASE_URL, WARD_API_KEY, and --model before a live run")
        client = make_live_client(base_url=base_url, api_key=api_key, model=model,
                                  timeout=args.timeout, ca_bundle=args.ca_bundle,
                                  interaction_mode="native_tool_call")
        transport = transport_security_mode(base_url)
        if args.protocol == "dispatcher":
            from dispatcher import KimiDispatcherClient
            client = KimiDispatcherClient(client)
    if args.resume_from:
        manifest = json.loads((args.resume_from / "run_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("case_id") != args.case:
            raise ValueError("continuation case ID mismatch")
        if manifest.get("run_summary", {}).get("terminal_reason") != "api_error":
            raise ValueError("continuation requires an API-interrupted source run")
        agent = manifest["agent"]
        if agent["requested_model"] != model:
            raise ValueError("continuation model mismatch")
        if (agent["max_actions"], agent["max_model_calls"]) != (max_actions, max_calls):
            raise ValueError("continuation must preserve configured action and call budgets")
        receipt_path = args.resume_from / "release_receipt.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["protocol"] != args.protocol:
                raise ValueError("continuation protocol mismatch")
        from eval import evaluate
        evaluate(args.resume_from, case)
    runner = importlib.import_module(f"world_model_patient_ehr_{case['adapter']}.agent_canary")
    result = runner.run_tool_agent_canary(
        client=client, **runtime_paths(case), output_dir=args.output,
        run_id=args.run_id, agent_id=args.agent_id, requested_model=model,
        transport_mode=transport, interaction_mode="native_tool_call",
        max_actions=max_actions, max_model_calls=max_calls,
        max_provider_attempts=args.max_provider_attempts, progress=args.progress,
        case_registry_path=None, resume_from_output_dir=args.resume_from,
    )
    receipt = {"release_version": "ward-interactive-anonymous-1.0", "case_id": args.case,
               "protocol": args.protocol, "adapter": case["adapter"],
               "scripted_test": args.scripted_actions is not None,
               "historical_run_reproduction_claimed": False,
               "continuation": args.resume_from is not None}
    (args.output / "release_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    args = arguments()
    try:
        result = execute(args)
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"Interaction run failed: {exc}") from None
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
