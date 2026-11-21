"""Portable command-line entry points for the benchmark execution boundary."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from . import __version__
from .contracts import read_submission_jsonl
from .gold import GoldPackage
from .judge import FrozenJudge
from .prompt_audit import audit_prompt_manifests
from .runner import prepare_model_preflight, run_model_preflight
from .scorer import score_submissions
from .temporal import evaluate_temporal
from .util import read_json, read_jsonl, sha256_file, write_json, write_jsonl


def run_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Checkpoint runner and semantic judge workflow.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Compile independently elicited checkpoint requests.")
    prepare.add_argument("--gold", required=True, type=Path)
    prepare.add_argument("--roster", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    execute = commands.add_parser("execute", help="Execute a prepared job using environment credentials.")
    execute.add_argument("--job", required=True, type=Path)
    execute.add_argument("--roster", required=True, type=Path)
    execute.add_argument("--output", required=True, type=Path)
    judge_prepare = commands.add_parser("judge-prepare", help="Prepare unresolved semantic pairs offline.")
    judge_prepare.add_argument("--gold", required=True, type=Path)
    judge_prepare.add_argument("--submissions", required=True, type=Path)
    judge_prepare.add_argument("--output", required=True, type=Path)
    judge_run = commands.add_parser("judge-execute", help="Run a JSON stdin/stdout judge adapter.")
    judge_run.add_argument("--job", required=True, type=Path)
    judge_run.add_argument("--output", required=True, type=Path)
    judge_run.add_argument("--judge-id", required=True, choices=["judge_a", "judge_b"])
    judge_run.add_argument("--adapter", required=True, type=Path)
    judge_run.add_argument("--adapter-arg", action="append", default=[])
    judge_run.add_argument("--model-manifest", required=True, type=Path)
    judge_run.add_argument("--timeout", type=int, default=600)
    judge_run.add_argument("--concurrency", type=int, default=1)
    finalize = commands.add_parser("judge-finalize", help="Build a semantic decision cache for scoring.")
    for name in ("job", "judge-a", "judge-b", "output"):
        finalize.add_argument(f"--{name}", required=True, type=Path)
    finalize.add_argument("--human-resolutions", type=Path)
    finalize.add_argument("--decision-policy", choices=["dual_consensus_human", "primary_judge_b"], default="primary_judge_b")
    args = parser.parse_args(argv)
    if args.command in {"prepare", "execute"}:
        roster = read_json(args.roster)
        if len(roster.get("models", [])) != 1:
            parser.error("Use one model per job so each output is one complete checkpoint submission set.")
        if args.command == "prepare":
            result = prepare_model_preflight(GoldPackage(args.gold), args.roster, args.output)
        else:
            result = run_model_preflight(args.job, args.roster, args.output)
    else:
        from .judge_jobs import prepare_judge_job, run_judge_adapter, finalize_judge_job
        if args.command == "judge-prepare":
            result = prepare_judge_job(GoldPackage(args.gold), args.submissions, args.output)
        elif args.command == "judge-execute":
            result = run_judge_adapter(args.job, args.output, judge_id=args.judge_id,
                adapter=args.adapter, adapter_args=args.adapter_arg,
                model_manifest_path=args.model_manifest, timeout_seconds=args.timeout,
                max_concurrency=args.concurrency)
        else:
            result = finalize_judge_job(args.job, args.judge_a, args.judge_b, args.output,
                human_resolutions=args.human_resolutions, decision_policy=args.decision_policy)
    print(json.dumps({"command": args.command, "status": result.get("status", "complete"), "output": str(args.output)}))
    return 2 if result.get("status") in {"blocked", "failed", "incomplete"} else 0


def eval_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline seven-dimension and six-metric longitudinal evaluation.")
    parser.add_argument("--gold", required=True, type=Path, help="Directory containing cases/ward_*.gold.json.")
    parser.add_argument("--submissions", required=True, type=Path)
    parser.add_argument("--judge-cache", type=Path)
    parser.add_argument("--prompt-manifests", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", default="model")
    parser.add_argument("--expected-cases", type=int, default=40)
    parser.add_argument("--allow-cache-misses", action="store_true", help="Write explicitly provisional scores when semantic decisions are missing.")
    args = parser.parse_args(argv)
    gold = GoldPackage(args.gold)
    if len(gold.cases) != args.expected_cases:
        parser.error(f"Expected {args.expected_cases} cases, found {len(gold.cases)}. Set --expected-cases explicitly for a synthetic/subset run.")
    text_audit = gold.text_audit()
    if not text_audit["formal_text_ready"]:
        parser.error("Gold has unscoreable text fields; repair the reference before scoring.")
    records, input_errors = read_submission_jsonl(args.submissions)
    judge = FrozenJudge(args.judge_cache)
    result = score_submissions(gold, records, judge=judge,
        expected_case_count=args.expected_cases, input_errors=input_errors)
    temporal_judge = FrozenJudge(args.judge_cache)
    temporal = evaluate_temporal(gold, records, result["checkpoints"], judge=temporal_judge, model_id=args.model_id)
    cache_misses = len({row["cache_key"] for row in [*judge.trace, *temporal_judge.trace] if row["source"] == "cache_miss"})
    prompt_audit = None
    if args.prompt_manifests:
        prompt_audit = audit_prompt_manifests(read_jsonl(args.prompt_manifests), gold)
    blockers = []
    if not result["run_structure_valid"]:
        blockers.append("invalid_run_structure")
    if cache_misses:
        blockers.append("semantic_cache_misses")
    if prompt_audit is not None and prompt_audit.get("status") != "passed":
        blockers.append("prompt_audit_failed")
    result["formal_result"] = not blockers and prompt_audit is not None
    result["release_evaluation"] = {
        "status": "provisional" if blockers else "complete",
        "implementation_version": __version__,
        "blockers": blockers, "cache_miss_count": cache_misses,
        "prompt_audit_performed": prompt_audit is not None,
        "temporal_judge_decision_count": len(temporal_judge.trace),
        "cache_miss_count_policy": "unique_cache_keys_across_checkpoint_and_temporal_evaluation",
        "submissions_sha256": sha256_file(args.submissions),
        "judge_cache_sha256": sha256_file(args.judge_cache) if args.judge_cache else None,
        "independent_checkpoint_predictions": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "eval_result.json", result)
    write_jsonl(args.output / "checkpoint_scores.jsonl", result["checkpoints"])
    write_jsonl(args.output / "case_scores.jsonl", result["aggregate"]["cases"])
    write_json(args.output / "temporal_metrics.json", temporal)
    write_jsonl(args.output / "temporal_case_metrics.jsonl", temporal["case_metrics"])
    if prompt_audit is not None:
        write_json(args.output / "prompt_audit.json", prompt_audit)
    print(json.dumps({"status": result["release_evaluation"]["status"],
        "case_count": len(gold.cases), "checkpoint_count": len(result["checkpoints"]),
        "valid_checkpoint_count": sum(row["valid_submission"] for row in result["checkpoints"]),
        "semantic_cache_miss_count": cache_misses, "output": str(args.output)}))
    fatal = set(blockers) - ({"semantic_cache_misses"} if args.allow_cache_misses else set())
    return 2 if fatal else 0
