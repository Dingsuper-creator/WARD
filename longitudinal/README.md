# Longitudinal benchmark

Python 3.11 or newer and `jsonschema>=4.23,<5` are required. The commands below run from the benchmark root. Both Python entry scripts also work from any working directory when given explicit input paths. Installing this component with `python -m pip install ./longitudinal` is optional; its wheel includes the schemas and scoring contracts.

Each checkpoint is elicited independently from cumulative visible evidence. The preparation step reads the reference package to compile visible requests. The execution step reads only the prepared job and provider configuration. Keep reference answers separate from the model execution environment.

## Checkpoint inference

Edit `longitudinal/config/model_roster.example.json`: set the model identifier, protocol, token budget, and sampling parameters. Use one model per job. Set the environment variables named by `gateway.base_url_env` and `gateway.api_key_env`. The base URL is the provider origin; the runner appends `/v1/chat/completions` or `/v1/messages`. A model-level `endpoint_url` can supply an exact Chat Completions URL.

```bash
python longitudinal/run.py prepare --gold data/longitudinal --roster longitudinal/config/model_roster.example.json --output output/checkpoint-job
python longitudinal/run.py execute --job output/checkpoint-job --roster longitudinal/config/model_roster.example.json --output output/checkpoint-run
```

The prepared job contains `requests.jsonl` and `prompt_request_manifests.jsonl`. Execution writes `raw_model_submissions.jsonl`, provider attempts, outcomes, and a manifest. Reusing the same execution output directory resumes completed requests. Invalid outputs retain explicit statuses; transport errors are recorded separately from model-format failures.

The roster supports `openai_chat` and `anthropic_messages`. Markdown-fence normalization is disabled in the example; enable `strip_outer_json_fence` only when it is part of the declared protocol. Provider error codes are used for retry decisions; a model-level `non_retryable_messages` list can add deployment-specific message compatibility without embedding local provider text in source code.

## Semantic judging

Use **DeepSeek-V4-Pro (Judge-B)** as the primary semantic evaluator and **Qwen3.7-Max (Judge-A)** for sensitivity analysis. Exact and registered-alias matches are deterministic; the judges resolve remaining clinical text pairs.

Set the deployment IDs and endpoints in `longitudinal/config/judge_a.example.json` and `judge_b.example.json`, and export `WARD_JUDGE_A_API_KEY` and `WARD_JUDGE_B_API_KEY`. The bundled adapter calls these models to create a reusable semantic cache.

<details>
<summary>Commands for generating the semantic cache</summary>

```bash
python longitudinal/run.py judge-prepare --gold data/longitudinal --submissions output/checkpoint-run/raw_model_submissions.jsonl --output output/judge-job
python longitudinal/run.py judge-execute --job output/judge-job --judge-id judge_a --adapter python --adapter-arg longitudinal/adapters/chat_completions.py --model-manifest longitudinal/config/judge_a.example.json --output output/judge-a
python longitudinal/run.py judge-execute --job output/judge-job --judge-id judge_b --adapter python --adapter-arg longitudinal/adapters/chat_completions.py --model-manifest longitudinal/config/judge_b.example.json --output output/judge-b
python longitudinal/run.py judge-finalize --job output/judge-job --judge-a output/judge-a --judge-b output/judge-b --decision-policy primary_judge_b --output output/judge-finalization
```

The resulting cache is `output/judge-finalization/private/judge_cache.v1.json`. If the command reports unresolved items, follow its review instructions before evaluating. Generate a new cache when the predictions or reference version changes.

When all pairs match deterministically, skip remote judging and omit `--judge-cache` during evaluation.

</details>

## Offline evaluation

```bash
python longitudinal/eval.py --gold data/longitudinal --submissions output/checkpoint-run/raw_model_submissions.jsonl --judge-cache output/judge-finalization/private/judge_cache.v1.json --prompt-manifests output/checkpoint-job/prompt_request_manifests.jsonl --model-id your-model-id --output output/checkpoint-scores
```

Outputs:

- `eval_result.json`: seven F1 dimensions, stopping accuracy, critical-action recall, validation, and semantic decision provenance.
- `checkpoint_scores.jsonl` and `case_scores.jsonl`: auditable counts before case aggregation.
- `temporal_metrics.json` and `temporal_case_metrics.jsonl`: six longitudinal metrics with case-level numerators, denominators, and contributing-case counts.
- `prompt_audit.json` when prompt manifests are supplied.

Seven F1 dimensions are Evidence, Utilization, Claims, Patient State, Actions, Safety, and Stopping blockers. Counts are pooled within each case before case F1 values are macro-averaged. The default expected cohort is 40 cases; a synthetic or declared subset run must set `--expected-cases` explicitly.

The six temporal diagnostics are state-delta immediate response, linked-evidence nonresponse, stopping-change response, evidence subsequent match, evidence regression, and stopping-decision regression. Invalid checkpoint outputs count as empty predictions for checkpoint scoring and are unscorable for temporal transitions. Cases with zero denominators are excluded from the corresponding temporal metric. These transitions compare independently elicited predictions and do not imply an agent's autonomous correction.

Missing semantic cache entries produce provisional output and exit status 2. `--allow-cache-misses` permits diagnostic execution with status 0 but never changes the provisional status or makes the result formal. Structural or prompt-audit failures remain blocking. `formal_result` is true only when the input structure and supplied prompt audit pass and all requested semantic pairs have decisions. No composite total score is produced.

## Verification

```bash
PYTHONPATH=longitudinal/src python -m unittest discover -s longitudinal/tests -v
```

Tests use synthetic records, mocked transports, and frozen decisions. They do not call provider APIs. `tests/synthetic.py` is the reference generator for the small synthetic input contract. The historical case-specific polarity corrections remain in `gold.py`; they are part of scoring semantics and are independent of source identity.
