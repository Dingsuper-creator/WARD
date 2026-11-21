# WARD

WARD is a benchmark for evaluating clinical agents under accumulating evidence and sequential clinical actions.

![Motivation for evaluating evidence use, state updates and clinical actions](figs/fig-intro.png)


## Benchmark overview

The Longitudinal Track gives each checkpoint an independent model response based on the cumulative visible record. The Interactive Track lets an agent acquire evidence and advance the case through sequential tool calls. 
![WARD construction and the Longitudinal and Interactive evaluation tracks](figs/fig-frame.png)

 Benchmark construction, checkpoint assessment and tool-mediated clinical execution.

## Repository structure

```text
WARD/
├── README.md
├── requirements.txt              
├── data/
│   ├── longitudinal/       # checkpoint references and visible-record source
│   ├── interactive/        # scenarios, tools, transitions and case configurations
│   └── prepare.py          # Prepare a separate runtime copy from an authorized archive
├── longitudinal/
│   ├── run.py              # Target-model and semantic-judge execution
│   ├── eval.py             # Checkpoint scores and temporal diagnostics
│   ├── config/             # Model configuration example
│   └── src/ward_eval/      # Evaluation implementation and packaged schemas
└── interactive/
    ├── run.py              # Native tools, dispatcher and interrupted-run continuation
    ├── eval.py             # Trace validation and metric aggregation
    └── runtime/            # Tool interfaces and deterministic environment
```

The reference answers, unreleased evidence and hidden transition graph are evaluator-side data. The runners construct the model-visible inputs. Do not send complete reference files to a target model.

### Data example

The following synthetic example illustrates the model-visible checkpoint input. Field names follow the runner's payload; the clinical text is illustrative. The runner includes only evidence available at `public_time`.

```json
{
  "case_id": "ward_001",
  "checkpoint_id": "ward_001_cp_002",
  "public_time": "2125-01-02T09:00:00",
  "public_context": {
    "age_years": 62,
    "sex": "female",
    "setting": "inpatient",
    "initial_time": "2125-01-01T08:00:00",
    "admission_diagnosis": "Dyspnea under investigation"
  },
  "visible_evidence": [
    {
      "evidence_id": "doc:001",
      "record_kind": "document",
      "title": "Chest imaging report",
      "clinical_time": "2125-01-02T08:00:00",
      "available_time": "2125-01-02T08:30:00",
      "content": "A new pulmonary infiltrate is present."
    }
  ]
}
```

The model returns structured evidence use, claims, patient state, actions, safety constraints and a stopping decision. In the Interactive Track, the same kinds of records are exposed through tools as the case progresses. An illustrative action in the runner's canonical trace format is:

```json
{
  "tool": "ehr.read",
  "intent": "Review the available chest imaging report",
  "args": {
    "module": "imaging_results",
    "query": "Review the current chest imaging report"
  },
  "rationale": "Clarify the newly available imaging findings.",
  "evidence_ids": ["doc:001"]
}
```

Reference answers and hidden transition conditions remain evaluator-side. See the [data instructions](data/README.md) for the stored files and complete format details.

## Setup

Use Python 3.11 or newer. Run the examples below from the repository root.

```bash
python -m pip install -r requirements.txt
```

For model execution, configure an OpenAI-compatible endpoint through environment variables:

```bash
export WARD_API_KEY="YOUR_API_KEY"
export WARD_BASE_URL="https://api.example.com"
export WARD_API_BASE_URL="https://api.example.com/v1"
```

Credentials are not part of the repository. Replace the example endpoint and model name with the service you intend to evaluate. The checkpoint runner takes the provider origin and appends `/v1/chat/completions`; the interactive runner takes the API base ending in `/v1`.

## Longitudinal track

Each model response is a JSON checkpoint submission containing evidence, evidence-utilization links, claims, patient state, actions, safety constraints and stopping decisions. The executable format is in `longitudinal/src/ward_eval/schemas/checkpoint_submission.v1.schema.json`.

Copy `longitudinal/config/model_roster.example.json` to `models.local.json` and set `models[0].model_id` and the inference parameters. Use one model per job.

### Run a model

```bash
python longitudinal/run.py prepare \
  --gold data/longitudinal \
  --roster models.local.json \
  --output output/checkpoint-job

python longitudinal/run.py execute \
  --job output/checkpoint-job \
  --roster models.local.json \
  --output output/checkpoint-run
```

The runner writes `raw_model_submissions.jsonl`, request and attempt records, and a run manifest. Checkpoints are elicited independently; the model does not receive its predictions from earlier checkpoints.

### Evaluate predictions

Use **DeepSeek-V4-Pro (Judge-B)** as the primary semantic evaluator and **Qwen3.7-Max (Judge-A)** for sensitivity analysis, as in the paper. Set their deployment IDs and endpoints in `longitudinal/config/judge_b.example.json` and `judge_a.example.json`; [cache-generation commands](longitudinal/README.md#semantic-judging) are provided in the track instructions.

Evaluate predictions with the resulting semantic cache:

```bash
python longitudinal/eval.py \
  --gold data/longitudinal \
  --submissions output/checkpoint-run/raw_model_submissions.jsonl \
  --judge-cache output/judge-finalization/private/judge_cache.v1.json \
  --prompt-manifests output/checkpoint-job/prompt_request_manifests.jsonl \
  --model-id your-model-id \
  --output output/checkpoint-scores
```

The evaluator reports:

- Evidence, Utilization, Claims, Patient State, Actions, Safety and Stopping-blocker F1; stopping-decision accuracy and critical-action recall.
- State-delta immediate response, linked-evidence nonresponse, stopping-change response, evidence subsequent match, evidence regression and stopping-decision regression.

Checkpoint counts are pooled within each case before computing F1, then averaged across cases. Temporal metrics exclude cases without eligible opportunities and report their denominators.

Outputs include `eval_result.json`, `checkpoint_scores.jsonl`, `case_scores.jsonl`, `temporal_metrics.json` and `temporal_case_metrics.jsonl`.

## Interactive track

The environment exposes seven tools: `patient.ask`, `ehr.search`, `ehr.read`, `ehr.order`, `procedure.perform`, `treatment.start` and `care.transition`. It checks prerequisites, releases evidence and advances the recorded clinical state.

### Validate and run a case

```bash
python interactive/run.py --case ward_001 --dry-run

python interactive/run.py \
  --case ward_001 \
  --model your-model-id \
  --output output/interactive/your-model-id/ward_001
```

`--dry-run` validates the assets and resets the environment without network calls. For a provider requiring a single tool dispatcher, add `--protocol dispatcher`. An API-interrupted run can be continued with `--resume-from PATH`; the model, case, protocol and budgets must remain compatible.

Action and model-call budgets are recorded per case. In the supplied subset, `ward_007` has a 20-action budget and the other nine cases have 40. Use the same case configurations when comparing models. A completed process or exhausted budget does not imply clinical task completion.

The [Interactive instructions](interactive/README.md) describe transport configuration, continuation, artifacts and offline graph validation.

### Evaluate a run

```bash
python interactive/eval.py \
  --run output/interactive/your-model-id/ward_001 \
  --output output/interactive/scores.json
```

Repeat `--run` to aggregate selected final runs. Do not count an interrupted prefix and its continuation as separate cases.

The evaluator validates artifact hashes and computes milestone coverage, productive budget yield, reference-evidence recall, no-new-evidence repeat rate, longest nonproductive-run fraction, post-evidence advance rate and pending recovery. It also reports executed-action outcomes and termination status.

Clinical quality is a separate 1–7 human rating. Supply independently collected ratings through `--clinical-ratings`; absent ratings remain missing and are not estimated from environment progress. A CSV can use the columns `model_id,case_id,clinical_quality`; a JSON file can contain a list of objects with the same fields. A unique `run_id` can replace the model/case key. Duplicate, unmatched or out-of-range ratings are rejected.

