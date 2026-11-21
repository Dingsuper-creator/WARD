# Interactive benchmark

A deterministic clinical environment with seven public tools, a model runner, and offline evaluation. Python 3.10 or newer is required; this component uses the standard library only.

The public tools are `patient.ask`, `ehr.search`, `ehr.read`, `ehr.order`, `procedure.perform`, `treatment.start`, and `care.transition`. Only recorded evidence and transitions can be released. The runner constructs model messages from public fields and keeps the prerequisite graph and future evidence private.

## Validate data

From the repository root:

```sh
python interactive/run.py --case ward_001 --dry-run
```

The default data root is `data/interactive`, resolved relative to the installed source files, independently of the working directory. Use `--data-root` for another prepared bundle. The runner reads the bundle's `cases.json` index when present. `--case-config` accepts a JSON object containing `case_id`, `adapter` (`v055` or `v056`), optional budgets, and an `artifacts` mapping for `scenario`, `freeze_manifest`, `interface`, `ontology`, `graph`, `tool_map`, and `prompt_contract`. Paths are relative to the configuration file or its optional `data_root` field. Embedded route and policy paths are relative to that root.

The standard case configuration uses the phase adapter for `ward_007` with 20 actions and 24 model calls. The other cases use the ontology adapter with 40 actions and 48 model calls. Both limits can be explicitly overridden. This preserves the configured case budgets rather than silently changing historical denominators.

## Run a model

Set `WARD_API_BASE_URL` and `WARD_API_KEY` in the environment, then run:

```sh
python interactive/run.py --case ward_001 --model MODEL_ID --output runs/model/ward_001
```

The endpoint must provide a compatible Chat Completions interface. The runner uses temperature 0 and a maximum of 640 output tokens. A local compatible endpoint may omit its API key. Live model execution incurs the provider's normal usage.

For a single-dispatcher interface such as the Kimi adapter:

```sh
python interactive/run.py --case ward_001 --model MODEL_ID --protocol dispatcher --output runs/model/ward_001
```

The dispatcher packages the same seven actions into `clinical_action`; canonical validation and environment execution remain in the runner. Its protocol is recorded separately. Run cases in separate processes: the inherited runtime temporarily selects successor implementations through module globals and is not designed for concurrent threads in one process.

An API-interrupted run can be resumed into an empty directory:

```sh
python interactive/run.py --case ward_001 --model MODEL_ID --protocol dispatcher --resume-from runs/interrupted --output runs/continued
```

Continuation validates the source artifact bindings, preserves model, protocol and budgets, replays executed actions locally, and sends only new model calls. The final continuation already contains its prefix. Do not aggregate the prefix and continuation as separate trajectories.

## Evaluate selected runs

```sh
python interactive/eval.py --run runs/model/ward_001 --output scores.json
```

Repeat `--run` to aggregate selected model/case trajectories. The evaluator checks input hashes, trace chains and run bindings. It reports milestone coverage, productive budget yield, reference-evidence recall, pending recovery, post-evidence advancement, evidence-free repetition, and the longest nonproductive streak. Outcome shares pool executed actions; other model metrics are case means with explicit valid-case counts. Unused action budgets remain in the productive-yield denominator. Undefined ratios remain `null`.

Clinical quality requires independent human ratings. Provide CSV or JSON with `clinical_quality` from 1 through 7 and either `run_id` or both `model_id` and `case_id`:

```csv
model_id,case_id,clinical_quality
MODEL_ID,ward_001,5
```

```sh
python interactive/eval.py --run runs/model/ward_001 --clinical-ratings ratings.csv --output scores.json
```

A JSON input may be a list of rating objects or an object with a `ratings` list. `overall_clinical_quality_1_to_7` is also accepted as the score field. Invalid ranges, duplicate ratings, and ambiguous or unmatched keys are rejected. Cases without a supplied rating stay `null` and are excluded from the clinical-quality mean. The evaluator does not generate clinician judgments or infer clinical quality from completion.

## Offline verification

```sh
python -m unittest discover -s interactive/tests -v
python interactive/tests/validate_data.py
python interactive/tests/fixture.py /tmp/ward-synthetic
python interactive/run.py --case ward_001 --case-config /tmp/ward-synthetic/case.json --scripted-actions /tmp/ward-synthetic/actions.json --output /tmp/ward-synthetic-run
python interactive/eval.py --run /tmp/ward-synthetic-run --case-config /tmp/ward-synthetic/case.json
```

The fixture contains two artificial transitions and no patient-derived text. Tests cover prerequisite recovery, successful completion, metric denominators, hash tampering, continuation without repeated provider calls, the dispatcher, and clinical-rating validation. No model calls are made by these tests.

`validate_data.py` checks every prepared case's input bindings, graph dependencies, reference-concept execution, evidence release, and final visibility. It reads private reference actions as a data-integrity check; it is not a model policy or a claim that every public-tool trajectory completes.

## Release identity

This is an anonymous successor release. Schema and protocol identifiers use `ward-`, cases use `ward_`, paths are portable, and data hashes are regenerated after preparation. Historical artifact hashes and earlier model outputs must not be presented as runs of this release. `release_receipt.json` explicitly records that distinction.

The `runtime/` layers retain their internal versioned module names because the current environment composes their implementations. The supported user interfaces are `run.py` and `eval.py`; report generators, paper figures, historic case builders, and frozen model-result packages are outside this component.
