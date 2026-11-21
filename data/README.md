# Data

This directory contains the local anonymous successor dataset used by both runnable tracks:

| Track | Cases | Checkpoints | Evidence records | Reference transitions |
|---|---:|---:|---:|---:|
| Longitudinal | 40 | 240 | 730 | 501 |
| Interactive | 10 | — | 271 | 141 |

The interactive cases are `ward_001` through `ward_010`. The two tracks retain their own reference structures. The interactive graph must not be replaced with the longitudinal reference transitions.

## Files and visibility

- `longitudinal/cases/*.gold.json`: evaluator-side reference answers and cumulative evidence boundaries. The runner compiles the allowed public records from this package. The complete Gold file must never be sent to a model.
- `interactive/cases.json`: the 10-case index. `interactive/cases/ward_001.json` is a portable case configuration with `data_root` relative to the configuration file.
- `interactive/outputs/`: server-side scenarios, interfaces, ontologies, action graphs, state routes, tool maps, and protocol contracts. Hidden actions, prerequisites, and unreleased records stay inside the environment.
- Each track's `DATASET_MANIFEST.json` binds every delivered runtime asset by SHA-256 and byte size.
- `interactive/TRANSLATION_LINEAGE.json` records the successor dataset's temporal and graph preservation policy.

Review identities, signatures, local source paths, translation audit identities, and unused annotation history are excluded from runtime cases. Deidentification mappings remain in a separate private archive.

## Version and clinical content

Longitudinal data retain all 240 scoring projections and all 8,408 longitudinal atoms. Seventeen free-text state-status annotations were translated; this affects the auxiliary status-agreement diagnostic when comparing with older predictions that used untranslated status text. The seven principal scoring dimensions and six longitudinal diagnostics are unchanged by that status translation. Unused historical annotation fields were removed.

Interactive evidence was aligned by case ID, evidence ID, and record kind to approved reference text. All 271 evidence records are represented. Existing structured runtime dates were shifted by exactly 100 years to agree with the deidentified evidence timeline; no numeric dose or free-standing number was changed. Four evidence-time fields that differed between source versions retain the interactive visibility contract, including missing terminal timestamps. The original 141 transitions, prerequisites, and evidence-release sets are preserved. Reference aliases and clinical matching phrases replace bilingual duplicates.

This is a new version with new asset hashes. Historical trajectories bind their historical versions; do not label a run against these successor assets as a byte-identical replay of an earlier release.

## Verify the package

From the repository root:

```bash
python data/verify.py data
```

For a local release review with the private identity denylist:

```bash
python data/verify.py data --policy /path/to/private-policy.json --report /path/to/scan-report.json
```

The verifier independently checks file bindings, snapshot hashes, evidence references, runtime dependency hashes, untranslated text, identity-bearing metadata, local user paths, email addresses, professional name patterns, and identifier-like numbers. It prints locations and categories without echoing matched identity strings. A passing result covers these checks and the supplied identity rules; it is not a blanket certification of all possible free-text identities.

## Rebuild from an authorized archive

`prepare.py` uses only the Python standard library. It always requires a new output directory and leaves the source archive unchanged. Private rules and reviewed translations are supplied explicitly and are never copied into the output. Examples use neutral placeholders; they do not refer to a developer's filesystem.

```bash
python data/prepare.py longitudinal \
  --source /path/to/authorized-gold \
  --output /path/to/new-longitudinal \
  --policy /path/to/private-policy.json \
  --translations /path/to/reviewed-status-translations.json

python data/prepare.py interactive \
  --source /path/to/authorized-runtime \
  --output /path/to/new-interactive \
  --policy /path/to/private-policy.json \
  --reference-data /path/to/new-longitudinal \
  --translations /path/to/reviewed-runtime-translations.json \
  --alignment-exceptions /path/to/reviewed-time-exceptions.json
```

The policy accepts `replacements` (`match`, `replace`, optional `ignore_case`), regex `patterns` (`pattern`, `replace`), and `forbidden_terms`. Translation files contain `entries` with `text` and `translation`. Time exceptions require exact `case_id`, `evidence_id`, `field`, `source_value`, and `reference_value`; an unexpected difference stops the import. The importer verifies evidence identity alignment, rejects untranslated text and unresolved deny terms, updates all dependency hashes, and writes a new manifest. It does not translate clinical text with a model.

Data access and redistribution terms are separate from the code license. This local preparation does not upload or publish clinical records.
