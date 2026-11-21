#!/usr/bin/env python3
"""Create a separate, versioned runtime copy from an authorized local archive.

The private policy is never copied to the output. Clinical language and dates
are versioned explicitly. This utility does not certify clinical content for publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

REMOVED_KEYS = {
    "source", "source_metadata", "normalization_provenance", "root_hints",
    "review", "clinical_review", "review_provenance_v0_3",
    "legacy_clinical_review_migration", "model_preannotation",
    "reviewer_id", "reviewer_ids", "annotator_id", "annotator_ids", "typed_signature",
    "reviewed_at", "created_at", "created_at_utc", "generated_at_utc",
    "author", "authors", "patient_id", "encounter_id", "source_semantic_task_ids",
    "supersedes_tool_map_sha256", "supersedes_state_routes_sha256", "change_control",
}
CASE_KEYS = {
    "case_id", "checkpoints", "normalized_evidence", "process_gold", "milestone_candidates",
}
CHECKPOINT_KEYS = {"checkpoint_id", "sequence_index", "question", "answer"}
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
ITEM_KEYS = {
    "evidence_id", "role", "interpretation_en", "claim_ids", "state_ids", "action_ids", "expected_outcome",
    "claim_id", "canonical_en", "acceptable_aliases_en", "status", "linked_evidence_ids", "supporting_evidence_ids",
    "state_id", "action_id", "canonical_action_en", "category", "action_scope", "critical_omission_if_missing",
    "canonical_prohibited_action_en", "problem_en", "indication_en", "target_and_laterality_en",
    "deadline_or_time_window_en", "prerequisites_en",
}


def runtime_answer(answer):
    def item(value):
        return {k: v for k, v in value.items() if k in ITEM_KEYS} if value is not None else None
    result = {key: [item(row) for row in answer[key]] for key in (
        "layer1_evidence", "layer2_utilization_links", "layer3_claims", "layer5_actions",
    )}
    result["layer4_state"] = {key: [item(row) for row in value] if isinstance(value, list) else item(value) for key, value in answer["layer4_state"].items() if key in {"diagnoses", "active_problems", "resolved_problems", "blockers", "critical_facts_to_retain", "facts_to_retract_or_correct", "required_delta_from_previous_checkpoint", "treatment_readiness", "management_state"}}
    for action, original in zip(result["layer5_actions"], answer["layer5_actions"]):
        for field in ("problem", "indication", "target_and_laterality", "deadline_or_time_window", "prerequisites"):
            action[field + "_en"] = original.get(field + "_en", original.get(field))
    result["safety"] = {field + "_en": answer["safety"].get(field + "_en", answer["safety"].get(field, [])) for field in ("hard_failure_rules", "critical_omission_rules", "preliminary_risk_triggers")}
    result["stopping"] = {key: value for key, value in answer["stopping"].items() if key in {"decision", "remaining_blockers_en", "rationale_en"}}
    result["stopping"]["remaining_blockers_en"] = answer["stopping"].get("remaining_blockers_en", answer["stopping"].get("remaining_blockers", []))
    return result


def shift_structured_times(node, field=""):
    if isinstance(node, dict):
        return {key: shift_structured_times(value, key) for key, value in node.items()}
    if isinstance(node, list):
        return [shift_structured_times(value, field) for value in node]
    if isinstance(node, str) and re.search(r"time|clock|date", field, re.IGNORECASE) and re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}(?:T.*)?", node):
        parsed = datetime.fromisoformat(node.replace("Z", "+00:00"))
        # Fail instead of silently changing an invalid leap day.
        parsed.replace(year=parsed.year + 100)
        return str(parsed.year + 100) + node[4:]
    return node


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def encode(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical_digest(value) -> str:
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode(value))


class Transform:
    def __init__(self, policy: dict, ids: dict[str, str], translations: dict[str, str] | None = None):
        self.counts = Counter()
        self.ids = ids
        self.translations = translations or {}
        self.rules = sorted(policy.get("replacements", []), key=lambda x: len(x["match"]), reverse=True)
        self.patterns = [(re.compile(row["pattern"]), row["replace"]) for row in policy.get("patterns", [])]
        self.denied = policy.get("forbidden_terms", [])
        if not self.rules and not self.patterns:
            raise ValueError("A reviewed private replacement policy is required")
        if any(not x.get("match") for x in self.rules) or any(not x for x in self.denied):
            raise ValueError("Policy terms must be nonempty")

    def text(self, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value):
            return value
        for old, new in sorted(self.ids.items(), key=lambda x: len(x[0]), reverse=True):
            value = value.replace(old, new)
        for row in self.rules:
            old, new = row["match"], row["replace"]
            if row.get("ignore_case", False):
                value, n = re.subn(re.escape(old), lambda _: new, value, flags=re.IGNORECASE)
            else:
                n = value.count(old)
                value = value.replace(old, new)
            self.counts["literal_replacements"] += n
        for pattern, new in self.patterns:
            value, n = pattern.subn(new, value)
            self.counts["pattern_replacements"] += n
        return self.translations.get(value, value)

    def value(self, node, field=""):

        if isinstance(node, dict):
            result = {}
            for key, value in node.items():
                if key in REMOVED_KEYS or key.startswith("model_preannotation_provenance") or key.endswith(("translation_review", "_zh")):
                    self.counts["metadata_fields_removed"] += 1
                    continue
                if key in {"answer_finalization_transform", "legacy_question_ids"}:
                    self.counts["metadata_fields_removed"] += 1
                    continue
                # Keep the explicitly translated scoring field, avoiding duplicate legacy text.
                if key + "_en" in node:
                    self.counts["duplicate_language_fields_removed"] += 1
                    continue
                new_key = self.text(key)
                if new_key in result:
                    raise ValueError("Policy produced duplicate object keys")
                result[new_key] = self.value(value, key)
            return result
        if isinstance(node, list):
            result = [self.value(x, field) for x in node]
            if field in {"accepted_target_aliases", "accepted_action_type_aliases", "acceptable_aliases_en", "aliases", "phrases", "keywords", "terms"} and all(isinstance(x, str) for x in result):
                result = list(dict.fromkeys(result))
            return result
        return self.text(node) if isinstance(node, str) else node

    def validate_text(self, files: dict[str, bytes]):
        findings = []
        patterns = [
            ("absolute_local_path", re.compile(r"/(?:Users|home|Volumes)/[^\s\"<>]+")),
            ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
            ("national_identifier", re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)")),
        ]
        for relative, content in files.items():
            text = content.decode("utf-8")
            for index, term in enumerate(self.denied):
                if term.casefold() in text.casefold() or term.casefold() in relative.casefold():
                    findings.append({"path": relative, "category": "denied_term", "rule_index": index})
            scan_text = re.sub(r"\b[0-9a-f]{64}\b", "[digest]", text)
            for category, pattern in patterns:
                if pattern.search(scan_text):
                    findings.append({"path": relative, "category": category})
        if findings:
            # Do not echo matching source text or private policy terms.
            raise ValueError(json.dumps({"status": "blocked", "findings": findings}, ensure_ascii=True))


def neutral_id(value: str) -> str:
    match = re.search(r"_(\d{3})$", value)
    if not match:
        raise ValueError("Expected a three-digit case suffix")
    return "ward_" + match.group(1)


def longitudinal(source: Path, transform: Transform) -> dict[str, bytes]:
    files = {}
    case_paths = sorted((source / "cases").glob("*.gold.json"))
    if not case_paths:
        raise ValueError("No case files found")
    for path in case_paths:
        original = read_json(path)
        selected = {k: v for k, v in original.items() if k in CASE_KEYS}
        selected["checkpoints"] = [
            {k: v for k, v in cp.items() if k in CHECKPOINT_KEYS}
            for cp in original["checkpoints"]
        ]
        for cp in selected["checkpoints"]:
            cp["answer"] = runtime_answer(cp["answer"])
        result = transform.value(selected)
        result["schema_version"] = "ward-runtime-gold-1.0"
        result["gold_version"] = "ward-anonymous-runtime-1.0"
        for cp in result["checkpoints"]:
            cp["question"]["public_snapshot_sha256"] = canonical_digest(cp["question"]["public_snapshot"])
        name = f"cases/{result['case_id']}.gold.json"
        if name in files:
            raise ValueError("Duplicate output case identity")
        files[name] = encode(result)
    return files


def replace_hashes(node, updates):
    if isinstance(node, dict):
        return {key: replace_hashes(value, updates) for key, value in node.items()}
    if isinstance(node, list):
        return [replace_hashes(value, updates) for value in node]
    return updates.get(node, node) if isinstance(node, str) else node


def interactive(source: Path, transform: Transform, case_ids: list[str], reference_data: Path, alignment_exceptions: list[dict]) -> dict[str, bytes]:
    documents = {}
    originals = {}
    configs = []
    allowed_differences = {(row["case_id"], row["evidence_id"], row["field"]): row for row in alignment_exceptions}
    observed_differences = set()
    root = source.resolve()
    references = {row["case_id"]: row for row in (read_json(path) for path in (reference_data / "cases").glob("*.gold.json"))}


    def add(relative: str) -> str:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing dependency or dependency outside input archive")
        output_relative = transform.text(relative)
        if output_relative not in documents:
            original = read_json(path)
            if path.name == "FREEZE_MANIFEST.json":
                original = {k: v for k, v in original.items() if k in {
                    "manifest_version", "freeze_id", "status", "frozen_versions", "capability_boundaries", "baseline_dataset", "official_scoring_allowed",
                }}
            original = shift_structured_times(original)
            result = transform.value(original)
            if path.name.startswith("PROMPT_CONTRACT."):
                result["engineering_language"] = "en"
            if path.name.endswith(".patient_ehr.json"):
                reference = references[result["case_id"]]
                evidence = {row["evidence_id"]: row for row in reference["normalized_evidence"]}
                if set(evidence) != set(result["evidence_store"]):
                    raise ValueError("Approved evidence ID set does not match the scenario")
                for evidence_id, record in result["evidence_store"].items():
                    approved = evidence[evidence_id]
                    for key in ("record_kind", "clinical_time", "available_time"):
                        if record.get(key) != approved.get(key):
                            identity = (result["case_id"], evidence_id, key)
                            exception = allowed_differences.get(identity)
                            if exception is None or exception["source_value"] != record.get(key) or exception["reference_value"] != approved.get(key):
                                raise ValueError("Approved evidence identity or time alignment failed")
                            observed_differences.add(identity)
                    record["content"] = approved["raw_content"]
                    record["title"] = approved["title"]
                context = reference["checkpoints"][0]["question"]["public_snapshot"]["public_context"]
                for key, value in result["public_context"].items():
                    if isinstance(value, str) and CJK.search(value):
                        result["public_context"][key] = context[key]
            documents[output_relative] = result
            originals[output_relative] = digest(path.read_bytes())
        return output_relative

    prefix = "outputs/world_model_patient_ehr_"
    freeze = add("outputs/world_model_interaction_scorer_freeze_v01/FREEZE_MANIFEST.json")
    prompt = add(prefix + "v05/PROMPT_CONTRACT.v0.5.4.json")
    for case_id in case_ids:
        version = "0.5.5" if case_id.endswith("_007") else "0.5.6"
        ontology = prefix + f"v02/action_ontology.{case_id}.v0.2.json"
        if not (root / ontology).is_file():
            ontology = prefix + "v02/action_ontology.v0.2.json"
        paths = {
            "scenario": prefix + f"v01/cases/{case_id}.patient_ehr.json",
            "interface": prefix + f"v02/cases/{case_id}.action_interface.v0.2.json",
            "ontology": ontology,
            "graph": prefix + f"v03/cases/{case_id}.action_graph.v0.3.json",
            "tool_map": prefix + f"v05/cases/{case_id}.tool_map.v{version}.json",
        }
        tool_map = read_json(root / paths["tool_map"])
        for key in ("state_routes_path", "lifecycle_policy_path", "event_relative_phase_policy_path"):
            if key in tool_map:
                add(tool_map[key])
        config = {key: add(value) for key, value in paths.items()}
        config.update(case_id=transform.ids[case_id], adapter="v055" if version == "0.5.5" else "v056", freeze_manifest=freeze, prompt_contract=prompt)
        configs.append(config)

    # Resolve the acyclic asset hash graph until every dependency binds its new bytes.
    base_documents = documents
    for _ in range(len(documents) + 2):
        hashes = {old: digest(encode(documents[path])) for path, old in originals.items()}
        revised = {path: replace_hashes(value, hashes) for path, value in base_documents.items()}
        if revised == documents:
            break
        documents = revised
    else:
        raise ValueError("Asset hash dependencies did not converge")
    files = {path: encode(value) for path, value in documents.items()}
    for config in configs:
        files[f"cases/{config['case_id']}.json"] = encode({**config, "data_root": ".."})
    files["cases.json"] = encode({"schema_version": "ward-interactive-index-1.0", "cases": configs})
    if observed_differences != set(allowed_differences):
        raise ValueError("Declared alignment exceptions do not exactly match observed differences")
    files["TRANSLATION_LINEAGE.json"] = encode({
        "schema_version": "ward-successor-lineage-1.0",
        "method": "same-case evidence identity alignment to approved reference text",
        "original_visibility_contract_preserved": True,
        "original_transition_graph_preserved": True,
        "reference_transition_graph_imported": False,
        "structured_clock_offset_years": 100,
        "retained_source_time_field_count": len(observed_differences),
        "reference_data_manifest_sha256": digest((reference_data / "DATASET_MANIFEST.json").read_bytes()),
    })
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("track", choices=("longitudinal", "interactive"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True, help="Private reviewed replacements and forbidden terms")
    parser.add_argument("--reference-data", type=Path, help="Approved aligned Gold; required for interactive import")
    parser.add_argument("--alignment-exceptions", type=Path, help="Reviewed, exact evidence-time differences to preserve from the runtime source")
    parser.add_argument("--translations", type=Path, help="Reviewed short-text translation entries")
    parser.add_argument("--case-numbers", default="1,2,3,4,5,6,7,8,9,10")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output must be a new directory; frozen inputs are never overwritten")
    if args.output.resolve().is_relative_to(args.source.resolve()):
        raise ValueError("Output must not be nested inside the source archive")
    if args.track == "longitudinal":
        paths = sorted((args.source / "cases").glob("*.gold.json"))
    else:
        selected = {int(n) for n in args.case_numbers.split(",")}
        paths = sorted((args.source / "outputs/world_model_patient_ehr_v01/cases").glob("*.patient_ehr.json"))
        paths = [path for path in paths if int(read_json(path)["case_id"].rsplit("_", 1)[1]) in selected]
        if len(paths) != len(selected):
            raise ValueError("Selected case set is incomplete")
    ids = {read_json(path)["case_id"]: neutral_id(read_json(path)["case_id"]) for path in paths}
    if not ids or len(set(ids.values())) != len(ids):
        raise ValueError("Empty or colliding case identity map")
    translations = {}
    if args.translations:
        entries = read_json(args.translations)["entries"]
        if any(not row.get("translation") or CJK.search(row["translation"]) for row in entries):
            raise ValueError("Every reviewed translation must be nonempty and contain no untranslated text")
        translations = {row["text"]: row["translation"] for row in entries}
    transform = Transform(read_json(args.policy), ids, translations)
    if args.track == "interactive" and not args.reference_data:
        raise ValueError("Interactive import requires approved aligned reference evidence")
    files = longitudinal(args.source, transform) if args.track == "longitudinal" else interactive(args.source, transform, list(ids), args.reference_data, read_json(args.alignment_exceptions) if args.alignment_exceptions else [])
    transform.validate_text(files)
    untranslated_text = [path for path, content in files.items() if CJK.search(content.decode("utf-8"))]
    if untranslated_text:
        raise ValueError(json.dumps({"status": "blocked", "category": "untranslated_text", "paths": untranslated_text}))
    manifest = {
        "schema_version": "ward-data-manifest-1.0", "track": args.track,
        "dataset_kind": "clinical_runtime_copy", "status": "local_anonymous_successor",
        "case_count": len(ids), "case_ids": sorted(ids.values()),
        "clinical_translation_performed": bool(translations) or args.track == "interactive", "date_shift_performed": args.track == "interactive",
        "structured_date_offset_years": 100 if args.track == "interactive" else 0,
        "translation_method": "approved_evidence_id_alignment_and_reviewed_short_text",
        "identity_metadata_removed": True, "source_archive_modified": False,
        "source_policy_sha256": digest(args.policy.read_bytes()),
        "transformation_counts": dict(transform.counts),
        "files": [{"path": path, "sha256": digest(content), "size_bytes": len(content)} for path, content in sorted(files.items())],
    }
    args.output.mkdir(parents=True)
    for relative, content in files.items():
        path = args.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    write_json(args.output / "DATASET_MANIFEST.json", manifest)
    print(json.dumps({"status": "prepared", "track": args.track, "case_count": len(ids), "file_count": len(files)}))


if __name__ == "__main__":
    main()
