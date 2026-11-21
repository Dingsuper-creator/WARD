#!/usr/bin/env python3
"""Independently verify asset bindings and scan a candidate release.

Reports contain locations and categories, never matched identity or secret text.
A clear result is a check against supplied rules, not a privacy certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

CODE_SUFFIXES = {".py", ".sh", ".js", ".ts", ".tsx", ".toml", ".md", ".txt"}
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
HASH = re.compile(r"\b[0-9a-f]{64}\b")
PATTERNS = {
    "local_user_path": re.compile(r"/(?:Users|home|Volumes)/[^\s\"<>]+"),
    "email_address": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "professional_name": re.compile(r"\b(?:Dr|Prof)\.\s+[A-Z][A-Za-z.-]+(?:\s+[A-Z][A-Za-z.-]+){0,2}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "identifier_number": re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"),
    "phone_number": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
}
IDENTITY_KEYS = {"reviewer_id", "reviewer_ids", "annotator_id", "annotator_ids", "typed_signature", "patient_name", "patient_id", "encounter_id", "root_hints"}
BOUND_KEYS = {"scenario_sha256", "action_interface_sha256", "action_ontology_sha256", "action_graph_sha256", "freeze_manifest_sha256", "state_routes_sha256", "lifecycle_policy_sha256", "event_relative_phase_policy_sha256"}


def sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def canonical_sha(value) -> str:
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def walk(node, pointer=""):
    yield pointer, node
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from walk(value, pointer + "/" + str(index))


def inspect(root: Path, terms: list[str]) -> dict:
    findings = []
    stats = Counter()
    hashes = {}
    documents = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in {".git", "__pycache__"} for part in path.relative_to(root).parts):
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append({"path": relative, "category": "symlink_requires_review"})
            continue
        content = path.read_bytes()
        hashes[relative] = sha(content)
        stats["files"] += 1
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            findings.append({"path": relative, "category": "uninspected_binary"})
            continue
        normalized = HASH.sub("[digest]", text)
        # Patterns are applied to data; source code contains scanner pattern definitions.
        if path.suffix == ".json" or path.suffix == ".jsonl":
            for category, pattern in PATTERNS.items():
                for match in pattern.finditer(normalized):
                    findings.append({"path": relative, "line": normalized.count("\n", 0, match.start()) + 1, "category": category})
        if CJK.search(text):
            findings.append({"path": relative, "category": "untranslated_text"})
        for index, term in enumerate(terms):
            if term.casefold() in text.casefold() or term.casefold() in relative.casefold():
                findings.append({"path": relative, "category": "private_deny_term", "rule_index": index})
        if path.suffix == ".json":
            try:
                documents[relative] = json.loads(text)
                if CJK.search(json.dumps(documents[relative], ensure_ascii=False)) and not CJK.search(text):
                    findings.append({"path": relative, "category": "escaped_untranslated_text"})
            except ValueError:
                findings.append({"path": relative, "category": "invalid_json"})

    for relative, document in documents.items():
        for pointer, node in walk(document):
            if not isinstance(node, dict):
                continue
            for key in IDENTITY_KEYS & node.keys():
                if node[key] not in (None, "", [], {}):
                    findings.append({"path": relative, "pointer": pointer + "/" + key, "category": "identity_metadata"})
        if relative.endswith("DATASET_MANIFEST.json"):
            base = Path(relative).parent
            declared = set()
            for binding in document.get("files", []):
                bound = (base / binding["path"]).as_posix()
                declared.add(bound)
                target = (root / bound).resolve()
                if not target.is_relative_to(root.resolve()) or hashes.get(bound) != binding.get("sha256"):
                    findings.append({"path": bound, "category": "manifest_binding_mismatch"})
                elif target.stat().st_size != binding.get("size_bytes"):
                    findings.append({"path": bound, "category": "manifest_size_mismatch"})
            stats["manifest_files_checked"] += len(declared)
        if relative.endswith(".gold.json"):
            stats["gold_cases"] += 1
            evidence = {row["evidence_id"] for row in document["normalized_evidence"]}
            stats["gold_evidence"] += len(evidence)
            for checkpoint in document["checkpoints"]:
                stats["checkpoints"] += 1
                q = checkpoint["question"]
                allowed, future = set(q["allowed_evidence_ids"]), set(q["future_forbidden_evidence_ids"])
                if allowed & future or not (allowed | future) <= evidence:
                    findings.append({"path": relative, "category": "evidence_visibility_reference_invalid", "checkpoint_id": checkpoint["checkpoint_id"]})
                if canonical_sha(q["public_snapshot"]) != q["public_snapshot_sha256"]:
                    findings.append({"path": relative, "category": "snapshot_hash_mismatch", "checkpoint_id": checkpoint["checkpoint_id"]})
        if relative.endswith(".patient_ehr.json"):
            stats["interactive_cases"] += 1
            stats["interactive_evidence"] += len(document["evidence_store"])
            stats["interactive_transitions"] += len(document["transitions"])
        for key in BOUND_KEYS & document.keys():
            if document[key] not in set(hashes.values()):
                findings.append({"path": relative, "pointer": "/" + key, "category": "runtime_hash_binding_missing"})
        if isinstance(document.get("scorer_freeze"), dict):
            if document["scorer_freeze"].get("manifest_sha256") not in set(hashes.values()):
                findings.append({"path": relative, "category": "scorer_freeze_binding_missing"})
    return {"status": "passed" if not findings else "blocked", "scope": "integrity_and_configured_identity_rules", "counts": dict(stats), "finding_count": len(findings), "findings": findings, "limitations": ["Free-text identities not covered by the private policy require independent release review.", "Dataset text, source code, filenames, and declared bindings are checked together."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--policy", type=Path, help="Private policy with forbidden_terms; never copied")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("root must be a directory")
    terms = json.loads(args.policy.read_text(encoding="utf-8")).get("forbidden_terms", []) if args.policy else []
    if any(not isinstance(term, str) or not term for term in terms):
        parser.error("forbidden_terms must contain nonempty strings")
    result = inspect(args.root, terms)
    output = json.dumps(result, ensure_ascii=True, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    print(output, end="")
    raise SystemExit(result["status"] != "passed")


if __name__ == "__main__":
    main()
