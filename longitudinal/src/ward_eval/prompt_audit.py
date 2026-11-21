from __future__ import annotations

from typing import Any

from .gold import GoldCheckpoint
from .util import canonical_sha256, normalize_text, sha256_text


FORBIDDEN_EXPOSED_ROOTS = {
    "answer",
    "review",
    "model_preannotation",
    "process_gold",
    "acceptable_aliases_en",
    "future_forbidden_evidence_ids",
}


def audit_prompt_manifest(manifest: dict[str, Any], gold_cp: GoldCheckpoint) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    required = {
        "case_id",
        "checkpoint_id",
        "prompt_template_version",
        "system_prompt_sha256",
        "user_prompt_sha256",
        "tool_schema_sha256",
        "public_snapshot_sha256",
        "visible_evidence_sha256",
        "request_sha256",
        "gold_exposed",
        "future_evidence_exposed",
        "exposed_fields",
        "clinical_data_trust_policy",
    }
    for field in sorted(required - set(manifest)):
        findings.append({"severity": "critical", "code": "missing_manifest_field", "field": field})
    if manifest.get("case_id") != gold_cp.case_id:
        findings.append({"severity": "critical", "code": "prompt_case_mismatch"})
    if manifest.get("checkpoint_id") != gold_cp.checkpoint_id:
        findings.append({"severity": "critical", "code": "prompt_checkpoint_mismatch"})
    if manifest.get("public_snapshot_sha256") != gold_cp.snapshot_sha256:
        findings.append({"severity": "critical", "code": "public_snapshot_hash_mismatch"})
    if manifest.get("gold_exposed") is not False:
        findings.append({"severity": "critical", "code": "gold_exposed_flag_invalid"})
    if manifest.get("future_evidence_exposed") is not False:
        findings.append({"severity": "critical", "code": "future_evidence_flag_invalid"})
    if manifest.get("clinical_data_trust_policy") != "untrusted_delimited_data":
        findings.append({
            "severity": "critical",
            "code": "clinical_data_trust_boundary_missing",
            "message": "public EHR evidence must be explicitly delimited and treated as untrusted data",
        })

    for field in manifest.get("exposed_fields", []):
        components = {part for part in str(field).replace("[]", "").split(".") if part}
        overlap = components & FORBIDDEN_EXPOSED_ROOTS
        if overlap:
            findings.append({
                "severity": "critical",
                "code": "forbidden_exposed_field",
                "field": field,
                "forbidden_components": sorted(overlap),
            })

    payload = manifest.get("request_payload")
    prompt_reproducible = payload is not None
    if payload is None:
        findings.append({
            "severity": "high",
            "code": "prompt_not_available_for_reproduction",
            "message": "A hash alone cannot reproduce or leakage-audit the prompt payload.",
        })
    else:
        if canonical_sha256(payload) != manifest.get("request_sha256"):
            findings.append({"severity": "critical", "code": "request_hash_mismatch"})
        _audit_component_hashes(payload, manifest, findings)
        keys = _all_keys(payload)
        forbidden_keys = sorted(keys & FORBIDDEN_EXPOSED_ROOTS)
        if forbidden_keys:
            findings.append({
                "severity": "critical",
                "code": "forbidden_prompt_payload_keys",
                "keys": forbidden_keys,
            })
        strings = set(_all_strings(payload))
        leaked_future = sorted(strings & gold_cp.future_evidence_ids)
        if leaked_future:
            findings.append({
                "severity": "critical",
                "code": "future_evidence_id_in_prompt",
                "evidence_ids": leaked_future,
            })
        gold_internal_ids = _gold_internal_ids(gold_cp.checkpoint["answer"])
        leaked_internal = sorted(strings & gold_internal_ids)
        if leaked_internal:
            findings.append({
                "severity": "critical",
                "code": "gold_internal_id_in_prompt",
                "ids": leaked_internal,
            })
        overlap_scan_values = set(_all_strings({
            "system_prompt": payload.get("system_prompt"),
            "user_prompt": payload.get("user_prompt"),
            "visible_evidence": payload.get("visible_evidence"),
        }))
        normalized_payload_strings = {
            normalize_text(value) for value in overlap_scan_values if len(normalize_text(value)) >= 16
        }
        public_strings = {
            normalize_text(value)
            for value in _all_strings(gold_cp.checkpoint["question"]["public_snapshot"])
            if len(normalize_text(value)) >= 16
        }
        gold_only_strings = {
            normalize_text(value)
            for source in (
                gold_cp.checkpoint.get("answer", {}),
                gold_cp.checkpoint.get("review", {}),
                gold_cp.checkpoint.get("model_preannotation", {}),
                gold_cp.case.get("process_gold", {}),
            )
            for value in _all_strings(source)
            if len(normalize_text(value)) >= 16
        } - public_strings
        overlaps = sorted(normalized_payload_strings & gold_only_strings)
        if overlaps:
            findings.append({
                "severity": "critical",
                "code": "gold_only_text_overlap",
                "overlap_count": len(overlaps),
                "overlap_sha256": [sha256_text(value) for value in overlaps[:20]],
            })

    blocking = [finding for finding in findings if finding["severity"] in {"critical", "high"}]
    return {
        "status": "passed" if not blocking else "blocked",
        "prompt_reproducible": prompt_reproducible,
        "finding_count": len(findings),
        "findings": findings,
    }


def audit_prompt_manifests(manifests: list[dict[str, Any]], gold) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    unknown: list[dict[str, Any]] = []
    for manifest in manifests:
        checkpoint_id = manifest.get("checkpoint_id") if isinstance(manifest, dict) else None
        if checkpoint_id not in gold.checkpoint_by_id:
            unknown.append({"checkpoint_id": checkpoint_id, "code": "unknown_prompt_checkpoint"})
            continue
        grouped.setdefault(checkpoint_id, []).append(manifest)
    rows: list[dict[str, Any]] = []
    for checkpoint_id, gold_checkpoint in gold.checkpoint_by_id.items():
        current = grouped.get(checkpoint_id, [])
        if not current:
            rows.append({
                "checkpoint_id": checkpoint_id,
                "status": "blocked",
                "prompt_reproducible": False,
                "finding_count": 1,
                "findings": [{"severity": "critical", "code": "missing_prompt_manifest"}],
            })
        elif len(current) > 1:
            rows.append({
                "checkpoint_id": checkpoint_id,
                "status": "blocked",
                "prompt_reproducible": False,
                "finding_count": 1,
                "findings": [{"severity": "critical", "code": "duplicate_prompt_manifest", "count": len(current)}],
            })
        else:
            rows.append({"checkpoint_id": checkpoint_id, **audit_prompt_manifest(current[0], gold_checkpoint)})
    passed = sum(1 for row in rows if row["status"] == "passed")
    return {
        "status": "passed" if passed == len(gold.checkpoint_by_id) and not unknown else "blocked",
        "expected_checkpoint_count": len(gold.checkpoint_by_id),
        "provided_manifest_count": len(manifests),
        "passed_checkpoint_count": passed,
        "blocked_checkpoint_count": len(rows) - passed,
        "unknown": unknown,
        "rows": rows,
    }


def _all_keys(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            result.add(str(key))
            result.update(_all_keys(child))
    elif isinstance(value, list):
        for child in value:
            result.update(_all_keys(child))
    return result


def _all_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _all_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_strings(child)


def _gold_internal_ids(answer: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for item in answer.get("layer3_claims", []):
        ids.add(item["claim_id"])
    for item in answer.get("layer5_actions", []):
        ids.add(item["action_id"])
    for value in answer.get("layer4_state", {}).values():
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, dict) and item.get("state_id"):
                ids.add(item["state_id"])
    return ids


def _audit_component_hashes(payload: Any, manifest: dict[str, Any], findings: list[dict[str, Any]]) -> None:
    if not isinstance(payload, dict):
        findings.append({"severity": "critical", "code": "request_payload_not_object"})
        return
    required = {"system_prompt", "user_prompt", "tool_schema", "sampling_parameters", "visible_evidence"}
    for field in sorted(required - set(payload)):
        findings.append({"severity": "critical", "code": "missing_request_component", "field": field})
    checks = (
        ("system_prompt", "system_prompt_sha256", sha256_text),
        ("user_prompt", "user_prompt_sha256", sha256_text),
        ("tool_schema", "tool_schema_sha256", canonical_sha256),
        ("visible_evidence", "visible_evidence_sha256", canonical_sha256),
    )
    for payload_field, manifest_field, hash_function in checks:
        if payload_field not in payload:
            continue
        value = payload[payload_field]
        if payload_field in {"system_prompt", "user_prompt"} and not isinstance(value, str):
            findings.append({"severity": "critical", "code": "prompt_component_not_string", "field": payload_field})
            continue
        if hash_function(value) != manifest.get(manifest_field):
            findings.append({"severity": "critical", "code": "prompt_component_hash_mismatch", "field": payload_field})
