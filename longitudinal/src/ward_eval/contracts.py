from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .util import contains_cjk


SUBMISSION_VERSION = "ward-checkpoint-submission-1.1"
OUTPUT_STATUSES = {
    "complete",
    "schema_invalid",
    "missing_output",
    "refusal",
    "truncated",
    "multiple_submissions",
    "unsupported_format",
    "translation_failed",
    "transport_error",
}
STATE_BUCKETS = (
    "diagnoses",
    "active_problems",
    "resolved_problems",
    "blockers",
    "critical_facts_to_retain",
    "facts_to_retract_or_correct",
    "required_delta_from_previous_checkpoint",
)
SINGLE_STATE_BUCKETS = ("treatment_readiness", "management_state")
REQUIRED_COMPLETE_FIELDS = {
    "key_evidence",
    "utilization_links",
    "claims",
    "patient_state",
    "actions",
    "safety",
    "stopping",
}

SCHEMA_PATH = Path(__file__).resolve().parent / "schemas/checkpoint_submission.v1.schema.json"
with SCHEMA_PATH.open("r", encoding="utf-8") as _schema_handle:
    CHECKPOINT_SUBMISSION_SCHEMA = json.load(_schema_handle)
Draft202012Validator.check_schema(CHECKPOINT_SUBMISSION_SCHEMA)
SUBMISSION_VALIDATOR = Draft202012Validator(CHECKPOINT_SUBMISSION_SCHEMA)


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def validate_submission(record: Any, *, require_target_language: bool = True) -> list[ValidationIssue]:
    """Validate one canonical submission with the executable public JSON Schema."""
    issues = [_schema_issue(error, record) for error in sorted(
        SUBMISSION_VALIDATOR.iter_errors(record),
        key=lambda error: (list(error.absolute_path), error.validator, error.message),
    )]
    if issues:
        return issues

    if record["checkpoint_id"].split("_cp_", 1)[0] != record["case_id"]:
        issues.append(ValidationIssue(
            path="$.checkpoint_id",
            code="case_checkpoint_mismatch",
            message="checkpoint must belong to case",
        ))
    if record["submission_status"] == "complete" and require_target_language:
        if record["language"] != "en":
            issues.append(ValidationIssue(
                path="$.language",
                code="non_target_language_submission",
                message="complete formal submissions must use language code en",
            ))
        for path, value in _submission_texts(record):
            if contains_cjk(value):
                issues.append(ValidationIssue(
                    path=path,
                    code="cjk_in_target_language_submission",
                    message="formal submission text contains CJK characters",
                ))
    return issues


def read_submission_jsonl(path: Path) -> tuple[list[Any], list[dict[str, Any]]]:
    """Read JSONL without guessing malformed records or aborting the whole run."""
    rows: list[Any] = []
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append({
                    "code": "invalid_json",
                    "line_number": line_number,
                    "line_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                    "message": str(exc),
                })
    return rows, errors


def _schema_issue(error: Any, record: Any) -> ValidationIssue:
    path = _json_path(error.absolute_path)
    code = "schema_invalid"
    if error.validator == "required":
        missing = error.message.split("'", 2)[1]
        path = f"{path}.{missing}"
        code = (
            "missing_dimension"
            if isinstance(record, dict)
            and record.get("submission_status") == "complete"
            and missing in REQUIRED_COMPLETE_FIELDS
            else "missing_field"
        )
    elif error.validator == "additionalProperties":
        code = "unknown_field"
    elif error.validator == "const" and path == "$.schema_version":
        code = "wrong_version"
    elif error.validator == "type":
        code = "wrong_type"
    elif error.validator == "pattern":
        code = "invalid_format"
    elif error.validator == "enum":
        code = "invalid_enum"
    return ValidationIssue(path=path, code=code, message=error.message)


def _json_path(parts: Any) -> str:
    result = "$"
    for part in parts:
        result += f"[{part}]" if isinstance(part, int) else f".{part}"
    return result


def _submission_texts(record: dict[str, Any]):
    for i, item in enumerate(record["key_evidence"]):
        if isinstance(item.get("interpretation_en"), str):
            yield f"$.key_evidence[{i}].interpretation_en", item["interpretation_en"]
    for i, item in enumerate(record["utilization_links"]):
        if isinstance(item.get("expected_outcome_en"), str):
            yield f"$.utilization_links[{i}].expected_outcome_en", item["expected_outcome_en"]
    for i, item in enumerate(record["claims"]):
        yield f"$.claims[{i}].text_en", item["text_en"]
    state = record["patient_state"]
    for bucket in STATE_BUCKETS:
        for i, item in enumerate(state[bucket]):
            yield f"$.patient_state.{bucket}[{i}].text_en", item["text_en"]
    for bucket in SINGLE_STATE_BUCKETS:
        item = state[bucket]
        if item is not None:
            yield f"$.patient_state.{bucket}.text_en", item["text_en"]
    for i, item in enumerate(record["actions"]):
        for field in (
            "text_en", "problem_en", "indication_en", "target_and_laterality_en",
            "deadline_or_time_window_en", "prohibited_action_en",
        ):
            if isinstance(item.get(field), str):
                yield f"$.actions[{i}].{field}", item[field]
        for j, text in enumerate(item.get("prerequisites_en", [])):
            yield f"$.actions[{i}].prerequisites_en[{j}]", text
    for field, values in record["safety"].items():
        for i, text in enumerate(values):
            yield f"$.safety.{field}[{i}]", text
    for i, text in enumerate(record["stopping"]["remaining_blockers"]):
        yield f"$.stopping.remaining_blockers[{i}]", text
    yield "$.stopping.rationale_en", record["stopping"]["rationale_en"]
