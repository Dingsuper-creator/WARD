from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import SINGLE_STATE_BUCKETS, STATE_BUCKETS
from .util import is_scoreable_text, read_json, sha256_file


ACTION_EN_OVERRIDES = {
    "ward_006_cp_011_action_005": (
        "Do not remove the drain while output remains above 10 mL/day."
    ),
}


PROHIBITED_ACTION_EN_OVERRIDES = {
    "ward_026_cp_023_action_006": (
        "Label the tiny pulmonary nodule or lymph nodes as metastases or escalate "
        "treatment based on these findings alone."
    ),
}


@dataclass
class GoldCheckpoint:
    case_id: str
    checkpoint_id: str
    case: dict[str, Any]
    checkpoint: dict[str, Any]

    @property
    def snapshot_sha256(self) -> str:
        return self.checkpoint["question"]["public_snapshot_sha256"]

    @property
    def allowed_evidence_ids(self) -> set[str]:
        return set(self.checkpoint["question"]["allowed_evidence_ids"])

    @property
    def future_evidence_ids(self) -> set[str]:
        return set(self.checkpoint["question"]["future_forbidden_evidence_ids"])


class GoldPackage:
    def __init__(self, root: Path):
        self.root = root
        self.case_paths = sorted((root / "cases").glob("ward_*.gold.json"))
        if not self.case_paths:
            raise ValueError(f"no WARD gold cases found under {root / 'cases'}")
        self.cases = [read_json(path) for path in self.case_paths]
        self.case_by_id: dict[str, dict[str, Any]] = {}
        self.checkpoint_by_id: dict[str, GoldCheckpoint] = {}
        for case in self.cases:
            case_id = case.get("case_id")
            if case_id in self.case_by_id:
                raise ValueError(f"duplicate gold case_id: {case_id}")
            self.case_by_id[case_id] = case
            for checkpoint in case.get("checkpoints", []):
                checkpoint_id = checkpoint.get("checkpoint_id")
                if checkpoint_id in self.checkpoint_by_id:
                    raise ValueError(f"duplicate gold checkpoint_id: {checkpoint_id}")
                self.checkpoint_by_id[checkpoint_id] = GoldCheckpoint(
                    case_id=case_id,
                    checkpoint_id=checkpoint_id,
                    case=case,
                    checkpoint=checkpoint,
                )

    def profile(self) -> dict[str, Any]:
        checkpoints = [cp for case in self.cases for cp in case["checkpoints"]]
        return {
            "case_count": len(self.cases),
            "checkpoint_count": len(checkpoints),
            "evidence_count": sum(len(case["normalized_evidence"]) for case in self.cases),
            "claim_count": sum(len(cp["answer"]["layer3_claims"]) for cp in checkpoints),
            "state_item_count": sum(_state_count(cp["answer"]["layer4_state"]) for cp in checkpoints),
            "action_count": sum(len(cp["answer"]["layer5_actions"]) for cp in checkpoints),
            "transition_count": sum(len(case["process_gold"]["transitions"]) for case in self.cases),
            "case_file_sha256": {path.name: sha256_file(path) for path in self.case_paths},
        }

    def text_audit(self) -> dict[str, Any]:
        missing: list[dict[str, str]] = []
        counts: dict[str, dict[str, int]] = {}

        def check(category: str, path: str, value: Any) -> None:
            bucket = counts.setdefault(category, {"total": 0, "scoreable": 0, "missing": 0})
            bucket["total"] += 1
            if is_scoreable_text(value):
                bucket["scoreable"] += 1
            else:
                bucket["missing"] += 1
                reason = "null_or_empty" if value is None or value == "" else "contains_cjk_or_non_string"
                missing.append({"category": category, "path": path, "reason": reason})

        for case in self.cases:
            case_id = case["case_id"]
            for cp in case["checkpoints"]:
                cp_id = cp["checkpoint_id"]
                base = f"cases/{case_id}/checkpoints/{cp_id}/answer"
                answer = cp["answer"]
                for i, claim in enumerate(answer["layer3_claims"]):
                    check("layer3_claim", f"{base}/layer3_claims/{i}/canonical_en", claim.get("canonical_en"))
                state = answer["layer4_state"]
                for bucket in STATE_BUCKETS:
                    for i, item in enumerate(state[bucket]):
                        check("layer4_state", f"{base}/layer4_state/{bucket}/{i}/canonical_en", item.get("canonical_en"))
                for bucket in SINGLE_STATE_BUCKETS:
                    item = state.get(bucket)
                    if item is not None:
                        check("layer4_state", f"{base}/layer4_state/{bucket}/canonical_en", item.get("canonical_en"))
                for i, action in enumerate(answer["layer5_actions"]):
                    check("layer5_action", f"{base}/layer5_actions/{i}/canonical_action_en", action.get("canonical_action_en"))
                    check("layer5_problem", f"{base}/layer5_actions/{i}/problem_en", action.get("problem_en", action.get("problem")))
                    check("layer5_indication", f"{base}/layer5_actions/{i}/indication_en", action.get("indication_en", action.get("indication")))
                    check("layer5_target", f"{base}/layer5_actions/{i}/target_and_laterality_en", action.get("target_and_laterality_en", action.get("target_and_laterality")))
                    check("layer5_deadline", f"{base}/layer5_actions/{i}/deadline_or_time_window_en", action.get("deadline_or_time_window_en", action.get("deadline_or_time_window")))
                    prerequisites = action.get("prerequisites_en", action.get("prerequisites", [])) or []
                    for j, text in enumerate(prerequisites):
                        check("layer5_prerequisite", f"{base}/layer5_actions/{i}/prerequisites_en/{j}", text)
                for field in ("hard_failure_rules", "critical_omission_rules", "preliminary_risk_triggers"):
                    values = answer["safety"].get(field + "_en", answer["safety"].get(field, [])) or []
                    for i, text in enumerate(values):
                        check("safety", f"{base}/safety/{field}_en/{i}", text)
                stopping = answer["stopping"]
                check("stopping_rationale", f"{base}/stopping/rationale_en", stopping.get("rationale_en"))
                blockers = stopping.get("remaining_blockers_en", stopping.get("remaining_blockers", [])) or []
                for i, text in enumerate(blockers):
                    check("stopping_blocker", f"{base}/stopping/remaining_blockers_en/{i}", text)
            for i, milestone in enumerate(case["milestone_candidates"]):
                for field in ("completion_criteria", "partial_criteria", "safety_invalidation_rules"):
                    values = milestone.get(field + "_en", milestone.get(field, [])) or []
                    for j, text in enumerate(values):
                        check("milestone_criteria", f"cases/{case_id}/milestone_candidates/{i}/{field}_en/{j}", text)

        return {
            "status": "passed" if not missing else "blocked",
            "formal_text_ready": not missing,
            "counts": counts,
            "missing_count": len(missing),
            "missing": missing,
        }

    def project_checkpoint_en(self, gold_cp: GoldCheckpoint) -> dict[str, Any]:
        return self._project_answer_en(gold_cp, gold_cp.checkpoint["answer"])

    def project_model_preannotation_en(self, gold_cp: GoldCheckpoint) -> dict[str, Any]:
        return self._project_answer_en(gold_cp, gold_cp.checkpoint["model_preannotation"]["answer"])

    def _project_answer_en(self, gold_cp: GoldCheckpoint, answer: dict[str, Any]) -> dict[str, Any]:
        missing: list[str] = []

        claims = []
        for item in answer["layer3_claims"]:
            text = item.get("canonical_en")
            if not is_scoreable_text(text):
                missing.append(f"layer3_claims/{item.get('claim_id')}/canonical_en")
                continue
            claims.append({
                "id": item["claim_id"],
                "text": text,
                "aliases": [alias for alias in item.get("acceptable_aliases_en", []) if is_scoreable_text(alias)],
                "status": item.get("status"),
                "evidence_ids": sorted(set(item.get("linked_evidence_ids", [])) | set(item.get("supporting_evidence_ids", []))),
            })

        state_projection: dict[str, Any] = {}
        state = answer["layer4_state"]
        for bucket in STATE_BUCKETS:
            projected = []
            for item in state[bucket]:
                text = item.get("canonical_en")
                if not is_scoreable_text(text):
                    missing.append(f"layer4_state/{bucket}/{item.get('state_id')}/canonical_en")
                    continue
                projected.append(_project_state(item, text))
            state_projection[bucket] = projected
        for bucket in SINGLE_STATE_BUCKETS:
            item = state.get(bucket)
            if item is None:
                state_projection[bucket] = None
            elif is_scoreable_text(item.get("canonical_en")):
                state_projection[bucket] = _project_state(item, item["canonical_en"])
            else:
                missing.append(f"layer4_state/{bucket}/{item.get('state_id')}/canonical_en")
                state_projection[bucket] = None

        actions = []
        for item in answer["layer5_actions"]:
            text = ACTION_EN_OVERRIDES.get(
                item["action_id"], item.get("canonical_action_en")
            )
            if not is_scoreable_text(text):
                missing.append(f"layer5_actions/{item.get('action_id')}/canonical_action_en")
                continue
            actions.append({
                "id": item["action_id"],
                "text": text,
                "aliases": [alias for alias in item.get("acceptable_aliases_en", []) if is_scoreable_text(alias)],
                "category": item.get("category"),
                "action_scope": item.get("action_scope"),
                "evidence_ids": sorted(set(item.get("linked_evidence_ids", []))),
                "critical_omission_if_missing": bool(item.get("critical_omission_if_missing")),
                "prohibited_action_en": PROHIBITED_ACTION_EN_OVERRIDES.get(
                    item["action_id"], item.get("canonical_prohibited_action_en")
                ),
            })

        safety_projection: dict[str, list[str]] = {}
        for field in ("hard_failure_rules", "critical_omission_rules", "preliminary_risk_triggers"):
            projected = []
            values = answer["safety"].get(field + "_en", answer["safety"].get(field, [])) or []
            for index, value in enumerate(values):
                if is_scoreable_text(value):
                    projected.append(value)
                else:
                    missing.append(f"safety/{field}/{index}/text")
            safety_projection[field] = projected

        blockers = []
        source_blockers = answer["stopping"].get("remaining_blockers_en", answer["stopping"].get("remaining_blockers", [])) or []
        for index, value in enumerate(source_blockers):
            if is_scoreable_text(value):
                blockers.append(value)
            else:
                missing.append(f"stopping/remaining_blockers/{index}/text")

        valid_claim_ids = {item["id"] for item in claims}
        valid_state_ids = {
            item["id"]
            for bucket in (*STATE_BUCKETS, *SINGLE_STATE_BUCKETS)
            for item in (
                state_projection[bucket]
                if isinstance(state_projection[bucket], list)
                else ([state_projection[bucket]] if state_projection[bucket] is not None else [])
            )
        }
        valid_action_ids = {item["id"] for item in actions}
        layer2_projection = []
        for link in answer["layer2_utilization_links"]:
            layer2_projection.append({
                "evidence_id": link["evidence_id"],
                "claim_ids": [value for value in (link.get("claim_ids") or []) if value in valid_claim_ids],
                "state_ids": [value for value in (link.get("state_ids") or []) if value in valid_state_ids],
                "action_ids": [value for value in (link.get("action_ids") or []) if value in valid_action_ids],
                "expected_outcome": link.get("expected_outcome"),
            })

        return {
            "case_id": gold_cp.case_id,
            "checkpoint_id": gold_cp.checkpoint_id,
            "snapshot_sha256": gold_cp.snapshot_sha256,
            "allowed_evidence_ids": sorted(gold_cp.allowed_evidence_ids),
            "future_evidence_ids": sorted(gold_cp.future_evidence_ids),
            "layer1": [
                {
                    "evidence_id": item["evidence_id"],
                    "role": item["role"],
                    "interpretation_en": item.get("interpretation_en"),
                }
                for item in answer["layer1_evidence"]
            ],
            "layer2": layer2_projection,
            "layer3": claims,
            "layer4": state_projection,
            "layer5": actions,
            "safety": safety_projection,
            "stopping": {
                "decision": answer["stopping"]["decision"],
                "remaining_blockers": blockers,
                "rationale_en": answer["stopping"].get("rationale_en"),
            },
            "missing_text": missing,
        }


def _project_state(item: dict[str, Any], text: str) -> dict[str, Any]:
    return {
        "id": item["state_id"],
        "text": text,
        "aliases": [],
        "status": item.get("status"),
        "evidence_ids": sorted(set(item.get("linked_evidence_ids", [])) | set(item.get("supporting_evidence_ids", []))),
    }


def _state_count(state: dict[str, Any]) -> int:
    count = sum(len(state[bucket]) for bucket in STATE_BUCKETS)
    return count + sum(1 for bucket in SINGLE_STATE_BUCKETS if state.get(bucket) is not None)
