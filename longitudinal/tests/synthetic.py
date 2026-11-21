"""Small generated references with no patient or institutional source material."""
from __future__ import annotations
from pathlib import Path
from ward_eval.contracts import STATE_BUCKETS, SINGLE_STATE_BUCKETS
from ward_eval.gold import GoldPackage
from ward_eval.util import canonical_sha256, write_json


def make_gold(root: Path) -> GoldPackage:
    case_id = "ward_001"
    evidence_id = "ward_001_evidence_001"
    evidence = {
        "evidence_id": evidence_id, "record_kind": "document", "title": "Synthetic observation",
        "clinical_time": "2124-01-01T00:00:00", "available_time": "2124-01-01T00:00:00",
        "raw_content": "A generated measurement is available. This record represents no individual.",
    }
    checkpoints = []
    for index in range(3):
        cp_id = f"{case_id}_cp_{index:03d}"
        snapshot = {"public_time": f"2124-01-0{index + 1}T00:00:00",
                    "public_context": {"setting": "synthetic"}, "visible_evidence_ids": [evidence_id]}
        state = {**{bucket: [] for bucket in STATE_BUCKETS}, **{bucket: None for bucket in SINGLE_STATE_BUCKETS}}
        state_id = f"{cp_id}_state_001"
        state["required_delta_from_previous_checkpoint"] = [{
            "state_id": state_id, "canonical_en": "A synthetic response must be recorded.",
            "status": "confirmed", "linked_evidence_ids": [evidence_id],
        }]
        claim_id, action_id = f"{cp_id}_claim_001", f"{cp_id}_action_001"
        answer = {
            "layer1_evidence": [{"evidence_id": evidence_id, "role": "critical", "interpretation_en": "Generated evidence."}],
            "layer2_utilization_links": [{"evidence_id": evidence_id, "claim_ids": [claim_id], "state_ids": [state_id], "action_ids": [action_id], "expected_outcome": "Review complete."}],
            "layer3_claims": [{"claim_id": claim_id, "canonical_en": "The generated measurement is available.", "status": "confirmed", "linked_evidence_ids": [evidence_id]}],
            "layer4_state": state,
            "layer5_actions": [{"action_id": action_id, "canonical_action_en": "Review the generated measurement.", "category": "necessary", "action_scope": "prospective", "linked_evidence_ids": [evidence_id], "critical_omission_if_missing": True, "problem_en": "Synthetic task.", "indication_en": "Synthetic indication.", "target_and_laterality_en": "No laterality.", "deadline_or_time_window_en": "At this checkpoint."}],
            "safety": {"hard_failure_rules_en": ["Do not invent unavailable evidence."], "critical_omission_rules_en": [], "preliminary_risk_triggers_en": []},
            "stopping": {"decision": "continue_investigation" if index == 0 else "continue_treatment", "remaining_blockers_en": ["Synthetic follow-up remains."], "rationale_en": "The generated task remains open."},
        }
        checkpoints.append({"checkpoint_id": cp_id, "sequence_index": index,
            "question": {"public_snapshot": snapshot, "public_snapshot_sha256": canonical_sha256(snapshot), "allowed_evidence_ids": [evidence_id], "future_forbidden_evidence_ids": []}, "answer": answer})
    write_json(root / "cases" / f"{case_id}.gold.json", {"case_id": case_id,
        "normalized_evidence": [evidence], "checkpoints": checkpoints,
        "milestone_candidates": [], "process_gold": {"transitions": []}})
    return GoldPackage(root)
