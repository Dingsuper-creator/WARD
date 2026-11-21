"""Separate event-relative time from surgery-specific clinical phase."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from world_model_patient_ehr_v01.contracts import (
    InteractionContractError,
    credential_findings,
    read_json,
    sha256_file,
)
from world_model_patient_ehr_v02.ontology import normalize
from world_model_patient_ehr_v053.lifecycle import (
    build_lifecycle_action_envelope,
)


EVENT_RELATIVE_PHASE_POLICY_VERSION = (
    "ward-patient-ehr-event-relative-phase-policy-0.1.0"
)
EVENT_PHASE_FRAME_VERSION = (
    "ward-patient-ehr-event-phase-frame-0.1.0"
)
SURGICAL_PHASES = ("preoperative", "operative", "postoperative")


def validate_event_phase_policy(policy: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if policy.get("policy_version") != EVENT_RELATIVE_PHASE_POLICY_VERSION:
        errors.append("wrong_event_phase_policy_version")
    if credential_findings(policy):
        errors.append("credential_like_material_present")
    for key in (
        "gold_action_visible",
        "dependency_graph_visible",
        "future_evidence_visible",
        "automatic_semantic_judge_scoring",
        "official_scoring_allowed",
    ):
        if policy.get(key) is not False:
            errors.append(f"invalid_boundary:{key}")
    if policy.get("single_total_score_absent") is not True:
        errors.append("single_total_score_boundary_invalid")
    if policy.get("surgical_phase_mapping") != {
        "before": "preoperative",
        "during": "operative",
        "after": "postoperative",
    }:
        errors.append("invalid_surgical_phase_mapping")
    anchor_patterns = policy.get("surgical_temporal_anchor_patterns")
    structured_patterns = policy.get(
        "canonical_structured_phase_patterns"
    )
    for namespace, groups, labels in (
        (
            "surgical_anchor",
            anchor_patterns,
            {"before", "during", "after"},
        ),
        (
            "structured_phase",
            structured_patterns,
            set(SURGICAL_PHASES),
        ),
    ):
        if not isinstance(groups, dict) or set(groups) != labels:
            errors.append(f"invalid_{namespace}_groups")
            continue
        for label, patterns in groups.items():
            if not isinstance(patterns, list) or not patterns:
                errors.append(f"invalid_{namespace}_patterns:{label}")
                continue
            for pattern in patterns:
                try:
                    re.compile(pattern, flags=re.I)
                except (re.error, TypeError):
                    errors.append(f"invalid_{namespace}_regex:{label}")
    inference = policy.get("inference_policy", {})
    for key, expected in {
        "generic_before_after_during_implies_surgical_phase": False,
        "surgery_anchor_required_for_text_derived_surgical_phase": True,
        "canonical_structured_surgical_phase_allowed": True,
        "post_biopsy_is_postoperative": False,
        "phase_conflicts_audited": True,
        "phase_conflict_automatically_overrides_structured_field": False,
        "background_fields_used_for_phase_inference": False,
        "llm_semantic_judge_used": False,
    }.items():
        if inference.get(key) is not expected:
            errors.append(f"invalid_inference_policy:{key}")
    return errors


def load_event_phase_policy(
    path: Path,
) -> tuple[dict[str, Any], str]:
    policy = read_json(path)
    errors = validate_event_phase_policy(policy)
    if errors:
        raise InteractionContractError(
            "invalid event-relative phase policy: " + "; ".join(errors)
        )
    return policy, sha256_file(path)


def _matched_labels(
    text: str,
    groups: dict[str, list[str]],
) -> list[str]:
    return sorted(
        label
        for label, patterns in groups.items()
        if any(re.search(pattern, text, flags=re.I) for pattern in patterns)
    )


def _canonical_structured_phases(
    value: str,
    policy: dict[str, Any],
) -> set[str]:
    normalized = normalize(value)
    return {
        phase
        for phase, patterns in policy[
            "canonical_structured_phase_patterns"
        ].items()
        if any(
            re.search(pattern, normalized, flags=re.I)
            for pattern in patterns
        )
    }


def build_event_phase_action_envelope(
    action: dict[str, Any],
    lifecycle_policy: dict[str, Any],
    event_phase_policy: dict[str, Any],
) -> dict[str, Any]:
    """Correct only surgery-phase inference; preserve event lifecycle data."""

    envelope = build_lifecycle_action_envelope(
        action,
        lifecycle_policy,
    )
    values = envelope["command_values"]
    query = normalize(str(values.get("query", "")))
    structured_value = str(values.get("clinical_phase", ""))
    canonical_structured = _canonical_structured_phases(
        structured_value,
        event_phase_policy,
    )
    surgical_relations = _matched_labels(
        query,
        event_phase_policy["surgical_temporal_anchor_patterns"],
    )
    text_surgical_phases = {
        event_phase_policy["surgical_phase_mapping"][relation]
        for relation in surgical_relations
    }

    command_features = set(envelope["command_features"])
    sources = copy.deepcopy(envelope["command_feature_sources"])
    for phase in SURGICAL_PHASES:
        retained_sources = [
            source
            for source in sources.get(phase, [])
            if source not in {"event_frame", "clinical_phase"}
        ]
        if phase in canonical_structured:
            retained_sources.append("clinical_phase")
        if phase in text_surgical_phases:
            retained_sources.append("surgery_event_frame")

        # Query lexicon matches such phrases as post-biopsy. Query-derived
        # surgical phase is retained only with an explicit surgery anchor.
        if "query" in retained_sources and phase not in text_surgical_phases:
            retained_sources.remove("query")
        retained_sources = sorted(set(retained_sources))
        if retained_sources:
            sources[phase] = retained_sources
            command_features.add(phase)
        else:
            sources.pop(phase, None)
            command_features.discard(phase)

    structured_phases = sorted(canonical_structured)
    text_phases = sorted(text_surgical_phases)
    phase_conflict = bool(
        structured_phases
        and text_phases
        and set(structured_phases).isdisjoint(text_phases)
    )
    event_frame = copy.deepcopy(envelope["event_frame"])
    event_frame.update(
        {
            "event_phase_frame_version": EVENT_PHASE_FRAME_VERSION,
            "event_relative_phase_policy_version": (
                EVENT_RELATIVE_PHASE_POLICY_VERSION
            ),
            "generic_temporal_relations": list(
                event_frame["temporal_relations"]
            ),
            "surgical_temporal_relations": surgical_relations,
            "text_derived_phases": text_phases,
            "structured_phases": structured_phases,
            "phase_conflict": phase_conflict,
            "event_relative_phase_decoupled": True,
            "post_biopsy_is_postoperative": False,
            "background_fields_used_for_phase_inference": False,
        }
    )
    envelope["command_features"] = sorted(command_features)
    envelope["command_feature_sources"] = {
        key: sorted(set(value))
        for key, value in sorted(sources.items())
    }
    envelope["event_frame"] = event_frame
    return envelope


__all__ = [
    "EVENT_PHASE_FRAME_VERSION",
    "EVENT_RELATIVE_PHASE_POLICY_VERSION",
    "build_event_phase_action_envelope",
    "load_event_phase_policy",
    "validate_event_phase_policy",
]
