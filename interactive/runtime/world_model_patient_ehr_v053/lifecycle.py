"""Case-independent clinical event and lifecycle extraction."""

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
from world_model_patient_ehr_v052.environment import build_action_envelope


LIFECYCLE_POLICY_VERSION = (
    "ward-patient-ehr-clinical-event-lifecycle-policy-0.1.0"
)
EVENT_FRAME_VERSION = (
    "ward-patient-ehr-clinical-event-frame-0.1.0"
)


def validate_lifecycle_policy(policy: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if policy.get("policy_version") != LIFECYCLE_POLICY_VERSION:
        errors.append("wrong_lifecycle_policy_version")
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
    relations = policy.get("temporal_relations")
    statuses = policy.get("event_statuses")
    phases = policy.get("phase_mapping")
    families = policy.get("event_feature_families")
    generic_terms = policy.get("generic_event_terms")
    if not isinstance(relations, dict) or set(relations) != {
        "before",
        "after",
        "during",
    }:
        errors.append("invalid_temporal_relations")
    if not isinstance(statuses, dict) or not statuses:
        errors.append("invalid_event_statuses")
    if not isinstance(phases, dict) or set(phases) != {
        "before",
        "after",
        "during",
    }:
        errors.append("invalid_phase_mapping")
    if not isinstance(families, dict) or not families:
        errors.append("invalid_event_feature_families")
    if not isinstance(generic_terms, dict) or set(generic_terms) != set(
        families or {}
    ):
        errors.append("invalid_generic_event_terms")
    for namespace, groups in (
        ("relation", relations),
        ("status", statuses),
        ("generic_event", generic_terms),
    ):
        if not isinstance(groups, dict):
            continue
        for label, patterns in groups.items():
            if (
                not isinstance(label, str)
                or not isinstance(patterns, list)
                or not patterns
            ):
                errors.append(f"invalid_{namespace}_patterns:{label}")
                continue
            for pattern in patterns:
                if not isinstance(pattern, str) or not pattern:
                    errors.append(f"invalid_{namespace}_pattern:{label}")
                    continue
                try:
                    re.compile(pattern, flags=re.I)
                except re.error:
                    errors.append(f"invalid_{namespace}_regex:{label}")
    inference = policy.get("inference_policy", {})
    for key, expected in {
        "temporal_relation_requires_referenced_event_feature": True,
        "after_relation_implies_referenced_event_completed": True,
        "structured_and_text_phase_conflicts_are_audited": True,
        "phase_conflict_automatically_overrides_structured_field": False,
        "unique_safe_state_candidate_required_for_advance": True,
        "ambiguous_candidates_advance": False,
        "background_fields_used_for_routing": False,
        "llm_semantic_judge_used": False,
    }.items():
        if inference.get(key) is not expected:
            errors.append(f"invalid_inference_policy:{key}")
    return errors


def load_lifecycle_policy(path: Path) -> tuple[dict[str, Any], str]:
    policy = read_json(path)
    errors = validate_lifecycle_policy(policy)
    if errors:
        raise InteractionContractError(
            "invalid clinical lifecycle policy: " + "; ".join(errors)
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


def build_lifecycle_action_envelope(
    action: dict[str, Any],
    policy: dict[str, Any],
) -> dict[str, Any]:
    """Add event, status, and temporal relations to the v0.5.2 envelope."""

    envelope = build_action_envelope(action)
    command_values = envelope["command_values"]
    query = normalize(str(command_values.get("query", "")))
    structured_phase = normalize(
        str(command_values.get("clinical_phase", ""))
    )
    command_features = set(envelope["command_features"])
    feature_sources = copy.deepcopy(envelope["command_feature_sources"])

    feature_to_family: dict[str, str] = {}
    for family, features in policy["event_feature_families"].items():
        for feature in features:
            feature_to_family[feature] = family
    referenced_feature_set = command_features & set(feature_to_family)
    generic_families = _matched_labels(
        query,
        policy["generic_event_terms"],
    )
    for family in generic_families:
        generic_feature = f"generic_{family}"
        if generic_feature in feature_to_family:
            referenced_feature_set.add(generic_feature)
            command_features.add(generic_feature)
            feature_sources.setdefault(generic_feature, []).append(
                "event_frame_generic_term"
            )
    referenced_features = sorted(referenced_feature_set)
    referenced_families = sorted(
        {feature_to_family[feature] for feature in referenced_features}
    )

    relations = _matched_labels(query, policy["temporal_relations"])
    if (
        policy["inference_policy"][
            "temporal_relation_requires_referenced_event_feature"
        ]
        and not referenced_features
    ):
        relations = []
    statuses = _matched_labels(query, policy["event_statuses"])
    if (
        "after" in relations
        and policy["inference_policy"][
            "after_relation_implies_referenced_event_completed"
        ]
        and not any(
            status in statuses
            for status in ("pending", "resulted", "completed", "performed")
        )
    ):
        statuses.append("completed")
    statuses = [
        item
        for item in policy["status_precedence"]
        if item in set(statuses)
    ]

    derived_phases = sorted(
        {
            policy["phase_mapping"][relation]
            for relation in relations
            if relation in policy["phase_mapping"]
        }
    )
    for phase in derived_phases:
        command_features.add(phase)
        feature_sources.setdefault(phase, []).append("event_frame")

    structured_phases = sorted(
        feature
        for feature in ("preoperative", "postoperative", "operative")
        if feature in command_features
        and "clinical_phase" in feature_sources.get(feature, [])
    )
    text_phases = sorted(set(derived_phases))
    phase_conflict = bool(
        structured_phases
        and text_phases
        and set(structured_phases).isdisjoint(text_phases)
    )
    primary_status = statuses[0] if statuses else None
    event_frame = {
        "event_frame_version": EVENT_FRAME_VERSION,
        "referenced_event_features": referenced_features,
        "referenced_event_families": referenced_families,
        "temporal_relations": relations,
        "event_status_candidates": statuses,
        "primary_event_status": primary_status,
        "referenced_event_completed_implied": (
            "after" in relations
            and "pending" not in statuses
            and policy["inference_policy"][
                "after_relation_implies_referenced_event_completed"
            ]
        ),
        "text_derived_phases": text_phases,
        "structured_phases": structured_phases,
        "phase_conflict": phase_conflict,
        "phase_conflict_automatically_overrides_structured_field": False,
        "background_fields_used_for_event_inference": False,
    }
    envelope["command_features"] = sorted(command_features)
    envelope["command_feature_sources"] = {
        key: sorted(set(value))
        for key, value in sorted(feature_sources.items())
    }
    envelope["event_frame"] = event_frame
    return envelope


__all__ = [
    "EVENT_FRAME_VERSION",
    "LIFECYCLE_POLICY_VERSION",
    "build_lifecycle_action_envelope",
    "load_lifecycle_policy",
    "validate_lifecycle_policy",
]
