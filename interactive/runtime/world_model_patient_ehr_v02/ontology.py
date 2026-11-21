"""Deterministic, extensible semantic resolution for free-text clinical actions."""

from __future__ import annotations

import copy
import difflib
import re
import unicodedata
from typing import Any

from world_model_patient_ehr_v01.contracts import InteractionContractError


ONTOLOGY_VERSION = "ward-patient-ehr-action-ontology-0.2.0"
RESOLVER_VERSION = "ward-patient-ehr-action-resolver-0.2.0"


def normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = re.sub(r"[_/\\-]+", " ", text)
    text = re.sub(r"[^\w\u3400-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains(normalized_target: str, candidate: str) -> bool:
    normalized_candidate = normalize(candidate)
    return bool(normalized_candidate) and normalized_candidate in normalized_target


def validate_ontology(ontology: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if ontology.get("ontology_version") != ONTOLOGY_VERSION:
        errors.append("wrong_ontology_version")
    concepts = ontology.get("concepts")
    if not isinstance(concepts, dict) or not concepts:
        return [*errors, "concepts_must_be_nonempty_object"]
    for concept_id, concept in concepts.items():
        if not isinstance(concept_id, str) or not concept_id:
            errors.append("invalid_concept_id")
            continue
        if not isinstance(concept, dict):
            errors.append(f"invalid_concept:{concept_id}")
            continue
        action_types = concept.get("action_types")
        if (
            not isinstance(action_types, list)
            or not action_types
            or any(not isinstance(item, str) or not item for item in action_types)
        ):
            errors.append(f"invalid_action_types:{concept_id}")
        phrases = concept.get("phrases")
        if (
            not isinstance(phrases, list)
            or not phrases
            or any(not isinstance(item, str) or not item for item in phrases)
        ):
            errors.append(f"invalid_phrases:{concept_id}")
        groups = concept.get("anchor_groups")
        if (
            not isinstance(groups, list)
            or not groups
            or any(
                not isinstance(group, list)
                or not group
                or any(not isinstance(item, str) or not item for item in group)
                for group in groups
            )
        ):
            errors.append(f"invalid_anchor_groups:{concept_id}")
            groups = []
        minimum = concept.get("minimum_anchor_groups")
        if (
            not isinstance(minimum, int)
            or isinstance(minimum, bool)
            or minimum < 1
            or minimum > len(groups)
        ):
            errors.append(f"invalid_minimum_anchor_groups:{concept_id}")
        threshold = concept.get("similarity_threshold", 0.72)
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not 0 <= float(threshold) <= 1
        ):
            errors.append(f"invalid_similarity_threshold:{concept_id}")
    return errors


def resolve_against_concept(
    action: dict[str, Any],
    concept_id: str,
    ontology: dict[str, Any],
) -> dict[str, Any]:
    concepts = ontology["concepts"]
    if concept_id not in concepts:
        raise InteractionContractError(f"unknown action concept: {concept_id}")
    concept = concepts[concept_id]
    action_type = str(action.get("action_type", ""))
    target = normalize(str(action.get("target", "")))
    action_type_match = action_type in concept["action_types"]
    normalized_phrases = [normalize(item) for item in concept["phrases"]]
    exact_phrase_match = target in normalized_phrases
    similarity = max(
        (
            difflib.SequenceMatcher(None, target, phrase).ratio()
            for phrase in normalized_phrases
        ),
        default=0.0,
    )
    anchor_hits = [
        index
        for index, group in enumerate(concept["anchor_groups"])
        if any(_contains(target, candidate) for candidate in group)
    ]
    anchor_match = len(anchor_hits) >= concept["minimum_anchor_groups"]
    similarity_match = similarity >= float(
        concept.get("similarity_threshold", 0.72)
    )
    equivalent = action_type_match and (
        exact_phrase_match or anchor_match or similarity_match
    )
    if exact_phrase_match:
        match_mode = "controlled_phrase"
    elif anchor_match:
        match_mode = "semantic_anchor"
    elif similarity_match:
        match_mode = "lexical_similarity"
    elif action_type_match:
        match_mode = "action_type_only"
    else:
        match_mode = "unresolved"
    return {
        "resolver_version": RESOLVER_VERSION,
        "concept_id": concept_id,
        "equivalent": equivalent,
        "action_type_match": action_type_match,
        "match_mode": match_mode,
        "anchor_group_hits": anchor_hits,
        "anchor_group_count": len(concept["anchor_groups"]),
        "similarity": round(similarity, 6),
    }


def resolve_all_concepts(
    action: dict[str, Any],
    ontology: dict[str, Any],
) -> list[dict[str, Any]]:
    matches = [
        resolve_against_concept(action, concept_id, ontology)
        for concept_id in ontology["concepts"]
    ]
    matches = [item for item in matches if item["equivalent"]]
    return sorted(
        (copy.deepcopy(item) for item in matches),
        key=lambda item: (
            item["match_mode"] == "controlled_phrase",
            len(item["anchor_group_hits"]),
            item["similarity"],
            item["concept_id"],
        ),
        reverse=True,
    )

