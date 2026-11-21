from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .judge import FrozenJudge
from .util import normalize_text


@dataclass
class SemanticMatchResult:
    tp: int
    fp: int
    fn: int
    mapping: dict[str, str]
    equivalent_pairs: list[dict[str, Any]]
    partial_pairs: list[dict[str, Any]]
    contradiction_pairs: list[dict[str, Any]]
    abstain_pair_count: int


def semantic_match(
    predictions: list[dict[str, Any]],
    gold_items: list[dict[str, Any]],
    *,
    dimension: str,
    checkpoint_id: str,
    judge: FrozenJudge,
) -> SemanticMatchResult:
    exact_edges, gold_to_prediction = _exact_matches(predictions, gold_items)
    initially_matched_predictions = set(gold_to_prediction.values())
    initially_matched_gold = set(gold_to_prediction)

    decisions: dict[tuple[int, int], str] = {}
    remaining_after_exact_predictions = [
        pi for pi in range(len(predictions)) if pi not in initially_matched_predictions
    ]
    remaining_after_exact_gold = [
        gi for gi in range(len(gold_items)) if gi not in initially_matched_gold
    ]
    for pi in remaining_after_exact_predictions:
        prediction = predictions[pi]
        for gi in remaining_after_exact_gold:
            gold = gold_items[gi]
            decision = judge.decide(
                dimension=dimension,
                prediction=prediction["text"],
                gold=gold["text"],
                aliases=gold.get("aliases", []),
                checkpoint_id=checkpoint_id,
            )
            decisions[(pi, gi)] = decision.verdict

    cached_equivalent_edges = {
        pi: [gi for gi in remaining_after_exact_gold if decisions.get((pi, gi)) == "equivalent"]
        for pi in remaining_after_exact_predictions
    }
    cached_matches = _maximum_matching(cached_equivalent_edges, remaining_after_exact_predictions)
    gold_to_prediction.update(cached_matches)

    equivalent_pairs: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    matched_predictions: set[int] = set()
    matched_gold: set[int] = set()
    for gi, pi in sorted(gold_to_prediction.items()):
        matched_predictions.add(pi)
        matched_gold.add(gi)
        prediction = predictions[pi]
        gold = gold_items[gi]
        mapping[prediction["id"]] = gold["id"]
        equivalent_pairs.append({
            "prediction_id": prediction["id"],
            "gold_id": gold["id"],
            "prediction_text": prediction["text"],
            "gold_text": gold["text"],
        })

    remaining_predictions = [i for i in range(len(predictions)) if i not in matched_predictions]
    remaining_gold = [i for i in range(len(gold_items)) if i not in matched_gold]
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    partial_pairs: list[dict[str, Any]] = []
    contradiction_pairs: list[dict[str, Any]] = []
    for verdict, output in (("contradiction", contradiction_pairs), ("partial", partial_pairs)):
        for pi in remaining_predictions:
            if pi in used_pred:
                continue
            candidate = next(
                (gi for gi in remaining_gold if gi not in used_gold and decisions.get((pi, gi)) == verdict),
                None,
            )
            if candidate is None:
                continue
            used_pred.add(pi)
            used_gold.add(candidate)
            output.append({
                "prediction_id": predictions[pi]["id"],
                "gold_id": gold_items[candidate]["id"],
                "prediction_text": predictions[pi]["text"],
                "gold_text": gold_items[candidate]["text"],
            })

    abstain_count = sum(
        1
        for pi in remaining_predictions
        for gi in remaining_gold
        if decisions.get((pi, gi)) == "abstain"
    )
    tp = len(equivalent_pairs)
    return SemanticMatchResult(
        tp=tp,
        fp=len(predictions) - tp,
        fn=len(gold_items) - tp,
        mapping=mapping,
        equivalent_pairs=equivalent_pairs,
        partial_pairs=partial_pairs,
        contradiction_pairs=contradiction_pairs,
        abstain_pair_count=abstain_count,
    )


def exact_match_partition(
    predictions: list[dict[str, Any]],
    gold_items: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    """Return stable exact/alias-unmatched indexes without calling a Judge."""
    _, gold_to_prediction = _exact_matches(predictions, gold_items)
    matched_predictions = set(gold_to_prediction.values())
    matched_gold = set(gold_to_prediction)
    return (
        [index for index in range(len(predictions)) if index not in matched_predictions],
        [index for index in range(len(gold_items)) if index not in matched_gold],
    )


def _exact_matches(
    predictions: list[dict[str, Any]],
    gold_items: list[dict[str, Any]],
) -> tuple[dict[int, list[int]], dict[int, int]]:
    exact_edges = {
        pi: [
            gi
            for gi, gold in enumerate(gold_items)
            if normalize_text(prediction["text"])
            in {normalize_text(gold["text"]), *(normalize_text(alias) for alias in gold.get("aliases", []))}
        ]
        for pi, prediction in enumerate(predictions)
    }
    return exact_edges, _maximum_matching(exact_edges, range(len(predictions)))


def _maximum_matching(edges: dict[int, list[int]], prediction_indexes) -> dict[int, int]:
    gold_to_prediction: dict[int, int] = {}

    def augment(prediction_index: int, seen: set[int]) -> bool:
        # Prefer an unused Gold item before displacing an earlier prediction.
        # This keeps duplicate exact strings aligned by stable input order while
        # retaining maximum-cardinality matching for overlapping alias sets.
        for gold_index in edges.get(prediction_index, []):
            if gold_index not in seen and gold_index not in gold_to_prediction:
                seen.add(gold_index)
                gold_to_prediction[gold_index] = prediction_index
                return True
        for gold_index in edges.get(prediction_index, []):
            if gold_index in seen:
                continue
            seen.add(gold_index)
            if augment(gold_to_prediction[gold_index], seen):
                gold_to_prediction[gold_index] = prediction_index
                return True
        return False

    for prediction_index in prediction_indexes:
        augment(prediction_index, set())
    return gold_to_prediction
