from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .util import canonical_sha256, normalize_text, read_json


VERDICTS = {"equivalent", "partial", "contradiction", "unrelated", "abstain"}


@dataclass(frozen=True)
class JudgeDecision:
    verdict: str
    source: str
    cache_key: str
    confidence: float | None = None


class FrozenJudge:
    """Deterministic exact/alias matcher plus an optional frozen decision cache.

    This evaluator never calls a remote model. A separate audited job may fill
    the cache, whose content is then frozen and hash-bound to the score run.
    """

    def __init__(self, cache_path: Path | None = None):
        self.cache_path = cache_path
        self.cache: dict[str, dict[str, Any]] = {}
        if cache_path is not None:
            payload = read_json(cache_path)
            version = payload.get("schema_version") if isinstance(payload, dict) else None
            if version not in (None, "ward-judge-cache-1.0"):
                raise ValueError(f"unsupported Judge cache schema_version: {version}")
            self.cache = payload.get("decisions", payload)
        self.trace: list[dict[str, Any]] = []

    def decide(
        self,
        *,
        dimension: str,
        prediction: str,
        gold: str,
        aliases: list[str] | None = None,
        checkpoint_id: str,
    ) -> JudgeDecision:
        aliases = aliases or []
        key_payload = {
            "dimension": dimension,
            "prediction": prediction,
            "gold": gold,
            "aliases": aliases,
            "checkpoint_id": checkpoint_id,
        }
        cache_key = canonical_sha256(key_payload)
        normalized_prediction = normalize_text(prediction)
        accepted = {normalize_text(gold), *(normalize_text(alias) for alias in aliases)}
        if normalized_prediction and normalized_prediction in accepted:
            decision = JudgeDecision("equivalent", "exact_or_alias", cache_key, 1.0)
        elif cache_key in self.cache:
            cached = self.cache[cache_key]
            verdict = cached.get("verdict")
            if verdict not in VERDICTS:
                raise ValueError(f"invalid cached Judge verdict for {cache_key}: {verdict}")
            decision = JudgeDecision(
                verdict=verdict,
                source="frozen_cache",
                cache_key=cache_key,
                confidence=cached.get("confidence"),
            )
        else:
            decision = JudgeDecision("abstain", "cache_miss", cache_key, None)
        provenance = self.cache.get(cache_key, {}) if decision.source == "frozen_cache" else {}
        self.trace.append({
            **key_payload,
            **decision.__dict__,
            "decision_source": provenance.get("source", decision.source),
            "semantic_pair_id": provenance.get("item_id"),
            "reason_codes": provenance.get("reason_codes", []),
        })
        return decision


def judge_cache_sha256(judge: FrozenJudge) -> str | None:
    if judge.cache_path is None:
        return None
    from .util import sha256_file

    return sha256_file(judge.cache_path)
