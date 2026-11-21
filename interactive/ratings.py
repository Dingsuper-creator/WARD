"""Validate independent clinical ratings without generating judgments."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def apply_ratings(rows: list[dict], path: Path) -> None:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            ratings = list(csv.DictReader(handle))
    else:
        ratings = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(ratings, dict):
            ratings = ratings.get("ratings")
    if not isinstance(ratings, list):
        raise ValueError("clinical ratings must be a list or a ratings object")
    assigned = set()
    values = []
    for index, rating in enumerate(ratings):
        if not isinstance(rating, dict):
            raise ValueError(f"clinical rating {index} must be an object")
        run_id = rating.get("run_id")
        model, case = rating.get("model_id"), rating.get("case_id")
        if not run_id and not (model and case):
            raise ValueError("clinical ratings require run_id or both model_id and case_id")
        matches = [row for row in rows if (not run_id or row["run_id"] == run_id)
                   and (not model or row["model_id"] == model) and (not case or row["case_id"] == case)]
        if len(matches) != 1:
            raise ValueError("clinical rating must match exactly one selected run")
        matched = matches[0]
        identity = (matched["model_id"], matched["case_id"])
        if identity in assigned:
            raise ValueError("duplicate clinical rating for a selected trajectory")
        assigned.add(identity)
        raw = rating.get("clinical_quality", rating.get("overall_clinical_quality_1_to_7"))
        if isinstance(raw, bool) or raw is None or raw == "":
            raise ValueError("clinical quality must be a number from 1 through 7")
        value = float(raw)
        if not math.isfinite(value) or not 1 <= value <= 7:
            raise ValueError("clinical quality must be a number from 1 through 7")
        values.append((matched, value))
    # Apply only after all keys and scores have been validated.
    for matched, value in values:
        matched["clinical_quality"] = value
        matched["clinical_quality_source"] = "independent ratings supplied by evaluator"
