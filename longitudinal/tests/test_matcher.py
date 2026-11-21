from __future__ import annotations

import tempfile

import unittest

from pathlib import Path

from ward_eval.gold import GoldPackage

from ward_eval.judge import FrozenJudge

from ward_eval.matcher import semantic_match

from ward_eval.prompt_audit import audit_prompt_manifest

from ward_eval.util import canonical_sha256, sha256_text, write_json

class JudgeContractTests(unittest.TestCase):
    def _judge(self, verdict: str):
        key_payload = {
            "dimension": "layer3_claims",
            "prediction": "The lesion is benign.",
            "gold": "The lesion is malignant.",
            "aliases": [],
            "checkpoint_id": "ward_001_cp_000",
        }
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "judge.json"
        write_json(path, {"decisions": {canonical_sha256(key_payload): {"verdict": verdict}}})
        return directory, FrozenJudge(path)

    def test_partial_is_one_fp_and_one_fn_without_half_credit(self):
        directory, judge = self._judge("partial")
        self.addCleanup(directory.cleanup)
        result = semantic_match(
            [{"id": "pred_1", "text": "The lesion is benign."}],
            [{"id": "gold_1", "text": "The lesion is malignant.", "aliases": []}],
            dimension="layer3_claims", checkpoint_id="ward_001_cp_000", judge=judge,
        )
        self.assertEqual((result.tp, result.fp, result.fn), (0, 1, 1))
        self.assertEqual(len(result.partial_pairs), 1)

    def test_contradiction_is_one_fp_one_fn_and_guardrail_event(self):
        directory, judge = self._judge("contradiction")
        self.addCleanup(directory.cleanup)
        result = semantic_match(
            [{"id": "pred_1", "text": "The lesion is benign."}],
            [{"id": "gold_1", "text": "The lesion is malignant.", "aliases": []}],
            dimension="layer3_claims", checkpoint_id="ward_001_cp_000", judge=judge,
        )
        self.assertEqual((result.tp, result.fp, result.fn), (0, 1, 1))
        self.assertEqual(len(result.contradiction_pairs), 1)

    def test_duplicate_exact_strings_keep_stable_pair_order(self):
        result = semantic_match(
            [
                {"id": "pred_1", "text": "Same state."},
                {"id": "pred_2", "text": "Same state."},
            ],
            [
                {"id": "gold_1", "text": "Same state.", "aliases": []},
                {"id": "gold_2", "text": "Same state.", "aliases": []},
            ],
            dimension="layer4_state.active_problems",
            checkpoint_id="ward_001_cp_000",
            judge=FrozenJudge(),
        )
        self.assertEqual(result.mapping, {"pred_1": "gold_1", "pred_2": "gold_2"})
