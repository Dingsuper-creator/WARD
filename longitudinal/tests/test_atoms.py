from __future__ import annotations

import unittest

from ward_eval.longitudinal_atoms import stable_atom_id, synthetic_transition_labels

class StableAtomIdTests(unittest.TestCase):
    def test_atom_id_is_deterministic_and_dimension_scoped(self) -> None:
        first = stable_atom_id("ward_001", "state", "same text")
        self.assertEqual(first, stable_atom_id("ward_001", "state", "same text"))
        self.assertNotEqual(first, stable_atom_id("ward_001", "claim", "same text"))

class TransitionSemanticsTests(unittest.TestCase):
    def test_immediate_correction(self) -> None:
        result = synthetic_transition_labels([0, 1])
        self.assertEqual(result["episodes"][0]["correction_lag"], 1)
        self.assertEqual(result["episodes"][0]["immediate_correction"], 1)

    def test_delayed_correction(self) -> None:
        result = synthetic_transition_labels([0, 0, 1])
        self.assertEqual(result["episodes"][0]["correction_lag"], 2)
        self.assertEqual(result["episodes"][0]["delayed_correction"], 1)

    def test_persistent_error(self) -> None:
        result = synthetic_transition_labels([0, 0, 0])
        self.assertEqual(result["episodes"][0]["persistent_error"], 1)
        self.assertEqual(result["episodes"][0]["right_censored"], 1)

    def test_regression(self) -> None:
        result = synthetic_transition_labels([1, 0])
        self.assertEqual(result["regression_count"], 1)
        self.assertEqual(result["episodes"][0]["initial_deficit"], 0)

    def test_regression_then_recovery_is_oscillation(self) -> None:
        result = synthetic_transition_labels([1, 0, 1])
        self.assertEqual(result["regression_count"], 1)
        self.assertTrue(result["oscillation"])
        self.assertEqual(result["episodes"][0]["corrected"], 1)

    def test_correction_then_regression_is_oscillation(self) -> None:
        result = synthetic_transition_labels([0, 1, 0])
        self.assertTrue(result["oscillation"])
        self.assertEqual(len(result["episodes"]), 2)
        self.assertEqual(result["episodes"][0]["corrected"], 1)
        self.assertEqual(result["episodes"][1]["persistent_error"], 1)

    def test_unscorable_breaks_immediacy(self) -> None:
        result = synthetic_transition_labels([0, None, 1])
        self.assertEqual(result["episodes"][0]["correction_lag"], 2)
        self.assertEqual(result["episodes"][0]["delayed_correction"], 1)
