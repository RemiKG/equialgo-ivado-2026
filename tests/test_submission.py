"""Behavioral checks for budget allocation, policy invariants and submission IDs."""
import itertools
import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from model_corrige import (ROOT, TARGET, RepairedScore, allocate, groups,
                          load_data, validate_data, verify_predictions)


class AllocationTests(unittest.TestCase):
    def test_optimizer_matches_exhaustive_binary_search(self):
        rng = np.random.default_rng(18)
        for _ in range(12):
            scores = rng.normal(size=10)
            group = np.array([0]*6 + [1]*4)
            pred = allocate(scores, group, np.array([f"C{i}" for i in range(10)]), max_gap=.25)
            best = -np.inf
            for chosen in itertools.combinations(range(10), 4):
                p = np.zeros(10, dtype=int)
                p[list(chosen)] = 1
                if abs(p[group == 0].mean() - p[group == 1].mean()) <= .25:
                    best = max(best, scores @ p)
            self.assertAlmostEqual(scores @ pred, best, places=12)
            self.assertEqual(pred.sum(), 4)

    def test_ties_and_decisions_are_invariant_to_row_order(self):
        group = np.array([0]*15 + [1]*10)
        ids = np.array([f"C{i:06d}" for i in range(25)])
        scores = np.ones(25)
        permutation = np.random.default_rng(42).permutation(25)
        p = allocate(scores, group, ids)
        shuffled = allocate(scores[permutation], group[permutation], ids[permutation])
        np.testing.assert_array_equal(p[permutation], shuffled)

    def test_infeasible_exact_parity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "infeasible"):
            allocate(np.arange(10), np.array([0]*7+[1]*3), np.arange(10), max_gap=0)

    def test_invalid_inputs_are_rejected(self):
        base = (np.arange(10, dtype=float), np.array([0]*5+[1]*5), np.arange(10))
        for rate in [.2, .5, 1.0]:
            with self.assertRaises(ValueError):
                allocate(*base, grant_rate=rate)
        with self.assertRaises(ValueError):
            allocate(np.full(10, np.nan), base[1], base[2])
        with self.assertRaises(ValueError):
            allocate(base[0], np.zeros(10), base[2])
        with self.assertRaises(ValueError):
            allocate(base[0], base[1], np.zeros(10))


class SubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history, cls.evaluation = load_data()
        cls.model = RepairedScore.fit(cls.history)
        cls.submission = pd.read_csv(ROOT / "predictions.csv")

    def test_exact_official_submission(self):
        report = verify_predictions(self.evaluation, self.submission)
        self.assertEqual(report["grants"], 1600)
        self.assertLessEqual(report["demographic_parity_gap"], .02)
        self.assertIsNone(report["reference_accuracy"])

    def test_submission_reproduces_from_supplied_data(self):
        p = allocate(self.model.score(self.evaluation), groups(self.evaluation), self.evaluation.id_candidat.to_numpy())
        np.testing.assert_array_equal(p, self.submission[TARGET].to_numpy())

    def test_single_row_and_subset_scoring_match_full_batch(self):
        scores = self.model.score(self.evaluation)
        for i in [0, 1, 43, 876, 3999]:
            self.assertAlmostEqual(self.model.score(self.evaluation.iloc[[i]])[0], scores[i], places=12)

    def test_score_has_no_geographic_or_program_contribution(self):
        changed = self.evaluation.copy()
        changed["region_administrative"] = "Cote-Nord"
        changed["code_postal_3"] = "G4R"
        changed["distance_domicile_campus_km"] = 600
        changed["programme_etudes"] = "Genie"
        np.testing.assert_allclose(self.model.score(changed), self.model.score(self.evaluation), atol=1e-12)

    def test_monotonic_policy_score(self):
        base = self.model.score(self.evaluation)
        for feature, delta in [("cote_r_equivalent", 1), ("heures_travail_semaine", 1),
                               ("premiere_generation_universitaire", 1), ("revenu_familial_estime", -1000)]:
            changed = self.evaluation.copy()
            changed[feature] += delta
            self.assertTrue(np.all(self.model.score(changed) >= base - 1e-12), feature)

    def test_reordered_or_nonbinary_submission_is_rejected(self):
        with self.assertRaises(ValueError):
            verify_predictions(self.evaluation, self.submission.iloc[::-1].reset_index(drop=True))
        bad = self.submission.copy()
        bad.loc[0, TARGET] = 2
        with self.assertRaises(ValueError):
            verify_predictions(self.evaluation, bad)

    def test_unknown_region_or_missing_inputs_are_rejected(self):
        bad = self.evaluation.copy()
        bad.loc[0, "region_administrative"] = "Unknown"
        with self.assertRaises(ValueError):
            validate_data(bad)
        with self.assertRaises(ValueError):
            validate_data(self.evaluation.drop(columns="code_postal_3"))

    def test_history_keeps_original_predictions_and_comparison(self):
        from prediction_history import archive_prediction, preserve_existing
        with tempfile.TemporaryDirectory() as temporary:
            history_dir = Path(temporary) / "versions"
            first = archive_prediction(self.evaluation, self.submission, "test-first", history_dir=history_dir)
            original = (first / "predictions.csv").read_bytes()
            changed = self.submission.copy()
            yes = changed.index[changed[TARGET] == 1][0]
            no = changed.index[changed[TARGET] == 0][0]
            changed.loc[[yes, no], TARGET] = [0, 1]
            second = archive_prediction(self.evaluation, changed, "test-second", history_dir=history_dir)
            self.assertEqual((first / "predictions.csv").read_bytes(), original)
            metadata = json.loads((second / "metadata.json").read_text())
            self.assertEqual(metadata["changed_decisions_from_previous"], 2)
            self.assertEqual(metadata["csv_sha256"], hashlib.sha256((second / "predictions.csv").read_bytes()).hexdigest())
            self.assertTrue((second / metadata["model_source_filename"]).exists())
            comparison = pd.read_csv(history_dir / "comparison.csv")
            self.assertEqual(len(comparison), 2)
            self.assertTrue(comparison.official_accuracy.isna().all())
            preserve_existing(first / "predictions.csv", self.evaluation, history_dir)
            self.assertEqual(len(list(history_dir.glob("*/metadata.json"))), 2)


if __name__ == "__main__":
    unittest.main()
