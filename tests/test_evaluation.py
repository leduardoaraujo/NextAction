"""Classifier evaluation report."""

import unittest
from pathlib import Path

from opendecision.evaluation import evaluate, metrics
from opendecision.i18n import LocalizedError

DATA = Path(__file__).resolve().parents[1] / "data" / "flights.csv"


class EvaluationTest(unittest.TestCase):
    def test_cross_validation_evaluates_every_example(self):
        result = evaluate(DATA)
        self.assertEqual(result["evaluated_examples"], 48)
        self.assertEqual(sum(a["support"] for a in result["per_action"].values()), 48)
        self.assertEqual(set(result["per_action"]), {"ask", "search", "present_options", "confirm"})
        self.assertIn("warning", result)

    def test_recent_fraction_tests_on_the_last_rows(self):
        result = evaluate(DATA, recent_fraction=0.25)
        self.assertEqual(result["evaluated_examples"], 12)
        self.assertEqual(result["train_examples"], 36)

    def test_threshold_splits_automated_from_review(self):
        result = metrics(["a", "b", "a"], [[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]], ["a", "b"], majority="a")
        at_07 = next(row for row in result["thresholds"] if row["threshold"] == 0.7)
        self.assertEqual(at_07["automated"], round(2 / 3, 3))
        self.assertEqual(at_07["unreviewed_errors"], 1)
        self.assertEqual(at_07["sent_to_review"], 1)
        self.assertEqual(result["confusion_matrix"]["b"], {"a": 1})

    def test_report_text_is_localized(self):
        result = evaluate(DATA, lang="pt")
        self.assertIn("validação cruzada", result["method"])
        self.assertTrue(result["warning"].startswith("Só"))

    def test_rejects_two_test_modes(self):
        with self.assertRaises(LocalizedError):
            evaluate(DATA, test=DATA, recent_fraction=0.2)


if __name__ == "__main__":
    unittest.main()
