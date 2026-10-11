"""Decision log and label export."""

import csv
import tempfile
import unittest
from pathlib import Path

from opendecision.decision_log import export_labels, read, record_decision, record_feedback


def _decision(action):
    return {"action": action, "probabilities": {action: 0.6}, "needs_review": True}


class DecisionLogTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.log = Path(self.temp.name) / "decisions.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def test_exports_only_decisions_with_feedback_and_uses_the_latest(self):
        first = record_decision(self.log, "flights", "date missing", _decision("search"))
        record_decision(self.log, "flights", "no feedback", _decision("search"))
        record_decision(self.log, "research", "other scenario", _decision("search"))
        record_feedback(self.log, first, "search")
        record_feedback(self.log, first, "perguntar_dados", outcome="corrected")

        destination = Path(self.temp.name) / "labels.csv"
        summary = export_labels(self.log, destination, "flights")

        self.assertEqual(summary["decisions"], 2)
        self.assertEqual(summary["labeled"], 1)
        self.assertEqual(summary["corrected_suggestions"], 1)
        with destination.open(encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual([(r["state"], r["action"]) for r in rows], [("date missing", "ask")])

    def test_feedback_requires_a_recorded_decision(self):
        with self.assertRaises(KeyError):
            record_feedback(self.log, "unknown", "search")
        self.assertEqual(read(self.log), [])


if __name__ == "__main__":
    unittest.main()
