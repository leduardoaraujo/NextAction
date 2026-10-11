"""Synthetic orchestration data, policy and benchmark contracts."""

import csv
import tempfile
import unittest
from pathlib import Path

from demos.orchestrator.benchmark import keyword_router, read_axis_predictions, summary
from demos.orchestrator.generate_data import generate, state
from nextaction.policy import apply
from nextaction.scenarios import get_scenario


class OrchestratorDemoTest(unittest.TestCase):
    def test_alias_and_no_source_mask(self):
        scenario = get_scenario("orquestrador")
        self.assertEqual(scenario.name, "orchestrator")
        encoded = state("Mostre vendas", {"datasource": "missing", "last_error": "none", "missing": "none"})
        decision = apply({"data_query": .9, "clarification_needed": .1}, encoded, scenario.policy)
        self.assertEqual(decision["blocked_actions"], ["data_query"])
        self.assertEqual(decision["action"], "clarification_needed")

    def test_phrase_families_do_not_cross_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            generate(Path(folder))
            splits = {}
            for split in ("train", "validation", "test"):
                with (Path(folder) / f"orchestrator_{split}.csv").open(encoding="utf-8", newline="") as file:
                    splits[split] = list(csv.DictReader(file))
                self.assertEqual({r["action"] for r in splits[split]},
                                 {"data_query", "pure_chat", "clarification_needed", "error_recovery"})
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
                self.assertFalse({r["utterance"] for r in splits[a]} & {r["utterance"] for r in splits[b]})

    def test_keyword_router_and_route_metric(self):
        self.assertEqual(keyword_router("Mostre um gráfico de vendas"), "data_query")
        self.assertEqual(keyword_router("Olá, tudo bem?"), "pure_chat")
        scores = summary(["clarification_needed", "data_query"], ["pure_chat", "data_query"])
        self.assertEqual(scores["accuracy"], .5)
        self.assertEqual(scores["agent_accuracy"], 1.0)

    def test_axis_predictions_require_complete_matching_families(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.csv"
            path.write_text("family,action\na,data_query\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_axis_predictions(path, [{"family": "a"}, {"family": "b"}])


if __name__ == "__main__":
    unittest.main()
