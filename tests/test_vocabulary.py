"""Canonical vocabulary, Portuguese aliases and backward compatibility."""

import tempfile
import unittest
from pathlib import Path

import joblib

from nextaction.data import load_examples
from nextaction.i18n import LocalizedError
from nextaction.model import decide, new_model
from nextaction.scenarios import normalize_scenario
from nextaction.vocabulary import normalize_action, normalize_state


class VocabularyTest(unittest.TestCase):
    def test_portuguese_actions_map_to_canonical_labels(self):
        self.assertEqual(normalize_action("buscar_voos"), "search")
        self.assertEqual(normalize_action("pedir_confirmacao"), "confirm")
        self.assertEqual(normalize_action("search"), "search")
        self.assertEqual(normalize_action("custom_action"), "custom_action")

    def test_scenarios_accept_portuguese_names(self):
        self.assertEqual(normalize_scenario("Agente"), "agent")
        self.assertEqual(normalize_scenario("flights"), "flights")
        with self.assertRaises(LocalizedError):
            normalize_scenario("hotels")

    def test_state_tags_and_speakers_are_normalized(self):
        state = "[intencao FindFlight] [preenchidos origin] [faltando date] [pedidos nenhum] || usuario: oi || assistente: olá"
        self.assertEqual(
            normalize_state(state),
            "[intent FindFlight] [filled origin] [missing date] [requested none] || user: oi || assistant: olá",
        )

    def test_free_text_is_not_rewritten(self):
        self.assertEqual(normalize_state("nenhum voo encontrado"), "nenhum voo encontrado")

    def test_csv_with_portuguese_columns_and_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.csv"
            rows = ["estado,acao"] + [f"falta dado {i},perguntar_dados" for i in range(4)]
            rows += [f"pronto para buscar {i},buscar_voos" for i in range(4)]
            path.write_text("\n".join(rows), encoding="utf-8")
            _, actions = load_examples(path)
        self.assertEqual(set(actions), {"ask", "search"})

    def test_model_with_portuguese_labels_returns_english_actions(self):
        states = [f"falta dado {i}" for i in range(4)] + [f"pronto para buscar {i}" for i in range(4)]
        old_model = new_model().fit(states, ["perguntar_dados"] * 4 + ["buscar_voos"] * 4)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.joblib"
            joblib.dump(old_model, path)
            result = decide(path, "falta dado")
        self.assertEqual(set(result["probabilities"]), {"ask", "search"})
        self.assertEqual(result["action"], "ask")


if __name__ == "__main__":
    unittest.main()
