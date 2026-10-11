"""Command line interface in English, with Portuguese aliases and localized messages."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nextaction.cli import main
from nextaction.i18n import MESSAGES, resolve_language

ROOT = Path(__file__).resolve().parents[1]


def run(*argv):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        main(list(argv))
    return json.loads(output.getvalue())


def run_error(*argv):
    errors = io.StringIO()
    with contextlib.redirect_stderr(errors), contextlib.suppress(SystemExit):
        main(list(argv))
    return errors.getvalue()


class CliTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.model = str(Path(self.temp.name) / "flights.joblib")
        self.jobs = patch.dict(os.environ, {"NEXTACTION_JOBS_DIR": self.temp.name})
        self.jobs.start()

    def tearDown(self):
        self.jobs.stop()
        self.temp.cleanup()

    def test_train_and_decide_in_english(self):
        result = run("train", "flights", "--model", self.model)
        self.assertEqual(result["examples"], 48)
        decision = run("decide", "flights", "Tenho origem GRU destino REC e ainda não consultei voos",
                       "--model", self.model)
        self.assertEqual(decision["action"], "search")

    def test_portuguese_commands_options_and_scenarios(self):
        run("treinar", "passagens", "--modelo", self.model)
        decision = run("decidir", "passagens", "O viajante escolheu o voo das 8h e quer reservar",
                       "--modelo", self.model, "--limiar", "0.99")
        self.assertTrue(decision["needs_review"])

    def test_rank_flights_with_portuguese_json(self):
        trip = Path(self.temp.name) / "viagem.json"
        trip.write_text(json.dumps({"pedido": {"origem": "GRU", "destino": "REC"}}), encoding="utf-8")
        self.assertEqual(run("avaliar-passagens", str(trip))["action"], "ask")

    def test_errors_follow_the_language_option(self):
        missing = str(Path(self.temp.name) / "missing.joblib")
        self.assertIn("Train the scenario first", run_error("decide", "flights", "x", "--model", missing))
        self.assertIn("Treine o cenário antes", run_error("--lang", "pt", "decide", "flights", "x", "--model", missing))

    def test_language_falls_back_to_environment(self):
        with patch.dict(os.environ, {"NEXTACTION_LANG": "pt-BR"}):
            self.assertEqual(resolve_language(None), "pt")
        self.assertEqual(resolve_language("en-US"), "en")

    def test_every_message_has_both_languages(self):
        incomplete = [key for key, texts in MESSAGES.items() if set(texts) != {"en", "pt"}]
        self.assertEqual(incomplete, [])


if __name__ == "__main__":
    unittest.main()
