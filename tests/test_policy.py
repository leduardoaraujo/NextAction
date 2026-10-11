"""Decision policy, scenario configs, tuning and calibration."""

import contextlib
import io
import json
import os
import shutil
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from opendecision.cli import main
from opendecision.policy import ActionPolicy, Policy, Rule, apply, calibrate, evaluate_policy, parse_state
from opendecision.scenarios import ROOT, get_scenario, load_scenario
from opendecision.tuning import update_config

MISSING = "[intent BookFlight] [filled origin] [missing date] [requested none] || user: book a flight"
COMPLETE = "[intent BookFlight] [filled date origin] [missing none] [requested none] || user: book it"
POLICY = Policy(
    default_threshold=0.6, review_cost=1.0,
    actions={
        "execute": ActionPolicy(risk="irreversible", min_confidence=0.95, unreviewed_error_cost=50,
                                requires_confirmation=True),
        "ask": ActionPolicy(unreviewed_error_cost=2),
    },
    rules=(Rule(id="no_execute_with_missing", deny=("execute", "search"), when_missing_any=True),),
)


class PolicyTest(unittest.TestCase):
    def test_parse_state_reads_tags_in_both_languages(self):
        parsed = parse_state("[intencao X] [preenchidos a b] [faltando c] [pedidos nenhum] || usuario: oi")
        self.assertEqual((parsed.intent, parsed.filled, parsed.missing, parsed.requested),
                         ("X", frozenset({"a", "b"}), frozenset({"c"}), frozenset()))
        self.assertIsNone(parse_state("plain text without tags"))

    def test_rule_blocks_actions_and_renormalizes(self):
        decision = apply({"execute": 0.7, "search": 0.1, "ask": 0.2}, MISSING, POLICY)
        self.assertEqual(decision["action"], "ask")
        self.assertEqual(decision["blocked_actions"], ["execute", "search"])
        self.assertEqual(decision["probabilities"], {"ask": 1.0})
        self.assertIn("masked_by_rule:no_execute_with_missing", decision["reasons"])

    def test_irreversible_action_needs_high_confidence_and_confirmation(self):
        confident = apply({"execute": 0.97, "ask": 0.03}, COMPLETE, POLICY)
        unsure = apply({"execute": 0.9, "ask": 0.1}, COMPLETE, POLICY)
        self.assertFalse(confident["needs_review"])
        self.assertTrue(confident["requires_confirmation"])
        self.assertIn("irreversible_action", confident["reasons"])
        self.assertTrue(unsure["needs_review"])
        self.assertEqual(unsure["threshold"], 0.95)

    def test_explicit_threshold_overrides_policy(self):
        self.assertFalse(apply({"execute": 0.9, "ask": 0.1}, COMPLETE, POLICY, threshold=0.5)["needs_review"])

    def test_policy_simulation_counts_cost(self):
        result = evaluate_policy(
            [COMPLETE, COMPLETE, MISSING], ["execute", "ask", "ask"],
            [[0.99, 0.01], [0.97, 0.03], [0.8, 0.2]], ["execute", "ask"], POLICY,
        )
        self.assertEqual(result["unreviewed_errors"], 1)
        self.assertEqual(result["decisions_with_blocked_actions"], 1)
        self.assertEqual(result["expected_cost_per_decision"], round(50 / 3, 4))

    def test_calibration_raises_threshold_when_errors_are_expensive(self):
        probabilities = [[0.9, 0.1]] * 5 + [[0.99, 0.01]] * 5
        actual = ["ask"] * 5 + ["execute"] * 5
        thresholds = calibrate(actual, probabilities, ["execute", "ask"], POLICY)
        self.assertGreater(thresholds["execute"], 0.9)


class ScenarioConfigTest(unittest.TestCase):
    def test_bundled_configs_load(self):
        agent = get_scenario("agente")
        self.assertEqual(agent.params.min_df, 5)
        self.assertEqual(agent.policy.for_action("execute").risk, "irreversible")
        self.assertTrue(agent.policy.rules)

    def test_update_config_keeps_comments(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "flights.toml"
            shutil.copyfile(ROOT / "configs" / "flights.toml", path)
            update_config(path, {("model", "C"): 3.0, ("policy", "actions", "ask", "min_confidence"): 0.55})
            text = path.read_text(encoding="utf-8")
            config = tomllib.loads(text)
            self.assertIn("# Flight booking assistant", text)
            self.assertEqual(config["model"]["C"], 3.0)
            self.assertEqual(load_scenario(path).policy.threshold("ask"), 0.55)


class TuningCliTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        folder = Path(self.temp.name)
        shutil.copyfile(ROOT / "configs" / "flights.toml", folder / "flights.toml")
        self.config = folder / "flights.toml"
        self.env = patch.dict(os.environ, {"OPENDECISION_JOBS_DIR": self.temp.name})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def run_cli(self, *argv):
        scenario = load_scenario(self.config)
        output = io.StringIO()
        with patch("opendecision.cli.get_scenario", return_value=scenario), contextlib.redirect_stdout(output):
            main(["--no-progress", *argv])
        return json.loads(output.getvalue())

    def test_tune_writes_best_parameters(self):
        result = self.run_cli("tune", "flights", "--write")
        self.assertIn(result["best"]["params"]["C"], [0.3, 1.0, 3.0, 10.0])
        written = tomllib.loads(self.config.read_text(encoding="utf-8"))["model"]
        self.assertEqual(written["C"], result["best"]["params"]["C"])

    def test_calibrate_policy_reports_cost_before_and_after(self):
        result = self.run_cli("calibrate-policy", "flights", "--write")
        self.assertEqual(result["method"], "cross-validation")
        self.assertLessEqual(result["after"]["expected_cost_per_decision"],
                             result["before"]["expected_cost_per_decision"])
        actions = tomllib.loads(self.config.read_text(encoding="utf-8"))["policy"]["actions"]
        self.assertTrue(all("min_confidence" in actions[a] for a in result["thresholds"]))


if __name__ == "__main__":
    unittest.main()
