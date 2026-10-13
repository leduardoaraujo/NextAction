"""Outcome learning requires logged actions with known selection probabilities."""

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from nextaction.bandit import fit_candidate, observations, preview_candidate
from nextaction.decision_log import record_decision, record_feedback
from nextaction.schemas import FeedbackRequest


class BanditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.log = Path(self.temp.name) / "decisions.jsonl"
        self.model = Path(self.temp.name) / "candidate.joblib"

    def test_feedback_requires_a_complete_logged_observation(self):
        with self.assertRaises(ValidationError):
            FeedbackRequest.model_validate({"final_action": "ask", "reward": 1})
        with self.assertRaises(ValidationError):
            FeedbackRequest.model_validate({"final_action": "ask", "action_taken": "ask",
                                            "reward": 1, "propensity": 0})
        request = FeedbackRequest.model_validate({"acao_final": "perguntar_dados",
                                                  "acao_executada": "perguntar_dados",
                                                  "recompensa": 1, "probabilidade_escolha": 0.5})
        self.assertEqual(request.action_taken, "ask")
        self.assertEqual(request.propensity, 0.5)

    def test_candidate_requires_outcomes_and_stays_separate_from_serving(self):
        decision = {"action": "ask", "probabilities": {"ask": 0.8}, "needs_review": False}
        unlabeled = record_decision(self.log, "flights", "needs date", decision)
        record_feedback(self.log, unlabeled, "ask")
        self.assertEqual(observations(self.log, "flights"), [])
        self.assertEqual(fit_candidate(self.log, "flights", self.model)["status"], "insufficient_data")
        self.assertFalse(self.model.exists())

        for index in range(240):
            state = "red urgent" if index % 2 else "blue routine"
            action = "ask" if index % 4 < 2 else "answer"
            reward = float((state.startswith("red") and action == "ask") or
                           (state.startswith("blue") and action == "answer"))
            key = record_decision(self.log, "flights", state, decision)
            record_feedback(self.log, key, action, action_taken=action,
                            reward=reward, propensity=0.5)

        report = fit_candidate(self.log, "flights", self.model)
        self.assertEqual(report["status"], "candidate_ready")
        self.assertEqual(report["observations"], 240)
        self.assertGreater(report["matched_held_out"], 0)
        self.assertTrue(report["offline_estimate_usable"])
        self.assertGreater(report["candidate_snips_reward"], report["logging_mean_reward"])
        self.assertTrue(self.model.exists())
        preview = preview_candidate(self.model, "red urgent")
        self.assertTrue(preview["candidate_only"])
        self.assertEqual(preview["suggested"], "ask")


if __name__ == "__main__":
    unittest.main()
