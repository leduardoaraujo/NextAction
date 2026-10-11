"""Pac-Man demo: game rules, state description and the policy's deny_tag rules."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demos" / "pacman"))

from expert import expert_action  # noqa: E402
from features import describe  # noqa: E402
from game import ACTIONS, Game  # noqa: E402

from nextaction.policy import Policy, Rule, apply, parse_state  # noqa: E402
from nextaction.scenarios import get_scenario  # noqa: E402


class PacmanDemoTest(unittest.TestCase):
    def test_games_are_deterministic_per_seed(self):
        def play(seed):
            game = Game(seed=seed)
            while game.status == "playing":
                game.step(expert_action(game))
            return game.status, game.score, game.steps

        self.assertEqual(play(3), play(3))

    def test_walls_stop_pacman_and_pellets_score(self):
        game = Game(seed=0)
        start = game.pacman
        blocked = next(a for a in ACTIONS if a not in game.legal_actions())
        game.step(blocked)
        self.assertEqual(game.pacman, start)
        open_move = game.legal_actions()[0]
        game.step(open_move)
        self.assertGreaterEqual(game.score, 0)
        self.assertNotEqual(game.pacman, start)

    def test_state_lists_blocked_moves_for_the_policy(self):
        game = Game(seed=1)
        state = describe(game)
        blocked = parse_state(state).tags["blocked"]
        self.assertEqual(blocked, frozenset(a for a in ACTIONS if a not in game.legal_actions()))
        self.assertIn("[features ", state)

    def test_deny_tag_rules_remove_walls_and_ghosts(self):
        policy = Policy(default_threshold=0.5, rules=(Rule(id="walls", deny_tag="blocked"),
                                                      Rule(id="ghosts", deny_tag="danger")))
        state = "[blocked up left] [danger right] [features food_down_0]"
        decision = apply({"up": 0.6, "down": 0.1, "left": 0.2, "right": 0.1}, state, policy)
        self.assertEqual(decision["action"], "down")
        self.assertEqual(decision["blocked_actions"], ["left", "right", "up"])
        self.assertEqual(decision["reasons"], ["masked_by_rule:walls", "masked_by_rule:ghosts"])

    def test_pacman_config_loads_its_policy(self):
        scenario = get_scenario("pacman")
        self.assertEqual({rule.deny_tag for rule in scenario.policy.rules}, {"blocked", "danger"})


if __name__ == "__main__":
    unittest.main()
