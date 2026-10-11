"""Connect Four demo: rules, search player and the state the policy reads."""

import sys
import unittest
from pathlib import Path

# Both demos have modules named game and features: drop any already imported.
for _module in ("game", "features", "expert", "play"):
    sys.modules.pop(_module, None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demos" / "connect4"))

from features import analyse, describe  # noqa: E402
from game import WIN, Board, evaluate, negamax, search_move  # noqa: E402

from nextaction.policy import parse_state  # noqa: E402
from nextaction.scenarios import get_scenario  # noqa: E402


def board_after(*columns: int) -> Board:
    board = Board()
    for col in columns:
        board.play(col)
    return board


class Connect4DemoTest(unittest.TestCase):
    def test_four_in_a_row_wins(self):
        board = board_after(0, 6, 1, 6, 2, 6, 3)
        self.assertEqual(board.winner, 1)

    def test_search_takes_wins_and_blocks_threats(self):
        threat = board_after(0, 6, 1, 6, 2)  # player 1 threatens c4
        self.assertEqual(search_move(threat, 4), 3)
        threat.play(5)
        self.assertEqual(search_move(threat, 4), 3)

    def test_alpha_beta_matches_plain_minimax(self):
        def plain(board, depth):
            if board.winner is not None:
                return 0 if board.winner == 0 else -(WIN + depth)
            if depth == 0:
                return evaluate(board, board.to_move)
            values = []
            for col in board.legal():
                child = board.copy()
                child.play(col)
                values.append(-plain(child, depth - 1))
            return max(values)

        board = board_after(3, 3, 2, 4, 1)
        self.assertEqual(negamax(board, 3, -10 * WIN, 10 * WIN), plain(board, 3))

    def test_state_lists_full_columns_and_gifted_wins(self):
        full = board_after(*[0] * 6)
        self.assertIn("c1", parse_state(describe(full)).tags["blocked"])
        # Player 2 has three in a row on the second level (c2–c4) and c5 is empty: if player 1
        # drops in c5, player 2 completes the line on top of it.
        gift = Board()
        gift.cells[0][1], gift.cells[0][2], gift.cells[0][3] = 1, 2, 1
        gift.cells[1][1], gift.cells[1][2], gift.cells[1][3] = 2, 2, 2
        gift.to_move, gift.moves = 1, 6
        self.assertIn("c5", analyse(gift)["danger"])

    def test_connect4_config_loads_with_alias(self):
        scenario = get_scenario("lig4")
        self.assertEqual(scenario.name, "connect4")
        self.assertEqual({rule.deny_tag for rule in scenario.policy.rules}, {"blocked", "danger"})


if __name__ == "__main__":
    unittest.main()
