"""Label Connect Four positions with the teacher's move (alpha-beta search, depth 4).

    python demos/connect4/generate_data.py     # 1,500 training games, 300 test games

Positions come from games between varied players (random, depth 2, depth 4 with noise), so
the data covers good and bad positions; each position is labeled with the teacher's move.
Training and test games use different seeds. Output goes to data/external/ (git-ignored).
"""

import argparse
import csv
import random
from pathlib import Path

from features import describe
from game import ACTIONS, Board, search_move

ROOT = Path(__file__).resolve().parents[2]
TEACHER_DEPTH = 4
TEST_SEED_OFFSET = 100_000


def opponent_move(board: Board, style: str, rng: random.Random) -> int:
    if style == "random" or rng.random() < 0.15:  # some noise so games branch
        return rng.choice(board.legal())
    return search_move(board, 2 if style == "depth2" else TEACHER_DEPTH, rng)


def record(path: Path, seeds: range) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["state", "action", "game"])
        for seed in seeds:
            rng = random.Random(seed)
            board = Board()
            styles = {1: rng.choice(("random", "depth2", "teacher")), 2: rng.choice(("random", "depth2", "teacher"))}
            while board.winner is None:
                writer.writerow([describe(board), ACTIONS[search_move(board, TEACHER_DEPTH, rng)], seed])
                rows += 1
                board.play(opponent_move(board, styles[board.to_move], rng))
    return {"file": str(path.relative_to(ROOT)), "games": len(seeds), "examples": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-games", type=int, default=1500)
    parser.add_argument("--test-games", type=int, default=300)
    args = parser.parse_args()
    external = ROOT / "data" / "external"
    print(record(external / "connect4_train.csv", range(args.train_games)), flush=True)
    print(record(external / "connect4_test.csv", range(TEST_SEED_OFFSET, TEST_SEED_OFFSET + args.test_games)))


if __name__ == "__main__":
    main()
