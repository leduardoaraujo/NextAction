"""Let the expert play and record every move as a ``state,action`` example.

    python demos/pacman/generate_data.py            # 400 training games, 100 test games

Training and test games use different seeds, so the test set holds positions from games
the model never saw. Output goes to data/external/ (git-ignored).
"""

import argparse
import csv
from pathlib import Path

from expert import expert_action
from features import describe
from game import Game

ROOT = Path(__file__).resolve().parents[2]
TEST_SEED_OFFSET = 100_000


def record(path: Path, seeds: range) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = wins = 0
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["state", "action", "game"])
        for seed in seeds:
            game = Game(seed=seed)
            while game.status == "playing":
                action = expert_action(game)
                writer.writerow([describe(game), action, seed])
                game.step(action)
                rows += 1
            wins += game.status == "won"
    return {"file": str(path.relative_to(ROOT)), "games": len(seeds), "examples": rows, "expert_wins": wins}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-games", type=int, default=400)
    parser.add_argument("--test-games", type=int, default=100)
    args = parser.parse_args()
    external = ROOT / "data" / "external"
    print(record(external / "pacman_train.csv", range(args.train_games)))
    print(record(external / "pacman_test.csv", range(TEST_SEED_OFFSET, TEST_SEED_OFFSET + args.test_games)))


if __name__ == "__main__":
    main()
