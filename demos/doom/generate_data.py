"""Record the rule-based teacher in ViZDoom's Deathmatch arena."""

import argparse
import csv
from pathlib import Path

import vizdoom as vzd

from common import BUTTONS, DECISION_TICS, describe, open_game, teacher

ROOT = Path(__file__).resolve().parents[2]
TEST_SEED_OFFSET = 100_000


def record(path: Path, seeds: range) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    kills = 0
    survivors = 0
    game = open_game(seeds.start)
    try:
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["state", "action", "game"])
            for index, seed in enumerate(seeds):
                if index:
                    game.set_seed(seed)
                    game.new_episode()
                while not game.is_episode_finished() and game.get_episode_time() < 600:
                    state = game.get_state()
                    if state is None:
                        break
                    action = teacher(state)
                    writer.writerow([describe(state), action, seed])
                    rows += 1
                    game.make_action(BUTTONS[action], DECISION_TICS)
                kills += int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
                survivors += not game.is_player_dead()
    finally:
        game.close()
    return {"file": str(path.relative_to(ROOT)), "games": len(seeds), "examples": rows,
            "kills": kills, "survived": survivors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-games", type=int, default=20)
    parser.add_argument("--test-games", type=int, default=5)
    args = parser.parse_args()
    external = ROOT / "data" / "external"
    print(record(external / "doom_train.csv", range(args.train_games)), flush=True)
    print(record(external / "doom_test.csv", range(TEST_SEED_OFFSET, TEST_SEED_OFFSET + args.test_games)))


if __name__ == "__main__":
    main()
