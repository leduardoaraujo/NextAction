"""Play ViZDoom Deathmatch and record continuous gameplay to a GIF."""

import argparse
import random
import sys
from pathlib import Path

import joblib
import vizdoom as vzd
from PIL import Image, ImageDraw, ImageFont

from common import ACTIONS, BUTTONS, DECISION_TICS, describe, open_game, teacher

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nextaction.policy import apply  # noqa: E402
from nextaction.scenarios import get_scenario  # noqa: E402


def frame(screen, action: str, info: dict, tics: int, health: int, kills: int, font):
    game_image = Image.fromarray(screen).resize((480, 360), Image.Resampling.NEAREST)
    image = Image.new("RGB", (730, 360), (14, 17, 27))
    image.paste(game_image, (0, 0))
    draw = ImageDraw.Draw(image)
    x = 496
    draw.text((x, 12), "NextAction / DOOM", font=font, fill=(255, 221, 0))
    draw.text((x, 49), f"time {tics / 35:.1f}s", font=font, fill="white")
    draw.text((x, 76), f"health {health}   kills {kills}", font=font, fill="white")
    draw.text((x, 114), action.replace("_", " ").upper(), font=font, fill=(104, 226, 150))
    draw.text((x, 142), info.get("source", ""), font=font, fill=(170, 181, 195))
    probabilities = info.get("probabilities", {})
    for index, name in enumerate(ACTIONS):
        y = 161 + index * 30
        draw.text((x, y), name.replace("_", " "), font=font, fill="white")
        p = probabilities.get(name, 0.0)
        draw.rectangle((x, y + 19, x + round(178 * p), y + 23), fill=(84, 199, 112))
        draw.text((x + 184, y), f"{p:.2f}", font=font, fill="white")
    return image


def run(agent: str, seed: int, gif: Path | None, fps: int, max_tics: int, record_from: int = 0) -> dict:
    scenario = get_scenario("doom")
    model = joblib.load(scenario.model) if agent in ("model", "model-only") else None
    game = open_game(seed)
    rng = random.Random(seed)
    frames = []
    font = ImageFont.load_default(size=17)
    reviews = decisions = tics = 0
    try:
        while not game.is_episode_finished() and tics < max_tics:
            state = game.get_state()
            if state is None:
                break
            if agent == "teacher":
                action, info = teacher(state), {"source": "teacher"}
            elif agent == "random":
                action, info = rng.choice(ACTIONS), {"source": "random"}
            else:
                raw = dict(zip(model.classes_, map(float, model.predict_proba([describe(state)])[0])))
                decision = apply(raw, describe(state), scenario.policy)
                action = decision["action"]
                info = {"source": "model", "probabilities": decision["probabilities"]}
                if decision["needs_review"] and agent == "model":
                    action = teacher(state)
                    info["source"] = "teacher review"
                    reviews += 1
            decisions += 1
            game.set_action(BUTTONS[action])
            for _ in range(DECISION_TICS):
                if game.is_episode_finished() or tics >= max_tics:
                    break
                game.advance_action()
                tics += 1
                if gif and tics >= record_from and (tics % 2 == 0 or game.is_episode_finished()):
                    next_state = game.get_state()
                    if next_state is not None:
                        health, _, kills = map(int, next_state.game_variables)
                        frames.append(frame(next_state.screen_buffer, action, info,
                                            tics, health, kills, font))
        kills = int(game.get_game_variable(vzd.GameVariable.KILLCOUNT))
        result = {"agent": agent, "seed": seed, "kills": kills,
                  "survived": not game.is_player_dead(),
                  "reward": round(game.get_total_reward(), 1), "tics": tics,
                  "decisions": decisions, "reviews": reviews,
                  "gif": str(gif) if gif else None}
        if gif and frames:
            gif.parent.mkdir(parents=True, exist_ok=True)
            palette_frames = [f.convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for f in frames]
            palette_frames[0].save(gif, save_all=True,
                                   append_images=palette_frames[1:] + [palette_frames[-1]] * fps,
                                   duration=round(1000 / fps), loop=0, optimize=True)
            result["frames"] = len(frames)
        return result
    finally:
        game.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("model", "model-only", "teacher", "random"), default="model")
    parser.add_argument("--seed", type=int, default=100_001)
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--max-tics", type=int, default=600)
    parser.add_argument("--record-from", type=int, default=0,
                        help="skip this many game tics before recording the GIF")
    args = parser.parse_args()
    if args.fps < 1 or args.max_tics < 1 or not 0 <= args.record_from < args.max_tics:
        parser.error("fps and max-tics must be positive; record-from must be within the episode")
    print(run(args.agent, args.seed, args.gif, args.fps, args.max_tics, args.record_from))


if __name__ == "__main__":
    main()
