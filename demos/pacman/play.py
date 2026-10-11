"""Watch an agent play Pac-Man and see every decision it makes.

    python demos/pacman/play.py                        # model + policy, live in the terminal
    python demos/pacman/play.py --agent expert         # the rule-based expert
    python demos/pacman/play.py --gif demos/pacman/pacman.gif --pick-win

Agents:
  model       NextAction model + decision policy; low-confidence moves go to the expert
  model-only  model + policy, never asks the expert
  raw         model alone: no policy, no expert
  expert      the rule-based player that labeled the training data
  random      a random legal move

The side panel shows the probability of each move, the moves the policy blocked (walls and
moves next to a ghost), the threshold, and who made the decision.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import joblib

from expert import expert_action, random_action
from features import describe
from game import ACTIONS, MAZE, Game

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nextaction.policy import apply  # noqa: E402
from nextaction.scenarios import get_scenario  # noqa: E402

AGENTS = ("model", "model-only", "raw", "expert", "random")


class Agent:
    def __init__(self, kind: str, seed: int = 0):
        self.kind = kind
        self.rng = random.Random(seed)
        if kind in ("model", "model-only", "raw"):
            scenario = get_scenario("pacman")
            self.model = joblib.load(scenario.model)
            self.policy = scenario.policy

    def decide(self, game: Game) -> tuple[str, dict]:
        """Return the move and what the panel shows about the decision."""
        if self.kind == "expert":
            return expert_action(game), {"source": "expert"}
        if self.kind == "random":
            return random_action(game, self.rng), {"source": "random"}
        state = describe(game)
        probabilities = dict(zip(self.model.classes_, map(float, self.model.predict_proba([state])[0])))
        if self.kind == "raw":
            action = max(probabilities, key=probabilities.get)
            return action, {"source": "model", "probabilities": probabilities, "blocked": [], "state": state}
        decision = apply(probabilities, state, self.policy)
        info = {
            "source": "model", "probabilities": decision["probabilities"], "raw": probabilities,
            "blocked": decision["blocked_actions"], "threshold": decision["threshold"], "state": state,
            "confidence": decision["probabilities"][decision["action"]], "model_action": decision["action"],
        }
        if decision["needs_review"] and self.kind == "model":
            info["source"] = "expert review"
            return expert_action(game), info
        return decision["action"], info


def run_game(kind: str, seed: int, on_step=None) -> dict:
    game, agent = Game(seed=seed), Agent(kind, seed)
    reviews = corrections = 0
    while game.status == "playing":
        action, info = agent.decide(game)
        reviews += info["source"] == "expert review"
        # The policy corrected the model when its own top choice was a blocked move.
        raw = info.get("raw")
        corrections += bool(raw) and max(raw, key=raw.get) in info.get("blocked", [])
        if on_step:
            on_step(game, action, info)
        game.step(action)
    if on_step:
        on_step(game, None, {"source": "end"})
    return {"agent": kind, "seed": seed, "status": game.status, "score": game.score,
            "steps": game.steps, "reviews": reviews, "corrections": corrections}


# Terminal view -----------------------------------------------------------------------
def terminal_view(game: Game, action: str | None, info: dict):
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.text import Text

    ghosts = {g.position for g in game.ghosts}
    board = Text()
    for row in range(len(MAZE)):
        for col in range(len(MAZE[0])):
            cell = (row, col)
            if cell == game.pacman:
                board.append("C ", style="bold yellow")
            elif cell in ghosts:
                board.append("G ", style="bold blue" if game.frightened else "bold red")
            elif cell in game.walls:
                board.append("██", style="blue")
            elif cell in game.power_pills:
                board.append("● ", style="bold white")
            elif cell in game.pellets:
                board.append("· ", style="white")
            else:
                board.append("  ")
        board.append("\n")
    panel = Text()
    panel.append(f"step {game.steps}   score {game.score}\n", style="bold")
    if action is None:
        panel.append(f"\n{game.status.upper()}\n", style="bold green" if game.status == "won" else "bold red")
    else:
        source = info["source"]
        panel.append(f"decided by: {source}\n", style="yellow" if source == "expert review" else "green")
        panel.append(f"move: {action}\n\n", style="bold")
        probabilities = info.get("probabilities")
        if probabilities is not None:
            for a in ACTIONS:
                if a in info.get("blocked", []):
                    panel.append(f"{a:>5}  blocked\n", style="dim")
                    continue
                p = probabilities.get(a, 0.0)
                panel.append(f"{a:>5}  {'█' * round(p * 20):<20} {p:.2f}\n",
                             style="bold green" if a == info.get("model_action", action) else "")
            if "threshold" in info:
                panel.append(f"\nthreshold {info['threshold']:.2f}\n", style="dim")
    return Columns([Panel(board, title="Pac-Man"), Panel(panel, title="NextAction decision", width=44)])


# GIF frames ---------------------------------------------------------------------------
CELL, PANEL_W = 28, 300
COLORS = {"bg": (12, 12, 20), "wall": (33, 64, 160), "pellet": (255, 214, 170), "pac": (255, 221, 0),
          "ghost": [(235, 64, 52), (255, 134, 196)], "scared": (70, 110, 255), "text": (230, 230, 235),
          "dim": (120, 120, 135), "bar": (72, 199, 116), "review": (255, 179, 64), "chosen": (255, 221, 0)}


def frame(game: Game, action: str | None, info: dict, font, small):
    from PIL import Image, ImageDraw

    width, height = 15 * CELL + PANEL_W, 13 * CELL
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    for (r, c) in game.walls:
        draw.rectangle([c * CELL + 1, r * CELL + 1, (c + 1) * CELL - 2, (r + 1) * CELL - 2], fill=COLORS["wall"])
    for (r, c) in game.pellets:
        cx, cy = c * CELL + CELL // 2, r * CELL + CELL // 2
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=COLORS["pellet"])
    for (r, c) in game.power_pills:
        cx, cy = c * CELL + CELL // 2, r * CELL + CELL // 2
        draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=COLORS["pellet"])
    for index, ghost in enumerate(game.ghosts):
        r, c = ghost.position
        color = COLORS["scared"] if game.frightened else COLORS["ghost"][index % 2]
        x0, y0 = c * CELL + 3, r * CELL + 3
        draw.pieslice([x0, y0, x0 + CELL - 6, y0 + CELL - 6], 180, 360, fill=color)
        draw.rectangle([x0, y0 + (CELL - 6) // 2, x0 + CELL - 6, y0 + CELL - 6], fill=color)
        for ex in (x0 + 7, x0 + 15):
            draw.ellipse([ex - 3, y0 + 7, ex + 3, y0 + 13], fill=(255, 255, 255))
    r, c = game.pacman
    mouth = {"right": 0, "down": 90, "left": 180, "up": 270}.get(action or "right", 0)
    open_angle = 35 if game.steps % 2 else 10
    draw.pieslice([c * CELL + 3, r * CELL + 3, (c + 1) * CELL - 3, (r + 1) * CELL - 3],
                  mouth + open_angle, mouth - open_angle + 360, fill=COLORS["pac"])

    x = 15 * CELL + 16
    draw.text((x, 10), "NextAction", font=font, fill=COLORS["chosen"])
    draw.text((x, 34), f"step {game.steps}    score {game.score}", font=small, fill=COLORS["text"])
    if action is None:
        label = {"won": "WON", "lost": "CAUGHT", "timeout": "TIMEOUT"}[game.status]
        draw.text((x, 70), label, font=font, fill=COLORS["bar"] if game.status == "won" else COLORS["ghost"][0])
        return image
    source = info["source"]
    draw.text((x, 60), f"decided by: {source}", font=small,
              fill=COLORS["review"] if source == "expert review" else COLORS["bar"])
    draw.text((x, 80), f"move: {action}", font=font, fill=COLORS["text"])
    probabilities = info.get("probabilities") or {}
    y = 116
    for a in ACTIONS:
        draw.text((x, y), a, font=small, fill=COLORS["text"])
        if a in info.get("blocked", []):
            draw.text((x + 60, y), "blocked by policy", font=small, fill=COLORS["dim"])
        else:
            p = probabilities.get(a, 0.0)
            color = COLORS["chosen"] if a == info.get("model_action", action) else COLORS["bar"]
            draw.rectangle([x + 60, y + 3, x + 60 + int(p * 150), y + 15], fill=color)
            draw.text((x + 220, y), f"{p:.2f}", font=small, fill=COLORS["text"])
        y += 26
    if "threshold" in info:
        draw.text((x, y + 6), f"confidence {info['confidence']:.2f}  threshold {info['threshold']:.2f}",
                  font=small, fill=COLORS["dim"])
    return image


def save_gif(frames, path: Path, fps: int) -> None:
    frames = [f.convert("P", palette=1, colors=32) for f in frames]  # adaptive palette
    frames[0].save(path, save_all=True, append_images=frames[1:] + [frames[-1]] * fps * 2,
                   duration=int(1000 / fps), loop=0, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", choices=AGENTS, default="model")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--gif", type=Path, help="record the game to this GIF instead of the terminal")
    parser.add_argument("--pick-win", action="store_true", help="use the first seed from --seed the agent wins")
    args = parser.parse_args()

    seed = args.seed
    if args.pick_win:
        while run_game(args.agent, seed)["status"] != "won":
            seed += 1
    if args.gif:
        from PIL import ImageFont

        font, small = ImageFont.load_default(size=18), ImageFont.load_default(size=13)
        frames = []
        result = run_game(args.agent, seed, lambda g, a, i: frames.append(frame(g, a, i, font, small)))
        save_gif(frames, args.gif, args.fps)
        print({**result, "gif": str(args.gif), "frames": len(frames)})
        return

    from rich.live import Live

    with Live(refresh_per_second=args.fps * 2) as live:
        def show(game, action, info):
            live.update(terminal_view(game, action, info))
            time.sleep(1 / args.fps)
        result = run_game(args.agent, seed, show)
    print(result)


if __name__ == "__main__":
    main()
