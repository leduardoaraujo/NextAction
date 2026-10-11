"""Watch an agent play Connect Four and see every decision it makes.

    python demos/connect4/play.py                               # model vs depth-4 search, terminal
    python demos/connect4/play.py --gif demos/connect4/connect4.gif --pick-win

Agents:
  model                 model + policy; unsure moves are reviewed by a depth-6 search (stronger
                        than the depth-4 teacher it learned from)
  model-teacher-review  model + policy; unsure moves go back to the depth-4 teacher
  model-only            model + policy, no review
  teacher               the depth-4 search that labeled the training data
  strong                a depth-6 search
  random                a random legal column

The opponent is a depth-4 search. The panel shows each column's probability, the columns the
policy blocked (full, or handing the opponent a win), the threshold and who decided.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import joblib

from features import describe
from game import ACTIONS, COLS, ROWS, Board, search_move

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nextaction.policy import apply  # noqa: E402
from nextaction.scenarios import get_scenario  # noqa: E402

AGENTS = ("model", "model-teacher-review", "model-only", "teacher", "strong", "random")
TEACHER_DEPTH, REVIEW_DEPTH, OPPONENT_DEPTH = 4, 6, 4
OPENING_PLIES = 2  # random opening moves, the same for every agent on a given seed


class Agent:
    def __init__(self, kind: str, seed: int = 0):
        self.kind, self.rng = kind, random.Random(seed)
        if kind.startswith("model"):
            scenario = get_scenario("connect4")
            self.model, self.policy = joblib.load(scenario.model), scenario.policy

    def decide(self, board: Board) -> tuple[int, dict]:
        if self.kind == "random":
            return self.rng.choice(board.legal()), {"source": "random"}
        if self.kind in ("teacher", "strong"):
            depth = TEACHER_DEPTH if self.kind == "teacher" else REVIEW_DEPTH
            return search_move(board, depth, self.rng), {"source": f"search depth {depth}"}
        state = describe(board)
        probabilities = dict(zip(self.model.classes_, map(float, self.model.predict_proba([state])[0])))
        decision = apply(probabilities, state, self.policy)
        info = {"source": "model", "probabilities": decision["probabilities"], "blocked": decision["blocked_actions"],
                "threshold": decision["threshold"], "model_action": decision["action"],
                "confidence": decision["probabilities"][decision["action"]], "raw": probabilities}
        if decision["needs_review"] and self.kind != "model-only":
            depth = REVIEW_DEPTH if self.kind == "model" else TEACHER_DEPTH
            info["source"] = f"review: search depth {depth}"
            return search_move(board, depth, self.rng), info
        return ACTIONS.index(decision["action"]), info


def run_game(kind: str, seed: int, agent_first: bool = True, on_step=None) -> dict:
    """One game against the depth-4 opponent. Returns the result from the agent's side.

    The first ``OPENING_PLIES`` moves are random but fixed by the seed, so every agent faces
    the same varied openings; otherwise searches replay a handful of opening lines.
    """
    board, agent, rng = Board(), Agent(kind, seed), random.Random(seed + 1)
    agent_player = 1 if agent_first else 2
    opening = random.Random(seed + 2)
    for _ in range(OPENING_PLIES):
        board.play(opening.choice(board.legal()))
    reviews = moves = 0
    while board.winner is None:
        if board.to_move == agent_player:
            col, info = agent.decide(board)
            moves += 1
            reviews += info["source"].startswith("review")
        else:
            col, info = search_move(board, OPPONENT_DEPTH, rng), {"source": "opponent"}
        if on_step:
            on_step(board, col, info, agent_player)
        board.play(col)
    if on_step:
        on_step(board, None, {"source": "end"}, agent_player)
    result = "draw" if board.winner == 0 else "won" if board.winner == agent_player else "lost"
    return {"agent": kind, "seed": seed, "first": agent_first, "result": result, "moves": moves, "reviews": reviews}


# Terminal view -----------------------------------------------------------------------
def terminal_view(board: Board, col: int | None, info: dict, agent_player: int):
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.text import Text

    grid = Text()
    for r in reversed(range(ROWS)):
        for c in range(COLS):
            cell = board.cells[r][c]
            grid.append("● " if cell else "· ", style=("bold red" if cell == agent_player else "bold yellow") if cell else "dim")
        grid.append("\n")
    grid.append(" ".join(str(c + 1) for c in range(COLS)))
    panel = Text(f"move {board.moves}\n", style="bold")
    if col is None:
        result = "draw" if board.winner == 0 else "agent won" if board.winner == agent_player else "agent lost"
        panel.append(f"\n{result.upper()}\n", style="bold")
    elif "probabilities" in info:
        panel.append(f"decided by: {info['source']}\n", style="yellow" if info["source"].startswith("review") else "green")
        panel.append(f"plays: column {col + 1}\n\n", style="bold")
        for name in ACTIONS:
            if name in info["blocked"]:
                panel.append(f"{name}  blocked\n", style="dim")
            else:
                p = info["probabilities"].get(name, 0.0)
                panel.append(f"{name}  {'█' * round(p * 20):<20} {p:.2f}\n",
                             style="bold green" if name == info["model_action"] else "")
    else:
        panel.append(f"{info['source']} plays column {col + 1}\n")
    return Columns([Panel(grid, title="Connect Four"), Panel(panel, title="NextAction decision", width=44)])


# GIF frames ---------------------------------------------------------------------------
CELL, PANEL_W = 56, 320
COLORS = {"bg": (12, 12, 20), "board": (33, 64, 160), "hole": (12, 12, 20), "agent": (235, 64, 52),
          "opponent": (255, 214, 0), "text": (230, 230, 235), "dim": (120, 120, 135), "bar": (72, 199, 116),
          "review": (255, 179, 64), "chosen": (255, 221, 0)}


def frame(board: Board, col: int | None, info: dict, agent_player: int, font, small):
    from PIL import Image, ImageDraw

    width, height = COLS * CELL + PANEL_W, ROWS * CELL + 30
    image = Image.new("RGB", (width, height), COLORS["bg"])
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle([0, 30, COLS * CELL, height - 1], 10, fill=COLORS["board"])
    for r in range(ROWS):
        for c in range(COLS):
            x, y = c * CELL + 6, 30 + (ROWS - 1 - r) * CELL + 6
            cell = board.cells[r][c]
            color = COLORS["hole"] if not cell else COLORS["agent"] if cell == agent_player else COLORS["opponent"]
            draw.ellipse([x, y, x + CELL - 12, y + CELL - 12], fill=color)
    if col is not None:
        cx = col * CELL + CELL // 2
        draw.polygon([(cx - 9, 6), (cx + 9, 6), (cx, 22)],
                     fill=COLORS["agent"] if board.to_move == agent_player else COLORS["opponent"])
    x = COLS * CELL + 16
    draw.text((x, 8), "NextAction", font=font, fill=COLORS["chosen"])
    draw.text((x, 32), f"move {board.moves}   red = model", font=small, fill=COLORS["text"])
    if col is None:
        label = "DRAW" if board.winner == 0 else "MODEL WINS" if board.winner == agent_player else "MODEL LOSES"
        draw.text((x, 70), label, font=font, fill=COLORS["bar"] if board.winner == agent_player else COLORS["text"])
        return image
    if "probabilities" not in info:
        draw.text((x, 70), f"opponent plays c{col + 1}", font=small, fill=COLORS["dim"])
        return image
    source = info["source"]
    draw.text((x, 58), f"decided by: {source}", font=small,
              fill=COLORS["review"] if source.startswith("review") else COLORS["bar"])
    draw.text((x, 78), f"plays: column {col + 1}", font=font, fill=COLORS["text"])
    y = 112
    for name in ACTIONS:
        draw.text((x, y), name, font=small, fill=COLORS["text"])
        if name in info["blocked"]:
            draw.text((x + 40, y), "blocked by policy", font=small, fill=COLORS["dim"])
        else:
            p = info["probabilities"].get(name, 0.0)
            color = COLORS["chosen"] if name == info["model_action"] else COLORS["bar"]
            draw.rectangle([x + 40, y + 3, x + 40 + int(p * 180), y + 14], fill=color)
            draw.text((x + 232, y), f"{p:.2f}", font=small, fill=COLORS["text"])
        y += 24
    draw.text((x, y + 6), f"confidence {info['confidence']:.2f}  threshold {info['threshold']:.2f}",
              font=small, fill=COLORS["dim"])
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", choices=AGENTS, default="model")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--second", action="store_true", help="the agent moves second")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--gif", type=Path)
    parser.add_argument("--pick-win", action="store_true", help="first seed from --seed the agent wins with a review")
    args = parser.parse_args()

    seed, first = args.seed, not args.second
    if args.pick_win:
        while True:
            result = run_game(args.agent, seed, first)
            if result["result"] == "won" and result["reviews"]:
                break
            seed += 1
    if args.gif:
        from PIL import ImageFont

        font, small = ImageFont.load_default(size=18), ImageFont.load_default(size=13)
        frames = []
        result = run_game(args.agent, seed, first, lambda b, c, i, p: frames.append(frame(b, c, i, p, font, small)))
        frames = [f.convert("P", palette=1, colors=32) for f in frames]
        frames[0].save(args.gif, save_all=True, append_images=frames[1:] + [frames[-1]] * args.fps * 3,
                       duration=int(1000 / args.fps), loop=0, optimize=True)
        print({**result, "gif": str(args.gif), "frames": len(frames)})
        return

    from rich.live import Live

    with Live(refresh_per_second=8) as live:
        def show(board, col, info, player):
            live.update(terminal_view(board, col, info, player))
            time.sleep(1 / args.fps)
        print(run_game(args.agent, seed, first, show))


if __name__ == "__main__":
    main()
