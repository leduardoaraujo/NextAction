"""Where and why the model goes wrong: error analysis over the benchmark games.

    python demos/pacman/analysis.py --games 300

Plays the benchmark seeds with the model (policy + expert on low confidence) and with the
expert, logging every move. At each model move it also asks the expert what it would do, so
"disagreement" means the model's choice differs from its teacher's. Writes ANALYSIS.md and
the charts in demos/pacman/analysis/.
"""

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from expert import expert_action
from features import describe, measure
from game import ACTIONS, Game
from play import Agent, run_game

HERE = Path(__file__).resolve().parent
CHARTS = HERE / "analysis"
SEED_OFFSET = 200_000
# Reference palette (validated for color-vision deficiencies), light mode.
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODEL_COLOR, EXPERT_COLOR = "#2a78d6", "#eb6834"
THREAT_ORDER = ["0", "1", "2", "close", "mid", "far"]
THREAT_LABELS = {"0": "0", "1": "1", "2": "2", "close": "3–4", "mid": "5–8", "far": "9+"}


def threat_bucket(distance: int) -> str:
    return "0" if distance <= 0 else "1" if distance == 1 else "2" if distance == 2 else \
        "close" if distance <= 4 else "mid" if distance <= 8 else "far"


def play_logged(kind: str, seed: int) -> dict:
    """Play one game, recording each move with its context."""
    game, agent = Game(seed=seed), Agent(kind, seed)
    moves = []
    while game.status == "playing":
        m = measure(game)
        action, info = agent.decide(game)
        expert = expert_action(game) if kind == "model" else action
        model = info.get("model_action")
        moves.append({
            "step": game.steps, "chosen": action, "source": info["source"],
            "model": model, "confidence": info.get("confidence"), "expert": expert,
            # Every open move is next to a dangerous ghost: no move avoids being caught.
            "trapped": bool(m.legal) and all(m.ghost[a] <= 1 for a in m.legal),
            # Model and expert moves are equally good by the expert's own measures.
            "tie": model is not None and model in m.legal and expert in m.legal and model != expert
                   and m.food[model] == m.food[expert] and min(m.ghost[model], 6) == min(m.ghost[expert], 6)
                   and min(m.prey[model], 7) == min(m.prey[expert], 7),
            "threat": threat_bucket(m.threat), "junction": len(m.legal) >= 3,
            "power": bool(game.frightened), "state": describe(game) if kind == "model" else None,
        })
        game.step(action)
    return {"seed": seed, "status": game.status, "score": game.score, "steps": game.steps,
            "pellets_left": len(game.pellets) + len(game.power_pills), "moves": moves}


def rate(items, predicate) -> str:
    items = list(items)
    return f"{sum(map(predicate, items)) / len(items):.1%}" if items else "-"


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, pad=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def charts(model_games, expert_games, model_moves, threshold: float) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CHARTS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.facecolor": SURFACE})
    names = []

    # 1. Disagreement with the expert by distance to the nearest dangerous ghost.
    decided = [m for m in model_moves if m["model"] is not None]
    by_threat = defaultdict(list)
    for move in decided:
        by_threat[move["threat"]].append(move["model"] != move["expert"])
    buckets = [b for b in THREAT_ORDER if by_threat[b]]
    values = [100 * sum(by_threat[b]) / len(by_threat[b]) for b in buckets]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar([THREAT_LABELS[b] for b in buckets], values, color=MODEL_COLOR, width=0.6,
                  edgecolor=SURFACE, linewidth=2)
    for bar, value, bucket in zip(bars, values, buckets):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.3, f"{value:.1f}%\n(n={len(by_threat[bucket])})",
                ha="center", va="bottom", color=TEXT, fontsize=8)
    style(ax, "Disagreement is highest when ghosts are far and the choice is about food",
          "steps to the nearest dangerous ghost", "moves where model ≠ expert (%)")
    ax.set_ylim(0, max(values) * 1.35)
    fig.tight_layout()
    fig.savefig(CHARTS / "disagreement_by_ghost_distance.png", dpi=150)
    names.append("disagreement_by_ghost_distance.png")
    plt.close(fig)

    # 2. Confidence of agreeing vs disagreeing moves, with the review threshold.
    agree = [m["confidence"] for m in decided if m["model"] == m["expert"]]
    disagree = [m["confidence"] for m in decided if m["model"] != m["expert"]]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = [i / 20 for i in range(5, 21)]
    ax.hist(agree, bins=bins, weights=[100 / len(agree)] * len(agree), color=MODEL_COLOR,
            label=f"agrees with expert (n={len(agree):,})", edgecolor=SURFACE, linewidth=2)
    ax.hist(disagree, bins=bins, weights=[100 / len(disagree)] * len(disagree), histtype="step",
            color=EXPERT_COLOR, linewidth=2, label=f"disagrees (n={len(disagree):,})")
    ax.axvline(threshold, color=TEXT, linewidth=1, linestyle="--")
    ax.text(threshold + 0.01, ax.get_ylim()[1] * 0.92, f"review threshold {threshold}", color=TEXT, fontsize=8)
    style(ax, "Wrong moves are less confident, but most sit above the threshold",
          "model confidence", "share of moves in each group (%)")
    ax.legend(frameon=False, labelcolor=TEXT, loc="upper left")
    fig.tight_layout()
    fig.savefig(CHARTS / "confidence_agree_vs_disagree.png", dpi=150)
    names.append("confidence_agree_vs_disagree.png")
    plt.close(fig)

    # 3. Pellets left when Pac-Man is caught, model vs expert.
    lost_model = [g["pellets_left"] for g in model_games if g["status"] == "lost"]
    lost_expert = [g["pellets_left"] for g in expert_games if g["status"] == "lost"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    edges = list(range(0, 100, 10))
    ax.hist([lost_model, lost_expert], bins=edges, color=[MODEL_COLOR, EXPERT_COLOR],
            label=[f"model (n={len(lost_model)} losses)", f"expert (n={len(lost_expert)} losses)"],
            edgecolor=SURFACE, linewidth=2)
    style(ax, "Model and expert get caught at similar points in the game",
          "pellets left when caught", "lost games")
    ax.legend(frameon=False, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(CHARTS / "pellets_left_when_caught.png", dpi=150)
    names.append("pellets_left_when_caught.png")
    plt.close(fig)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=300)
    parser.add_argument("--thresholds", default="0.6,0.7,0.8,0.9", help="review thresholds to measure")
    args = parser.parse_args()
    seeds = range(SEED_OFFSET, SEED_OFFSET + args.games)
    threshold = Agent("model").policy.default_threshold

    model_games = [play_logged("model", s) for s in seeds]
    expert_games = [play_logged("expert", s) for s in seeds]
    moves = [m for g in model_games for m in g["moves"]]
    decided = [m for m in moves if m["model"] is not None]
    wrong = [m for m in decided if m["model"] != m["expert"]]
    reviewed = [m for m in moves if m["source"] == "expert review"]
    names = charts(model_games, expert_games, moves, threshold)
    sweep = []
    for value in map(float, args.thresholds.split(",")):
        results = [run_game("model", s, threshold=value) for s in seeds]
        steps = sum(r["steps"] for r in results)
        sweep.append((value, sum(r["status"] == "won" for r in results), sum(r["reviews"] for r in results) / steps,
                      sum(r["score"] for r in results) / len(results)))
    expert_wins = sum(g["status"] == "won" for g in expert_games)

    # Outcomes on the same seeds.
    pairs = Counter((m["status"] == "won", e["status"] == "won") for m, e in zip(model_games, expert_games))
    lost = [g for g in model_games if g["status"] == "lost"]
    fatal = [g["moves"][-1] for g in lost]
    confusion = Counter((m["expert"], m["model"]) for m in wrong)

    def situation(label, subset):
        subset = list(subset)
        return f"| {label} | {len(subset):,} | {rate(subset, lambda m: m['model'] != m['expert'])} |"

    lines = [
        "# Pac-Man error analysis",
        "",
        f"Generated by `python demos/pacman/analysis.py --games {args.games}` on the benchmark seeds "
        f"{seeds.start}–{seeds.stop - 1}. At every model move the expert is also asked what it would do; "
        "a *disagreement* is a move where the model's choice differs from its teacher's.",
        "",
        "## Summary",
        "",
        "| | |",
        "| --- | --- |",
        f"| Moves logged | {len(moves):,} in {args.games} games |",
        f"| Model disagrees with the expert | {len(wrong):,} moves ({len(wrong) / len(decided):.1%}) |",
        f"| Moves sent to the expert (confidence < {threshold}) | {len(reviewed):,} ({len(reviewed) / len(moves):.1%}) |",
        f"| Disagreements caught by the threshold | {rate(wrong, lambda m: m['confidence'] < threshold)} |",
        f"| Reviewed moves where the expert changed the move | {rate(reviewed, lambda m: m['model'] != m['expert'])} |",
        f"| Median confidence: agreeing / disagreeing moves | {statistics.median(m['confidence'] for m in decided if m['model'] == m['expert']):.2f} / {statistics.median(m['confidence'] for m in wrong):.2f} |",
        "",
        "## Where the model disagrees",
        "",
        f"![Disagreement by ghost distance](analysis/{names[0]})",
        "",
        "| Situation | Moves | Disagreement |",
        "| --- | --- | --- |",
        situation("all moves", decided),
        situation("ghost 0–2 steps away", (m for m in decided if m["threat"] in ("0", "1", "2"))),
        situation("ghost 3–8 steps away", (m for m in decided if m["threat"] in ("close", "mid"))),
        situation("ghost 9+ steps away", (m for m in decided if m["threat"] == "far")),
        situation("at a junction (3+ open moves)", (m for m in decided if m["junction"])),
        situation("in a corridor (≤2 open moves)", (m for m in decided if not m["junction"])),
        situation("power pill active", (m for m in decided if m["power"])),
        situation("no power pill", (m for m in decided if not m["power"])),
        "",
        f"**{rate(wrong, lambda m: m['tie'])} of disagreements are ties**: the model's move and the expert's "
        "are equally close to food and equally far from ghosts by the expert's own measures, so the "
        "expert's choice there is an arbitrary tie-break the model cannot learn. "
        f"Excluding ties, the model disagrees on {sum(not m['tie'] for m in wrong) / len(decided):.1%} of moves.",
        "",
        "Most common disagreements (expert's move → model's move):",
        "",
        "| Expert | Model | Moves |",
        "| --- | --- | --- |",
        *[f"| {e} | {mo} | {n} |" for (e, mo), n in confusion.most_common(6)],
        "",
        "## Does confidence flag the bad moves?",
        "",
        f"![Confidence of agreeing and disagreeing moves](analysis/{names[1]})",
        "",
        "| Confidence | Moves | Disagreement |",
        "| --- | --- | --- |",
        *[situation(f"{lo:.2f}–{min(hi, 1):.2f}", (m for m in decided if lo <= m["confidence"] < hi))
          for lo, hi in ((0.25, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01))],
        "",
        "What a different review threshold would do on these same moves (the games would change a little, "
        "since reviewed moves alter the trajectory):",
        "",
        "| Threshold | Moves sent to the expert | Disagreements caught | Disagreements left to the model |",
        "| --- | --- | --- | --- |",
        *[f"| {t} | {rate(decided, lambda m, t=t: m['confidence'] < t)} | "
          f"{rate(wrong, lambda m, t=t: m['confidence'] < t)} | "
          f"{sum(m['confidence'] >= t for m in wrong):,} |" for t in (0.6, 0.7, 0.8, 0.9)],
        "",
        "Measured by replaying all games with each threshold:",
        "",
        "| Threshold | Win rate | Moves decided by the expert | Avg. score |",
        "| --- | --- | --- | --- |",
        *[f"| {t} | {w / args.games:.0%} | {r:.1%} | {sc:.0f} |" for t, w, r, sc in sweep],
        "",
        f"Sending more moves to the expert does not win more games, because the expert is no stronger "
        f"than the model it taught (it wins {expert_wins / args.games:.0%} of the same games). Review pays off "
        "only when the reviewer is better than the model; here it mostly swaps one good move for another.",
        "",
        "## Why games are lost",
        "",
        f"![Pellets left when caught](analysis/{names[2]})",
        "",
        "| | Model | Expert |",
        "| --- | --- | --- |",
        f"| Games won | {sum(g['status'] == 'won' for g in model_games)} | {sum(g['status'] == 'won' for g in expert_games)} |",
        f"| Games lost (caught by a ghost) | {len(lost)} | {sum(g['status'] == 'lost' for g in expert_games)} |",
        f"| Median step when caught | {statistics.median(g['steps'] for g in lost):.0f} | {statistics.median(g['steps'] for g in expert_games if g['status'] == 'lost'):.0f} |",
        f"| Median pellets left when caught | {statistics.median(g['pellets_left'] for g in lost):.0f} | {statistics.median(g['pellets_left'] for g in expert_games if g['status'] == 'lost'):.0f} |",
        "",
        "The fatal move of each lost model game:",
        "",
        "| | Share of losses |",
        "| --- | --- |",
        f"| Pac-Man was already trapped (every open move next to a ghost) | {rate(fatal, lambda m: m['trapped'])} |",
        f"| Decided by the model | {rate(fatal, lambda m: m['source'] == 'model')} |",
        f"| Decided by the expert (review) | {rate(fatal, lambda m: m['source'] == 'expert review')} |",
        f"| The expert would have chosen the same move | {rate(fatal, lambda m: m['chosen'] == m['expert'])} |",
        f"| A ghost was 0–2 steps away before the move | {rate(fatal, lambda m: m['threat'] in ('0', '1', '2'))} |",
        f"| Pac-Man was in a corridor (≤2 open moves) | {rate(fatal, lambda m: not m['junction'])} |",
        "",
        "Outcomes on the same seeds:",
        "",
        "| | Expert won | Expert lost |",
        "| --- | --- | --- |",
        f"| **Model won** | {pairs[(True, True)]} | {pairs[(True, False)]} |",
        f"| **Model lost** | {pairs[(False, True)]} | {pairs[(False, False)]} |",
    ]
    (HERE / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[6:16]))
    print(f"saved {HERE / 'ANALYSIS.md'}")


if __name__ == "__main__":
    main()
