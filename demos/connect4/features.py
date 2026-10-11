"""Turn a Connect Four position into a NextAction state string, from the side to move.

- ``[blocked ...]`` lists full columns and ``[danger ...]`` lists columns that would let the
  opponent win right on top; policy rules deny both with ``deny_tag``.
- ``[features ...]`` holds tokens for the model: immediate wins and blocks, how each column
  changes the heuristic score, the column heights, and every occupied cell as ``me`` or ``op``.
"""

from game import ACTIONS, COLS, ROWS, Board, evaluate


def score_bucket(delta: int) -> str:
    for limit, name in ((-20, "awful"), (-5, "bad"), (0, "worse"), (1, "flat"), (6, "better"), (21, "good")):
        if delta < limit:
            return name
    return "great"


def analyse(board: Board) -> dict:
    me, op = board.to_move, 3 - board.to_move
    legal = set(board.legal())
    base = evaluate(board, me)
    info = {"blocked": [], "danger": [], "win": [], "block": [], "delta": {}}
    for col in range(COLS):
        name = ACTIONS[col]
        if col not in legal:
            info["blocked"].append(name)
            continue
        if board.wins_with(col, me):
            info["win"].append(name)
        if board.wins_with(col, op):
            info["block"].append(name)
        child = board.copy()
        child.play(col)
        # Playing here hands the opponent a winning cell right above.
        if child.winner is None and child.height(col) < ROWS and child.wins_with(col, op):
            info["danger"].append(name)
        info["delta"][name] = evaluate(child, me) - base if child.winner is None else 100
    return info


def describe(board: Board) -> str:
    info = analyse(board)
    me = board.to_move
    tokens = [f"win_{c}" for c in info["win"]] + [f"block_{c}" for c in info["block"]]
    tokens += [f"{c}_{score_bucket(d)}" for c, d in info["delta"].items()]
    tokens += [f"h{c}_{board.height(i)}" for i, c in enumerate(ACTIONS) if c not in info["blocked"]]
    tokens += [
        f"r{r}{ACTIONS[c]}_{'me' if board.cells[r][c] == me else 'op'}"
        for r in range(ROWS) for c in range(COLS) if board.cells[r][c]
    ]
    return (
        f"[blocked {' '.join(info['blocked']) or 'none'}] [danger {' '.join(info['danger']) or 'none'}] "
        f"[features {' '.join(tokens) or 'empty'}]"
    )
