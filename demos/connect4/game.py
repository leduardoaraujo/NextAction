"""Connect Four on the standard 6×7 board, plus an alpha-beta search player.

Moves are column names ``c1``…``c7``. The search player looks ``depth`` plies ahead with
a heuristic that counts open lines of two and three, tries central columns first, and adds
a little seeded noise between equally good moves so games differ from seed to seed.
"""

import random
from dataclasses import dataclass, field

ROWS, COLS = 6, 7
ACTIONS = tuple(f"c{c + 1}" for c in range(COLS))
CENTER_FIRST = (3, 2, 4, 1, 5, 0, 6)
WIN = 1_000_000


@dataclass
class Board:
    cells: list[list[int]] = field(default_factory=lambda: [[0] * COLS for _ in range(ROWS)])  # row 0 = bottom
    to_move: int = 1  # players are 1 and 2
    moves: int = 0
    winner: int | None = None  # 0 for a draw

    def copy(self) -> "Board":
        return Board([row[:] for row in self.cells], self.to_move, self.moves, self.winner)

    def height(self, col: int) -> int:
        return next((r for r in range(ROWS) if self.cells[r][col] == 0), ROWS)

    def legal(self) -> list[int]:
        return [c for c in CENTER_FIRST if self.height(c) < ROWS] if self.winner is None else []

    def play(self, col: int) -> None:
        row = self.height(col)
        self.cells[row][col] = self.to_move
        self.moves += 1
        if self.connects(row, col):
            self.winner = self.to_move
        elif self.moves == ROWS * COLS:
            self.winner = 0
        self.to_move = 3 - self.to_move

    def connects(self, row: int, col: int) -> bool:
        player = self.cells[row][col]
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < ROWS and 0 <= c < COLS and self.cells[r][c] == player:
                    count += 1
                    r, c = r + sign * dr, c + sign * dc
            if count >= 4:
                return True
        return False

    def wins_with(self, col: int, player: int) -> bool:
        """Would ``player`` connect four by dropping in ``col``?"""
        row = self.height(col)
        if row >= ROWS:
            return False
        self.cells[row][col] = player
        result = self.connects(row, col)
        self.cells[row][col] = 0
        return result


def windows():
    for r in range(ROWS):
        for c in range(COLS):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + i * dr, c + i * dc) for i in range(4)]
                if all(0 <= rr < ROWS and 0 <= cc < COLS for rr, cc in cells):
                    yield cells


WINDOWS = list(windows())


def evaluate(board: Board, player: int) -> int:
    """Symmetric heuristic from ``player``'s point of view: center control and open lines."""
    opponent = 3 - player
    score = 0
    for r in range(ROWS):
        if board.cells[r][3] == player:
            score += 3
        elif board.cells[r][3] == opponent:
            score -= 3
    for cells in WINDOWS:
        values = [board.cells[r][c] for r, c in cells]
        mine, theirs = values.count(player), values.count(opponent)
        if mine and not theirs:
            score += (0, 1, 4, 20, 1000)[mine]
        elif theirs and not mine:
            score -= (0, 1, 4, 20, 1000)[theirs]
    return score


def negamax(board: Board, depth: int, alpha: int, beta: int) -> int:
    if board.winner is not None:
        return 0 if board.winner == 0 else -(WIN + depth)  # the player to move has lost
    if depth == 0:
        return evaluate(board, board.to_move)
    best = -10 * WIN
    for col in board.legal():
        child = board.copy()
        child.play(col)
        best = max(best, -negamax(child, depth - 1, -beta, -alpha))
        alpha = max(alpha, best)
        if alpha >= beta:
            break
    return best


def search_move(board: Board, depth: int, rng: random.Random | None = None) -> int:
    scores = {}
    for col in board.legal():
        child = board.copy()
        child.play(col)
        scores[col] = -negamax(child, depth - 1, -10 * WIN, 10 * WIN)
    best = max(scores.values())
    tied = [c for c in board.legal() if scores[c] >= best - (2 if abs(best) < WIN else 0)]
    return rng.choice(tied) if rng else tied[0]
