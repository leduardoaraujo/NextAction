"""A small, deterministic Pac-Man on a grid, written in plain Python.

Rules: Pac-Man moves one cell per step; eating every pellet wins, touching a ghost loses.
A power pill frightens the ghosts for a while, and frightened ghosts can be eaten. Ghosts
chase Pac-Man most of the time and move a little slower than him.
"""

import random
from collections import deque
from dataclasses import dataclass, field

MAZE = [
    "###############",
    "#o.....#.....o#",
    "#.##.#.#.#.##.#",
    "#....#...#....#",
    "#.##.##.##.##.#",
    "#.....G.G.....#",
    "###.#.###.#.###",
    "#...#..P..#...#",
    "#.#.#.###.#.#.#",
    "#.#.........#.#",
    "#.###.#.#.###.#",
    "#o....#.#....o#",
    "###############",
]
ACTIONS = ("up", "down", "left", "right")
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
PELLET, POWER_PILL, GHOST, POWER_STEPS = 10, 50, 200, 20
MAX_STEPS = 600

Cell = tuple[int, int]


@dataclass
class Ghost:
    position: Cell
    start: Cell
    direction: str = "up"


@dataclass
class Game:
    seed: int = 0
    walls: set[Cell] = field(default_factory=set)
    pellets: set[Cell] = field(default_factory=set)
    power_pills: set[Cell] = field(default_factory=set)
    pacman: Cell = (0, 0)
    ghosts: list[Ghost] = field(default_factory=list)
    score: int = 0
    steps: int = 0
    frightened: int = 0
    status: str = "playing"  # playing | won | lost | timeout

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        for row, line in enumerate(MAZE):
            for col, char in enumerate(line):
                cell = (row, col)
                if char == "#":
                    self.walls.add(cell)
                elif char == ".":
                    self.pellets.add(cell)
                elif char == "o":
                    self.power_pills.add(cell)
                elif char == "P":
                    self.pacman = cell
                elif char == "G":
                    self.ghosts.append(Ghost(cell, cell))

    # Geometry -------------------------------------------------------------------------
    def neighbor(self, cell: Cell, action: str) -> Cell:
        dr, dc = MOVES[action]
        return (cell[0] + dr, cell[1] + dc)

    def legal_actions(self, cell: Cell | None = None) -> list[str]:
        cell = self.pacman if cell is None else cell
        return [a for a in ACTIONS if self.neighbor(cell, a) not in self.walls]

    def distances(self, source: Cell) -> dict[Cell, int]:
        """Shortest path length from ``source`` to every reachable cell."""
        seen, queue = {source: 0}, deque([source])
        while queue:
            cell = queue.popleft()
            for action in ACTIONS:
                nxt = self.neighbor(cell, action)
                if nxt not in self.walls and nxt not in seen:
                    seen[nxt] = seen[cell] + 1
                    queue.append(nxt)
        return seen

    # Dynamics -------------------------------------------------------------------------
    def step(self, action: str) -> None:
        if self.status != "playing":
            return
        self.steps += 1
        before = {id(g): g.position for g in self.ghosts}
        if action in self.legal_actions():
            self.pacman = self.neighbor(self.pacman, action)
        self._eat()
        self._collide()
        if self.status == "playing":
            self._move_ghosts()
            self._collide(swapped_from=before)
        if self.frightened:
            self.frightened -= 1
        if self.status == "playing" and not self.pellets and not self.power_pills:
            self.status = "won"
        elif self.status == "playing" and self.steps >= MAX_STEPS:
            self.status = "timeout"

    def _eat(self) -> None:
        if self.pacman in self.pellets:
            self.pellets.discard(self.pacman)
            self.score += PELLET
        elif self.pacman in self.power_pills:
            self.power_pills.discard(self.pacman)
            self.score += POWER_PILL
            self.frightened = POWER_STEPS

    def _collide(self, swapped_from: dict | None = None) -> None:
        for ghost in self.ghosts:
            crossed = swapped_from and swapped_from[id(ghost)] == self.pacman
            if ghost.position == self.pacman or crossed:
                if self.frightened:
                    self.score += GHOST
                    ghost.position = ghost.start
                else:
                    self.status = "lost"
                    return

    def _move_ghosts(self) -> None:
        # Ghosts skip every fourth step (every other step while frightened).
        if self.steps % 4 == 0 or (self.frightened and self.steps % 2 == 0):
            return
        from_pacman = self.distances(self.pacman)
        for ghost in self.ghosts:
            options = [a for a in self.legal_actions(ghost.position)]
            reverse = {"up": "down", "down": "up", "left": "right", "right": "left"}[ghost.direction]
            if len(options) > 1 and reverse in options:
                options.remove(reverse)
            if self.frightened:
                pick = max if self.rng.random() < 0.8 else self.rng.choice
            else:
                pick = min if self.rng.random() < 0.7 else self.rng.choice
            if pick in (min, max):
                key = lambda a: from_pacman.get(self.neighbor(ghost.position, a), 99)
                action = pick(options, key=key)
            else:
                action = pick(options)
            ghost.direction = action
            ghost.position = self.neighbor(ghost.position, action)
