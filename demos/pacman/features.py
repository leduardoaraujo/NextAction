"""Turn a game position into a NextAction state string, and the measurements behind it.

The state has two kinds of tags:

- ``[blocked ...]`` and ``[danger ...]`` list actions by name; the decision policy denies them
  with ``deny_tag`` rules (walls, and moves next to a non-frightened ghost);
- ``[features ...]`` holds tokens for the model, each tied to a direction so they do not mix,
  for example ``food_left_1`` (a pellet one step after moving left) or ``ghost_up_close``.
"""

from collections import deque
from dataclasses import dataclass

from game import ACTIONS, Game

FAR = 99


def multi_source(game: Game, sources) -> dict:
    """Distance from every cell to the nearest source cell."""
    seen = {cell: 0 for cell in sources}
    queue = deque(seen)
    while queue:
        cell = queue.popleft()
        for action in ACTIONS:
            nxt = game.neighbor(cell, action)
            if nxt not in game.walls and nxt not in seen:
                seen[nxt] = seen[cell] + 1
                queue.append(nxt)
    return seen


@dataclass
class Measures:
    legal: list[str]
    ghost: dict[str, int]  # distance to the nearest dangerous ghost after each move
    food: dict[str, int]   # distance to the nearest pellet after each move
    prey: dict[str, int]   # distance to the nearest frightened ghost after each move
    threat: int            # distance from Pac-Man to the nearest dangerous ghost


def measure(game: Game) -> Measures:
    ghosts = [g.position for g in game.ghosts]
    danger_map = multi_source(game, ghosts) if ghosts and not game.frightened else {}
    prey_map = multi_source(game, ghosts) if ghosts and game.frightened else {}
    food_map = multi_source(game, game.pellets | game.power_pills)
    legal = game.legal_actions()
    after = {a: game.neighbor(game.pacman, a) for a in legal}
    return Measures(
        legal=legal,
        ghost={a: danger_map.get(c, FAR) for a, c in after.items()},
        food={a: food_map.get(c, FAR) for a, c in after.items()},
        prey={a: prey_map.get(c, FAR) for a, c in after.items()},
        threat=danger_map.get(game.pacman, FAR),
    )


def bucket(distance: int) -> str:
    if distance <= 2:
        return str(distance)
    if distance <= 4:
        return "close"
    if distance <= 8:
        return "mid"
    return "far"


def describe(game: Game) -> str:
    m = measure(game)
    blocked = [a for a in ACTIONS if a not in m.legal]
    danger = [a for a in m.legal if m.ghost[a] <= 1]
    tokens = [f"wall_{a}" for a in blocked]
    for a in m.legal:
        tokens.append(f"food_{a}_{bucket(m.food[a])}")
        tokens.append(f"ghost_{a}_{bucket(m.ghost[a])}")
        if game.frightened:
            tokens.append(f"prey_{a}_{bucket(m.prey[a])}")
    if m.legal:
        tokens.append(f"closest_food_{min(m.legal, key=lambda a: (m.food[a], ACTIONS.index(a)))}")
        tokens.append(f"safest_{max(m.legal, key=lambda a: (m.ghost[a], -ACTIONS.index(a)))}")
    tokens.append(f"threat_{bucket(m.threat)}")
    tokens.append("power_yes" if game.frightened else "power_no")
    return (
        f"[blocked {' '.join(blocked) or 'none'}] [danger {' '.join(danger) or 'none'}] "
        f"[features {' '.join(tokens)}]"
    )
