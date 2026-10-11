"""A rule-based Pac-Man player that labels the training data.

It plays the role a human reviewer would in a real agent: the model learns to imitate it,
and the decision policy hands it the move whenever the model is not confident.
"""

import random

from features import FAR, measure
from game import ACTIONS, Game


def expert_action(game: Game) -> str:
    m = measure(game)
    if not m.legal:
        return "up"
    order = lambda a: ACTIONS.index(a)
    safe = [a for a in m.legal if m.ghost[a] > 2] or [
        a for a in m.legal if m.ghost[a] == max(m.ghost[b] for b in m.legal)
    ]
    hunting = [a for a in safe if m.prey[a] <= 6]
    if game.frightened and hunting:
        return min(hunting, key=lambda a: (m.prey[a], m.food[a], order(a)))
    if m.threat <= 3:
        # Run first, eat second.
        return min(safe, key=lambda a: (-min(m.ghost[a], 6), m.food[a], order(a)))
    return min(safe, key=lambda a: (m.food[a], -min(m.ghost[a], FAR), order(a)))


def random_action(game: Game, rng: random.Random) -> str:
    return rng.choice(game.legal_actions() or list(ACTIONS))
