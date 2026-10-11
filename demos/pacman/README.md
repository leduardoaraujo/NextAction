# Pac-Man demo

NextAction picking Pac-Man's next move in real time. The panel shows every decision: the probability of each move, the moves the decision policy blocked, the confidence against the threshold, and who decided. When the model is unsure, the move goes to a rule-based expert, the game's stand-in for a human reviewer.

![NextAction playing Pac-Man](pacman.gif)

The game uses the same pipeline as every other scenario: labeled `state,action` examples, `nextaction train`, `nextaction evaluate` and a decision policy in [`configs/pacman.toml`](../../configs/pacman.toml).

## Run it

From the repository root:

```bash
pip install -e ".[demo]"

python demos/pacman/generate_data.py       # expert plays 400 training and 100 test games
nextaction train pacman                    # ~50k examples, a few seconds
nextaction evaluate pacman                 # accuracy on moves from unseen games

python demos/pacman/play.py                # watch the model play in the terminal
python demos/pacman/play.py --agent expert # or the expert, or random, raw, model-only
python demos/pacman/play.py --gif demos/pacman/pacman.gif --pick-win
python demos/pacman/benchmark.py --games 300
python demos/pacman/analysis.py --games 300    # error analysis and charts
```

| Script | What it does |
| --- | --- |
| [`game.py`](game.py) | the game: a 15×13 maze, two ghosts that chase most of the time, power pills; deterministic per seed |
| [`features.py`](features.py) | turns a position into a NextAction state string |
| [`expert.py`](expert.py) | the rule-based player that labels the training data and reviews unsure moves |
| [`generate_data.py`](generate_data.py) | records expert games as training and test CSVs (different seeds) |
| [`play.py`](play.py) | runs one game with any agent, live in the terminal or recorded to a GIF |
| [`benchmark.py`](benchmark.py) | plays hundreds of unseen games per agent and writes [RESULTS.md](RESULTS.md) |
| [`analysis.py`](analysis.py) | logs every move, compares it with the expert's, and writes [ANALYSIS.md](ANALYSIS.md) with charts |

## State and policy

Each position becomes a state such as:

```text
[blocked up down] [danger none] [features wall_up wall_down food_left_0 ghost_left_close food_right_0 ghost_right_close closest_food_left safest_left threat_mid power_no]
```

- `[features ...]` are tokens for the model, each tied to a direction: distance to the nearest pellet (`food_left_0`), to the nearest dangerous ghost (`ghost_up_close`), to a frightened ghost (`prey_right_2`), the closest-food and safest directions, and whether a power pill is active.
- `[blocked ...]` and `[danger ...]` list moves by name: walls, and cells next to a ghost that is not frightened. Two policy rules with `deny_tag` remove them before the model's choice, so a wall or a ghost can never be chosen.
- A move below 0.6 confidence goes to the expert.

## Results

Imitation accuracy on 12,734 moves from 100 unseen games: **0.973** (always choosing the most common move: 0.251).

Playing 300 games that never appear in training or test data:

| Agent | Win rate | 95% interval | Avg. score | Moves decided by the expert |
| --- | --- | --- | --- | --- |
| model + policy, expert on low confidence | 39% | 33%–44% | 3079 | 3.3% |
| model + policy, alone | 40% | 35%–46% | 2928 | — |
| model alone, no policy | 40% | 35%–46% | 2928 | — |
| rule-based expert | 35% | 30%–41% | 3258 | — |
| random legal move | 0% | 0%–1% | 43 | — |

- **The model plays as well as the expert it learned from.** Win rates overlap within the 95% intervals, so neither is better; the expert scores a bit more because it hunts frightened ghosts more often.
- **It asks for help on 3.3% of moves**, the positions where its confidence falls below the threshold.
- **The policy is a safety net here, not a crutch.** The model almost never picks a wall or a move next to a ghost on its own (0.003 corrections per game), so with or without the policy it wins the same share of games. The rules guarantee that it cannot, which matters more when a mistake is expensive.

Full table: [RESULTS.md](RESULTS.md).

## What goes wrong

[ANALYSIS.md](ANALYSIS.md) logs all 36,746 moves of the benchmark games and compares each with what the expert would do. The main findings:

- **The model disagrees with its teacher on 3.1% of moves**, and mostly when ghosts are far away (4.7%) or a power pill is active (5.1%), the moments where the choice is about which food to chase rather than survival. Near a ghost it disagrees on only 2.2%.
- **Confidence is informative but not decisive.** Agreeing moves have a median confidence of 0.99 and disagreeing ones 0.69, yet the 0.6 threshold catches only a third of the disagreements.
- **More review does not win more games.** Replaying with thresholds from 0.6 to 0.9 sends up to 15.9% of moves to the expert, and the win rate stays at 37–39%: the expert is no stronger than the model it taught. Review only pays off when the reviewer is better than the model.
- **Every loss ends in a trap.** In all lost games, Pac-Man's last position had every open move next to a ghost, so the fatal mistake happened earlier, by walking into a dead end, which per-move features cannot see.

![Confidence of agreeing and disagreeing moves](analysis/confidence_agree_vs_disagree.png)

## Limits

- The expert is a simple greedy player and wins only about a third of the games, so the model learns its mistakes too. A stronger teacher, or learning from game outcomes, would raise the ceiling.
- The features describe distances, not the whole maze, so the model cannot plan long routes.
