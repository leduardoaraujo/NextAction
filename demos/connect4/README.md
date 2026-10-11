# Connect Four demo

NextAction picking the column to play, against an alpha-beta search. The panel shows the probability of each column, the columns the policy blocked (full, or handing the opponent a win), the confidence against the threshold, and who decided: when the model is unsure, a deeper search reviews the move.

![NextAction playing Connect Four](connect4.gif)

This demo asks the question the [Pac-Man demo](../pacman/README.md) left open. There, sending unsure moves to the expert did not win more games, because the expert was no stronger than the model it taught. Here the model learns from a depth-4 search, and unsure moves can be reviewed either by that same teacher or by a deeper, depth-6 search.

## Run it

From the repository root:

```bash
pip install -e ".[demo]"

python demos/connect4/generate_data.py      # positions from 1,500 training and 300 test games (~10 min)
nextaction train lig4                       # or: nextaction train connect4
nextaction evaluate connect4

python demos/connect4/play.py               # watch the model play in the terminal
python demos/connect4/play.py --agent model-only
python demos/connect4/play.py --gif demos/connect4/connect4.gif --pick-win
python demos/connect4/benchmark.py --games 200   # ~30 min, depth-6 search is slow in Python
```

| Script | What it does |
| --- | --- |
| [`game.py`](game.py) | the 6×7 board and an alpha-beta search player with a symmetric heuristic |
| [`features.py`](features.py) | turns a position into a NextAction state string, from the side to move |
| [`generate_data.py`](generate_data.py) | plays varied games and labels every position with the depth-4 teacher's move |
| [`play.py`](play.py) | one game against a depth-4 search, live in the terminal or recorded to a GIF |
| [`benchmark.py`](benchmark.py) | every agent against the depth-4 search and writes [RESULTS.md](RESULTS.md) |

## State and policy

```text
[blocked none] [danger c6] [features block_c3 c1_bad c3_great c4_better ... h3_1 h4_2 r0c4_me r0c3_op ...]
```

- `[features ...]`: immediate wins (`win_c4`) and blocks (`block_c3`), how each column changes a heuristic score (`c3_great`), column heights, and every occupied cell as `me` or `op`.
- `[blocked ...]` lists full columns and `[danger ...]` lists columns that let the opponent win right on top. Two `deny_tag` rules in [`configs/connect4.toml`](../../configs/connect4.toml) remove both before the model chooses.
- A move below 0.5 confidence is reviewed by a search.

## Results

Imitation accuracy on 6,716 positions from 300 unseen games: **0.668** (always the most common column: 0.282). Many positions have several equally good columns, so a different choice from the teacher is not always a worse one.

200 games per agent against a depth-4 search, half playing first. Each game starts from two random opening moves, the same for every agent on a seed:

| Agent | Wins | Draws | Losses | Win rate (95% interval) | Moves reviewed |
| --- | --- | --- | --- | --- | --- |
| model + policy, reviewed by depth-6 search | 79 | 54 | 67 | 40% (33–46%) | 39.4% |
| model + policy, reviewed by the depth-4 teacher | 66 | 58 | 76 | 33% (27–40%) | 41.3% |
| model + policy, no review | 38 | 39 | 123 | 19% (14–25%) | — |
| depth-4 search (the teacher) | 76 | 58 | 66 | 38% (32–45%) | — |
| depth-6 search | 83 | 41 | 76 | 42% (35–48%) | — |
| random legal column | 0 | 0 | 200 | 0% (0–2%) | — |

- **Alone, the model is much weaker than its teacher** (19% wins, 123 losses). Matching two thirds of a search's moves is not enough in a tactical game, where one bad move loses.
- **With review it plays at the teacher's level**: reviewed by a depth-6 search it wins 40%, close to the depth-4 search playing every move (38%), while searching on only 39% of its moves.
- **A stronger reviewer helps more than an equal one**: 13 more wins and 9 fewer losses than reviewing with the teacher, at the same review rate. The 95% intervals overlap, so this is suggestive, not conclusive; together with the Pac-Man result it points the same way: review adds value when the reviewer knows more than the model.

Full table: [RESULTS.md](RESULTS.md).

## Notes

- Without random openings, every game replayed two to four opening lines, which made the results depend on a handful of positions; fixed random openings per seed fixed that.
- The first heuristic was asymmetric, and a deeper search did not beat a shallower one; the symmetric version makes depth 6 beat depth 4 by 22 to 8 in head-to-head games.
