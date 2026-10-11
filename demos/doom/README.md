# Doom Deathmatch demo

NextAction plays ViZDoom's **Deathmatch** arena. It walks around the map, turns away from nearby walls, aims at enemies as they spawn, and fires. The GIF records the actual game every two tics (about 17.5 frames per second), including motion between model decisions.

![NextAction playing Deathmatch](doom.gif)

From the repository root:

```powershell
pip install -e ".[demo]"
pip install "vizdoom>=1.3,<2"
python demos/doom/generate_data.py
nextaction train doom
nextaction evaluate doom
python demos/doom/play.py --gif demos/doom/doom.gif --record-from 270
```

Use `--agent teacher`, `--agent model-only`, or `--agent random` for comparison. `--seed` selects a match, `--max-tics` controls match length, and `--record-from` trims the quiet opening from the GIF while the agent still plays it. `--fps` controls playback. The default agent sends low-confidence decisions to the teacher.

The teacher uses ViZDoom's visible-object labels, center depth, health, and ammo to choose among advance, turn, aim and fire. NextAction imitates those decisions. Difficulty is set to 1 for a short, watchable match. This is an imitation-learning demonstration, not a general Doom player or a model that reads raw pixels.

Training and test matches use separate seeds. Their CSVs are saved under `data/external/`; the trained model is saved under `models/`.

With 20 training and 5 test matches, the model reproduced all 580 teacher decisions in the test set (majority-action baseline: 71.9%). This measures imitation of the simple teacher, not general Deathmatch skill. In the recorded held-out match, the model made three kills and survived the 600-tic run.
