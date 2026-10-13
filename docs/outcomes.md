# Bilingual model and outcome learning

## One English/Portuguese model

`agent_bilingual` uses the same action vocabulary and policy interface as `agent`.
Its training set combines the official English SGD train split with a translated
Portuguese sample from that **same train split**. The smaller Portuguese set is
repeated four times. The translated test dialogues come only from the official
test split; `scripts/build_bilingual_sgd.py` also selects their English counterparts
for a paired comparison. The external data and trained models are git-ignored.

```bash
# First download the official SGD train/test splits into data/external/sgd/.
pip install -e ".[translate]"
python scripts/translate_sgd_sample.py data/external/sgd/train data/external/sgd_pt/train --count 800
python scripts/translate_sgd_sample.py data/external/sgd/test data/external/sgd_pt/test --count 300
nextaction import-sgd data/external/sgd_pt/train --output data/external/sgd_pt_train.csv
nextaction import-sgd data/external/sgd_pt/test --output data/external/sgd_pt_test.csv
python scripts/build_bilingual_sgd.py
nextaction train agent_bilingual
python scripts/evaluate_bilingual.py
nextaction decide agente_bilingue "[intencao SearchOnewayFlight] [faltando destination_airport] || usuario: Quero viajar"
```

On 2,883 paired turns per language, the English model scored 0.891 English and
0.609 Portuguese accuracy. The bilingual model scored **0.895 English and 0.837
Portuguese**. Its expected calibration error was 0.068 English and 0.073
Portuguese. Recreating the translated sample or changing the translation model
may change these numbers. The Portuguese sample is translated dialogue data,
so these numbers do not establish accuracy on real Portuguese conversations.
The bilingual scenario requires review by default. Measure and tune its
thresholds on representative local decisions before automation.

## Learning from observed outcomes

The feedback endpoint still accepts `final_action`, the action a reviewer says
was correct. For outcome learning it additionally accepts all three fields:

- `action_taken`: the action the application actually performed;
- `reward`: the observed result on a scale from 0 to 1;
- `propensity`: the probability that the **logging policy selected that action**
  in that state, between 0 and 1.

```json
{
  "final_action": "ask",
  "action_taken": "ask",
  "reward": 1.0,
  "propensity": 0.5
}
```

The application's action-selection logic must supply `propensity`. Classifier
confidence is **not** a propensity. A deterministic policy usually has no action
overlap, so it cannot support a comparison of alternative actions. Add controlled
randomization among policy-allowed actions, with user confirmation for irreversible
actions, only in a setting where experimentation is appropriate. Log outcomes
only after they are actually measured. A reviewer correction is not a reward.

```bash
nextaction fit-bandit agent --log logs/decisions.jsonl
nextaction preview-bandit agent "[intent SearchOnewayFlight] || user: I need a flight"
```

`fit-bandit` needs at least 200 complete observations, including 20 for each of
at least two actions. It fits per-action reward estimates on the earlier 80% of
observations and reports a self-normalized inverse propensity estimate on the
later 20%, alongside the logged mean reward, matches and effective sample size.
The result is a **candidate file for review**. The decision API never loads it
and the preview command never executes an action. Small effective sample sizes,
poor action overlap, and nonrandom or incorrect propensities can make the offline
estimate misleading. No real outcome observations are bundled with this project.
