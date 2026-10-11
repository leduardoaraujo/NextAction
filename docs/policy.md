# Scenario configs and the decision policy

A classifier only says how likely each action is. A real decision system also needs to know **which actions are allowed right now, how sure it must be before acting alone, and how bad each kind of mistake is**. In OpenDecision those answers live next to the model parameters, in one TOML file per scenario: `configs/<scenario>.toml`. Adding a scenario means adding a file.

## Anatomy of a config

```toml
[scenario]
name = "agent"
alias = "agente"                                   # Portuguese name accepted everywhere
languages = ["en"]
data = "data/external/sgd_generic_train.csv"
test = "data/external/sgd_generic_test.csv"        # used by `evaluate`
validation = "data/external/sgd_generic_dev.csv"   # used by `calibrate-policy`
model = "models/agent.joblib"

[model]                  # hyperparameters
ngram_range = [1, 2]
min_df = 5
C = 1.0
class_weight = "none"    # or "balanced"
max_iter = 1000
calibration = "none"     # "sigmoid" or "isotonic" (CalibratedClassifierCV)

[tuning]                 # grid searched by `opendecision tune`
sample = 40000
folds = 3
metric = "f1_macro"      # or "accuracy"
C = [0.3, 1.0, 3.0]
min_df = [2, 5]

[policy]
default_threshold = 0.7
review_cost = 1.0        # cost of one human review: the unit for every other cost

[policy.actions.execute]
risk = "irreversible"            # low | medium | irreversible
min_confidence = 0.86
unreviewed_error_cost = 50       # a wrong unreviewed execution costs 50 reviews
requires_confirmation = true     # confirm with the user even when confident

[[policy.rules]]
id = "no_search_or_execute_with_missing_fields"
when_missing_any = true
deny = ["search", "execute"]
```

## How a decision is made

1. The model predicts a probability for every action.
2. **Rules** whose conditions all hold remove their `deny` actions, and the rest are renormalized. Conditions read the state tags:

   | Condition | Holds when |
   | --- | --- |
   | `when_missing_any = true` | any required slot is still missing |
   | `when_missing = ["date"]` | any of these slots is missing |
   | `when_filled_all = ["origin", "date"]` | all of these slots are filled |
   | `when_intent = ["BookFlight"]` | the active intent is one of these |

3. The top action is chosen. It is automated only if its probability reaches that action's `min_confidence` (or `default_threshold`). A `threshold` sent in the request overrides every per-action value.
4. The response explains itself. For `[missing destination_airport departure_date] || user: I need a flight from Boston.`:

```json
{
  "type": "choice",
  "action": "ask",
  "probabilities": { "answer": 0.0011, "ask": 0.9947, "confirm": 0.0, "end_conversation": 0.0004, "offer_more_help": 0.0011, "present_options": 0.0019, "suggest_next_step": 0.0008 },
  "needs_review": false,
  "requires_confirmation": false,
  "risk": "low",
  "threshold": 0.33,
  "reasons": ["masked_by_rule:no_search_or_execute_with_missing_fields"],
  "blocked_actions": ["execute", "search"]
}
```

`reasons` can contain `masked_by_rule:<id>`, `below_threshold`, `irreversible_action` and `all_actions_blocked` (every action was denied, so the rules are ignored and the case goes to review).

## Tuning the model

```bash
opendecision tune flights            # search the [tuning] grid with cross-validation
opendecision tune flights --write    # store the best values in [model]
```

The search reports progress per candidate and fold, and prints the top five combinations with their mean score and standard deviation. On large datasets, `sample` limits the search to a stratified subsample.

## Calibrating the policy from error costs

```bash
opendecision calibrate-policy agent            # show the thresholds and their effect
opendecision calibrate-policy agent --write    # store them as min_confidence
```

For every action, the command tries thresholds from 0.30 to 0.99. For each one it adds the cost of the wrong decisions that would pass unreviewed (`unreviewed_error_cost`) and the cost of the ones sent to review (`review_cost`), and keeps the cheapest threshold. It uses the scenario's `validation` file when there is one, cross-validation on small datasets, and an 80/20 split otherwise.

Calibrate on data that looks like production. Calibrating the `agent` scenario on a slice of its own training data gave thresholds that were too optimistic: 94.8% accuracy there, against 90% on services never seen in training. That is why it uses the SGD `dev` split.

## Results on the SGD test split

`opendecision evaluate agent` adds a `policy` section that simulates the configured policy on held-out predictions:

| Policy | Accuracy | Automated | Accuracy when automated | Unreviewed errors | Expected cost per decision |
| --- | --- | --- | --- | --- | --- |
| Single threshold 0.7, no rules | 0.901 | 74.7% | 97.8% | 690 | — |
| Hand-set per-action thresholds + rule | 0.905 | 69.4% | 98.2% | 516 | 0.351 |
| Thresholds calibrated on dev + rule | 0.905 | 85.1% | 95.2% | 1,722 | **0.309** |

The rule alone lifts accuracy from 0.901 to 0.905 by removing impossible choices. Calibration then trades more unreviewed low-cost mistakes for far fewer reviews, which lowers the total cost by 12% under the configured costs. If unreviewed errors are worse for your product than these costs say, raise `unreviewed_error_cost` and calibrate again.
