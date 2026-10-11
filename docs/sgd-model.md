# Generic agent model on Schema-Guided Dialogue

The `agent` scenario learns the next action of a task-oriented assistant from the [Schema-Guided Dialogue dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) (SGD) by Google: about 18k English conversations between people and a virtual assistant across 20 domains (flights, hotels, restaurants, banking, events…).

## From dialogue acts to agent actions

Every assistant turn in SGD is annotated with dialogue acts and, when the assistant queried or changed something, with a `service_call`. `nextaction/sgd.py` turns each assistant turn into one training example, deriving the action from those annotations:

| Action | Derived from | Meaning for an agent |
| --- | --- | --- |
| `search` | a `service_call` that is not a transaction | query a tool or API before answering |
| `execute` | `NOTIFY_SUCCESS` / `NOTIFY_FAILURE` | run the transaction and report the outcome |
| `confirm` | `CONFIRM` | confirm details before a transaction |
| `ask` | `REQUEST` | ask the user for missing information |
| `suggest_next_step` | `OFFER_INTENT` | propose the next step ("want me to book it?") |
| `present_options` | `OFFER`, `INFORM_COUNT` without a new query | present an option from results already fetched |
| `answer` | `INFORM` | answer a question about an item |
| `offer_more_help` | `REQ_MORE` | ask whether the user needs anything else |
| `end_conversation` | `GOODBYE` | close the conversation |

When a turn has several acts, the first match in the table wins; `search` comes first because a query preceding the reply is the decision that matters. These labels are the project-wide vocabulary in `nextaction/vocabulary.py`, so the `flights` and `research` scenarios use the same names.

## State format

The model sees the dialogue state followed by the last three turns:

```text
[intent SearchOnewayFlight] [filled origin_airport] [missing destination_airport departure_date] [requested none]
|| user: I need a flight from Boston.
```

- `intent`: the user's active intent.
- `filled`: slots the user already provided (names only, no values).
- `missing`: required slots of that intent, from the service schema, that are still empty.
- `requested`: slots the user asked about.

To use the model in your own agent, build the same string from the state you already track. Portuguese tags (`[intencao] [preenchidos] [faltando] [pedidos]`, `usuario:`, `assistente:`) are rewritten to this format automatically. `--no-state` produces conversation-only text, and `--context N` changes how many turns are included.

## Reproduce

```bash
git clone --depth 1 https://github.com/google-research-datasets/dstc8-schema-guided-dialogue data/external/sgd
for split in train dev test; do
  nextaction import-sgd data/external/sgd/$split --output data/external/sgd_generic_$split.csv
done

nextaction evaluate agent      # trains on train, evaluates on the official test split
nextaction train agent         # writes models/agent.joblib
```

Training on 165k examples takes a few minutes on a laptop CPU. The scenario uses `min_df=5`, which drops terms seen in fewer than five turns: the vocabulary shrinks from 178k to 54k terms with no loss of accuracy (0.900 → 0.902), and with skops compression the exported model goes from 125 MB to 4.7 MB.

`--schema flights` keeps only flight turns and the four actions of the flight scenario (`ask`, `search`, `present_options`, `confirm`).

## Results

Published model (`min_df=5`), trained on `train` (164,982 examples), evaluated on the official `test` split (42,297 examples, including services absent from training).

| Action | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| `answer` | 0.984 | 0.964 | 0.974 | 4,170 |
| `ask` | 0.999 | 0.942 | 0.970 | 9,016 |
| `confirm` | 0.903 | 0.849 | 0.875 | 5,697 |
| `end_conversation` | 0.925 | 0.893 | 0.909 | 4,201 |
| `execute` | 0.939 | 0.924 | 0.932 | 5,006 |
| `offer_more_help` | 0.732 | 0.765 | 0.748 | 2,684 |
| `present_options` | 0.684 | 0.711 | 0.697 | 1,030 |
| `search` | 0.850 | 0.959 | 0.901 | 8,233 |
| `suggest_next_step` | 0.796 | 0.777 | 0.787 | 2,260 |

Accuracy 0.902 against a 0.213 majority-class baseline; expected calibration error 0.084. On the `dev` split (same services as training), accuracy is about 0.93.

| Review threshold | Decisions automated | Accuracy on automated |
| --- | --- | --- |
| 0.5 | 90.2% | 94.3% |
| 0.7 | 75.1% | 97.8% |
| 0.8 | 65.4% | 98.8% |
| 0.9 | 50.0% | 99.6% |

### Ablation: does the dialogue state matter?

Both runs use the full vocabulary (`min_df=1`).

| Input | Accuracy | F1 `ask` | F1 `search` | Automated at 0.9 |
| --- | --- | --- | --- | --- |
| Conversation + state | 0.900 | 0.968 | 0.898 | 48.5% |
| Conversation only | 0.772 | 0.707 | 0.744 | 28.9% |

From text alone, "ask for more details" and "search now" look alike. Knowing which required fields are missing separates them. The weakest actions, `present_options` and `offer_more_help`, depend on things the state string does not capture yet, such as whether results were already shown.

## Caveats

- SGD conversations were written by crowd workers from simulated dialogue outlines. They are cleaner and more regular than real traffic.
- The dataset is licensed CC BY-SA 4.0. It is not redistributed in this repository, and models trained on it should keep that license and cite Rastogi et al. (2020).
