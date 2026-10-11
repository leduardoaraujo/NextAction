# Generic agent model on Schema-Guided Dialogue

The `agente` scenario learns the next action of a task-oriented assistant from the [Schema-Guided Dialogue dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) (SGD) by Google: about 18k English conversations between people and a virtual assistant across 20 domains (flights, hotels, restaurants, banking, events…).

## From dialogue acts to agent actions

Every assistant turn in SGD is annotated with dialogue acts and, when the assistant queried or changed something, with a `service_call`. `opendecision/sgd.py` turns each assistant turn into one training example. The action is derived from those annotations:

| Action | Derived from | Meaning for an agent |
| --- | --- | --- |
| `buscar` | a `service_call` that is not a transaction | query a tool or API before answering |
| `executar` | `NOTIFY_SUCCESS` / `NOTIFY_FAILURE` | run the transaction and report the outcome |
| `confirmar` | `CONFIRM` | confirm details before a transaction |
| `perguntar` | `REQUEST` | ask the user for missing information |
| `sugerir_proxima_etapa` | `OFFER_INTENT` | propose the next step (for example, "want me to book it?") |
| `apresentar_opcao` | `OFFER`, `INFORM_COUNT` without a new query | present another option from results already fetched |
| `responder` | `INFORM` | answer a question about an item |
| `oferecer_mais_ajuda` | `REQ_MORE` | ask whether the user needs anything else |
| `encerrar` | `GOODBYE` | close the conversation |

When a turn has several acts, the first match in the table wins. `buscar` comes first because a query preceding the reply is the decision that matters.

## State format

The model sees the dialogue state followed by the last three turns:

```text
[intencao SearchOnewayFlight] [preenchidos origin_airport] [faltando destination_airport departure_date] [pedidos nenhum]
|| usuario: I need a flight from Boston.
```

- `intencao`: the user's active intent.
- `preenchidos`: slots the user already provided (names only, no values).
- `faltando`: required slots of that intent, taken from the service schema, that are still empty.
- `pedidos`: slots the user asked about.

To use the model in your own agent, build the same string from the state you already track. `--sem-estado` produces conversation-only text, and `--contexto N` changes how many turns are included.

## Reproduce

```bash
git clone --depth 1 https://github.com/google-research-datasets/dstc8-schema-guided-dialogue dados/externos/sgd
for split in train dev test; do
  python -m opendecision importar-sgd dados/externos/sgd/$split --destino dados/externos/sgd_generico_$split.csv
done

python -m opendecision avaliar --dados dados/externos/sgd_generico_train.csv \
  --teste dados/externos/sgd_generico_test.csv --min-df 5
python -m opendecision treinar --dados dados/externos/sgd_generico_train.csv \
  --modelo modelos/agente.joblib --min-df 5
```

Training on 165k examples takes a few minutes on a laptop CPU. `--min-df 5` drops terms seen in fewer than five turns. It cuts the vocabulary from 178k to 54k terms with no loss of accuracy (0.900 → 0.902). Together with skops compression, the exported model shrinks from 125 MB to 4.7 MB.

`--esquema passagens` keeps only flight turns and maps them to the flight scenario actions (`perguntar_dados`, `buscar_voos`, `apresentar_opcoes`, `pedir_confirmacao`). Note that the text is in English.

## Results

Published model (`min_df=5`), trained on `train` (164,982 examples), evaluated on the official `test` split (42,297 examples, including services absent from training).

| Action | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| `apresentar_opcao` | 0.684 | 0.711 | 0.697 | 1,030 |
| `buscar` | 0.850 | 0.959 | 0.901 | 8,233 |
| `confirmar` | 0.903 | 0.849 | 0.875 | 5,697 |
| `encerrar` | 0.925 | 0.893 | 0.909 | 4,201 |
| `executar` | 0.939 | 0.924 | 0.932 | 5,006 |
| `oferecer_mais_ajuda` | 0.732 | 0.765 | 0.748 | 2,684 |
| `perguntar` | 0.999 | 0.942 | 0.970 | 9,016 |
| `responder` | 0.984 | 0.964 | 0.974 | 4,170 |
| `sugerir_proxima_etapa` | 0.796 | 0.777 | 0.787 | 2,260 |

Accuracy 0.902 against a 0.213 majority-class baseline; expected calibration error 0.084. On the `dev` split (same services as training), accuracy is about 0.93.

| Review threshold | Decisions automated | Accuracy on automated |
| --- | --- | --- |
| 0.5 | 90.2% | 94.3% |
| 0.7 | 75.1% | 97.8% |
| 0.8 | 65.4% | 98.8% |
| 0.9 | 50.0% | 99.6% |

### Ablation: does the dialogue state matter?

Both runs use the full vocabulary (`min_df=1`).

| Input | Accuracy | F1 `perguntar` | F1 `buscar` | Automated at 0.9 |
| --- | --- | --- | --- | --- |
| Conversation + state | 0.900 | 0.968 | 0.898 | 48.5% |
| Conversation only | 0.772 | 0.707 | 0.744 | 28.9% |

From text alone, "ask for more details" and "search now" look alike. Knowing which required fields are missing separates them. The weakest actions, `apresentar_opcao` and `oferecer_mais_ajuda`, depend on things the state string does not capture yet, such as whether results were already shown.

## Caveats

- SGD conversations were written by crowd workers from simulated dialogue outlines. They are cleaner and more regular than real traffic.
- The dataset is licensed CC BY-SA 4.0. It is not redistributed in this repository, and models trained on it should keep that license and cite Rastogi et al. (2020).
