# REST API, feedback loop and Rust client

## Running the server

Train the models you want to serve, then start the API from the repository root:

```bash
python -m opendecision treinar                                                     # pesquisa
python -m opendecision treinar --dados dados/passagens.csv --modelo modelos/passagens.joblib
uvicorn opendecision.api:app --reload
```

Interactive OpenAPI docs are served at `http://127.0.0.1:8000/docs`. The `agente` model requires the SGD data; see [sgd-model.md](sgd-model.md).

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | liveness check |
| `GET` | `/v1/models` | which scenario models are trained locally |
| `POST` | `/v1/actions/decide` | next action for `agente`, `passagens` or `pesquisa` |
| `POST` | `/v1/decisions/{decisao_id}/feedback` | record the action that should have been taken |
| `POST` | `/v1/flights/evaluate` | filter and rank the flight offers sent in the body |

## Deciding

```bash
curl -s http://127.0.0.1:8000/v1/actions/decide -H "Content-Type: application/json" -d '{
  "cenario": "passagens",
  "estado": "Tenho origem GRU destino REC ida 12/11 e ainda não consultei voos",
  "limiar": 0.7
}'
```

The response contains `proxima_acao`, `probabilidades_estimadas`, `pedir_revisao` (true when the top probability is below `limiar`) and `decisao_id`. A missing model returns `503`; invalid input returns `422`.

## Feedback loop

Each decision is appended to `registros/decisoes.jsonl` with its state, suggestion and probabilities. When a person or the agent knows the right action, send it back:

```bash
curl -s http://127.0.0.1:8000/v1/decisions/<decisao_id>/feedback -H "Content-Type: application/json" \
  -d '{"acao_final": "pedir_confirmacao", "comentario": "the traveler had already chosen a flight"}'
```

- `acao_final` may be an action the model does not know yet. That is how missing actions surface.
- The log is append-only; if a decision receives several feedback entries, the last one wins.
- States can contain personal data. The log is git-ignored, and `OPENDECISION_REGISTRO=""` disables it.

Turn reviewed decisions into training data and evaluate on the most recent cases:

```bash
python -m opendecision exportar-rotulos passagens          # → dados/rotulados_passagens.csv
python -m opendecision avaliar --dados dados/rotulados_passagens.csv --fracao-final 0.2
```

The export also reports how many suggestions were corrected, which is a first production error rate.

## Flight offer ranking

`/v1/flights/evaluate` (and `python -m opendecision avaliar-passagens dados/viagem_exemplo.json`) applies explicit rules: ask for missing fields, search when there are no offers, discard offers that break budget, stop or baggage limits, rank the rest by weighted price, duration and stops, and ask for confirmation once an offer is chosen. The score only orders offers; it is not a probability. Details, in Portuguese, are in [cenarios/passagens.md](../cenarios/passagens.md).

## Rust client

A typed client in `clients/rust` consumes the same contract:

```bash
cargo run --manifest-path clients/rust/Cargo.toml -- action passagens "Tenho origem GRU destino REC e ainda não consultei voos"
cargo run --manifest-path clients/rust/Cargo.toml -- flights dados/viagem_exemplo.json
```

It targets `http://127.0.0.1:8000` by default; set `OPENDECISION_API_URL` to change it.
