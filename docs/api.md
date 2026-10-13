# REST API, feedback loop, languages and Rust client

## Running the server

Install the API extra, train the models you want to serve, then start the server from the repository root:

```bash
pip install -e ".[api]"
nextaction train flights
nextaction train research
uvicorn nextaction.api:app --reload
```

Interactive OpenAPI docs are served at `http://127.0.0.1:8000/docs`. The `agent` model requires the SGD data; see [sgd-model.md](sgd-model.md).

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | liveness check |
| `GET` | `/v1/models` | which scenario models are trained locally |
| `GET` | `/v1/jobs` | training and import jobs with their progress ([progress.md](progress.md)) |
| `GET` | `/dashboard` | live web page for those jobs |
| `POST` | `/v1/actions/decide` | next action for `agent`, `flights` or `research` |
| `POST` | `/v1/decisions/{decision_id}/feedback` | record the action that should have been taken |
| `POST` | `/v1/flights/evaluate` | filter and rank the flight offers sent in the body |

## Deciding

```bash
curl -s http://127.0.0.1:8000/v1/actions/decide -H "Content-Type: application/json" -d '{
  "scenario": "flights",
  "state": "Tenho origem GRU destino REC ida 12/11 e ainda não consultei voos",
  "threshold": 0.7
}'
```

```json
{
  "type": "choice",
  "action": "search",
  "probabilities": { "ask": 0.2085, "confirm": 0.183, "present_options": 0.1922, "search": 0.4163 },
  "needs_review": true,
  "decision_id": "3f9c…"
}
```

`threshold` is optional: without it, the scenario's per-action thresholds apply. The response also carries `requires_confirmation`, `risk`, the `threshold` used, `reasons` and `blocked_actions` from the decision policy ([policy.md](policy.md)). A missing model returns `503`; invalid input returns `422`.

## Feedback loop

Each decision is appended to `logs/decisions.jsonl` with its state, suggestion and probabilities. When a person or the agent knows the right action, send it back:

```bash
curl -s http://127.0.0.1:8000/v1/decisions/<decision_id>/feedback -H "Content-Type: application/json" \
  -d '{"final_action": "confirm", "comment": "the traveler had already chosen a flight"}'
```

- `final_action` may be an action the model does not know yet. That is how missing actions surface.
- The log is append-only; if a decision receives several feedback entries, the last one wins.
- To log a measured outcome for offline learning, also send `action_taken`, `reward` (0 to 1), and `propensity` (the actual selection probability, greater than 0 and at most 1) together. See [outcome learning](outcomes.md). The `final_action` correction and the observed reward have different meanings.
- States can contain personal data. The log is git-ignored, and `NEXTACTION_LOG=""` disables it.

Turn reviewed decisions into training data and evaluate on the most recent cases:

```bash
nextaction export-labels flights          # → data/labeled_flights.csv
nextaction evaluate flights --data data/labeled_flights.csv --recent-fraction 0.2
```

The export also reports how many suggestions were corrected, which is a first production error rate.

## Languages

English is canonical; Portuguese is a permanent alias everywhere.

**Requests** accept either field name. These two calls are equivalent:

```json
{"scenario": "flights", "state": "…", "threshold": 0.7}
{"cenario": "passagens", "estado": "…", "limiar": 0.7}
```

| Endpoint | English fields | Portuguese aliases |
| --- | --- | --- |
| decide | `scenario`, `state`, `threshold` | `cenario`, `estado`, `limiar` |
| feedback | `final_action`, `outcome`, `comment`, `action_taken`, `reward`, `propensity` | `acao_final`, `resultado`, `comentario`, `acao_executada`, `recompensa`, `probabilidade_escolha` |
| flights | `request`, `offers`, `selected_offer` | `pedido`, `ofertas`, `oferta_escolhida` |
| flight request | `origin`, `destination`, `departure_date`, `return_date`, `passengers`, `max_budget`, `max_stops`, `baggage_required`, `weights` | `origem`, `destino`, `data_ida`, `data_volta`, `passageiros`, `orcamento_max`, `max_escalas`, `bagagem_obrigatoria`, `pesos` |
| offer | `total_price`, `duration_min`, `stops`, `baggage_included` | `preco_total`, `duracao_min`, `escalas`, `bagagem_incluida` |
| weights | `price`, `duration`, `stops` | `preco`, `duracao`, `escalas` |

Action labels sent as feedback are normalized too: `pedir_confirmacao` is stored as `confirm`. Models trained before the English vocabulary still return English actions, because their Portuguese labels are mapped on the fly. If such a model was trained on Portuguese state tags (`[faltando …]`), retrain it: incoming states are normalized to the English tags, which that model never saw.

**Responses** always use English names. Human-readable text (`detail` in errors, `notice`, evaluation warnings) follows the `Accept-Language` header:

```bash
curl -s -H "Accept-Language: pt-BR" http://127.0.0.1:8000/v1/decisions/unknown/feedback \
  -H "Content-Type: application/json" -d '{"final_action": "search"}'
# {"detail": "Decisão não encontrada."}
```

The CLI uses `--lang pt` or the `NEXTACTION_LANG` environment variable, and its commands have Portuguese aliases (`nextaction decidir passagens "…"`).

## Flight offer ranking

`/v1/flights/evaluate` and `nextaction rank-flights data/trip_example.json` apply explicit rules: ask for missing fields, search when there are no offers, discard offers that break budget, stop or baggage limits (`over_budget`, `too_many_stops`, `missing_required_baggage`), rank the rest by weighted price, duration and stops, and ask for confirmation once an offer is chosen. The score only orders offers; it is not a probability. See [scenarios/flights.md](scenarios/flights.md).

## Rust client

A typed client in `clients/rust` consumes the same contract. Trip files may use English or Portuguese keys, and `NEXTACTION_LANG` is forwarded as `Accept-Language`.

```bash
cargo run --manifest-path clients/rust/Cargo.toml -- action flights "Tenho origem GRU destino REC e ainda não consultei voos"
cargo run --manifest-path clients/rust/Cargo.toml -- flights data/trip_example.json
```

It targets `http://127.0.0.1:8000` by default; set `NEXTACTION_API_URL` to change it.
