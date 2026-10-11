# Scenario: next action of a flight booking agent

This scenario is **fictional**. The agent helps a traveler find flights and chooses the next step of the conversation; the model does not query airlines or buy tickets. The bundled examples in `data/flights.csv` are written in Portuguese.

## Decision

Input: a description of the current state of the conversation and of the search results.

Output: one of four actions with estimated probabilities.

| Action | When |
| --- | --- |
| `ask` | origin, destination, date, number of travelers or another essential requirement is missing |
| `search` | the essential requirements are known and up-to-date offers are missing |
| `present_options` | flights were compared on total price, schedule and conditions, and the traveler can choose |
| `confirm` | the traveler picked a specific offer; details and explicit authorization are needed before any purchase |

Example flow: the traveler asks for a ticket from São Paulo to Recife without a date → `ask`; gives the dates and number of passengers → `search`; after an external search and comparison → `present_options`; picks a flight → `confirm`.

The program only **suggests** an action. Even when the model suggests `search`, the application must check the required fields itself. Price, availability and fare rules must come from an up-to-date source, and purchasing requires a separate step of explicit authorization.

## Try it

```bash
opendecision train flights
opendecision decide flights "Tenho origem GRU destino REC ida 12/11 volta 19/11 e um adulto mas ainda não consultei voos"
opendecision evaluate flights
```

The examples are synthetic and few. Cross-validation gives 0.50 accuracy, and the model never reaches 0.5 confidence, so with the default threshold every decision goes to review. To improve it, replace the examples with reviewed conversation states, keep the most recent real cases for evaluation, and only then choose a review threshold.

## Rule-based ranking with several conditions

`rank-flights` takes a **structured** state and makes a more detailed decision. The example file is fictional and contains no real prices or availability.

```bash
opendecision rank-flights data/trip_example.json
```

The flow:

1. If origin, destination, departure date or number of passengers is missing, `ask` and list the `missing_fields`.
2. If there are no offers, `search`.
3. Discard offers that break the budget, the stop limit or the baggage requirement (`over_budget`, `too_many_stops`, `missing_required_baggage`).
4. If no offer is left, `revise_criteria`.
5. Rank eligible offers by total price, duration and stops; `present_options` with the best one and the alternatives.
6. If the traveler already chose an offer, `confirm` the details and the current price.

The score uses the configurable `request.weights`. For each eligible offer the components are `lowest_price/price`, `shortest_duration/duration` and `1/(1+stops)`. Their weighted average **only orders offers**; it is not a probability or a trained model. The total price must include every passenger, fees and required baggage for the comparison to make sense.

To simulate a choice, add `"selected_offer": "flight_a"` to the top-level JSON object. The result is `confirm`; the command never issues a ticket. Portuguese keys (`pedido`, `ofertas`, `oferta_escolhida`, `preco_total`…) are accepted too.

This ranker complements the next-action classifier. The next machine-learning step is to log the offers shown and which one the traveler chose. With real, consented data, a preference model could be trained and its ranking compared with this simple formula on held-out trips.
