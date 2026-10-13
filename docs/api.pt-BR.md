# API REST, ciclo de feedback, idiomas e cliente Rust

## Rodando o servidor

Instale o extra da API, treine os modelos que você quer servir e inicie o servidor a partir da raiz do repositório:

```bash
pip install -e ".[api]"
nextaction train flights
nextaction train research
uvicorn nextaction.api:app --reload
```

A documentação interativa do OpenAPI fica em `http://127.0.0.1:8000/docs`. O modelo `agent` precisa dos dados do SGD; veja [sgd-model.md](sgd-model.md).

| Método | Rota | Finalidade |
| --- | --- | --- |
| `GET` | `/health` | verificação de disponibilidade |
| `GET` | `/v1/models` | quais modelos de cenário estão treinados localmente |
| `GET` | `/v1/jobs` | tarefas de treino e importação com o progresso de cada uma ([progress.md](progress.md)) |
| `GET` | `/dashboard` | página web ao vivo dessas tarefas |
| `POST` | `/v1/actions/decide` | próxima ação para `agent`, `flights` ou `research` |
| `POST` | `/v1/decisions/{decision_id}/feedback` | registra a ação que deveria ter sido tomada |
| `POST` | `/v1/flights/evaluate` | filtra e ordena as ofertas de voo enviadas no corpo |

## Decidindo

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

`threshold` é opcional: sem ele, valem os limiares por ação do cenário. A resposta também traz `requires_confirmation`, `risk`, o `threshold` usado, `reasons` e `blocked_actions` vindos da política de decisão ([policy.md](policy.md)). Um modelo ausente retorna `503`; uma entrada inválida retorna `422`.

## Ciclo de feedback

Cada decisão é acrescentada a `logs/decisions.jsonl` com o estado, a sugestão e as probabilidades. Quando uma pessoa ou o agente sabe qual era a ação certa, envie-a de volta:

```bash
curl -s http://127.0.0.1:8000/v1/decisions/<decision_id>/feedback -H "Content-Type: application/json" \
  -d '{"final_action": "confirm", "comment": "the traveler had already chosen a flight"}'
```

- `final_action` pode ser uma ação que o modelo ainda não conhece. É assim que ações que faltam aparecem.
- O log só recebe acréscimos; se uma decisão recebe vários feedbacks, vale o último.
- Para registrar um resultado medido para aprendizado offline, envie também `action_taken`, `reward` (de 0 a 1) e `propensity` (a probabilidade real de escolha, maior que 0 e no máximo 1), todos juntos. Veja [aprendizado com resultados](outcomes.md). A correção em `final_action` e a recompensa observada têm significados diferentes.
- Os estados podem conter dados pessoais. O log é ignorado pelo git, e `NEXTACTION_LOG=""` o desativa.

Transforme as decisões revisadas em dados de treino e avalie nos casos mais recentes:

```bash
nextaction export-labels flights          # → data/labeled_flights.csv
nextaction evaluate flights --data data/labeled_flights.csv --recent-fraction 0.2
```

A exportação também informa quantas sugestões foram corrigidas, o que já é uma primeira taxa de erro em produção.

## Idiomas

O inglês é o idioma canônico; o português é um sinônimo permanente em todos os lugares.

**Requisições** aceitam qualquer um dos nomes de campo. Estas duas chamadas são equivalentes:

```json
{"scenario": "flights", "state": "…", "threshold": 0.7}
{"cenario": "passagens", "estado": "…", "limiar": 0.7}
```

| Endpoint | Campos em inglês | Sinônimos em português |
| --- | --- | --- |
| decide | `scenario`, `state`, `threshold` | `cenario`, `estado`, `limiar` |
| feedback | `final_action`, `outcome`, `comment`, `action_taken`, `reward`, `propensity` | `acao_final`, `resultado`, `comentario`, `acao_executada`, `recompensa`, `probabilidade_escolha` |
| flights | `request`, `offers`, `selected_offer` | `pedido`, `ofertas`, `oferta_escolhida` |
| pedido de voo | `origin`, `destination`, `departure_date`, `return_date`, `passengers`, `max_budget`, `max_stops`, `baggage_required`, `weights` | `origem`, `destino`, `data_ida`, `data_volta`, `passageiros`, `orcamento_max`, `max_escalas`, `bagagem_obrigatoria`, `pesos` |
| oferta | `total_price`, `duration_min`, `stops`, `baggage_included` | `preco_total`, `duracao_min`, `escalas`, `bagagem_incluida` |
| pesos | `price`, `duration`, `stops` | `preco`, `duracao`, `escalas` |

Os rótulos de ação enviados como feedback também são normalizados: `pedir_confirmacao` é armazenado como `confirm`. Modelos treinados antes do vocabulário em inglês continuam retornando ações em inglês, porque seus rótulos em português são mapeados na hora. Se um desses modelos foi treinado com tags de estado em português (`[faltando …]`), treine-o de novo: os estados recebidos são normalizados para as tags em inglês, que esse modelo nunca viu.

**Respostas** usam sempre os nomes em inglês. O texto para pessoas (`detail` nos erros, `notice`, avisos da avaliação) segue o header `Accept-Language`:

```bash
curl -s -H "Accept-Language: pt-BR" http://127.0.0.1:8000/v1/decisions/unknown/feedback \
  -H "Content-Type: application/json" -d '{"final_action": "search"}'
# {"detail": "Decisão não encontrada."}
```

A CLI usa `--lang pt` ou a variável de ambiente `NEXTACTION_LANG`, e seus comandos têm sinônimos em português (`nextaction decidir passagens "…"`).

## Ordenação de ofertas de voo

`/v1/flights/evaluate` e `nextaction rank-flights data/trip_example.json` aplicam regras explícitas: perguntam pelos campos que faltam, buscam quando não há ofertas, descartam as ofertas que estouram os limites de orçamento, escalas ou bagagem (`over_budget`, `too_many_stops`, `missing_required_baggage`), ordenam as restantes por preço, duração e escalas ponderados e pedem confirmação quando uma oferta é escolhida. A pontuação só ordena as ofertas; ela não é uma probabilidade. Veja [scenarios/flights.md](scenarios/flights.md).

## Cliente Rust

Um cliente tipado em `clients/rust` consome o mesmo contrato. Os arquivos de viagem podem usar chaves em inglês ou português, e `NEXTACTION_LANG` é repassado como `Accept-Language`.

```bash
cargo run --manifest-path clients/rust/Cargo.toml -- action flights "Tenho origem GRU destino REC e ainda não consultei voos"
cargo run --manifest-path clients/rust/Cargo.toml -- flights data/trip_example.json
```

Por padrão ele aponta para `http://127.0.0.1:8000`; defina `NEXTACTION_API_URL` para mudar.
