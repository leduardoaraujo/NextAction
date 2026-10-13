# Configuração de cenários e a política de decisão

Um classificador só diz qual a probabilidade de cada ação. Um sistema de decisão real também precisa saber **quais ações são permitidas agora, quanta certeza é preciso ter antes de agir sozinho e quão grave é cada tipo de erro**. No NextAction, essas respostas ficam junto dos parâmetros do modelo, em um arquivo TOML por cenário: `configs/<scenario>.toml`. Adicionar um cenário é adicionar um arquivo.

## Anatomia de uma configuração

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

[tuning]                 # grid searched by `nextaction tune`
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

## Como uma decisão é tomada

1. O modelo prevê uma probabilidade para cada ação.
2. As **regras** cujas condições são todas verdadeiras removem as ações de `deny`, e as restantes são renormalizadas. As condições leem as tags do estado:

   | Condição | Vale quando |
   | --- | --- |
   | `when_missing_any = true` | algum campo obrigatório ainda falta |
   | `when_missing = ["date"]` | algum destes campos falta |
   | `when_filled_all = ["origin", "date"]` | todos estes campos estão preenchidos |
   | `when_intent = ["BookFlight"]` | a intenção ativa é uma destas |

   Uma regra também pode tirar sua lista `deny` do próprio estado: `deny_tag = "blocked"` bloqueia as ações nomeadas numa tag `[blocked ...]`. Isso permite que o ambiente envie sua própria máscara de ações, como as paredes no [demo de Pac-Man](../demos/pacman/README.md):

   ```toml
   [[policy.rules]]
   id = "no_walls"
   deny_tag = "blocked"   # [blocked up left] denies up and left
   ```

3. A ação com maior probabilidade é escolhida. Ela só é automática se sua probabilidade atingir o `min_confidence` daquela ação (ou o `default_threshold`). Um `threshold` enviado na requisição substitui todos os valores por ação. Uma ação com `risk = "irreversible"` nunca é automática: ela sempre vai para revisão, qualquer que seja a confiança, e também precisa definir `requires_confirmation = true`.
4. A resposta se explica. Para `[missing destination_airport departure_date] || user: I need a flight from Boston.`:

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

`reasons` pode conter `masked_by_rule:<id>`, `below_threshold`, `irreversible_action` e `all_actions_blocked` (todas as ações foram bloqueadas, então as regras são ignoradas e o caso vai para revisão).

A aplicação que faz a chamada também pode enviar `allowed_actions` (`--allowed-actions` na CLI). As ações fora dessa lista são bloqueadas com o motivo `masked_by_rule:application_allowlist`. Com `require_allowed_actions = true` em `[policy]`, uma requisição sem a lista vai para revisão com `application_allowlist_required`.

## Ajustando o modelo

```bash
nextaction tune flights            # search the [tuning] grid with cross-validation
nextaction tune flights --write    # store the best values in [model]
```

A busca informa o progresso por candidato e por fold e mostra as cinco melhores combinações com a pontuação média e o desvio padrão. Em conjuntos de dados grandes, `sample` limita a busca a uma subamostra estratificada.

## Calibrando a política a partir dos custos de erro

```bash
nextaction calibrate-policy agent            # show the thresholds and their effect
nextaction calibrate-policy agent --write    # store them as min_confidence
```

Para cada ação, o comando testa limiares de 0,30 a 0,99. Para cada um, soma o custo das decisões erradas que passariam sem revisão (`unreviewed_error_cost`) e o custo das que iriam para revisão (`review_cost`), e fica com o limiar mais barato. Ele usa o arquivo `validation` do cenário quando existe, validação cruzada em conjuntos de dados pequenos e uma divisão 80/20 nos demais casos.

Calibre com dados parecidos com os de produção. Calibrar o cenário `agent` numa parte dos próprios dados de treino gerou limiares otimistas demais: 94,8% de acurácia ali, contra 90% em serviços nunca vistos no treino. Por isso ele usa o split `dev` do SGD.

## Resultados no split de teste do SGD

`nextaction evaluate agent` adiciona uma seção `policy` que simula a política configurada em previsões separadas para teste:

| Política | Acurácia | Automáticas | Acurácia nas automáticas | Erros sem revisão | Custo esperado por decisão |
| --- | --- | --- | --- | --- | --- |
| Limiar único 0,7, sem regras | 0,901 | 74,7% | 97,8% | 690 | — |
| Limiares por ação definidos à mão + regra | 0,905 | 69,4% | 98,2% | 516 | 0,351 |
| Limiares calibrados no dev + regra | 0,905 | 85,1% | 95,2% | 1.722 | **0,309** |
| Igual, com toda ação irreversível revisada (atual) | 0,905 | 78,0% | 94,8% | 1.701 | 0,355 |

Só a regra já eleva a acurácia de 0,901 para 0,905, ao remover escolhas impossíveis. A calibração então troca mais erros baratos sem revisão por muito menos revisões, o que reduz o custo total em 12% com os custos configurados. Mandar todo `execute` para revisão custa 0,046 a mais por decisão com esses custos, que não contam o prejuízo de uma reserva ou de um pagamento indesejado; o projeto aceita esse preço. Se os erros sem revisão forem piores para o seu produto do que esses custos indicam, aumente `unreviewed_error_cost` e calibre de novo.
