# Modelo genérico de agente no Schema-Guided Dialogue

O cenário `agent` aprende a próxima ação de um assistente orientado a tarefas a partir do [conjunto de dados Schema-Guided Dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) (SGD), do Google: cerca de 18 mil conversas em inglês entre pessoas e um assistente virtual em 20 domínios (voos, hotéis, restaurantes, bancos, eventos…).

## De atos de diálogo a ações do agente

Cada turno do assistente no SGD é anotado com atos de diálogo e, quando o assistente consultou ou alterou algo, com uma `service_call`. `nextaction/sgd.py` transforma cada turno do assistente em um exemplo de treino, derivando a ação dessas anotações:

| Ação | Derivada de | Significado para um agente |
| --- | --- | --- |
| `search` | uma `service_call` que não é uma transação | consultar uma ferramenta ou API antes de responder |
| `execute` | `NOTIFY_SUCCESS` / `NOTIFY_FAILURE` | executar a transação e informar o resultado |
| `confirm` | `CONFIRM` | confirmar os detalhes antes de uma transação |
| `ask` | `REQUEST` | pedir ao usuário as informações que faltam |
| `suggest_next_step` | `OFFER_INTENT` | propor o próximo passo ("quer que eu reserve?") |
| `present_options` | `OFFER`, `INFORM_COUNT` sem nova consulta | apresentar uma opção dos resultados já obtidos |
| `answer` | `INFORM` | responder a uma pergunta sobre um item |
| `offer_more_help` | `REQ_MORE` | perguntar se o usuário precisa de mais alguma coisa |
| `end_conversation` | `GOODBYE` | encerrar a conversa |

Quando um turno tem vários atos, vale a primeira correspondência na tabela; `search` vem primeiro porque a consulta que antecede a resposta é a decisão que importa. Esses rótulos são o vocabulário comum do projeto em `nextaction/vocabulary.py`, então os cenários `flights` e `research` usam os mesmos nomes.

## Formato do estado

O modelo vê o estado do diálogo seguido dos três últimos turnos:

```text
[intent SearchOnewayFlight] [filled origin_airport] [missing destination_airport departure_date] [requested none]
|| user: I need a flight from Boston.
```

- `intent`: a intenção ativa do usuário.
- `filled`: campos que o usuário já forneceu (apenas nomes, sem valores).
- `missing`: campos obrigatórios dessa intenção, segundo o esquema do serviço, que ainda estão vazios.
- `requested`: campos sobre os quais o usuário perguntou.

Para usar o modelo no seu próprio agente, monte a mesma string a partir do estado que você já acompanha. Tags em português (`[intencao] [preenchidos] [faltando] [pedidos]`, `usuario:`, `assistente:`) são reescritas para esse formato automaticamente. `--no-state` produz texto só com a conversa, e `--context N` altera quantos turnos são incluídos.

## Reproduzir

```bash
git clone --depth 1 https://github.com/google-research-datasets/dstc8-schema-guided-dialogue data/external/sgd
for split in train dev test; do
  nextaction import-sgd data/external/sgd/$split --output data/external/sgd_generic_$split.csv
done

nextaction evaluate agent      # trains on train, evaluates on the official test split
nextaction train agent         # writes models/agent.joblib
```

O treino com 165 mil exemplos leva alguns minutos na CPU de um notebook. O cenário usa `min_df=5`, que descarta termos vistos em menos de cinco turnos: o vocabulário cai de 178 mil para 54 mil termos sem perda de acurácia (0,900 → 0,902), e com a compressão do skops o modelo exportado passa de 125 MB para 4,7 MB.

`--schema flights` mantém apenas os turnos de voos e as quatro ações do cenário de voos (`ask`, `search`, `present_options`, `confirm`).

## Resultados

Modelo publicado (`min_df=5`), treinado em `train` (164.982 exemplos), avaliado na divisão oficial `test` (42.297 exemplos, incluindo serviços ausentes do treino).

| Ação | Precisão | Revocação | F1 | Suporte |
| --- | --- | --- | --- | --- |
| `answer` | 0,984 | 0,964 | 0,974 | 4.170 |
| `ask` | 0,999 | 0,942 | 0,970 | 9.016 |
| `confirm` | 0,903 | 0,849 | 0,875 | 5.697 |
| `end_conversation` | 0,925 | 0,893 | 0,909 | 4.201 |
| `execute` | 0,939 | 0,924 | 0,932 | 5.006 |
| `offer_more_help` | 0,732 | 0,765 | 0,748 | 2.684 |
| `present_options` | 0,684 | 0,711 | 0,697 | 1.030 |
| `search` | 0,850 | 0,959 | 0,901 | 8.233 |
| `suggest_next_step` | 0,796 | 0,777 | 0,787 | 2.260 |

Acurácia de 0,902 contra uma linha de base de 0,213 da classe majoritária; erro de calibração esperado de 0,084. Na divisão `dev` (mesmos serviços do treino), a acurácia fica em torno de 0,93.

| Limiar de revisão | Decisões automatizadas | Acurácia nas automatizadas |
| --- | --- | --- |
| 0,5 | 90,2% | 94,3% |
| 0,7 | 75,1% | 97,8% |
| 0,8 | 65,4% | 98,8% |
| 0,9 | 50,0% | 99,6% |

### Ablação: o estado do diálogo importa?

As duas execuções usam o vocabulário completo (`min_df=1`).

| Entrada | Acurácia | F1 `ask` | F1 `search` | Automatizadas em 0,9 |
| --- | --- | --- | --- | --- |
| Conversa + estado | 0,900 | 0,968 | 0,898 | 48,5% |
| Só a conversa | 0,772 | 0,707 | 0,744 | 28,9% |

Só pelo texto, "pedir mais detalhes" e "buscar agora" se parecem. Saber quais campos obrigatórios estão faltando separa os dois. As ações mais fracas, `present_options` e `offer_more_help`, dependem de coisas que a string de estado ainda não captura, como se os resultados já foram mostrados.

## Ressalvas

- As conversas do SGD foram escritas por trabalhadores de crowdsourcing a partir de roteiros de diálogo simulados. Elas são mais limpas e regulares que o tráfego real.
- O conjunto de dados tem licença CC BY-SA 4.0. Ele não é redistribuído neste repositório, e modelos treinados com ele devem manter essa licença e citar Rastogi et al. (2020).
