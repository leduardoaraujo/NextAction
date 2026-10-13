# Relatório de testes

Gerado por `python scripts/test_report.py`. O CI produz o mesmo relatório a cada push e pull request, no resumo da execução na [aba Actions](https://github.com/leduardoaraujo/NextAction/actions/workflows/ci.yml). Os nomes dos testes vêm do código e ficam em inglês.

| | |
| --- | --- |
| Resultado | **77 de 77 passaram** |
| Duração | 3,3 s |
| Cobertura (linhas + ramos) | **83,4%** (1170/1345 instruções, 248/356 ramos) |
| Commit base | `aa2331c` |
| Ambiente | Python 3.14.5 no Windows |
| Data | 2026-10-13 02:01 UTC |

## Suítes

| Suíte | O que verifica | Testes | Resultado |
| --- | --- | --- | --- |
| [`test_api`](../tests/test_api.py) | Contratos HTTP da API REST, em inglês e com os aliases em português. | 12 | ✅ 12/12 |
| [`test_bandit`](../tests/test_bandit.py) | O aprendizado por resultados exige ações registradas com probabilidade de escolha conhecida. | 2 | ✅ 2/2 |
| [`test_cli`](../tests/test_cli.py) | Interface de linha de comando em inglês, com aliases em português e mensagens localizadas. | 6 | ✅ 6/6 |
| [`test_connect4_demo`](../tests/test_connect4_demo.py) | Demo de Lig 4: regras, jogador com busca e o estado que a política lê. | 5 | ✅ 5/5 |
| [`test_decision_log`](../tests/test_decision_log.py) | Registro de decisões e exportação de rótulos. | 2 | ✅ 2/2 |
| [`test_evaluation`](../tests/test_evaluation.py) | Relatório de avaliação do classificador. | 5 | ✅ 5/5 |
| [`test_flights`](../tests/test_flights.py) | Ranking de ofertas de voo baseado em regras. | 7 | ✅ 7/7 |
| [`test_hub`](../tests/test_hub.py) | Exportação para o Hugging Face Hub. | 2 | ✅ 2/2 |
| [`test_orchestrator_demo`](../tests/test_orchestrator_demo.py) | Dados sintéticos de orquestração, política e contratos do benchmark. | 4 | ✅ 4/4 |
| [`test_pacman_demo`](../tests/test_pacman_demo.py) | Demo de Pac-Man: regras do jogo, descrição do estado e as regras deny_tag da política. | 5 | ✅ 5/5 |
| [`test_policy`](../tests/test_policy.py) | Política de decisão, configs de cenário, ajuste e calibração. | 10 | ✅ 10/10 |
| [`test_progress`](../tests/test_progress.py) | Progresso via arquivos de job, CLI e API. | 7 | ✅ 7/7 |
| [`test_sgd`](../tests/test_sgd.py) | Conversão do Schema-Guided Dialogue em exemplos state,action. | 4 | ✅ 4/4 |
| [`test_vocabulary`](../tests/test_vocabulary.py) | Vocabulário canônico, aliases em português e compatibilidade com versões anteriores. | 6 | ✅ 6/6 |

## Cobertura por módulo

| Módulo | Instruções | Não cobertas | Ramos | Cobertura |
| --- | --- | --- | --- | --- |
| `nextaction/__init__.py` | 0 | 0 | 0/0 | 100% |
| `nextaction/__main__.py` | 3 | 3 | 0/2 | 0% |
| `nextaction/api.py` | 58 | 6 | 4/6 | 88% |
| `nextaction/bandit.py` | 64 | 3 | 13/16 | 92% |
| `nextaction/cli.py` | 172 | 22 | 24/34 | 84% |
| `nextaction/data.py` | 22 | 3 | 3/6 | 79% |
| `nextaction/decision_log.py` | 58 | 3 | 15/18 | 92% |
| `nextaction/evaluation.py` | 86 | 6 | 18/24 | 89% |
| `nextaction/flights.py` | 99 | 17 | 45/60 | 80% |
| `nextaction/hub.py` | 92 | 17 | 11/22 | 75% |
| `nextaction/i18n.py` | 24 | 1 | 8/10 | 91% |
| `nextaction/model.py` | 66 | 7 | 9/16 | 83% |
| `nextaction/policy.py` | 133 | 9 | 33/42 | 90% |
| `nextaction/progress.py` | 113 | 11 | 19/26 | 87% |
| `nextaction/scenarios.py` | 65 | 3 | 3/4 | 94% |
| `nextaction/schemas.py` | 69 | 0 | 2/2 | 100% |
| `nextaction/sgd.py` | 65 | 4 | 24/28 | 91% |
| `nextaction/tuning.py` | 80 | 11 | 17/24 | 83% |
| `nextaction/vision.py` | 57 | 49 | 0/16 | 11% |
| `nextaction/vocabulary.py` | 19 | 0 | 0/0 | 100% |

## Todos os testes

### `test_api`

| | Teste | Tempo |
| --- | --- | --- |
| ✅ | Decide accepts portuguese fields and scenario | 40 ms |
| ✅ | Decide in english | 10 ms |
| ✅ | Feedback for unknown decision is localized | 13 ms |
| ✅ | Feedback links final action and normalizes portuguese label | 22 ms |
| ✅ | Health and openapi | 53 ms |
| ✅ | Invalid feedback action returns 422 | 4 ms |
| ✅ | Invalid offer returns 422 | 4 ms |
| ✅ | Missing model returns localized 503 | 8 ms |
| ✅ | Missing trip field asks | 6 ms |
| ✅ | Rank flights | 5 ms |
| ✅ | Rank flights accepts portuguese fields | 4 ms |
| ✅ | Unknown scenario and invalid threshold return 422 | 7 ms |

### `test_bandit`

| | Teste | Tempo |
| --- | --- | --- |
| ✅ | Candidate requires outcomes and stays separate from serving | 2027 ms |
| ✅ | Feedback requires a complete logged observation | 1 ms |

### `test_cli`

| | Teste | Tempo |
| --- | --- | --- |
| ✅ | Errors follow the language option | 11 ms |
| ✅ | Every message has both languages | 2 ms |
| ✅ | Language falls back to environment | 3 ms |
| ✅ | Portuguese commands options and scenarios | 36 ms |
| ✅ | Rank flights with portuguese json | 7 ms |
| ✅ | Train and decide in english | 34 ms |

### `test_connect4_demo`

| | Teste | Tempo |
| --- | --- | --- |
| ✅ | Alpha beta matches plain minimax | 11 ms |
| ✅ | Connect4 config loads with alias | 0 ms |
| ✅ | Four in a row wins | 0 ms |
| ✅ | Search takes wins and blocks threats | 12 ms |
| ✅ | State lists full columns and gifted wins | 1 ms |

### `test_decision_log`

| | Teste | Tempo |
| --- | --- | --- |
| ✅ | Exports only decisions with feedback and uses the latest | 29 ms |
| ✅ | Feedback requires a recorded decision | 1 ms |

### `test_evaluation`

| | Teste | Tempo |
| --- | --- | --- |
| ✅ | Cross validation evaluates every example | 49 ms |
| ✅ | Recent fraction tests on the last rows | 12 ms |
| ✅ | Rejects two test modes | 1 ms |
| ✅ | Report text is localized | 41 ms |
| ✅ | Threshold splits automated from review | 4 ms |

### `test_flights`

| | Teste | Tempo |
| --- | --- | --- |
| ✅ | Asks for required fields before searching | 0 ms |
| ✅ | Filters and explains the ranking | 0 ms |
| ✅ | No eligible offer asks to revise criteria | 0 ms |
| ✅ | Rejects invalid price | 0 ms |
| ✅ | Rejects return before departure with localized message | 0 ms |
| ✅ | Searches when there are no offers | 0 ms |
| ✅ | Selected offer asks for localized confirmation | 0 ms |

### `test_hub`

| | Teste | Tempo |
| --- | --- | --- |
| ✅ | Rejects unknown scenario | 1 ms |
| ✅ | Writes loadable skops model and model card | 99 ms |

### `test_orchestrator_demo`

| | Teste | Tempo |
| --- | --- | --- |
| ✅ | Alias and no source mask | 0 ms |
| ✅ | Axis predictions require complete matching families | 3 ms |
| ✅ | Keyword router and route metric | 3 ms |
| ✅ | Phrase families do not cross splits | 16 ms |

### `test_pacman_demo`

| | Teste | Tempo |
| --- | --- | --- |
| ✅ | Deny tag rules remove walls and ghosts | 0 ms |
| ✅ | Games are deterministic per seed | 70 ms |
| ✅ | Pacman config loads its policy | 0 ms |
| ✅ | State lists blocked moves for the policy | 0 ms |
| ✅ | Walls stop pacman and pellets score | 0 ms |

### `test_policy`

| | Teste | Tempo |
| --- | --- | --- |
| ✅ | Calibration raises threshold when errors are expensive | 1 ms |
| ✅ | Explicit threshold overrides policy | 0 ms |
| ✅ | Irreversible action always needs review and confirmation | 0 ms |
| ✅ | Parse state reads tags in both languages | 0 ms |
| ✅ | Policy simulation counts cost | 0 ms |
| ✅ | Rule blocks actions and renormalizes | 0 ms |
| ✅ | Bundled configs load | 0 ms |
| ✅ | Update config keeps comments | 26 ms |
| ✅ | Calibrate policy reports cost before and after | 68 ms |
| ✅ | Tune writes best parameters | 316 ms |

### `test_progress`

| | Teste | Tempo |
| --- | --- | --- |
| ✅ | Api lists jobs and serves dashboard | 23 ms |
| ✅ | Cli train records a job and lists it | 42 ms |
| ✅ | Failed job keeps the error | 11 ms |
| ✅ | Job file tracks stage percent and completion | 19 ms |
| ✅ | Labels are localized | 2 ms |
| ✅ | Running job without updates becomes stale | 16 ms |
| ✅ | Stage outside a job is a no op | 2 ms |

### `test_sgd`

| | Teste | Tempo |
| --- | --- | --- |
| ✅ | Flights schema drops execution and accepts portuguese alias | 7 ms |
| ✅ | Maps acts and service calls | 6 ms |
| ✅ | State shows missing slots | 5 ms |
| ✅ | Without state uses only the conversation | 6 ms |

### `test_vocabulary`

| | Teste | Tempo |
| --- | --- | --- |
| ✅ | Csv with portuguese columns and labels | 3 ms |
| ✅ | Free text is not rewritten | 0 ms |
| ✅ | Model with portuguese labels returns english actions | 20 ms |
| ✅ | Portuguese actions map to canonical labels | 0 ms |
| ✅ | Scenarios accept portuguese names | 0 ms |
| ✅ | State tags and speakers are normalized | 0 ms |

