# Benchmark de Connect Four (Lig 4)

200 partidas por agente contra uma busca alfa-beta de profundidade 4, metade jogando primeiro, cada uma começando com 2 movimentos de abertura aleatórios que são os mesmos para todos os agentes em uma semente. Sementes 200000–200199 (nunca usadas nos dados de treino ou de teste). Gerado por `python demos/connect4/benchmark.py --games 200` em 2026-10-11 (Python 3.14.5).

| Agente | Vitórias | Empates | Derrotas | Taxa de vitória | Intervalo de 95% | Movimentos revisados |
| --- | --- | --- | --- | --- | --- | --- |
| modelo + política, movimentos incertos revisados por busca de profundidade 6 | 79 | 54 | 67 | 40% | 33%–46% | 39,4% |
| modelo + política, movimentos incertos revisados pelo professor de profundidade 4 | 66 | 58 | 76 | 33% | 27%–40% | 41,3% |
| modelo + política, sem revisão | 38 | 39 | 123 | 19% | 14%–25% | 0,0% |
| busca de profundidade 4 (o professor) | 76 | 58 | 66 | 38% | 32%–45% | 0,0% |
| busca de profundidade 6 | 83 | 41 | 76 | 42% | 35%–48% | 0,0% |
| coluna legal aleatória | 0 | 0 | 200 | 0% | 0%–2% | 0,0% |

**Movimentos revisados**: fração dos movimentos do agente em que a confiança do modelo ficou abaixo do limiar da política e uma busca escolheu no lugar dele.
