# Benchmark de Pac-Man

300 partidas por agente nas sementes 200000–200299, que nunca aparecem nos dados de treino ou de teste. Gerado por `python demos/pacman/benchmark.py --games 300` em 2026-10-11 (Python 3.14.5).

| Agente | Taxa de vitória | Intervalo de 95% | Pontuação média | Passos médios | Movimentos decididos pelo especialista | Correções da política por partida |
| --- | --- | --- | --- | --- | --- | --- |
| modelo + política, especialista em baixa confiança | 39% | 33%–44% | 3079 | 122 | 3,3% | 0,00 |
| modelo + política, sozinho | 40% | 35%–46% | 2928 | 121 | 0,0% | 0,00 |
| modelo sozinho, sem política | 40% | 35%–46% | 2928 | 121 | 0,0% | 0,00 |
| especialista baseado em regras | 35% | 30%–41% | 3258 | 126 | 0,0% | 0,00 |
| movimento legal aleatório | 0% | 0%–1% | 43 | 9 | 0,0% | 0,00 |

- **Movimentos decididos pelo especialista**: fração dos movimentos em que a confiança do modelo ficou abaixo do limiar da política, de modo que o especialista (que no jogo faz o papel de um revisor humano) escolheu no lugar dele.
- **Correções da política**: movimentos em que a primeira escolha do próprio modelo era uma parede ou uma célula ao lado de um fantasma e a política a bloqueou.
