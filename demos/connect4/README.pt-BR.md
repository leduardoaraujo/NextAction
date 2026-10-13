# Demo de Connect Four (Lig 4)

O NextAction escolhendo a coluna em que jogar, contra uma busca alfa-beta. O painel mostra a probabilidade de cada coluna, as colunas que a política bloqueou (cheias, ou que entregam a vitória ao adversário), a confiança comparada ao limiar e quem decidiu: quando o modelo está inseguro, uma busca mais profunda revisa o movimento.

![NextAction jogando Lig 4](connect4.gif)

Esta demo responde à pergunta que a [demo de Pac-Man](../pacman/README.md) deixou em aberto. Lá, enviar os movimentos incertos ao especialista não venceu mais partidas, porque o especialista não era mais forte que o modelo que ensinou. Aqui o modelo aprende com uma busca de profundidade 4, e os movimentos incertos podem ser revisados pelo mesmo professor ou por uma busca mais profunda, de profundidade 6.

## Como executar

A partir da raiz do repositório:

```bash
pip install -e ".[demo]"

python demos/connect4/generate_data.py      # positions from 1,500 training and 300 test games (~10 min)
nextaction train lig4                       # or: nextaction train connect4
nextaction evaluate connect4

python demos/connect4/play.py               # watch the model play in the terminal
python demos/connect4/play.py --agent model-only
python demos/connect4/play.py --gif demos/connect4/connect4.gif --pick-win
python demos/connect4/benchmark.py --games 200   # ~30 min, depth-6 search is slow in Python
```

| Script | O que faz |
| --- | --- |
| [`game.py`](game.py) | o tabuleiro 6×7 e um jogador de busca alfa-beta com heurística simétrica |
| [`features.py`](features.py) | transforma uma posição em uma string de estado do NextAction, do ponto de vista de quem joga |
| [`generate_data.py`](generate_data.py) | joga partidas variadas e rotula cada posição com o movimento do professor de profundidade 4 |
| [`play.py`](play.py) | uma partida contra uma busca de profundidade 4, ao vivo no terminal ou gravada em GIF |
| [`benchmark.py`](benchmark.py) | todos os agentes contra a busca de profundidade 4, e escreve o [RESULTS.md](RESULTS.md) |

## Estado e política

```text
[blocked none] [danger c6] [features block_c3 c1_bad c3_great c4_better ... h3_1 h4_2 r0c4_me r0c3_op ...]
```

- `[features ...]`: vitórias imediatas (`win_c4`) e bloqueios (`block_c3`), como cada coluna altera uma pontuação heurística (`c3_great`), as alturas das colunas e cada célula ocupada como `me` ou `op`.
- `[blocked ...]` lista as colunas cheias e `[danger ...]` lista as colunas que deixam o adversário vencer logo em cima. Duas regras `deny_tag` em [`configs/connect4.toml`](../../configs/connect4.toml) removem ambas antes de o modelo escolher.
- Um movimento com confiança abaixo de 0,5 é revisado por uma busca.

## Resultados

Acurácia de imitação em 6.716 posições de 300 partidas inéditas: **0,668** (sempre a coluna mais comum: 0,282). Muitas posições têm várias colunas igualmente boas, então uma escolha diferente da do professor nem sempre é pior.

200 partidas por agente contra uma busca de profundidade 4, metade jogando primeiro. Cada partida começa com dois movimentos de abertura aleatórios, os mesmos para todos os agentes em uma semente:

| Agente | Vitórias | Empates | Derrotas | Taxa de vitória (intervalo de 95%) | Movimentos revisados |
| --- | --- | --- | --- | --- | --- |
| modelo + política, revisado por busca de profundidade 6 | 79 | 54 | 67 | 40% (33–46%) | 39,4% |
| modelo + política, revisado pelo professor de profundidade 4 | 66 | 58 | 76 | 33% (27–40%) | 41,3% |
| modelo + política, sem revisão | 38 | 39 | 123 | 19% (14–25%) | — |
| busca de profundidade 4 (o professor) | 76 | 58 | 66 | 38% (32–45%) | — |
| busca de profundidade 6 | 83 | 41 | 76 | 42% (35–48%) | — |
| coluna legal aleatória | 0 | 0 | 200 | 0% (0–2%) | — |

- **Sozinho, o modelo é bem mais fraco que seu professor** (19% de vitórias, 123 derrotas). Acertar dois terços dos movimentos de uma busca não basta em um jogo tático, em que um único movimento ruim perde a partida.
- **Com revisão, ele joga no nível do professor**: revisado por uma busca de profundidade 6, vence 40%, perto da busca de profundidade 4 jogando todos os movimentos (38%), enquanto busca em apenas 39% dos seus movimentos.
- **Um revisor mais forte ajuda mais que um equivalente**: 13 vitórias a mais e 9 derrotas a menos do que revisando com o professor, com a mesma taxa de revisão. Os intervalos de 95% se sobrepõem, então isso é um indício, não uma conclusão; junto com o resultado do Pac-Man, aponta na mesma direção: a revisão agrega valor quando o revisor sabe mais que o modelo.

Tabela completa: [RESULTS.md](RESULTS.md).

## Observações

- Sem aberturas aleatórias, toda partida repetia duas a quatro linhas de abertura, o que fazia os resultados dependerem de um punhado de posições; aberturas aleatórias fixas por semente resolveram isso.
- A primeira heurística era assimétrica, e uma busca mais profunda não vencia uma mais rasa; a versão simétrica faz a profundidade 6 vencer a profundidade 4 por 22 a 8 em confrontos diretos.
