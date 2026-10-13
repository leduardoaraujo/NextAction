# Demo de Pac-Man

O NextAction escolhendo o próximo movimento do Pac-Man em tempo real. O painel mostra cada decisão: a probabilidade de cada movimento, os movimentos que a política de decisão bloqueou, a confiança comparada ao limiar e quem decidiu. Quando o modelo está inseguro, o movimento vai para um especialista baseado em regras, que no jogo faz o papel de um revisor humano.

![NextAction jogando Pac-Man](pacman.gif)

O jogo usa o mesmo pipeline de todos os outros cenários: exemplos rotulados `state,action`, `nextaction train`, `nextaction evaluate` e uma política de decisão em [`configs/pacman.toml`](../../configs/pacman.toml).

## Como executar

A partir da raiz do repositório:

```bash
pip install -e ".[demo]"

python demos/pacman/generate_data.py       # expert plays 400 training and 100 test games
nextaction train pacman                    # ~50k examples, a few seconds
nextaction evaluate pacman                 # accuracy on moves from unseen games

python demos/pacman/play.py                # watch the model play in the terminal
python demos/pacman/play.py --agent expert # or the expert, or random, raw, model-only
python demos/pacman/play.py --gif demos/pacman/pacman.gif --pick-win
python demos/pacman/benchmark.py --games 300
python demos/pacman/analysis.py --games 300    # error analysis and charts
```

| Script | O que faz |
| --- | --- |
| [`game.py`](game.py) | o jogo: um labirinto de 15×13, dois fantasmas que perseguem na maior parte do tempo, pílulas de poder; determinístico por semente |
| [`features.py`](features.py) | transforma uma posição em uma string de estado do NextAction |
| [`expert.py`](expert.py) | o jogador baseado em regras que rotula os dados de treino e revisa os movimentos incertos |
| [`generate_data.py`](generate_data.py) | grava partidas do especialista como CSVs de treino e de teste (sementes diferentes) |
| [`play.py`](play.py) | executa uma partida com qualquer agente, ao vivo no terminal ou gravada em GIF |
| [`benchmark.py`](benchmark.py) | joga centenas de partidas inéditas por agente e escreve o [RESULTS.md](RESULTS.md) |
| [`analysis.py`](analysis.py) | registra cada movimento, compara com o do especialista e escreve o [ANALYSIS.md](ANALYSIS.md) com gráficos |

## Estado e política

Cada posição vira um estado como:

```text
[blocked up down] [danger none] [features wall_up wall_down food_left_0 ghost_left_close food_right_0 ghost_right_close closest_food_left safest_left threat_mid power_no]
```

- `[features ...]` são tokens para o modelo, cada um ligado a uma direção: distância até a pastilha mais próxima (`food_left_0`), até o fantasma perigoso mais próximo (`ghost_up_close`), até um fantasma assustado (`prey_right_2`), as direções com a comida mais próxima e a mais segura, e se uma pílula de poder está ativa.
- `[blocked ...]` e `[danger ...]` listam movimentos pelo nome: paredes e células vizinhas a um fantasma que não está assustado. Duas regras da política com `deny_tag` os removem antes da escolha do modelo, então uma parede ou um fantasma nunca podem ser escolhidos.
- Um movimento com confiança abaixo de 0,6 vai para o especialista.

## Resultados

Acurácia de imitação em 12.734 movimentos de 100 partidas inéditas: **0,973** (escolher sempre o movimento mais comum: 0,251).

Jogando 300 partidas que nunca aparecem nos dados de treino ou de teste:

| Agente | Taxa de vitória | Intervalo de 95% | Pontuação média | Movimentos decididos pelo especialista |
| --- | --- | --- | --- | --- |
| modelo + política, especialista em baixa confiança | 39% | 33%–44% | 3079 | 3,3% |
| modelo + política, sozinho | 40% | 35%–46% | 2928 | — |
| modelo sozinho, sem política | 40% | 35%–46% | 2928 | — |
| especialista baseado em regras | 35% | 30%–41% | 3258 | — |
| movimento legal aleatório | 0% | 0%–1% | 43 | — |

- **O modelo joga tão bem quanto o especialista com quem aprendeu.** As taxas de vitória se sobrepõem dentro dos intervalos de 95%, então nenhum é melhor; o especialista pontua um pouco mais porque caça fantasmas assustados com mais frequência.
- **Ele pede ajuda em 3,3% dos movimentos**, as posições em que sua confiança fica abaixo do limiar.
- **Aqui a política é uma rede de segurança, não uma muleta.** O modelo quase nunca escolhe sozinho uma parede ou um movimento ao lado de um fantasma (0,003 correções por partida), então com ou sem a política ele vence a mesma fração de partidas. As regras garantem que ele não consiga, o que importa mais quando um erro é caro.

Tabela completa: [RESULTS.md](RESULTS.md).

## O que dá errado

O [ANALYSIS.md](ANALYSIS.md) registra todos os 36.746 movimentos das partidas do benchmark e compara cada um com o que o especialista faria. As principais conclusões:

- **O modelo discorda do seu professor em 3,1% dos movimentos**, e principalmente quando os fantasmas estão longe (4,7%) ou uma pílula de poder está ativa (5,1%), os momentos em que a escolha é sobre qual comida perseguir e não sobre sobrevivência. Perto de um fantasma ele discorda em apenas 2,2%.
- **A confiança é informativa, mas não decisiva.** Movimentos concordantes têm confiança mediana de 0,99 e os discordantes, de 0,69, mas o limiar de 0,6 captura apenas um terço das discordâncias.
- **Mais revisão não vence mais partidas.** Rejogar com limiares de 0,6 a 0,9 envia até 15,9% dos movimentos ao especialista, e a taxa de vitória fica em 37–39%: o especialista não é mais forte que o modelo que ensinou. A revisão só compensa quando o revisor é melhor que o modelo.
- **Toda derrota termina em uma armadilha.** Em todas as partidas perdidas, na última posição do Pac-Man todo movimento livre estava ao lado de um fantasma, então o erro fatal aconteceu antes, ao entrar em um beco sem saída, algo que features por movimento não conseguem enxergar.

![Confiança dos movimentos concordantes e discordantes](analysis/confidence_agree_vs_disagree.png)

## Limitações

- O especialista é um jogador guloso simples e vence apenas cerca de um terço das partidas, então o modelo aprende os erros dele também. Um professor mais forte, ou aprender a partir dos resultados das partidas, elevaria o teto.
- As features descrevem distâncias, não o labirinto inteiro, então o modelo não consegue planejar rotas longas.
