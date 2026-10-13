# Avaliação

Um modelo de decisão só é útil se você sabe **com que frequência ele erra, em quais ações e quanto dá para automatizar com segurança**. `nextaction evaluate <scenario>` responde a essas três perguntas.

## Escolhendo os dados de teste

| Opção | Quando usar |
| --- | --- |
| *(padrão)* validação cruzada estratificada, até 5 folds | conjuntos de dados pequenos, em que todo exemplo é necessário para o treino |
| `--test file.csv` | um conjunto de teste fixo e separado; o cenário `agent` usa por padrão o split de teste oficial do SGD |
| `--recent-fraction 0.2` | dados exportados do log de decisões; os 20% mais recentes viram o conjunto de teste, sem embaralhar |

`--recent-fraction` é o mais próximo do uso real: você treina no passado e mede no futuro. Divisões aleatórias superestimam o desempenho quando o tráfego muda com o tempo.

## Campos do relatório

| Campo | Significado |
| --- | --- |
| `accuracy` | proporção de previsões corretas |
| `majority_baseline_accuracy` | acurácia de sempre prever a ação mais comum no treino; o modelo precisa superar isso com folga |
| `per_action` | precisão, recall, F1 e suporte por ação. Um recall baixo numa ação arriscada (como `confirm`) importa mais que a acurácia geral |
| `confusion_matrix` | `{actual: {predicted: count}}`, mostrando quais ações são confundidas |
| `expected_calibration_error` | ECE: diferença média entre a confiança declarada e a acurácia real, em 10 faixas de confiança. 0 significa que "70% de confiança" acerta 70% das vezes |
| `log_loss` | penaliza erros cometidos com confiança; útil para comparar modelos |
| `thresholds` | para cada limiar: proporção de decisões `automated`, `automated_accuracy`, `unreviewed_errors` e casos `sent_to_review` |
| `warning` | aparece quando menos de 100 exemplos foram avaliados, porque os números são instáveis |

`method` e `warning` são textos para pessoas e seguem `--lang`; todos os outros campos são iguais nos dois idiomas.

## Escolhendo um limiar de revisão

Leia a tabela `thresholds` como uma curva de compromisso. Cada linha diz: *se as decisões abaixo desta confiança forem para uma pessoa, X% são automáticas e Y erros acontecem sem revisão.*

1. Estime o custo de um erro sem revisão e o custo de uma revisão humana.
2. Escolha o menor limiar cuja quantidade de erros sem revisão seja aceitável.
3. Confira de novo com dados novos. As probabilidades não são calibradas, então um limiar que funciona num conjunto de dados pode se desviar em outro.

Exemplo no split de teste do SGD: um limiar de 0,9 automatiza 50,0% das decisões com 99,6% de acurácia, e 0,7 automatiza 75,1% com 97,8%.

## Um resultado que serve de alerta

Nos 48 exemplos sintéticos de voos, uma única divisão aleatória chegou a mostrar 0,667 de acurácia. A validação cruzada mostrou 0,50, e nenhuma previsão passou de 0,5 de confiança, então com um limiar de 0,70 toda decisão iria para revisão. Conjuntos de dados pequenos fazem os números de uma única divisão parecerem melhores do que são, e é por isso que `train` não mostra mais uma pontuação de teste e indica `evaluate` no lugar.
