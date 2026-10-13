# Demo de roteamento de um orquestrador de IA

Este cenário do NextAction prevê a **próxima ação de um orquestrador de IA**. Ele se
inspira na divisão usada no workspace Axis do Luiz: pedidos analíticos vão para um
agente analista de dados, conversa comum vai para um agente de conversa e intenção
de baixa confiança vai para um fallback. A demo acrescenta as ações explícitas
`clarification_needed` e `error_recovery` para que o próximo passo do roteador possa
ser avaliado, e não apenas o nome do agente final. Ela não modifica o Axis nem usa
seus dados privados.

## Reproduzir localmente

```powershell
.venv/Scripts/python.exe demos/orchestrator/generate_data.py
.venv/Scripts/python.exe -m nextaction train orchestrator
.venv/Scripts/python.exe -m nextaction calibrate-policy orchestrator --write
.venv/Scripts/python.exe -m nextaction evaluate orchestrator
.venv/Scripts/python.exe demos/orchestrator/benchmark.py
.venv/Scripts/python.exe demos/orchestrator/analysis.py
.venv/Scripts/python.exe demos/orchestrator/play.py "Compare revenue by region"
```

`orquestrador` é o alias do cenário em português. O [benchmark](RESULTS.md)
informa tanto a acurácia exata da próxima ação quanto a acurácia mais simples da
rota agente de dados versus conversa. A [análise de erros](ANALYSIS.md) lista cada
caso de teste errado e mostra a confiança por faixa.

## Estado e política

```text
[blocked none] [context datasource_available last_error_none missing_none] || user: Compare monthly revenue by region
```

O ambiente informa a disponibilidade da fonte, um erro de SQL anterior e qualquer
métrica ausente. `[blocked data_query]` impede que a política escolha despachar para
o agente de dados sem uma fonte de dados disponível. O resultado `review` significa
que outro tomador de decisão precisa tratar uma rota de baixa confiança; esse
resultado **não** é considerado correto no benchmark. As quatro ações do modelo são
`data_query`, `pure_chat`, `clarification_needed` e `error_recovery`. Elas
correspondem às duas rotas de agente, a um pedido de esclarecimento e a um passo de
recuperação.

## Dados e comparação

Os 159 casos são sintéticos e escritos à mão: 100 de treino, 25 de validação e 34
de teste. As famílias de frases são disjuntas entre as partições. O conjunto de teste
é aproximadamente balanceado por ação, então sua acurácia é um diagnóstico, não uma
estimativa de produção. Nenhuma mensagem, prompt, schema ou credencial do Axis está
incluída. O baseline de palavras-chave é uma recriação independente e congelada da
antiga heurística de visualização do Axis. Ele oferece apenas as rotas de dados e de
chat; informar tanto a acurácia de rota quanto a de ação exata deixa visível essa
taxonomia mais restrita.

O classificador LLM do Axis precisa do runtime e das credenciais do Axis, que não
estão presentes neste workspace. Seus resultados medidos podem ser adicionados sem
modificar o benchmark:

```powershell
# Run with the Axis backend's Python environment and configured API credentials.
python demos/orchestrator/export_axis_predictions.py --axis-backend PATH/TO/axis/backend
.venv/Scripts/python.exe demos/orchestrator/benchmark.py --axis-predictions data/external/axis_orchestrator_predictions.csv
```

O exportador faz 34 chamadas ao classificador e grava apenas `family`, `action` e
`latency_ms` em um CSV ignorado pelo git. Não o execute sem a intenção de fazer essas
chamadas. Para uma afirmação de portfólio sobre uma implantação real, substitua ou
amplie este conjunto sintético com traces do Axis representativos, rotulados
manualmente e seguros quanto a permissões, mantenha famílias de conversa inteiras
fora do treino e compare os três roteadores nos mesmos casos separados. A demo atual
mostra um protótipo reproduzível, não superioridade sobre o classificador LLM do Axis.
