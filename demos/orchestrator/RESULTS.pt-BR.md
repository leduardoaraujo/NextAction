# Benchmark de roteamento do orquestrador de IA

O conjunto de teste contém 34 famílias de pedidos escritas à mão e separadas do treino e da calibração da política. Cada família aparece uma vez. Todos os exemplos são sintéticos e aproximadamente balanceados entre quatro ações; os números não estimam a acurácia no tráfego de produção.

| Roteador | Acurácia da próxima ação | Intervalo de Wilson de 95% | Acurácia de rota de agente | F1 macro | Despachos errados ao agente de dados | Pedidos de dados perdidos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline de palavras-chave | 29,4% | 16,8%–46,2% | 70,6% | 0,177 | 3 | 7 |
| Modelo NextAction | 85,3% | 69,9%–93,6% | 91,2% | 0,848 | 1 | 2 |

A acurácia de rota de agente difere em +20,6% para o NextAction em relação ao baseline de palavras-chave (intervalo de bootstrap pareado por família de 95%: +5,9% a +35,3%). O intervalo exclui o zero neste conjunto sintético; isso não estabelece uma vantagem em produção.

Com a política calibrada, o NextAction automatiza 33/34 casos (97,1%); 29/33 das próximas ações automatizadas estão corretas. Outra decisão é necessária em 1 caso(s). A acurácia da revisão não é presumida.

A inferência local em lote levou 0,36 ms por caso para o modelo NextAction nesta máquina. Esta não é uma medida de latência ponta a ponta comparável à dos outros roteadores.

O classificador LLM do Axis não foi executado: as credenciais do serviço e o runtime não estão presentes aqui. Use `--axis-predictions` com uma classificação exportada por família de teste para adicionar o resultado medido dele.

A acurácia de rota de agente avalia apenas se o pedido chega ao analista de dados ou ao caminho de conversa. A acurácia da próxima ação distingue dados, chat, esclarecimento e recuperação. O baseline de palavras-chave é uma recriação independente e congelada da heurística de visualização legada do Axis; ele tem apenas saídas de dados/chat. A política do NextAction também lê tags de contexto explícitas e bloqueia o despacho ao agente de dados quando não há fonte de dados disponível. Esta é uma comparação de roteamento ponta a ponta, não uma ablação controlada apenas do modelo.
