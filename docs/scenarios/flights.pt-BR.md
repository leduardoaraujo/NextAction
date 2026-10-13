# Cenário: próxima ação de um agente de reserva de voos

Este cenário é **fictício**. O agente ajuda um viajante a encontrar voos e escolhe o próximo passo da conversa; o modelo não consulta companhias aéreas nem compra passagens. Os exemplos incluídos em `data/flights.csv` estão escritos em português.

## Decisão

Entrada: uma descrição do estado atual da conversa e dos resultados da busca.

Saída: uma de quatro ações, com probabilidades estimadas.

| Ação | Quando |
| --- | --- |
| `ask` | falta origem, destino, data, número de viajantes ou outro requisito essencial |
| `search` | os requisitos essenciais são conhecidos e faltam ofertas atualizadas |
| `present_options` | os voos foram comparados por preço total, horários e condições, e o viajante pode escolher |
| `confirm` | o viajante escolheu uma oferta específica; são necessários os detalhes e uma autorização explícita antes de qualquer compra |

Fluxo de exemplo: o viajante pede uma passagem de São Paulo para Recife sem data → `ask`; informa as datas e o número de passageiros → `search`; depois de uma busca externa e da comparação → `present_options`; escolhe um voo → `confirm`.

O programa apenas **sugere** uma ação. Mesmo quando o modelo sugere `search`, a aplicação deve verificar os campos obrigatórios por conta própria. Preço, disponibilidade e regras tarifárias devem vir de uma fonte atualizada, e a compra exige uma etapa separada de autorização explícita.

## Experimente

```bash
nextaction train flights
nextaction decide flights "Tenho origem GRU destino REC ida 12/11 volta 19/11 e um adulto mas ainda não consultei voos"
nextaction evaluate flights
```

Os exemplos são sintéticos e poucos. A validação cruzada dá acurácia de 0,50, e o modelo nunca chega a 0,5 de confiança, então, com o limiar padrão, todas as decisões vão para revisão. Para melhorá-lo, substitua os exemplos por estados de conversa revisados, reserve os casos reais mais recentes para avaliação e só então escolha um limiar de revisão.

## Ranqueamento por regras com várias condições

`rank-flights` recebe um estado **estruturado** e toma uma decisão mais detalhada. O arquivo de exemplo é fictício e não contém preços nem disponibilidade reais.

```bash
nextaction rank-flights data/trip_example.json
```

O fluxo:

1. Se faltar origem, destino, data de ida ou número de passageiros, `ask` e lista os `missing_fields`.
2. Se não houver ofertas, `search`.
3. Descarta as ofertas que estouram o orçamento, o limite de escalas ou a exigência de bagagem (`over_budget`, `too_many_stops`, `missing_required_baggage`).
4. Se não sobrar nenhuma oferta, `revise_criteria`.
5. Ranqueia as ofertas elegíveis por preço total, duração e escalas; `present_options` com a melhor e as alternativas.
6. Se o viajante já escolheu uma oferta, `confirm` os detalhes e o preço atual.

A pontuação usa os pesos configuráveis de `request.weights`. Para cada oferta elegível, os componentes são `lowest_price/price`, `shortest_duration/duration` e `1/(1+stops)`. A média ponderada deles **apenas ordena as ofertas**; não é uma probabilidade nem um modelo treinado. O preço total deve incluir todos os passageiros, taxas e a bagagem exigida para que a comparação faça sentido.

Para simular uma escolha, adicione `"selected_offer": "flight_a"` ao objeto JSON de nível superior. O resultado é `confirm`; o comando nunca emite uma passagem. Chaves em português (`pedido`, `ofertas`, `oferta_escolhida`, `preco_total`…) também são aceitas.

Esse ranqueador complementa o classificador de próxima ação. O próximo passo de aprendizado de máquina é registrar as ofertas mostradas e qual delas o viajante escolheu. Com dados reais e consentidos, um modelo de preferência poderia ser treinado e seu ranqueamento comparado com esta fórmula simples em viagens reservadas para teste.
