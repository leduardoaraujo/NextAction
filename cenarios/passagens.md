# Cenário: próxima ação de um agente de passagens aéreas

Este cenário é **fictício**. O agente ajuda a encontrar passagens e escolhe a próxima etapa do atendimento; o modelo não consulta companhias nem compra bilhetes.

## Decisão

Entrada: uma descrição do estado atual da conversa e dos resultados de busca.

Saída: uma das quatro ações abaixo e suas probabilidades estimadas.

| Ação | Quando usar |
| --- | --- |
| `perguntar_dados` | Falta origem, destino, data, quantidade de viajantes ou outro requisito essencial. |
| `buscar_voos` | Os requisitos essenciais estão definidos e faltam ofertas atualizadas. |
| `apresentar_opcoes` | Há voos comparados com preço total, horários e condições para o viajante avaliar. |
| `pedir_confirmacao` | O viajante escolheu uma oferta concreta e é preciso confirmar detalhes e obter autorização explícita antes de qualquer compra. |

Fluxo exemplo: o viajante pede uma passagem de São Paulo para Recife sem data → `perguntar_dados`; informa ida e volta e número de passageiros → `buscar_voos`; após uma busca externa e comparação → `apresentar_opcoes`; escolhe um voo → `pedir_confirmacao`.

O programa atual apenas **sugere** uma ação. Mesmo que o modelo sugira `buscar_voos`, o aplicativo que executará a ação precisa conferir os campos obrigatórios. Preço, disponibilidade e regras tarifárias devem vir de uma fonte atualizada. Compra ou emissão exige uma etapa separada de autorização explícita.

## Experimente

Na raiz do projeto:

```powershell
.\.venv\Scripts\python -m opendecision treinar --dados dados/passagens.csv --modelo modelos/passagens.joblib
.\.venv\Scripts\python -m opendecision decidir "Tenho origem GRU destino REC ida 12/11 volta 19/11 e um adulto mas ainda não consultei voos" --modelo modelos/passagens.joblib
```

Os exemplos são sintéticos e pequenos. Para melhorar o modelo, substitua-os por estados de atendimento revisados e reserve casos reais mais recentes para avaliar erros. Só depois defina um limiar de revisão com base nesses dados. O número exibido como probabilidade ainda não passou por calibração.

Na verificação inicial com quatro frases novas, o modelo acertou as três primeiras etapas e errou um caso em que o viajante já havia escolhido um voo. Esse caso foi sinalizado para revisão pelo limiar padrão. O erro mostra por que a ação não deve ser executada diretamente a partir da previsão.

## Decisão com várias condições

O comando abaixo usa um estado **estruturado** para tomar uma decisão mais detalhada. O arquivo de exemplo é fictício e não contém preços ou disponibilidade atuais.

```powershell
.\.venv\Scripts\python -m opendecision avaliar-passagens dados/viagem_exemplo.json
```

O fluxo é:

1. Se faltarem origem, destino, data de ida ou número de passageiros, pedir os dados.
2. Se não houver ofertas, buscar voos.
3. Excluir ofertas que violam orçamento, limite de escalas ou bagagem obrigatória.
4. Se nenhuma oferta restar, pedir revisão dos critérios.
5. Ordenar as ofertas elegíveis por preço total, duração e escalas; apresentar a melhor e as alternativas.
6. Se o viajante já escolheu uma oferta, pedir confirmação dos detalhes e do preço atualizado.

A pontuação usa pesos configuráveis em `pedido.pesos`. Para cada oferta elegível, os componentes são `menor_preco/preco`, `menor_duracao/duracao` e `1/(1+escalas)`. A média ponderada desses componentes serve **somente para ordenar ofertas**; não é uma probabilidade nem um modelo treinado. O preço total informado deve incluir todos os passageiros, taxas e bagagem necessária para que a comparação faça sentido.

Para simular uma escolha, acrescente `"oferta_escolhida": "voo_a"` ao objeto principal do JSON. O resultado será `pedir_confirmacao`; o comando não emite passagem.

Esse avaliador complementa o classificador de próxima ação. O próximo passo de aprendizado de máquina é registrar ofertas apresentadas e qual o viajante escolheu. Com dados reais e consentidos, poderemos treinar um modelo de preferência e comparar seu ranking com esta fórmula simples em viagens reservadas para teste.
