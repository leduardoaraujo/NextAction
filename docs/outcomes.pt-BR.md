# Modelo bilíngue e aprendizado com resultados

## Um modelo inglês/português

`agent_bilingual` usa o mesmo vocabulário de ações e a mesma interface de política que `agent`.
Seu conjunto de treino combina a divisão oficial de treino do SGD em inglês com uma
amostra traduzida para o português dessa **mesma divisão de treino**. O conjunto
menor em português é repetido quatro vezes. Os diálogos de teste traduzidos vêm
apenas da divisão oficial de teste; `scripts/build_bilingual_sgd.py` também seleciona
os correspondentes em inglês para uma comparação pareada. Os dados externos e os
modelos treinados são ignorados pelo git.

```bash
# First download the official SGD train/test splits into data/external/sgd/.
pip install -e ".[translate]"
python scripts/translate_sgd_sample.py data/external/sgd/train data/external/sgd_pt/train --count 800
python scripts/translate_sgd_sample.py data/external/sgd/test data/external/sgd_pt/test --count 300
nextaction import-sgd data/external/sgd_pt/train --output data/external/sgd_pt_train.csv
nextaction import-sgd data/external/sgd_pt/test --output data/external/sgd_pt_test.csv
python scripts/build_bilingual_sgd.py
nextaction train agent_bilingual
python scripts/evaluate_bilingual.py
nextaction decide agente_bilingue "[intencao SearchOnewayFlight] [faltando destination_airport] || usuario: Quero viajar"
```

Em 2.883 turnos pareados por idioma, o modelo em inglês obteve acurácia de 0,891
em inglês e 0,609 em português. O modelo bilíngue obteve **0,895 em inglês e 0,837
em português**. Seu erro de calibração esperado foi de 0,068 em inglês e 0,073 em
português. Recriar a amostra traduzida ou trocar o modelo de tradução pode alterar
esses números. A amostra em português é composta de diálogos traduzidos, então
esses números não comprovam a acurácia em conversas reais em português.
O cenário bilíngue exige revisão por padrão. Meça e ajuste seus limiares
com decisões locais representativas antes de automatizar.

## Aprendizado com resultados observados

O endpoint de feedback continua aceitando `final_action`, a ação que um revisor diz
ter sido a correta. Para o aprendizado com resultados, ele aceita adicionalmente os três campos:

- `action_taken`: a ação que a aplicação de fato executou;
- `reward`: o resultado observado, em uma escala de 0 a 1;
- `propensity`: a probabilidade de que a **política de registro tenha escolhido essa ação**
  naquele estado, entre 0 e 1.

```json
{
  "final_action": "ask",
  "action_taken": "ask",
  "reward": 1.0,
  "propensity": 0.5
}
```

A lógica de escolha de ações da aplicação deve fornecer `propensity`. A confiança
do classificador **não** é uma propensão. Uma política determinística geralmente não
tem sobreposição de ações, então não permite comparar ações alternativas. Adicione
aleatorização controlada entre as ações permitidas pela política, com confirmação do
usuário para ações irreversíveis, apenas em um contexto em que a experimentação seja
adequada. Registre resultados somente depois que eles forem de fato medidos. Uma
correção de revisor não é uma recompensa.

```bash
nextaction fit-bandit agent --log logs/decisions.jsonl
nextaction preview-bandit agent "[intent SearchOnewayFlight] || user: I need a flight"
```

`fit-bandit` precisa de pelo menos 200 observações completas, incluindo 20 para cada
uma de pelo menos duas ações. Ele ajusta estimativas de recompensa por ação nos 80%
mais antigos das observações e informa uma estimativa de propensão inversa
autonormalizada nos 20% mais recentes, junto com a recompensa média registrada, as
coincidências e o tamanho efetivo da amostra. O resultado é um **arquivo candidato para revisão**.
A API de decisão nunca o carrega e o comando de prévia nunca executa uma ação.
Tamanhos efetivos de amostra pequenos, pouca sobreposição de ações e propensões não
aleatórias ou incorretas podem tornar a estimativa offline enganosa. Nenhuma
observação de resultado real acompanha este projeto.
