# OpenDecision

Primeiro passo de um modelo de decisão inspirado na **interface** do Jev: receber um estado, escolher uma ação definida e devolver probabilidades. Este protótipo treina um classificador de texto local. Ele não reproduz a arquitetura nem o treinamento RLCD do Jev.

## Passo 1: definir a decisão

O exemplo responde: **qual deve ser a próxima ação de um agente de pesquisa?** As ações possíveis são `pesquisar`, `perguntar` e `responder`. O modelo escolhe uma delas; o código que executa a ação fica separado.

## Passo 2: reunir exemplos

O arquivo `dados/exemplos.csv` contém duas colunas: `estado` descreve o que o agente sabe no momento e `acao` é a escolha correta. Substitua os exemplos fictícios por decisões reais revisadas por você. Cada modelo aprende um conjunto fixo de ações.

## Passo 3: instalar e treinar

No PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python main.py treinar
```

O texto é transformado em características TF-IDF e usado para treinar uma regressão logística. O comando salva o modelo em `modelos/decisor.joblib`.

## Passo 4: pedir uma decisão

```powershell
.\.venv\Scripts\python main.py decidir "Preciso saber a cotação de hoje e ainda não consultei nenhuma fonte"
```

A saída traz a ação escolhida, a probabilidade estimada de cada ação e `pedir_revisao`. Por padrão, `pedir_revisao` é verdadeiro quando a maior probabilidade fica abaixo de `0.70`. Ajuste com `--limiar 0.80`.

## Passo 5: avaliar antes de automatizar

Os 24 exemplos incluídos são somente uma demonstração. A acurácia sobre poucos casos não mede desempenho real. As probabilidades da regressão logística não são automaticamente calibradas. Para um agente real, separe dados históricos de avaliação, meça erros por tipo de ação e calibre as probabilidades e o limiar com dados independentes. O arquivo `joblib` só deve ser aberto se foi treinado por você.

Próximos passos: adaptar ações e estados ao seu agente, montar avaliação confiável, adicionar ações permitidas por contexto e expor a decisão por API.

## Referências

- [Apresentação oficial do Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [Documentação oficial da API](https://api.typesafe.ai/redoc)
- [Documentação do scikit-learn para calibração](https://scikit-learn.org/stable/modules/calibration.html)
