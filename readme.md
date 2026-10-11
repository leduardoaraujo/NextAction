# OpenDecision

Primeiro passo de um modelo de decisão inspirado na **interface** do Jev: receber um estado, escolher uma ação definida e devolver probabilidades. Este protótipo treina um classificador de texto local. Ele não reproduz a arquitetura nem o treinamento RLCD do Jev.

## Estrutura

- `opendecision/dados.py`: lê e valida os exemplos em CSV.
- `opendecision/modelo.py`: treina, avalia e usa o classificador.
- `opendecision/passagens.py`: aplica restrições e ordena ofertas fictícias.
- `opendecision/api.py` e `api_schemas.py`: expõem contratos HTTP e validação JSON.
- `opendecision/cli.py`: recebe os comandos e imprime o resultado.
- `clients/rust/`: cliente de terminal em Rust que consome a API.
- `main.py`: entrada mantida para os comandos existentes; `python -m opendecision` também funciona.
- `dados/`: exemplos rotulados; `cenarios/`: descrição dos cenários.

## Passo 1: definir a decisão

O exemplo responde: **qual deve ser a próxima ação de um agente de pesquisa?** As ações possíveis são `pesquisar`, `perguntar` e `responder`. O modelo escolhe uma delas; o código que executa a ação fica separado.

## Passo 2: reunir exemplos

O arquivo `dados/exemplos.csv` contém duas colunas: `estado` descreve o que o agente sabe no momento e `acao` é a escolha correta. Substitua os exemplos fictícios por decisões reais revisadas por você. Cada modelo aprende um conjunto fixo de ações.

## Passo 3: instalar e treinar

No PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m opendecision treinar
```

O texto é transformado em características TF-IDF e usado para treinar uma regressão logística. O comando salva o modelo em `modelos/decisor.joblib`.

## Passo 4: pedir uma decisão

```powershell
.\.venv\Scripts\python -m opendecision decidir "Preciso saber a cotação de hoje e ainda não consultei nenhuma fonte"
```

A saída traz a ação escolhida, a probabilidade estimada de cada ação e `pedir_revisao`. Por padrão, `pedir_revisao` é verdadeiro quando a maior probabilidade fica abaixo de `0.70`. Ajuste com `--limiar 0.80`.

## Passo 5: avaliar antes de automatizar

Os 24 exemplos incluídos são somente uma demonstração. A acurácia sobre poucos casos não mede desempenho real. As probabilidades da regressão logística não são automaticamente calibradas. Para um agente real, separe dados históricos de avaliação, meça erros por tipo de ação e calibre as probabilidades e o limiar com dados independentes. O arquivo `joblib` só deve ser aberto se foi treinado por você.

Próximos passos: adaptar ações e estados ao seu agente, montar avaliação confiável, adicionar ações permitidas por contexto e expor a decisão por API.

## Cenário de passagens aéreas

Veja [o cenário de passagens](cenarios/passagens.md) e os exemplos em `dados/passagens.csv`. Para treinar esse modelo separado:

```powershell
.\.venv\Scripts\python -m opendecision treinar --dados dados/passagens.csv --modelo modelos/passagens.joblib
```

Para experimentar uma decisão com orçamento, bagagem e preferência por preço, duração e escalas:

```powershell
.\.venv\Scripts\python -m opendecision avaliar-passagens dados/viagem_exemplo.json
```

Essa avaliação usa regras e uma pontuação explícita; ainda não aprende a preferência entre voos. Veja os detalhes e limites no cenário.

## API REST e cliente Rust

Instale as dependências e treine os modelos de demonstração na raiz do projeto. **Terminal 1:** inicie o servidor e deixe esse terminal aberto:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m opendecision treinar
.\.venv\Scripts\python -m opendecision treinar --dados dados/passagens.csv --modelo modelos/passagens.joblib
.\.venv\Scripts\python -m uvicorn opendecision.api:app --reload
```

Com o servidor aberto, consulte `http://127.0.0.1:8000/docs` para experimentar os endpoints e ver o contrato OpenAPI. No Terminal 2, `Invoke-RestMethod http://127.0.0.1:8000/health` deve retornar `status: ok` antes de rodar o cliente.

| Método | Rota | Uso |
| --- | --- | --- |
| `GET` | `/health` | Verifica se o servidor responde. |
| `GET` | `/v1/models` | Mostra quais modelos locais foram treinados. |
| `POST` | `/v1/actions/decide` | Escolhe a próxima ação para `pesquisa` ou `passagens`. |
| `POST` | `/v1/flights/evaluate` | Aplica critérios e ordena ofertas informadas no corpo JSON. |

**Terminal 2:** com [Rust instalado](https://rust-lang.org/tools/install/), execute o cliente enquanto o servidor do Terminal 1 continua ativo:

```powershell
cargo run --manifest-path clients/rust/Cargo.toml -- flights dados/viagem_exemplo.json
cargo run --manifest-path clients/rust/Cargo.toml -- action passagens "Tenho origem GRU destino REC e ainda não consultei voos"
```

O cliente usa `http://127.0.0.1:8000` por padrão. Defina `OPENDECISION_API_URL` para outro endereço. A API devolve `503` se o classificador solicitado ainda não foi treinado. A avaliação de ofertas usa os dados enviados; não pesquisa preços nem emite passagens.

Para executar os testes Python:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m unittest discover -s tests -v
```

## Coletar decisões reais e avaliar

Para aprender com casos reais, cada resposta de `POST /v1/actions/decide` traz um `decisao_id`. A API guarda o estado recebido, a ação sugerida e as probabilidades em `registros/decisoes.jsonl`. Quando uma pessoa ou o agente souber qual ação deveria ter sido tomada, informe essa ação:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/v1/decisions/<decisao_id>/feedback `
  -ContentType "application/json" -Body '{"acao_final": "pedir_confirmacao", "comentario": "o viajante já tinha escolhido"}'
```

A `acao_final` pode ser uma ação que o modelo ainda não conhece; assim aparecem ações que faltam. Os estados podem conter dados pessoais: o arquivo fica fora do git e pode ser desativado com `$env:OPENDECISION_REGISTRO = ""` antes de iniciar o servidor.

Transforme as decisões revisadas em CSV de treino (ordem cronológica) e avalie:

```powershell
.\.venv\Scripts\python -m opendecision exportar-rotulos passagens
.\.venv\Scripts\python -m opendecision avaliar --dados dados/rotulados_passagens.csv --fracao-final 0.2
```

`avaliar` usa validação cruzada por padrão, `--teste` para um CSV separado ou `--fracao-final` para testar nos casos mais recentes, que é o mais próximo do uso futuro. O relatório mostra acurácia contra a classe majoritária, precisão e revocação por ação, matriz de confusão, erro de calibração e, para cada limiar, quantas decisões seriam automáticas, quantos erros passariam sem revisão e quantas iriam para revisão. Escolha o limiar pelo custo dos erros, não pela acurácia geral.

Na base sintética de passagens, a validação cruzada dá 0,50 de acurácia e nenhuma previsão chega a 0,50 de confiança: com o limiar padrão, tudo iria para revisão.

## Publicar no Hugging Face

O comando `exportar-hf` treina o cenário a partir do CSV e gera uma pasta pronta para o Hub: `model.skops`, `dados_treino.csv` e `README.md` (model card com metadados). O formato [skops](https://skops.readthedocs.io/) evita publicar pickle: quem baixa o modelo não executa código arbitrário ao carregá-lo.

```powershell
.\.venv\Scripts\python -m pip install -r requirements-hf.txt
.\.venv\Scripts\python -m opendecision exportar-hf passagens --repo seu-usuario/opendecision-passagens --licenca mit
```

Revise `hf/passagens/README.md` e depois envie a pasta com um token de escrita:

```powershell
.\.venv\Scripts\hf auth login
.\.venv\Scripts\hf upload seu-usuario/opendecision-passagens hf/passagens . --repo-type model
```

Troque `passagens` por `pesquisa` para o outro classificador. A pasta `hf/` fica fora do git.

## Referências

- [Apresentação oficial do Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [Documentação oficial da API](https://api.typesafe.ai/redoc)
- [Documentação do scikit-learn para calibração](https://scikit-learn.org/stable/modules/calibration.html)
