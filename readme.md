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

## Referências

- [Apresentação oficial do Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [Documentação oficial da API](https://api.typesafe.ai/redoc)
- [Documentação do scikit-learn para calibração](https://scikit-learn.org/stable/modules/calibration.html)
