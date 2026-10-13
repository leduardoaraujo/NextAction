# Publicação no Hugging Face

Modelos publicados: [nextaction-agent](https://huggingface.co/ludolua/nextaction-agent) (genérico, SGD), [nextaction-flights](https://huggingface.co/ludolua/nextaction-flights) (sintético), [nextaction-pacman](https://huggingface.co/ludolua/nextaction-pacman) e [nextaction-connect4](https://huggingface.co/ludolua/nextaction-connect4) (demos de jogos, com seus GIFs).

`export-hf` treina um cenário, avalia o modelo e grava uma pasta pronta para o Hub:

| Arquivo | Conteúdo |
| --- | --- |
| `model.skops` | o pipeline treinado no formato [skops](https://skops.readthedocs.io/), compactado |
| `README.md` | model card: metadados, exemplo de uso, origem dos dados, métricas por ação e tabela de limiares |
| `config.toml` | a configuração do cenário, incluindo a política de decisão descrita no card |
| `training_data.csv` | dados de treino, apenas para os cenários sintéticos incluídos no repositório |

## Por que skops em vez de joblib

Arquivos `joblib` são pickles: carregar um deles pode executar código arbitrário, e o Hub os sinaliza. O skops carrega apenas tipos permitidos. O exportador se recusa a gravar um arquivo que exija tipos não confiáveis, então os usuários podem simplesmente chamar `skops.io.load(path)`.

## Passos

```bash
pip install -e ".[hf]"
nextaction export-hf agent --repo your-user/nextaction-agent
```

Revise `hf/agent/README.md` e depois envie com um token de escrita:

```bash
hf auth login
hf upload your-user/nextaction-agent hf/agent . --repo-type model
```

As métricas do card vêm da avaliação honesta: o split de teste oficial para `agent` e validação cruzada para os cenários pequenos. O card do `agent` declara `license: cc-by-sa-4.0` porque ele é treinado com o SGD; `--license` substitui a licença nos outros cenários. A pasta `hf/` é ignorada pelo git.

## Usando um modelo publicado

O arquivo publicado é um pipeline comum do scikit-learn, então ele espera o formato de estado em inglês (`[intent] [filled] [missing] [requested]`, `user:`). Dentro do NextAction, as tags em português são normalizadas antes da previsão; fora dele, chame `nextaction.vocabulary.normalize_state` ou monte o formato em inglês você mesmo.

```python
from huggingface_hub import hf_hub_download
import skops.io as sio

model = sio.load(hf_hub_download("your-user/nextaction-agent", "model.skops"))
state = "[intent SearchOnewayFlight] [filled destination_airport origin_airport] [missing departure_date] [requested none] || user: I need a one way flight from New York to Seattle."
print(dict(zip(model.classes_, model.predict_proba([state])[0])))
```
