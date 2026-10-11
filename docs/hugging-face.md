# Publishing to Hugging Face

`exportar-hf` trains a scenario, evaluates it and writes a folder ready for the Hub:

| File | Content |
| --- | --- |
| `model.skops` | the fitted pipeline in [skops](https://skops.readthedocs.io/) format, compressed |
| `README.md` | model card: metadata, usage snippet, data source, per-action metrics and threshold table |
| `dados_treino.csv` | training data, only for the bundled synthetic scenarios |

## Why skops instead of joblib

`joblib` files are pickles: loading one can run arbitrary code, and the Hub flags them. skops loads only allowed types. The exporter refuses to write a file that would need untrusted types, so users can simply call `skops.io.load(path)`.

## Steps

```bash
pip install -r requirements-hf.txt
python -m opendecision exportar-hf agente --repo your-user/opendecision-agente
```

Review `hf/agente/README.md`, then upload with a write token:

```bash
hf auth login
hf upload your-user/opendecision-agente hf/agente . --repo-type model
```

Metrics in the card come from the honest evaluation: the official test split for `agente`, cross-validation for the small scenarios. The `agente` card declares `license: cc-by-sa-4.0` because it is trained on SGD; `--licenca` overrides the license for the other scenarios. The `hf/` folder is git-ignored.

## Using a published model

```python
from huggingface_hub import hf_hub_download
import skops.io as sio

modelo = sio.load(hf_hub_download("your-user/opendecision-agente", "model.skops"))
estado = "[intencao SearchOnewayFlight] [preenchidos destination_airport origin_airport] [faltando departure_date] [pedidos nenhum] || usuario: I need a one way flight from New York to Seattle."
print(dict(zip(modelo.classes_, modelo.predict_proba([estado])[0])))
```
