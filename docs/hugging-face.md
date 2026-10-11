# Publishing to Hugging Face

`export-hf` trains a scenario, evaluates it and writes a folder ready for the Hub:

| File | Content |
| --- | --- |
| `model.skops` | the fitted pipeline in [skops](https://skops.readthedocs.io/) format, compressed |
| `README.md` | model card: metadata, usage snippet, data source, per-action metrics and threshold table |
| `training_data.csv` | training data, only for the bundled synthetic scenarios |

## Why skops instead of joblib

`joblib` files are pickles: loading one can run arbitrary code, and the Hub flags them. skops loads only allowed types. The exporter refuses to write a file that would need untrusted types, so users can simply call `skops.io.load(path)`.

## Steps

```bash
pip install -e ".[hf]"
opendecision export-hf agent --repo your-user/opendecision-agent
```

Review `hf/agent/README.md`, then upload with a write token:

```bash
hf auth login
hf upload your-user/opendecision-agent hf/agent . --repo-type model
```

Metrics in the card come from the honest evaluation: the official test split for `agent`, cross-validation for the small scenarios. The `agent` card declares `license: cc-by-sa-4.0` because it is trained on SGD; `--license` overrides the license for the other scenarios. The `hf/` folder is git-ignored.

## Using a published model

The published file is a plain scikit-learn pipeline, so it expects the English state format (`[intent] [filled] [missing] [requested]`, `user:`). Inside OpenDecision, Portuguese tags are normalized before prediction; outside it, call `opendecision.vocabulary.normalize_state` or build the English format yourself.

```python
from huggingface_hub import hf_hub_download
import skops.io as sio

model = sio.load(hf_hub_download("your-user/opendecision-agent", "model.skops"))
state = "[intent SearchOnewayFlight] [filled destination_airport origin_airport] [missing departure_date] [requested none] || user: I need a one way flight from New York to Seattle."
print(dict(zip(model.classes_, model.predict_proba([state])[0])))
```
