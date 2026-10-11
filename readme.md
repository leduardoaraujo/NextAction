# OpenDecision

**A small, auditable next-action model for AI agents.** Given the current state of a conversation, OpenDecision predicts what the agent should do next (ask, search, present, confirm, execute, close) with a probability for every action and a threshold that sends uncertain cases to a human.

[![CI](https://github.com/leduardoaraujo/OpenDecision/actions/workflows/ci.yml/badge.svg)](https://github.com/leduardoaraujo/OpenDecision/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Rust client](https://img.shields.io/badge/client-Rust-000000?logo=rust)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

```text
state  →  [intencao SearchOnewayFlight] [preenchidos origin_airport] [faltando destination_airport departure_date]
          || usuario: I need a flight from Boston.
action →  perguntar (0.99)   ✓ automatic, above the 0.9 review threshold
```

## Why

LLM agents rarely fail at writing sentences; they fail at **control flow**: searching before they know the destination, booking without confirmation, answering when they should ask. OpenDecision isolates that decision into a model that is:

- **Explicit**: a fixed set of actions, so every choice can be logged, counted and tested.
- **Calibrated for hand-off**: each prediction carries probabilities, and a threshold decides what is automated and what goes to review.
- **Improvable from use**: every API decision gets an id; corrections sent back become training data.
- **Cheap**: TF-IDF + logistic regression. Trains on a laptop, predicts in milliseconds, ships as a few-MB file.

The interface (state in, one defined action out, with probabilities) is inspired by [Jev's System One models](https://typesafe.ai/blog/introducing-system-one-models-and-jev). This project does not reproduce Jev's architecture or its RLCD training.

## Results

The generic `agente` model is trained on the [Schema-Guided Dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) dataset (165k assistant turns, 20 domains) and evaluated on its official test split, which includes **services never seen during training** (42k turns).

| Input | Accuracy | Automated at 0.9 threshold | Accuracy when automated |
| --- | --- | --- | --- |
| Conversation + dialogue state | **0.90** | 50.0% | **99.6%** |
| Conversation only¹ | 0.77 | 28.9% | 96.8% |
| Always the most frequent action | 0.21 | | |

¹ Ablation run with the full vocabulary (`min_df=1`); the published model uses `min_df=5` and is 4.7 MB.

Adding the dialogue state (which required fields are still missing) is what moves the needle: F1 for `perguntar` (ask) goes from 0.71 to 0.97. Full per-action metrics, calibration and threshold curves are in [docs/sgd-model.md](docs/sgd-model.md).

## How it works

```mermaid
flowchart LR
    SGD[(Schema-Guided<br/>Dialogue)] -->|importar-sgd| CSV[labeled CSV<br/>estado, acao]
    CSV -->|treinar · avaliar| M[TF-IDF +<br/>logistic regression]
    Agent[Agent or app] -->|state| API[FastAPI<br/>/v1/actions/decide]
    API <--> M
    API -->|action + probabilities| Agent
    API -->|below threshold| Human[Human review]
    API --> Log[(decision log)]
    Human -->|/v1/decisions/id/feedback| Log
    Log -->|exportar-rotulos| CSV
    M -->|exportar-hf| Hub[Hugging Face Hub]
```

1. **Data**: labeled `estado,acao` pairs, either written by hand, imported from public dialogue data or exported from real decisions with feedback.
2. **Evaluation first**: `avaliar` reports per-action errors, the confusion matrix, calibration error and how many decisions each threshold would automate, compared against a majority-class baseline.
3. **Serving**: a FastAPI service returns the action, the probabilities and `pedir_revisao` (needs review). A typed Rust CLI consumes the same contract.
4. **Learning loop**: decisions and corrections are appended to a JSONL log and turned back into training data in chronological order.
5. **Publishing**: models are exported in the safe [skops](https://skops.readthedocs.io/) format (no pickle) with a generated model card.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m opendecision treinar --dados dados/passagens.csv --modelo modelos/passagens.joblib
python -m opendecision decidir "Tenho origem GRU destino REC ida 12/11 e ainda não consultei voos" \
  --modelo modelos/passagens.joblib
```

```json
{
  "tipo": "escolha",
  "proxima_acao": "buscar_voos",
  "probabilidades_estimadas": { "apresentar_opcoes": 0.1922, "buscar_voos": 0.4163, "pedir_confirmacao": 0.183, "perguntar_dados": 0.2085 },
  "pedir_revisao": true
}
```

The bundled Portuguese scenarios are tiny and synthetic, so the model is unsure and asks for review. To train the generic model on real dialogues, follow [docs/sgd-model.md](docs/sgd-model.md). To serve it over HTTP:

```bash
uvicorn opendecision.api:app --reload    # then open http://127.0.0.1:8000/docs
```

## Features

| | |
| --- | --- |
| **Scenarios** | `agente` (generic, English, real dialogues), `passagens` (flight booking, Portuguese), `pesquisa` (research agent, Portuguese) |
| **CLI** | `treinar`, `decidir`, `avaliar`, `importar-sgd`, `exportar-rotulos`, `exportar-hf`, `avaliar-passagens` |
| **REST API** | decide, feedback, flight-offer ranking, model status; OpenAPI docs at `/docs` |
| **Evaluation** | cross-validation, separate test set or most-recent slice; per-action F1, confusion matrix, ECE, threshold table |
| **Rule-based ranker** | filters and ranks flight offers by budget, baggage, stops, price and duration, with explanations |
| **Clients** | typed Rust CLI (`clients/rust`) |
| **Distribution** | skops export plus generated Hugging Face model card |

## Repository layout

```text
opendecision/
  modelo.py        training and inference (TF-IDF + logistic regression)
  avaliacao.py     evaluation report: per-action metrics, calibration, thresholds
  sgd.py           Schema-Guided Dialogue → estado,acao converter
  registro.py      append-only decision log and feedback export
  api.py           FastAPI service · api_schemas.py: request/response contracts
  hub.py           skops export and model card generation
  passagens.py     rule-based flight offer filter and ranker
  cli.py           command line entry point
clients/rust/      Rust client for the REST API
dados/             bundled synthetic examples (external data goes to dados/externos/, git-ignored)
docs/              detailed documentation
tests/             unit tests (python -m unittest discover -s tests)
```

## Documentation

- [Generic agent model on Schema-Guided Dialogue](docs/sgd-model.md): data mapping, state format, results, ablation
- [Evaluation](docs/evaluation.md): what each metric means and how to choose a threshold
- [REST API, feedback loop and Rust client](docs/api.md)
- [Publishing to Hugging Face](docs/hugging-face.md)
- [Flight booking scenario](cenarios/passagens.md) (in Portuguese)

## Limitations

- The model matches word patterns. It does not verify facts or enforce business rules; the calling app must check preconditions before acting.
- Probabilities are not calibrated (expected calibration error ≈ 0.08 on SGD). Pick thresholds from the evaluation table, not from intuition.
- SGD conversations were written by crowd workers from simulated scripts, so real users will be messier. Measure on your own logged decisions before automating.
- The `agente` model is English-only; the Portuguese scenarios use small synthetic datasets for demonstration.
- Never execute irreversible actions (payments, bookings) from a prediction alone.

## Roadmap

- [ ] Multilingual sentence embeddings, so the SGD-trained model also serves Portuguese states
- [ ] Probability calibration (`CalibratedClassifierCV`) with thresholds tuned to error costs
- [ ] Context-dependent action masks (actions allowed per state)
- [ ] Learning from outcomes (contextual bandits) once enough feedback is logged

## License and data

Code: [MIT](LICENSE). The Schema-Guided Dialogue dataset is © Google, released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), and is not redistributed here; models trained on it are published under the same license. If you use the dataset, cite Rastogi et al., *Towards Scalable Multi-domain Conversational Agents: The Schema-Guided Dialogue Dataset* (AAAI 2020).
