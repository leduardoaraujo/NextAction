"""Package a trained classifier for the Hugging Face Hub."""

import json
import shutil
import zipfile
from pathlib import Path

from .data import load_examples
from .evaluation import evaluate, metrics
from .i18n import LocalizedError
from .model import fit
from .scenarios import Scenario, get_scenario

MODEL_FILE = "model.skops"
DATA_FILE = "training_data.csv"
LANGUAGE_NAMES = {"en": "English", "pt": "Portuguese"}


def export(name: str, destination: Path, repo: str | None = None, license_id: str | None = None) -> dict:
    """Train the scenario, evaluate it and write the skops model and model card to ``destination``."""
    # Imported here so skops is only required for exporting.
    import skops.io as sio

    scenario = get_scenario(name)
    if not scenario.data.is_file():
        raise LocalizedError("data_file_missing", path=scenario.data)
    model, report = train_and_evaluate(scenario)

    destination.mkdir(parents=True, exist_ok=True)
    sio.dump(model, destination / MODEL_FILE, compression=zipfile.ZIP_DEFLATED, compresslevel=9)
    if sio.get_untrusted_types(file=destination / MODEL_FILE):
        raise LocalizedError("skops_untrusted")
    if scenario.ship_data:
        shutil.copyfile(scenario.data, destination / DATA_FILE)
    card = model_card(
        scenario, list(model.classes_), report,
        repo or f"<user>/opendecision-{scenario.name}", license_id or scenario.license,
    )
    (destination / "README.md").write_text(card, encoding="utf-8")
    return {
        "folder": str(destination), "files": sorted(p.name for p in destination.iterdir()),
        "method": report["method"], "accuracy": report["accuracy"],
    }


def train_and_evaluate(scenario: Scenario):
    model, states, actions = fit(scenario.data, scenario.min_df)
    if scenario.test is None:
        report = evaluate(scenario.data, min_df=scenario.min_df, lang="en")
        return model, {**report, "train_examples": len(states)}
    # With a separate test set, the published model is the evaluated model itself.
    test_x, test_y = load_examples(scenario.test)
    majority = max(set(actions), key=actions.count)
    report = metrics(test_y, model.predict_proba(test_x), list(model.classes_), majority, "en")
    method = f"separate test split ({scenario.test.name})"
    return model, {"method": method, "train_examples": len(states), **report}


def model_card(scenario: Scenario, actions: list[str], report: dict, repo: str, license_id: str | None) -> str:
    metadata = [f"language: [{', '.join(scenario.languages)}]", "library_name: sklearn",
                "pipeline_tag: text-classification"]
    if license_id:
        metadata.append(f"license: {license_id}")
    metadata += ["tags:", "- sklearn", "- skops", "- text-classification", "- agents"]
    action_list = "\n".join(f"- `{action}`" for action in actions)
    example = json.dumps(scenario.example, ensure_ascii=False)
    languages = " or ".join(LANGUAGE_NAMES[code] for code in scenario.languages)
    per_action = "\n".join(
        f"| `{action}` | {m['precision']} | {m['recall']} | {m['f1']} | {m['support']} |"
        for action, m in report["per_action"].items()
    )
    thresholds = "\n".join(
        f"| {row['threshold']} | {row['automated']:.1%} | "
        f"{'-' if row['automated_accuracy'] is None else format(row['automated_accuracy'], '.1%')} |"
        for row in report["thresholds"]
    )
    shipped = (
        f"`{DATA_FILE}` is included with the model." if scenario.ship_data
        else "The training data is not redistributed here; use the original source."
    )
    warning = f"\n\n> {report['warning']}" if "warning" in report else ""
    return f"""---
{chr(10).join(metadata)}
---

# OpenDecision: {scenario.name}

Text classifier that predicts **{scenario.decision}**.

Input: a description of the current state in {languages}. Output: one of the actions below, with estimated probabilities.

{action_list}

The model only **suggests** an action. The application that executes it must check the
required conditions and ask for human review when confidence is low.

## Usage

```python
from huggingface_hub import hf_hub_download
import skops.io as sio

path = hf_hub_download("{repo}", "{MODEL_FILE}")
model = sio.load(path)  # the file needs no types outside skops' trusted list

state = {example}
probabilities = dict(zip(model.classes_, model.predict_proba([state])[0]))
print(max(probabilities, key=probabilities.get), probabilities)
```

Dependencies: `scikit-learn`, `skops` and `huggingface_hub`. Use a scikit-learn version
compatible with the training one to avoid version warnings.

## Architecture

scikit-learn pipeline: `TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", min_df={scenario.min_df})` followed by
`LogisticRegression(max_iter=1000)`. Training, evaluation and API code: [OpenDecision](https://github.com/leduardoaraujo/OpenDecision).

## Training data

{report["train_examples"]} {scenario.data_source}. {shipped}

## Evaluation

Method: {report["method"]}, with {report["evaluated_examples"]} examples evaluated.
Accuracy **{report["accuracy"]}**; always choosing the most frequent action would give {report["majority_baseline_accuracy"]}.
Expected calibration error: {report["expected_calibration_error"]}.{warning}

| Action | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
{per_action}

Review threshold: decisions with confidence below it go to a person.

| Threshold | Decisions automated | Accuracy on automated |
| --- | --- | --- |
{thresholds}

## Limitations

- The model matches word patterns; it does not understand business rules or verify facts.
- Logistic regression probabilities are not calibrated.
- The action set is fixed.
- Do not execute irreversible actions (for example, purchases) from a prediction without explicit confirmation.
"""
