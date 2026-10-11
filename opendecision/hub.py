"""Package a trained classifier for the Hugging Face Hub."""

import json
import shutil
import zipfile
from pathlib import Path

from .data import load_examples
from .evaluation import evaluate, metrics
from .i18n import LocalizedError
from .model import fit
from .policy import evaluate_policy
from .scenarios import Scenario, get_scenario

MODEL_FILE = "model.skops"
DATA_FILE = "training_data.csv"
CONFIG_FILE = "config.toml"
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
    # The scenario config carries the decision policy the card documents.
    shutil.copyfile(scenario.config_path, destination / CONFIG_FILE)
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
    model, states, actions = fit(scenario.data, scenario.params)
    if scenario.test is None:
        report = evaluate(scenario.data, params=scenario.params, lang="en", policy=scenario.policy)
        return model, {**report, "train_examples": len(states)}
    # With a separate test set, the published model is the evaluated model itself.
    test_x, test_y = load_examples(scenario.test)
    majority = max(set(actions), key=actions.count)
    probabilities = model.predict_proba(test_x)
    report = metrics(test_y, probabilities, list(model.classes_), majority, "en")
    report["policy"] = evaluate_policy(test_x, test_y, probabilities, list(model.classes_), scenario.policy)
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
    policy_section = policy_card(scenario, actions, report.get("policy"), repo)
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

scikit-learn pipeline: `TfidfVectorizer(ngram_range={scenario.params.ngram_range}, strip_accents="unicode", min_df={scenario.params.min_df})` followed by
`LogisticRegression(C={scenario.params.C}, class_weight={scenario.params.class_weight!r}, max_iter={scenario.params.max_iter})`
(calibration: {scenario.params.calibration}). Training, evaluation and API code: [OpenDecision](https://github.com/leduardoaraujo/OpenDecision).

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

{policy_section}

## Limitations

- The model matches word patterns; it does not understand business rules or verify facts.
- Logistic regression probabilities are not calibrated.
- The action set is fixed.
- Do not execute irreversible actions (for example, purchases) from a prediction without explicit confirmation.
"""


def policy_card(scenario: Scenario, actions: list[str], simulation: dict | None, repo: str) -> str:
    """Card section describing the decision policy shipped in config.toml."""
    policy = scenario.policy
    rows = "\n".join(
        f"| `{action}` | {policy.for_action(action).risk} | {policy.threshold(action)} | "
        f"{policy.for_action(action).unreviewed_error_cost:g} | "
        f"{'yes' if policy.for_action(action).requires_confirmation else 'no'} |"
        for action in actions
    )
    rules = "\n".join(
        f"- `{rule.id}`: deny {', '.join(f'`{a}`' for a in rule.deny)} when "
        + " and ".join(filter(None, [
            "any required slot is missing" if rule.when_missing_any else "",
            f"any of {', '.join(rule.when_missing)} is missing" if rule.when_missing else "",
            f"all of {', '.join(rule.when_filled_all)} are filled" if rule.when_filled_all else "",
            f"the intent is {', '.join(rule.when_intent)}" if rule.when_intent else "",
        ]))
        for rule in policy.rules
    ) or "- none"
    result = ""
    if simulation and simulation["automated_accuracy"] is not None:
        result = (
            f"\n\nSimulated on the evaluation data above, the policy automates {simulation['automated']:.1%} of "
            f"decisions at {simulation['automated_accuracy']:.1%} accuracy, with an expected cost of "
            f"{simulation['expected_cost_per_decision']} reviews per decision."
        )
    return f"""## Decision policy

`{CONFIG_FILE}` holds the decision policy used with this model: rules that block impossible actions,
a minimum confidence per action, risk levels and error costs, in units of one human review
(`review_cost = {policy.review_cost:g}`). `opendecision calibrate-policy` re-fits the thresholds from the costs.

| Action | Risk | Min. confidence | Unreviewed error cost | Confirm with user |
| --- | --- | --- | --- | --- |
{rows}

Rules:

{rules}{result}

Apply it with the OpenDecision package (`pip install git+https://github.com/leduardoaraujo/OpenDecision`):

```python
import tomllib
from opendecision.policy import apply, from_dict

with open(hf_hub_download("{repo}", "{CONFIG_FILE}"), "rb") as file:
    policy = from_dict(tomllib.load(file)["policy"])
decision = apply(probabilities, state, policy)  # probabilities and state from the snippet above
print(decision["action"], decision["needs_review"], decision["reasons"])
```"""
