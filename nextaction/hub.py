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
from .vocabulary import ACTION_DESCRIPTIONS

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
    if scenario.media:
        shutil.copyfile(scenario.media, destination / scenario.media.name)
    report["model_bytes"] = (destination / MODEL_FILE).stat().st_size
    card = model_card(
        scenario, list(model.classes_), report,
        repo or f"<user>/nextaction-{scenario.name}", license_id or scenario.license,
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
    name = repo.split("/")[-1]
    languages = " or ".join(LANGUAGE_NAMES[code] for code in scenario.languages)
    macro_f1 = round(sum(m["f1"] for m in report["per_action"].values()) / len(report["per_action"]), 3)
    split = "test" if scenario.test else "cross-validation"
    dataset_type = scenario.dataset_id or f"nextaction/{scenario.name}"
    metadata = [f"language: [{', '.join(scenario.languages)}]"]
    if license_id:
        metadata.append(f"license: {license_id}")
    metadata += ["library_name: sklearn", "pipeline_tag: text-classification"]
    if scenario.dataset_id:
        metadata += ["datasets:", f"- {scenario.dataset_id}"]
    metadata += [
        "metrics: [accuracy, f1]",
        "tags: [sklearn, skops, text-classification, agents, decision-making, dialogue-policy]",
        "model-index:",
        f"- name: {name}",
        "  results:",
        "  - task:",
        "      type: text-classification",
        "      name: Next-action prediction",
        "    dataset:",
        f"      name: {scenario.dataset_name or scenario.name}",
        f"      type: {dataset_type}",
        f"      split: {split}",
        "    metrics:",
        f"    - {{type: accuracy, value: {report['accuracy']}, name: Accuracy}}",
        f"    - {{type: f1, value: {macro_f1}, name: Macro F1}}",
    ]
    example = json.dumps(scenario.example, ensure_ascii=False)
    descriptions = {**ACTION_DESCRIPTIONS, **scenario.action_descriptions}
    action_rows = "\n".join(f"| `{a}` | {descriptions.get(a, '')} |" for a in actions)
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
        f"`{DATA_FILE}` is included in this repository." if scenario.ship_data
        else "The training data is not redistributed here; download it from the original source."
    )
    warning = f"\n\n> {report['warning']}" if "warning" in report else ""
    params = scenario.params
    if scenario.input_format == "tagged":
        input_section = """## Input format

The state is the dialogue state followed by the last turns, separated by ` || `:

```text
[intent <ActiveIntent>] [filled <slots>] [missing <slots>] [requested <slots>] || user: <text> || assistant: <text> || user: <text>
```

| Tag | Content |
| --- | --- |
| `intent` | the user's active intent, e.g. `SearchOnewayFlight` |
| `filled` | slots the user already provided (names only, no values) |
| `missing` | required slots of the intent that are still empty |
| `requested` | slots the user asked about |

Use `none` for empty tags. Knowing which required slots are `missing` is the strongest signal:
without the tags, accuracy drops from 0.90 to 0.77. Portuguese tags (`[intencao] [preenchidos]
[faltando] [pedidos]`, `usuario:`) are accepted by `nextaction.vocabulary.normalize_state`,
which the NextAction package applies before predicting."""
    else:
        input_section = f"""## Input format

A free-text description of the current situation in {languages}, for example {example}."""
        if scenario.demo:
            input_section += (
                "\n\nStates are built by `features.describe()` in the "
                f"[{scenario.demo}](https://github.com/leduardoaraujo/NextAction/tree/main/{scenario.demo}) folder; "
                "use it to turn a position into the text this model expects."
            )
    data_citation = (
        "\n\nTraining data:\n\n```bibtex\n" + scenario.citation + "\n```" if scenario.citation else ""
    )
    risk_data = scenario.risk or (
        "The training conversations were written by crowd workers from simulated outlines in 20 service "
        "domains, so real users will be messier and other domains less accurate."
        if scenario.input_format == "tagged" else
        "The training data is small and synthetic; treat this model as a demonstration of the pipeline, "
        "not as a production model."
    )
    size_bytes = report.get("model_bytes", 0)
    size = f"{size_bytes / 1e6:.1f} MB" if size_bytes >= 1e6 else f"{size_bytes / 1e3:.0f} kB"
    direct_use = scenario.direct_use or (
        "Decide the next step of a conversational agent (ask, search, present options, confirm, execute, "
        "end the conversation), route uncertain decisions to human review, and log decisions with feedback "
        "to collect training data."
    )
    out_of_scope = scenario.out_of_scope or (
        "Executing irreversible actions (payments, bookings) from a prediction alone, domains far from the "
        "training data without re-evaluating, and any use as a source of facts: the model chooses an action, "
        "it does not generate or verify content."
    )
    demo_section = ""
    if scenario.media or scenario.demo_summary:
        demo_link = (f" Scripts, benchmark and analysis: "
                     f"[{scenario.demo}](https://github.com/leduardoaraujo/NextAction/tree/main/{scenario.demo})."
                     if scenario.demo else "")
        image = f"![{scenario.name} demo]({scenario.media.name})\n\n" if scenario.media else ""
        demo_section = f"\n## Demo\n\n{image}{scenario.demo_summary or ''}{demo_link}\n"
    return f"""---
{chr(10).join(metadata)}
---

# NextAction: {scenario.name}

A small, auditable classifier that predicts **{scenario.decision}**. Given the current state, it returns
one action from a fixed vocabulary with a probability for each, so an agent can act alone when it is
confident and hand the decision to a person when it is not. It ships with a decision policy
(`{CONFIG_FILE}`) that blocks impossible actions and sets a minimum confidence per action.
{demo_section}
## Model details

| | |
| --- | --- |
| Developed by | [Luiz Araujo](https://github.com/leduardoaraujo) |
| Model type | TF-IDF features + multinomial logistic regression (scikit-learn) |
| Task | next-action prediction for task-oriented agents (text classification) |
| Language | {languages} |
| License | {license_id or "not specified"} |
| Size | {size} |
| Code | [github.com/leduardoaraujo/NextAction](https://github.com/leduardoaraujo/NextAction) |

## Uses

**Direct use.** {direct_use}

**Out of scope.** {out_of_scope}

## How to get started

```python
from huggingface_hub import hf_hub_download
import skops.io as sio

model = sio.load(hf_hub_download("{repo}", "{MODEL_FILE}"))  # no types outside skops' trusted list

state = {example}
probabilities = dict(zip(model.classes_, model.predict_proba([state])[0]))
print(max(probabilities, key=probabilities.get), probabilities)
```

Dependencies: `scikit-learn`, `skops` and `huggingface_hub`. Use a scikit-learn version compatible with
the training one. The full toolkit (CLI, REST API, decision log, evaluation, policy) is in the
[NextAction repository](https://github.com/leduardoaraujo/NextAction).

{input_section}

## Actions

| Action | Meaning |
| --- | --- |
{action_rows}

## Training details

**Data.** {report["train_examples"]} {scenario.data_source}. {shipped}

**Preprocessing.** States are normalized to the English tag format, then vectorized with TF-IDF over
word unigrams and bigrams with accents stripped.

| Hyperparameter | Value |
| --- | --- |
| `ngram_range` | {params.ngram_range} |
| `min_df` | {params.min_df} |
| `C` | {params.C} |
| `class_weight` | {params.class_weight} |
| `max_iter` | {params.max_iter} |
| calibration | {params.calibration} |

The solver is L-BFGS, run in warm-started chunks so training progress can be reported. Hyperparameters
live in `{CONFIG_FILE}` and can be searched with `nextaction tune`.

## Evaluation

**Method.** {report["method"]}, with {report["evaluated_examples"]} examples evaluated.

**Results.** Accuracy **{report["accuracy"]}** and macro F1 **{macro_f1}**; always choosing the most
frequent action would give {report["majority_baseline_accuracy"]}. Expected calibration error:
{report["expected_calibration_error"]}.{warning}

| Action | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
{per_action}

**Review thresholds.** Decisions with confidence below the threshold go to a person:

| Threshold | Decisions automated | Accuracy on automated |
| --- | --- | --- |
{thresholds}

{policy_card(scenario, actions, report.get("policy"), repo)}

## Bias, risks and limitations

- The model matches word patterns. It does not understand business rules or verify facts, so the calling
  application must check preconditions before acting.
- Logistic regression probabilities are not calibrated; pick thresholds from the tables above, and
  re-evaluate on your own traffic before automating.
- {risk_data}
- The action vocabulary is fixed; new actions require new labeled data.

## Citation

```bibtex
@software{{araujo2026nextaction,
  author = {{Araujo, Luiz}},
  title  = {{NextAction: a small, auditable next-action model for AI agents}},
  year   = {{2026}},
  url    = {{https://github.com/leduardoaraujo/NextAction}}
}}
```{data_citation}

## Model card contact

Open an issue in the [NextAction repository](https://github.com/leduardoaraujo/NextAction/issues).
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
(`review_cost = {policy.review_cost:g}`). `nextaction calibrate-policy` re-fits the thresholds from the costs.

| Action | Risk | Min. confidence | Unreviewed error cost | Confirm with user |
| --- | --- | --- | --- | --- |
{rows}

Rules:

{rules}{result}

Apply it with the NextAction package (`pip install git+https://github.com/leduardoaraujo/NextAction`):

```python
import tomllib
from nextaction.policy import apply, from_dict

with open(hf_hub_download("{repo}", "{CONFIG_FILE}"), "rb") as file:
    policy = from_dict(tomllib.load(file)["policy"])
decision = apply(probabilities, state, policy)  # probabilities and state from the snippet above
print(decision["action"], decision["needs_review"], decision["reasons"])
```"""
