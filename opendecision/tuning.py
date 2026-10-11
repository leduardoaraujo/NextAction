"""Hyperparameter search and policy calibration, written back to the scenario's TOML config."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split

from .data import load_examples
from .evaluation import cross_validated_probabilities
from .i18n import LocalizedError
from .model import fit_pipeline, new_model
from .policy import calibrate, evaluate_policy, from_dict
from .progress import stage
from .scenarios import Scenario, model_params

TUNABLE = ("C", "min_df", "ngram_range", "class_weight")
METRICS = {
    "f1_macro": lambda y, p: f1_score(y, p, average="macro"),
    "accuracy": accuracy_score,
}
# Above this size, calibration uses one 80/20 split instead of cross-validation.
HOLDOUT_ABOVE = 5000


def _sample(states, actions, size: int | None):
    if not size or len(states) <= size:
        return states, actions
    states, _, actions, _ = train_test_split(states, actions, train_size=size, stratify=actions, random_state=42)
    return states, actions


def tune(scenario: Scenario, write: bool = False) -> dict:
    settings = scenario.tuning
    grid = {key: settings[key] for key in TUNABLE if isinstance(settings.get(key), list)}
    if not grid:
        raise LocalizedError("tuning_empty_grid")
    states, actions = _sample(*load_examples(scenario.data), settings.get("sample"))
    metric_name = settings.get("metric", "f1_macro")
    score = METRICS[metric_name]
    folds = min(int(settings.get("folds", 3)), *np.unique(actions, return_counts=True)[1])
    splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=42).split(states, actions))

    candidates = list(ParameterGrid(grid))
    results = []
    with stage("tune", total=len(candidates) * folds) as progress:
        for candidate in candidates:
            params = model_params({**asdict(scenario.params), **candidate})
            scores = []
            for train_index, test_index in splits:
                model = new_model(params).fit([states[i] for i in train_index], [actions[i] for i in train_index])
                scores.append(score([actions[i] for i in test_index], model.predict([states[i] for i in test_index])))
                progress.advance()
            results.append({"params": candidate, "score": round(float(np.mean(scores)), 4),
                            "std": round(float(np.std(scores)), 4)})
    results.sort(key=lambda r: -r["score"])
    best = results[0]
    if write:
        values = {key: list(value) if isinstance(value, tuple) else value for key, value in best["params"].items()}
        update_config(scenario.config_path, {("model", key): value for key, value in values.items()})
    return {
        "metric": metric_name, "examples": len(states), "folds": folds,
        "best": best, "top": results[:5], "written_to": str(scenario.config_path) if write else None,
    }


def calibrate_policy(scenario: Scenario, write: bool = False) -> dict:
    """Calibrate on the scenario's validation set when it has one, otherwise on held-out training data."""
    states, actions = load_examples(scenario.data)
    if scenario.validation is not None:
        model = fit_pipeline(new_model(scenario.params), states, actions)
        held_states, held_actions = load_examples(scenario.validation)
        probabilities, classes = model.predict_proba(held_states), list(model.classes_)
        method = f"validation set ({scenario.validation.name})"
    elif len(states) <= HOLDOUT_ABOVE:
        probabilities, classes, _ = cross_validated_probabilities(states, actions, scenario.params)
        held_states, held_actions, method = states, actions, "cross-validation"
    else:
        train_x, held_states, train_y, held_actions = train_test_split(
            states, actions, test_size=0.2, stratify=actions, random_state=42
        )
        model = fit_pipeline(new_model(scenario.params), train_x, train_y)
        probabilities, classes, method = model.predict_proba(held_states), list(model.classes_), "80/20 holdout"

    with stage("calibrate", total=1):
        thresholds = calibrate(held_actions, probabilities, classes, scenario.policy)
    raw = {"default_threshold": scenario.policy.default_threshold, "review_cost": scenario.policy.review_cost,
           "actions": {name: {**asdict(settings), "min_confidence": settings.min_confidence}
                       for name, settings in scenario.policy.actions.items()},
           "rules": [asdict(rule) for rule in scenario.policy.rules]}
    for action, value in thresholds.items():
        raw["actions"].setdefault(action, {})["min_confidence"] = value
    calibrated = from_dict(raw)
    if write:
        update_config(scenario.config_path, {
            ("policy", "actions", action, "min_confidence"): value for action, value in thresholds.items()
        })
    return {
        "method": method, "examples": len(held_actions), "thresholds": thresholds,
        "before": evaluate_policy(held_states, held_actions, probabilities, classes, scenario.policy),
        "after": evaluate_policy(held_states, held_actions, probabilities, classes, calibrated),
        "written_to": str(scenario.config_path) if write else None,
    }


def update_config(path: Path, updates: dict[tuple[str, ...], object]) -> None:
    """Set nested keys in a TOML file, keeping its comments and layout."""
    import tomlkit

    document = tomlkit.parse(path.read_text(encoding="utf-8"))
    for keys, value in updates.items():
        table = document
        for key in keys[:-1]:
            if key not in table:
                table[key] = tomlkit.table()
            table = table[key]
        table[keys[-1]] = value
    path.write_text(tomlkit.dumps(document), encoding="utf-8")

