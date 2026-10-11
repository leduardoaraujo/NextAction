"""Classifier evaluation: per-action errors, calibration and review threshold trade-offs."""

from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .data import load_examples
from .i18n import LocalizedError, t
from .model import new_model

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def evaluate(
    data: Path, test: Path | None = None, recent_fraction: float | None = None,
    min_df: int = 1, lang: str | None = None,
) -> dict:
    """Evaluate by cross-validation, on a separate test CSV, or on the most recent examples.

    ``recent_fraction`` uses the last rows of the CSV as the test set without shuffling: with
    data exported from the decision log (chronological order) it measures future performance.
    """
    states, actions = load_examples(data)
    if test is not None and recent_fraction is not None:
        raise LocalizedError("test_modes_exclusive")
    if test is not None:
        test_x, test_y = load_examples(test)
        method = t("method_test_file", lang, path=test)
        return _evaluate_split(states, actions, test_x, test_y, method, min_df, lang)
    if recent_fraction is not None:
        if not 0 < recent_fraction < 1:
            raise LocalizedError("fraction_range")
        cut = len(states) - max(1, round(len(states) * recent_fraction))
        method = t("method_recent", lang, count=len(states) - cut)
        return _evaluate_split(
            states[:cut], actions[:cut], states[cut:], actions[cut:], method, min_df, lang
        )

    folds = min(5, min(Counter(actions).values()))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    probabilities = cross_val_predict(
        new_model(min_df), states, actions, cv=splitter, method="predict_proba"
    )
    majority = Counter(actions).most_common(1)[0][0]
    return {
        "method": t("method_cross_validation", lang, folds=folds),
        **metrics(actions, probabilities, sorted(set(actions)), majority, lang),
    }


def _evaluate_split(train_x, train_y, test_x, test_y, method: str, min_df: int, lang) -> dict:
    if len(set(train_y)) < 2:
        raise LocalizedError("train_needs_two_actions")
    model = new_model(min_df).fit(train_x, train_y)
    majority = Counter(train_y).most_common(1)[0][0]
    result = {"method": method, "train_examples": len(train_x)}
    result.update(metrics(test_y, model.predict_proba(test_x), list(model.classes_), majority, lang))
    unseen = sorted(set(test_y) - set(model.classes_))
    if unseen:
        result["actions_missing_from_training"] = unseen
    return result


def metrics(actual, probabilities, classes: list[str], majority: str, lang: str | None = None) -> dict:
    actual = np.asarray(actual)
    probabilities = np.asarray(probabilities)
    predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    correct = predicted == actual
    labels = sorted(str(label) for label in set(classes) | set(actual))

    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(actual, predicted, labels=labels)
    result = {
        "evaluated_examples": len(actual),
        "accuracy": _round(correct.mean()),
        "majority_baseline_accuracy": _round(np.mean(actual == majority)),
        "per_action": {
            action: {"precision": _round(p), "recall": _round(r), "f1": _round(f), "support": int(s)}
            for action, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
        "confusion_matrix": {
            real: {guess: int(n) for guess, n in zip(labels, row) if n}
            for real, row in zip(labels, matrix)
        },
        "expected_calibration_error": _round(_calibration_error(confidence, correct)),
        "log_loss": (
            _round(log_loss(actual, probabilities, labels=classes))
            if set(actual) <= set(classes) else None
        ),
        "thresholds": [_threshold(limit, confidence, correct) for limit in THRESHOLDS],
    }
    if len(actual) < 100:
        result["warning"] = t("small_evaluation_warning", lang, count=len(actual))
    return result


def _threshold(limit: float, confidence, correct) -> dict:
    automated = confidence >= limit
    return {
        "threshold": limit,
        "automated": _round(automated.mean()),
        "automated_accuracy": _round(correct[automated].mean()) if automated.any() else None,
        "unreviewed_errors": int((automated & ~correct).sum()),
        "sent_to_review": int((~automated).sum()),
    }


def _calibration_error(confidence, correct, bins: int = 10) -> float:
    """Average gap between confidence and accuracy over confidence bins (ECE)."""
    indices = np.minimum((confidence * bins).astype(int), bins - 1)
    total = 0.0
    for current in range(bins):
        in_bin = indices == current
        if in_bin.any():
            total += in_bin.mean() * abs(confidence[in_bin].mean() - correct[in_bin].mean())
    return total


def _round(value: float) -> float:
    return round(float(value), 3)
