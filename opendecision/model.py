"""Training and inference for the action classifier."""

import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .data import load_examples
from .i18n import LocalizedError, t
from .progress import stage
from .vocabulary import normalize_action, normalize_state


def new_model(min_df: int = 1):
    """``min_df`` drops rare terms; use 2 or more on large datasets to shrink the model file."""
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", min_df=min_df),
        LogisticRegression(max_iter=1000),
    )


def slim(model):
    """Drop ``stop_words_``, the list of discarded terms kept only for inspection.

    On large datasets it can hold millions of entries; predictions do not use it.
    """
    vectorizer = model.steps[0][1]
    if hasattr(vectorizer, "stop_words_"):
        delattr(vectorizer, "stop_words_")
    return model


def fit_pipeline(model, states: list[str], actions: list[str], chunk: int = 50):
    """Fit the pipeline while reporting progress: vectorizing, then solver iterations.

    The solver runs in warm-started chunks of ``chunk`` iterations, so the stage shows
    iterations out of ``max_iter`` and jumps to 100% when it converges early.
    """
    vectorizer, classifier = model.steps[0][1], model.steps[-1][1]
    with stage("vectorize", total=len(states)):
        features = vectorizer.fit_transform(states)
    max_iter = classifier.max_iter
    classifier.set_params(warm_start=True, max_iter=chunk)
    iterations = 0
    with stage("fit", total=max_iter, estimate_eta=False) as fitting, warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        while iterations < max_iter:
            classifier.fit(features, actions)
            used = int(np.max(classifier.n_iter_))
            iterations += used
            fitting.update(min(iterations, max_iter))
            if used < chunk:
                break
    classifier.set_params(warm_start=False, max_iter=max_iter)
    if iterations >= max_iter:
        warnings.warn(f"The solver did not converge in {max_iter} iterations.", ConvergenceWarning)
    return model


def fit(data: Path, min_df: int = 1):
    states, actions = load_examples(data)
    return slim(fit_pipeline(new_model(min_df), states, actions)), states, actions


def train(data: Path, destination: Path, min_df: int = 1, lang: str | None = None) -> dict:
    model, states, actions = fit(data, min_df)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return {
        "model_path": str(destination),
        "examples": len(states),
        "actions": sorted(set(actions)),
        "note": t("train_note", lang),
    }


def decide(model_path: Path, state: str, threshold: float = 0.7) -> dict:
    if not 0 <= threshold <= 1:
        raise LocalizedError("threshold_range")
    if not model_path.is_file():
        raise LocalizedError("model_file_missing", path=model_path)
    # Only load models you trained yourself: joblib files can execute code.
    model = joblib.load(model_path)
    probabilities = model.predict_proba([normalize_state(state)])[0]
    # Models trained before the English vocabulary have Portuguese labels.
    distribution = {
        normalize_action(str(action)): round(float(probability), 4)
        for action, probability in zip(model.classes_, probabilities)
    }
    action = max(distribution, key=distribution.get)
    return {
        "type": "choice",
        "action": action,
        "probabilities": distribution,
        "needs_review": distribution[action] < threshold,
    }
