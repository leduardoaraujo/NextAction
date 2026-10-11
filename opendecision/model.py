"""Training and inference for the action classifier."""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .data import load_examples
from .i18n import LocalizedError, t
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


def fit(data: Path, min_df: int = 1):
    states, actions = load_examples(data)
    return slim(new_model(min_df).fit(states, actions)), states, actions


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
