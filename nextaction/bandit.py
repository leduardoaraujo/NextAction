"""Offline outcome learning from explicitly logged actions and propensities.

This produces a candidate reward model for review. It never replaces a scenario's
next-action classifier or bypasses its decision policy.
"""

from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from .decision_log import read

MIN_EXAMPLES = 200
MIN_PER_ACTION = 20


def observations(path: Path, scenario: str) -> list[dict]:
    """Join each decision to its latest feedback containing observed outcome data."""
    records = read(path)
    feedback = {r["id"]: r for r in records if r.get("type") == "feedback"}
    examples = []
    for decision in records:
        if decision.get("type") != "decision" or decision.get("scenario") != scenario:
            continue
        note = feedback.get(decision["id"], {})
        if not {"action_taken", "reward", "propensity"} <= note.keys():
            continue
        reward, propensity = note["reward"], note["propensity"]
        if not isinstance(reward, (int, float)) or not isinstance(propensity, (int, float)):
            continue
        if not 0 <= reward <= 1 or not 0 < propensity <= 1:
            continue
        examples.append({"state": decision["state"], "action": note["action_taken"],
                         "reward": float(reward), "propensity": float(propensity),
                         "at": decision["at"]})
    return examples


def fit_candidate(log: Path, scenario: str, destination: Path,
                  min_examples: int = MIN_EXAMPLES) -> dict:
    examples = observations(log, scenario)
    counts = Counter(row["action"] for row in examples)
    if len(examples) < min_examples or len(counts) < 2 or min(counts.values(), default=0) < MIN_PER_ACTION:
        return {"status": "insufficient_data", "scenario": scenario,
                "observations": len(examples), "per_action": dict(counts),
                "required_total": min_examples, "required_per_action": MIN_PER_ACTION}

    examples.sort(key=lambda row: row["at"])
    split = int(len(examples) * 0.8)
    train, test = examples[:split], examples[split:]
    train_counts = Counter(row["action"] for row in train)
    if len(train_counts) < 2 or min(train_counts.values()) < MIN_PER_ACTION:
        return {"status": "insufficient_training_coverage", "scenario": scenario,
                "training_per_action": dict(train_counts)}

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    features = vectorizer.fit_transform([row["state"] for row in train])
    actions = sorted(train_counts)
    models = {}
    for action in actions:
        indices = [index for index, row in enumerate(train) if row["action"] == action]
        rewards = np.array([train[index]["reward"] for index in indices])
        weights = np.minimum(1 / np.array([train[index]["propensity"] for index in indices]), 20)
        models[action] = Ridge(alpha=10).fit(features[indices], rewards, sample_weight=weights)

    test_features = vectorizer.transform([row["state"] for row in test])
    predicted_rewards = np.column_stack([models[action].predict(test_features) for action in actions])
    chosen = [actions[index] for index in np.argmax(predicted_rewards, axis=1)]
    weights = np.array([1 / row["propensity"] if row["action"] == action else 0.0
                        for row, action in zip(test, chosen)])
    rewards = np.array([row["reward"] for row in test])
    matched = int(np.count_nonzero(weights))
    effective_size = float(weights.sum() ** 2 / np.square(weights).sum()) if matched else 0.0
    candidate_reward = float(np.dot(weights, rewards) / weights.sum()) if matched else None
    report = {
        "status": "candidate_ready", "scenario": scenario, "observations": len(examples),
        "training": len(train), "held_out": len(test), "per_action": dict(counts),
        "logging_mean_reward": round(float(rewards.mean()), 4),
        "candidate_snips_reward": round(candidate_reward, 4) if candidate_reward is not None else None,
        "matched_held_out": matched, "effective_sample_size": round(effective_size, 1),
        "offline_estimate_usable": matched >= 20 and effective_size >= 20,
        "warning": "Offline estimate needs genuine randomized propensities and sufficient action overlap; review before any deployment.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "models": models, "actions": actions,
                 "report": report}, destination)
    report["candidate_model"] = str(destination)
    return report


def preview_candidate(path: Path, state: str) -> dict:
    """Show estimated rewards; the candidate is never used for automatic actions."""
    candidate = joblib.load(path)
    features = candidate["vectorizer"].transform([state])
    estimates = {action: round(float(candidate["models"][action].predict(features)[0]), 4)
                 for action in candidate["actions"]}
    return {"suggested": max(estimates, key=estimates.get), "estimated_rewards": estimates,
            "candidate_only": True}
