"""Compare the bilingual model with the English baseline on paired SGD test turns."""

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib

from nextaction.data import load_examples
from nextaction.evaluation import metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bilingual-model", type=Path, default=Path("models/agent_bilingual.joblib"))
    parser.add_argument("--english-model", type=Path, default=Path("models/agent.joblib"))
    parser.add_argument("--english-test", type=Path, default=Path("data/external/sgd_en_paired_test.csv"))
    parser.add_argument("--portuguese-test", type=Path, default=Path("data/external/sgd_pt_test.csv"))
    args = parser.parse_args()
    report = {}
    for model_name, model_path in (("english", args.english_model), ("bilingual", args.bilingual_model)):
        model = joblib.load(model_path)
        report[model_name] = {}
        for language, test_path in (("en", args.english_test), ("pt", args.portuguese_test)):
            states, actions = load_examples(test_path)
            result = metrics(actions, model.predict_proba(states), list(model.classes_),
                             Counter(actions).most_common(1)[0][0])
            report[model_name][language] = {
                "examples": result["evaluated_examples"],
                "accuracy": result["accuracy"],
                "ece": result["expected_calibration_error"],
                "thresholds": result["thresholds"],
            }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
