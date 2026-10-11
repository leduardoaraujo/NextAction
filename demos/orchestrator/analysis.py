"""Inspect held-out routing errors and probability calibration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np

from demos.orchestrator.benchmark import ACTIONS, keyword_router, read_cases
from nextaction.policy import apply
from nextaction.scenarios import get_scenario

HERE = Path(__file__).resolve().parent
CHARTS = HERE / "analysis"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def analysis(output: Path = HERE / "ANALYSIS.md") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenario = get_scenario("orchestrator")
    cases = read_cases(scenario.test)
    model = joblib.load(scenario.model)
    states = [row["state"] for row in cases]
    probabilities = model.predict_proba(states)
    classes = list(model.classes_)
    predicted = [classes[int(np.argmax(row))] for row in probabilities]
    baseline = [keyword_router(row["utterance"]) for row in cases]
    choices = [apply(dict(zip(classes, map(float, proba))), state, scenario.policy)
               for state, proba in zip(states, probabilities)]
    actual = [row["action"] for row in cases]
    matrix = np.array([[sum(y == truth and p == guess for y, p in zip(actual, predicted))
                        for guess in ACTIONS] for truth in ACTIONS])
    CHARTS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=8)
    ax.set_xticks(range(4), ACTIONS, rotation=25, ha="right")
    ax.set_yticks(range(4), ACTIONS)
    ax.set_xlabel("NextAction prediction")
    ax.set_ylabel("Labeled next action")
    errors = sum(a != p for a, p in zip(actual, predicted))
    ax.set_title(f"Data/chat confusion drives {errors} routing errors", loc="left")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] >= 5 else "#0b0b0b")
    fig.colorbar(image, ax=ax, label="Cases")
    fig.tight_layout()
    fig.savefig(CHARTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    confidence = probabilities.max(axis=1)
    correct = np.array([a == p for a, p in zip(actual, predicted)])
    edges = np.array([0, .3, .4, .5, .6, .7, .8, 1.00001])
    bins = []
    for low, high in zip(edges, edges[1:]):
        mask = (confidence >= low) & (confidence < high)
        if mask.any():
            bins.append((f"{low:.1f}–{min(high, 1):.1f}", int(mask.sum()),
                         float(confidence[mask].mean()), float(correct[mask].mean())))
    fig, ax = plt.subplots(figsize=(6.5, 3.7))
    x = np.arange(len(bins))
    ax.bar(x - .18, [row[2] for row in bins], .36, color=BLUE, label="Mean model confidence")
    ax.bar(x + .18, [row[3] for row in bins], .36, color=ORANGE, label="Observed accuracy")
    ax.set_xticks(x, [row[0] for row in bins])
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Confidence band")
    ax.set_ylabel("Share")
    ax.set_title(f"Confidence is only a rough guide on {len(cases)} test cases", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(CHARTS / "confidence.png", dpi=150)
    plt.close(fig)

    lines = ["# AI orchestrator error analysis", "",
             f"All {len(cases)} test families are synthetic and held out from training. Counts are case counts, "
             "not estimates of production frequency.", "",
             "![Confusion matrix](analysis/confusion_matrix.png)", "",
             "## Incorrect next actions", "",
             "| User message | Expected | Predicted | Confidence | Policy review? | Keyword router |",
             "| --- | --- | --- | ---: | --- | --- |"]
    for case, truth, guess, proba, decision, rule in zip(cases, actual, predicted, probabilities, choices, baseline):
        if truth != guess:
            lines.append(f"| {case['utterance']} | `{truth}` | `{guess}` | {max(proba):.1%} | "
                         f"{'yes' if decision['needs_review'] else 'no'} | `{rule}` |")
    lines += ["", "## Confidence", "", "![Confidence](analysis/confidence.png)", "",
              "| Confidence band | Cases | Mean confidence | Observed accuracy |",
              "| --- | ---: | ---: | ---: |"]
    for band, n, mean, accuracy in bins:
        lines.append(f"| {band} | {n} | {mean:.1%} | {accuracy:.1%} |")
    lines += ["", "These bins are too small for a strong calibration claim. The validation "
              "thresholds choose when to hand the route to another decision maker; the review outcome "
              "is not included in the accuracy score.", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    analysis()
    print("saved", HERE / "ANALYSIS.md")
