"""Compare NextAction with a frozen keyword router on unseen request families.

Optional --axis-predictions accepts real Axis LLM classifications exported separately.
No LLM result is invented when that file is unavailable.
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from nextaction.policy import apply
from nextaction.scenarios import get_scenario

HERE = Path(__file__).resolve().parent
ACTIONS = ("data_query", "pure_chat", "clarification_needed", "error_recovery")


def keyword_router(message: str) -> str:
    """Independent two-route recreation of Axis's legacy visualization heuristic."""
    text = message.lower()
    if any(phrase in text for phrase in ("quem sou eu", "meu perfil", "o que você consegue", "what can this assistant")):
        return "pure_chat"
    strong = ("gráfico", "grafico", "chart", "kpi", "visualização", "visualizacao", "tabela", "table", "dashboard")
    if any(word in text for word in strong):
        return "data_query"
    metrics = ("vendas", "receita", "faturamento", "margem", "pedidos", "indicador", "revenue", "orders", "churn")
    intent = ("mostre", "mostrar", "compare", "comparar", "liste", "listar", "por ", "show", "plot", "compute", "break down")
    if any(word in text for word in metrics) and any(word in text for word in intent):
        return "data_query"
    quantitative = ("quanto", "quantos", "total", "how many")
    temporal = ("ano", "mês", "mensal", "year", "month", "quarter")
    return "data_query" if any(word in text for word in quantitative) and any(word in text for word in temporal) else "pure_chat"


def read_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_axis_predictions(path: Path, cases: list[dict]) -> tuple[list[str], float | None]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    by_family = {row["family"]: row for row in rows}
    expected = {row["family"] for row in cases}
    if set(by_family) != expected or len(rows) != len(expected):
        raise ValueError("Axis predictions must have exactly one row for every test family")
    actions = [by_family[row["family"]]["action"] for row in cases]
    if any(action not in ACTIONS + ("unknown",) for action in actions):
        raise ValueError("Axis predictions contain an unknown action")
    durations = [by_family[row["family"]].get("latency_ms", "") for row in cases]
    latency = sum(float(value) for value in durations) / len(durations) if all(durations) else None
    return actions, latency


def summary(actual: list[str], predicted: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "agent_accuracy": sum(p != "unknown" and ((p == "data_query") == (y == "data_query"))
                              for y, p in zip(actual, predicted)) / len(actual),
        "macro_f1": float(f1_score(actual, predicted, labels=ACTIONS, average="macro", zero_division=0)),
        "wrong_data_dispatch": sum(p == "data_query" and y != "data_query" for y, p in zip(actual, predicted)),
        "missed_data": sum(y == "data_query" and p != "data_query" for y, p in zip(actual, predicted)),
    }


def wilson(correct: int, total: int) -> str:
    z = 1.96
    p = correct / total
    middle = (p + z*z/(2*total)) / (1+z*z/total)
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1+z*z/total)
    return f"{max(0, middle-half):.1%}–{min(1, middle+half):.1%}"


def paired_route_interval(actual: list[str], challenger: list[str], baseline: list[str]) -> tuple[float, float]:
    """Paired bootstrap on held-out request families, not independent proportions."""
    truth = np.array([action == "data_query" for action in actual])
    gain = ((np.array(challenger) == "data_query") == truth).astype(int)
    gain -= ((np.array(baseline) == "data_query") == truth).astype(int)
    rng = np.random.default_rng(42)
    samples = gain[rng.integers(0, len(gain), size=(10000, len(gain)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, (.025, .975)))


def benchmark(axis_predictions: Path | None = None, output: Path = HERE / "RESULTS.md") -> dict:
    scenario = get_scenario("orchestrator")
    cases = read_cases(scenario.test)
    model = joblib.load(scenario.model)
    states = [row["state"] for row in cases]
    actual = [row["action"] for row in cases]
    start = time.perf_counter()
    probabilities = model.predict_proba(states)
    inference_ms = (time.perf_counter() - start) * 1000 / len(states)
    classes = list(model.classes_)
    model_predictions = [classes[int(np.argmax(row))] for row in probabilities]
    baseline_predictions = [keyword_router(row["utterance"]) for row in cases]
    policy_decisions = [apply(dict(zip(classes, map(float, proba))), state, scenario.policy)
                        for state, proba in zip(states, probabilities)]
    automated = [not decision["needs_review"] for decision in policy_decisions]
    policy_predictions = [decision["action"] for decision in policy_decisions]
    scores = {"Keyword baseline": summary(actual, baseline_predictions),
              "NextAction model": summary(actual, model_predictions)}
    if axis_predictions:
        axis_actions, axis_latency = read_axis_predictions(axis_predictions, cases)
        scores["Axis LLM classifier"] = summary(actual, axis_actions)
    else:
        axis_latency = None
    auto_count = sum(automated)
    auto_correct = sum(a == p for a, p, auto in zip(actual, policy_predictions, automated) if auto)
    route_gain = scores["NextAction model"]["agent_accuracy"] - scores["Keyword baseline"]["agent_accuracy"]
    route_low, route_high = paired_route_interval(actual, model_predictions, baseline_predictions)
    lines = [
        "# AI orchestrator routing benchmark", "",
        f"The test set contains {len(cases)} hand-authored request families held out from training and policy calibration. "
        "Each family appears once. All examples are synthetic and roughly balanced across four actions; "
        "the figures do not estimate production traffic accuracy.", "",
        "| Router | Next-action accuracy | 95% Wilson interval | Agent-route accuracy | Macro F1 | Wrong data-agent dispatches | Data requests missed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in scores.items():
        lines.append(f"| {name} | {row['accuracy']:.1%} | {wilson(round(row['accuracy']*len(actual)), len(actual))} | "
                     f"{row['agent_accuracy']:.1%} | {row['macro_f1']:.3f} | "
                     f"{row['wrong_data_dispatch']} | {row['missed_data']} |")
    interpretation = ("The interval includes zero; this set does not establish a routing advantage."
                      if route_low <= 0 <= route_high else
                      "The interval excludes zero on this synthetic set; it does not establish a production advantage.")
    lines += ["", f"Agent-route accuracy differs by {route_gain:+.1%} for NextAction versus the keyword baseline "
              f"(95% paired family-bootstrap interval {route_low:+.1%} to {route_high:+.1%}). "
              + interpretation, "",
              f"With calibrated policy, NextAction automates {auto_count}/{len(actual)} cases "
              f"({auto_count/len(actual):.1%}); {auto_correct}/{auto_count} automated next actions are correct. "
              f"Another decision is needed for {len(actual)-auto_count} case(s). Review accuracy is not assumed.", "",
              f"Local batch inference took {inference_ms:.2f} ms per case for the NextAction model "
              "on this machine. This is not a comparable end-to-end latency measurement for the other routers.", ""]
    if axis_predictions is None:
        lines += ["The Axis LLM classifier was not run: its service credentials and runtime are not present here. "
                  "Use `--axis-predictions` with one exported classification per test family to add its measured result.", ""]
    elif axis_latency is not None:
        lines += [f"The supplied Axis LLM predictions report mean classifier latency of {axis_latency:.1f} ms.", ""]
    lines += ["Agent-route accuracy asks only whether the request reaches the data analyst or the "
              "conversation path. Next-action accuracy distinguishes data, chat, clarification and recovery. "
              "The keyword baseline is an independent frozen recreation of the legacy Axis visualization "
              "heuristic; it has only data/chat outputs. The NextAction policy also reads explicit context "
              "tags and blocks data-agent dispatch when no data source is available. This is an end-to-end "
              "routing comparison, not a controlled model-only ablation.", ""]
    output.write_text("\n".join(lines), encoding="utf-8")
    return {"scores": scores, "automated": auto_count, "automated_correct": auto_correct,
            "cases": len(cases), "inference_ms": inference_ms}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axis-predictions", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "RESULTS.md")
    args = parser.parse_args()
    result = benchmark(args.axis_predictions, args.output)
    print(result)


if __name__ == "__main__":
    main()
