"""Export measured Axis LLM classifications for the shared synthetic test cases.

Run with the Axis backend's Python environment and configured model credentials:
    python demos/orchestrator/export_axis_predictions.py --axis-backend PATH/TO/axis/backend

The output has only family, action and classifier latency; no business data.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def export(axis_backend: Path, output: Path) -> int:
    if not (axis_backend / "app" / "services" / "ai" / "intent_classifier.py").is_file():
        raise FileNotFoundError("--axis-backend must point to the Axis backend folder")
    sys.path.insert(0, str(axis_backend))
    from app.services.ai.intent_classifier import classify_intent  # type: ignore[import-not-found]

    with (ROOT / "data" / "orchestrator_test.csv").open(encoding="utf-8", newline="") as file:
        cases = list(csv.DictReader(file))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, ("family", "action", "latency_ms"))
        writer.writeheader()
        for case in cases:
            start = time.perf_counter()
            classification, _usage = classify_intent(case["utterance"])
            elapsed = (time.perf_counter() - start) * 1000
            writer.writerow({"family": case["family"],
                             "action": classification.intent if classification else "unknown",
                             "latency_ms": f"{elapsed:.2f}"})
            file.flush()
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axis-backend", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "external" / "axis_orchestrator_predictions.csv")
    args = parser.parse_args()
    print(f"exported {export(args.axis_backend, args.output)} classifications to {args.output}")


if __name__ == "__main__":
    main()
