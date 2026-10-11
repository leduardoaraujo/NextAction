"""Build transparent synthetic train, validation and test CSVs for routing."""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demos.orchestrator.cases import CASES, context_for

ROOT = Path(__file__).resolve().parents[2]


def state(utterance: str, context: dict[str, str]) -> str:
    blocked = "data_query" if context["datasource"] == "missing" else "none"
    return (f"[blocked {blocked}] [context datasource_{context['datasource']} "
            f"last_error_{context['last_error']} missing_{context['missing']}] || user: {utterance}")


def generate(output_dir: Path = ROOT / "data") -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, actions in CASES.items():
        path = output_dir / f"orchestrator_{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("state", "action", "utterance", "family"))
            for action, utterances in actions.items():
                for index, utterance in enumerate(utterances):
                    writer.writerow((state(utterance, context_for(utterance)), action,
                                     utterance, f"{split}/{action}/{index}"))
        counts[split] = sum(map(len, actions.values()))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    print(generate(args.output_dir))


if __name__ == "__main__":
    main()
