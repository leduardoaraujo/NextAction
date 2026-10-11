"""Show NextAction's route for one message and an explicit runtime context."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demos.orchestrator.generate_data import state
from nextaction.model import decide
from nextaction.scenarios import get_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("message")
    parser.add_argument("--datasource", choices=("available", "missing"), default="available")
    parser.add_argument("--last-error", choices=("none", "sql_error"), default="none")
    parser.add_argument("--missing", choices=("none", "metric"), default="none")
    args = parser.parse_args()
    scenario = get_scenario("orchestrator")
    encoded = state(args.message, {"datasource": args.datasource,
                                   "last_error": args.last_error, "missing": args.missing})
    result = decide(scenario.model, encoded, policy=scenario.policy)
    result["final_step"] = "review" if result["needs_review"] else result["action"]
    result["state"] = encoded
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
