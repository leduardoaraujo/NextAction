"""Convert the Schema-Guided Dialogue dataset (Google, CC BY-SA 4.0) into ``state,action`` examples.

Every assistant turn becomes one example. The state is the conversation so far plus the
dialogue state (intent, filled slots, required slots still missing), and the action is what
the assistant did, grouped into generic agent actions.
"""

import csv
import json
from collections import Counter
from pathlib import Path

from .i18n import LocalizedError
from .progress import stage

# Priority order when a turn has several acts.
GENERIC_ACTIONS = {
    "NOTIFY_SUCCESS": "execute",
    "NOTIFY_FAILURE": "execute",
    "CONFIRM": "confirm",
    "REQUEST": "ask",
    "OFFER_INTENT": "suggest_next_step",
    "OFFER": "present_options",
    "INFORM_COUNT": "present_options",
    "INFORM": "answer",
    "REQ_MORE": "offer_more_help",
    "GOODBYE": "end_conversation",
}
FLIGHT_ACTIONS = {"ask", "search", "present_options", "confirm"}
SCHEMAS = ("generic", "flights")
SCHEMA_ALIASES = {"generico": "generic", "passagens": "flights"}


def turn_action(turn: dict) -> str | None:
    acts = {action["act"] for frame in turn["frames"] for action in frame["actions"]}
    queried = any(
        "service_call" in frame and not acts & {"NOTIFY_SUCCESS", "NOTIFY_FAILURE"}
        for frame in turn["frames"]
    )
    if queried:
        # The assistant called the service before replying: the decision was to search.
        return "search"
    return next((action for act, action in GENERIC_ACTIONS.items() if act in acts), None)


def describe_state(history: list[dict], required: dict, context: int, with_state: bool) -> str:
    parts = []
    last_user = next((turn for turn in reversed(history) if turn["speaker"] == "USER"), None)
    if with_state and last_user:
        for frame in last_user["frames"]:
            state = frame.get("state")
            if not state or state["active_intent"] == "NONE":
                continue
            intent = state["active_intent"]
            filled = sorted(state["slot_values"])
            missing = [slot for slot in required.get((frame["service"], intent), []) if slot not in filled]
            parts.append(
                f"[intent {intent}] [filled {' '.join(filled) or 'none'}] "
                f"[missing {' '.join(missing) or 'none'}] "
                f"[requested {' '.join(state['requested_slots']) or 'none'}]"
            )
    for turn in history[-context:]:
        speaker = "user" if turn["speaker"] == "USER" else "assistant"
        parts.append(f"{speaker}: {turn['utterance']}")
    return " || ".join(parts)


def convert(
    split_dir: Path, destination: Path, schema: str = "generic",
    context: int = 3, with_state: bool = True,
) -> dict:
    schema = SCHEMA_ALIASES.get(schema, schema)
    if schema not in SCHEMAS:
        raise LocalizedError("unknown_schema", name=schema, options=", ".join(SCHEMAS))
    files = sorted(split_dir.glob("dialogues_*.json"))
    if not files:
        raise LocalizedError("sgd_no_dialogues", path=split_dir)
    services = json.loads((split_dir / "schema.json").read_text(encoding="utf-8"))
    required = {
        (service["service_name"], intent["name"]): intent["required_slots"]
        for service in services for intent in service["intents"]
    }

    counts = Counter()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output, stage("import", total=len(files)) as progress:
        writer = csv.writer(output)
        writer.writerow(["state", "action", "dialogue"])
        for path in files:
            progress.advance()
            for dialogue in json.loads(path.read_text(encoding="utf-8")):
                if schema == "flights" and not any(s.startswith("Flights") for s in dialogue["services"]):
                    continue
                turns = dialogue["turns"]
                for index, turn in enumerate(turns):
                    if turn["speaker"] != "SYSTEM" or index == 0:
                        continue
                    action = turn_action(turn)
                    if schema == "flights":
                        flight_turn = all(f["service"].startswith("Flights") for f in turn["frames"])
                        action = action if flight_turn and action in FLIGHT_ACTIONS else None
                    if action is None:
                        continue
                    state = describe_state(turns[:index], required, context, with_state)
                    writer.writerow([state, action, dialogue["dialogue_id"]])
                    counts[action] += 1
    return {"file": str(destination), "examples": sum(counts.values()), "per_action": dict(counts.most_common())}
