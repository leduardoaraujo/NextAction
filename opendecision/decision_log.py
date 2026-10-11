"""Append-only log of decisions and the feedback that becomes training data."""

import csv
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .vocabulary import normalize_action

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with _lock, path.open("a", encoding="utf-8") as output:
        output.write(line + "\n")


def read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def record_decision(path: Path, scenario: str, state: str, decision: dict) -> str:
    decision_id = uuid.uuid4().hex
    _append(path, {
        "type": "decision", "id": decision_id, "at": _now(), "scenario": scenario,
        "state": state, "suggested": decision["action"],
        "probabilities": decision["probabilities"], "needs_review": decision["needs_review"],
    })
    return decision_id


def record_feedback(
    path: Path, decision_id: str, final_action: str,
    outcome: str | None = None, comment: str | None = None,
) -> dict:
    """Store the action that should have been taken; the latest feedback for a decision wins."""
    if not any(r["type"] == "decision" and r["id"] == decision_id for r in read(path)):
        raise KeyError(decision_id)
    record = {"type": "feedback", "id": decision_id, "at": _now(), "final_action": normalize_action(final_action)}
    if outcome is not None:
        record["outcome"] = outcome
    if comment is not None:
        record["comment"] = comment
    _append(path, record)
    return record


def export_labels(path: Path, destination: Path, scenario: str) -> dict:
    """Write a chronological ``state,action`` CSV with the decisions that received feedback."""
    records = read(path)
    feedback = {r["id"]: r for r in records if r["type"] == "feedback"}
    decisions = [r for r in records if r["type"] == "decision" and r["scenario"] == scenario]
    labeled = [(d, feedback[d["id"]]) for d in decisions if d["id"] in feedback]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["state", "action", "at", "suggested"])
        for decision, note in labeled:
            writer.writerow([decision["state"], note["final_action"], decision["at"], decision["suggested"]])
    corrected = sum(d["suggested"] != n["final_action"] for d, n in labeled)
    return {
        "file": str(destination), "decisions": len(decisions), "labeled": len(labeled),
        "corrected_suggestions": corrected,
    }
