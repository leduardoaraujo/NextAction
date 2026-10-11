"""Loading and validation of labeled examples."""

import csv
from collections import Counter
from pathlib import Path

from .i18n import LocalizedError
from .vocabulary import normalize_action, normalize_state

COLUMNS = (("state", "action"), ("estado", "acao"))


def load_examples(path: Path) -> tuple[list[str], list[str]]:
    """Read a ``state,action`` CSV (``estado,acao`` also works) with canonical English labels."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = next((pair for pair in COLUMNS if set(pair) <= set(reader.fieldnames or ())), None)
        if columns is None:
            raise LocalizedError("csv_missing_columns")
        state_column, action_column = columns
        rows = [(row[state_column].strip(), row[action_column].strip()) for row in reader]

    if not rows or any(not state or not action for state, action in rows):
        raise LocalizedError("csv_empty_fields")
    states = [normalize_state(state) for state, _ in rows]
    actions = [normalize_action(action) for _, action in rows]
    counts = Counter(actions)
    if len(counts) < 2 or min(counts.values()) < 4:
        raise LocalizedError("csv_too_few_examples")
    return states, actions
