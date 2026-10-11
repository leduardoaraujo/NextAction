"""Schema-Guided Dialogue conversion into state,action examples."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from nextaction.sgd import convert


def _user(text, filled):
    state = {"active_intent": "SearchOnewayFlight", "slot_values": {s: ["x"] for s in filled}, "requested_slots": []}
    return {"speaker": "USER", "utterance": text,
            "frames": [{"service": "Flights_1", "actions": [{"act": "INFORM"}], "state": state}]}


def _system(text, acts, called=False):
    frame = {"service": "Flights_1", "actions": [{"act": act} for act in acts]}
    if called:
        frame["service_call"] = {"method": "SearchOnewayFlight"}
    return {"speaker": "SYSTEM", "utterance": text, "frames": [frame]}


DIALOGUE = {"dialogue_id": "1_00000", "services": ["Flights_1"], "turns": [
    _user("I need a flight to Seattle", ["destination"]),
    _system("Where from?", ["REQUEST"]),
    _user("From New York", ["destination", "origin"]),
    _system("I found 3 flights", ["INFORM_COUNT", "OFFER"], called=True),
    _user("Book it", ["destination", "origin"]),
    _system("Please confirm", ["CONFIRM"]),
    _user("Yes", ["destination", "origin"]),
    _system("Booked", ["NOTIFY_SUCCESS"], called=True),
]}
SCHEMA = [{"service_name": "Flights_1", "intents": [
    {"name": "SearchOnewayFlight", "required_slots": ["origin", "destination"]},
]}]


class ConvertSgdTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        (self.folder / "dialogues_001.json").write_text(json.dumps([DIALOGUE]), encoding="utf-8")
        (self.folder / "schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def _rows(self, **options):
        destination = self.folder / "output.csv"
        convert(self.folder, destination, **options)
        with destination.open(encoding="utf-8") as source:
            return list(csv.DictReader(source))

    def test_maps_acts_and_service_calls(self):
        self.assertEqual([r["action"] for r in self._rows()], ["ask", "search", "confirm", "execute"])

    def test_state_shows_missing_slots(self):
        first, second = self._rows()[:2]
        self.assertIn("[missing origin]", first["state"])
        self.assertIn("[missing none]", second["state"])
        self.assertTrue(first["state"].endswith("user: I need a flight to Seattle"))

    def test_without_state_uses_only_the_conversation(self):
        self.assertNotIn("[intent", self._rows(with_state=False)[0]["state"])

    def test_flights_schema_drops_execution_and_accepts_portuguese_alias(self):
        actions = [r["action"] for r in self._rows(schema="passagens")]
        self.assertEqual(actions, ["ask", "search", "confirm"])


if __name__ == "__main__":
    unittest.main()
