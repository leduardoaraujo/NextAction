"""Rule-based flight offer ranking."""

import json
import unittest
from copy import deepcopy
from pathlib import Path

from nextaction.flights import rank_flights
from nextaction.i18n import LocalizedError

EXAMPLE = Path(__file__).resolve().parents[1] / "data" / "trip_example.json"


class RankFlightsTest(unittest.TestCase):
    def setUp(self):
        self.trip = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_asks_for_required_fields_before_searching(self):
        del self.trip["request"]["departure_date"]
        self.assertEqual(rank_flights(self.trip), {"action": "ask", "missing_fields": ["departure_date"]})

    def test_searches_when_there_are_no_offers(self):
        self.trip["offers"] = []
        self.assertEqual(rank_flights(self.trip)["action"], "search")

    def test_filters_and_explains_the_ranking(self):
        result = rank_flights(self.trip)
        self.assertEqual(result["action"], "present_options")
        self.assertEqual(result["recommendation"], "flight_a")
        self.assertEqual([offer["id"] for offer in result["ranking"]], ["flight_a", "flight_c"])
        self.assertEqual(result["discarded"], [{"id": "flight_b", "reasons": ["missing_required_baggage"]}])

    def test_selected_offer_asks_for_localized_confirmation(self):
        self.trip["selected_offer"] = "flight_a"
        result = rank_flights(self.trip, lang="pt")
        self.assertEqual(result["action"], "confirm")
        self.assertEqual(result["selected_offer"]["id"], "flight_a")
        self.assertTrue(result["notice"].startswith("Confirme"))

    def test_no_eligible_offer_asks_to_revise_criteria(self):
        self.trip["request"]["max_budget"] = 500
        self.assertEqual(rank_flights(self.trip)["action"], "revise_criteria")

    def test_rejects_invalid_price(self):
        trip = deepcopy(self.trip)
        trip["offers"][0]["total_price"] = -1
        with self.assertRaisesRegex(LocalizedError, "total_price"):
            rank_flights(trip)

    def test_rejects_return_before_departure_with_localized_message(self):
        self.trip["request"]["return_date"] = "2026-11-01"
        with self.assertRaises(LocalizedError) as raised:
            rank_flights(self.trip)
        self.assertIn("return_date", str(raised.exception))
        self.assertIn("volta", raised.exception.message("pt"))


if __name__ == "__main__":
    unittest.main()
