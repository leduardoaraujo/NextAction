"""HTTP contracts of the REST API, in English and with Portuguese aliases."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from nextaction.api import MODEL_PATHS, app
from nextaction.model import train

ROOT = Path(__file__).resolve().parents[1]
SEARCH_STATE = "Tenho origem GRU destino REC ida 12/11 e ainda não consultei voos"


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.model_path = Path(cls.temp.name) / "flights.joblib"
        train(ROOT / "data" / "flights.csv", cls.model_path)
        cls.models = patch.dict(MODEL_PATHS, {"flights": cls.model_path})
        cls.log = patch("nextaction.api.DECISION_LOG", Path(cls.temp.name) / "decisions.jsonl")
        cls.models.start()
        cls.log.start()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.log.stop()
        cls.models.stop()
        cls.temp.cleanup()

    def trip(self):
        return json.loads((ROOT / "data" / "trip_example.json").read_text(encoding="utf-8"))

    def test_health_and_openapi(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        routes = self.client.get("/openapi.json").json()["paths"]
        self.assertIn("/v1/actions/decide", routes)
        self.assertIn("/v1/flights/evaluate", routes)

    def test_decide_in_english(self):
        response = self.client.post("/v1/actions/decide", json={"scenario": "flights", "state": SEARCH_STATE})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["action"], "search")
        self.assertEqual(body["type"], "choice")
        self.assertIn("probabilities", body)
        self.assertIn("decision_id", body)

    def test_decide_accepts_portuguese_fields_and_scenario(self):
        response = self.client.post(
            "/v1/actions/decide", json={"cenario": "passagens", "estado": SEARCH_STATE, "limiar": 0.5}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["action"], "search")

    def test_feedback_links_final_action_and_normalizes_portuguese_label(self):
        decision = self.client.post(
            "/v1/actions/decide", json={"scenario": "flights", "state": "O viajante escolheu o voo das 8h"}
        ).json()
        response = self.client.post(
            f"/v1/decisions/{decision['decision_id']}/feedback",
            json={"acao_final": "pedir_confirmacao", "comentario": "já havia escolha"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["final_action"], "confirm")

    def test_feedback_for_unknown_decision_is_localized(self):
        response = self.client.post(
            "/v1/decisions/missing/feedback", json={"final_action": "search"},
            headers={"Accept-Language": "pt-BR,pt;q=0.9"},
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Decisão não encontrada.")

    def test_invalid_feedback_action_returns_422(self):
        response = self.client.post("/v1/decisions/x/feedback", json={"final_action": "Search Flights!"})
        self.assertEqual(response.status_code, 422)

    def test_missing_model_returns_localized_503(self):
        with patch.dict(MODEL_PATHS, {"flights": self.model_path.parent / "missing.joblib"}):
            english = self.client.post("/v1/actions/decide", json={"scenario": "flights", "state": "x"})
            portuguese = self.client.post(
                "/v1/actions/decide", json={"scenario": "flights", "state": "x"},
                headers={"Accept-Language": "pt"},
            )
        self.assertEqual(english.status_code, 503)
        self.assertIn("unavailable", english.json()["detail"])
        self.assertIn("indisponível", portuguese.json()["detail"])

    def test_unknown_scenario_and_invalid_threshold_return_422(self):
        for body in ({"scenario": "hotels", "state": "x"}, {"scenario": "flights", "state": "x", "threshold": 1.5}):
            self.assertEqual(self.client.post("/v1/actions/decide", json=body).status_code, 422)

    def test_rank_flights(self):
        response = self.client.post("/v1/flights/evaluate", json=self.trip())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["recommendation"], "flight_a")
        self.assertEqual(response.json()["discarded"][0]["id"], "flight_b")

    def test_rank_flights_accepts_portuguese_fields(self):
        response = self.client.post("/v1/flights/evaluate", json={
            "pedido": {"origem": "GRU", "destino": "REC", "data_ida": "2026-11-12", "passageiros": 1},
            "ofertas": [{"id": "a", "preco_total": 900, "duracao_min": 180, "escalas": 0, "bagagem_incluida": True}],
            "oferta_escolhida": "a",
        }, headers={"Accept-Language": "pt"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["action"], "confirm")
        self.assertIn("Confirme", response.json()["notice"])

    def test_missing_trip_field_asks(self):
        response = self.client.post("/v1/flights/evaluate", json={"request": {"origin": "GRU", "destination": "REC"}})
        self.assertEqual(response.json()["action"], "ask")
        self.assertIn("departure_date", response.json()["missing_fields"])

    def test_invalid_offer_returns_422(self):
        trip = self.trip()
        trip["offers"][0]["total_price"] = -1
        self.assertEqual(self.client.post("/v1/flights/evaluate", json=trip).status_code, 422)


if __name__ == "__main__":
    unittest.main()
