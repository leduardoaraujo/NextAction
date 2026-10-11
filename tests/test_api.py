"""Contratos HTTP da API REST."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from opendecision.api import MODEL_PATHS, app
from opendecision.modelo import treinar


ROOT = Path(__file__).resolve().parents[1]


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.model_path = Path(cls.temp.name) / "passagens.joblib"
        treinar(ROOT / "dados" / "passagens.csv", cls.model_path)
        cls.registro = patch("opendecision.api.REGISTRO", Path(cls.temp.name) / "decisoes.jsonl")
        cls.registro.start()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.registro.stop()
        cls.temp.cleanup()

    def test_health_e_openapi(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        routes = self.client.get("/openapi.json").json()["paths"]
        self.assertIn("/v1/actions/decide", routes)
        self.assertIn("/v1/flights/evaluate", routes)

    def test_decisao_de_acao(self):
        with patch.dict(MODEL_PATHS, {"passagens": self.model_path}):
            response = self.client.post("/v1/actions/decide", json={
                "cenario": "passagens",
                "estado": "Tenho origem GRU destino REC ida 12/11 e ainda não consultei voos",
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["proxima_acao"], "buscar_voos")
        self.assertIn("probabilidades_estimadas", response.json())

    def test_feedback_vincula_acao_final_a_decisao(self):
        with patch.dict(MODEL_PATHS, {"passagens": self.model_path}):
            decisao = self.client.post("/v1/actions/decide", json={
                "cenario": "passagens", "estado": "O viajante escolheu o voo das 8h",
            }).json()
        response = self.client.post(
            f"/v1/decisions/{decisao['decisao_id']}/feedback",
            json={"acao_final": "pedir_confirmacao", "comentario": "já havia escolha"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["acao_final"], "pedir_confirmacao")

    def test_feedback_de_decisao_inexistente_retorna_404(self):
        response = self.client.post("/v1/decisions/naoexiste/feedback", json={"acao_final": "buscar_voos"})
        self.assertEqual(response.status_code, 404)

    def test_feedback_com_acao_invalida_retorna_422(self):
        response = self.client.post("/v1/decisions/x/feedback", json={"acao_final": "Buscar Voos!"})
        self.assertEqual(response.status_code, 422)

    def test_modelo_ausente_retorna_503(self):
        with patch.dict(MODEL_PATHS, {"passagens": self.model_path.parent / "ausente.joblib"}):
            response = self.client.post("/v1/actions/decide", json={
                "cenario": "passagens", "estado": "Preciso buscar uma passagem",
            })
        self.assertEqual(response.status_code, 503)

    def test_limiar_invalido_retorna_422(self):
        response = self.client.post("/v1/actions/decide", json={
            "cenario": "passagens", "estado": "Teste", "limiar": 1.5,
        })
        self.assertEqual(response.status_code, 422)

    def test_avaliacao_de_voos(self):
        cenario = json.loads((ROOT / "dados" / "viagem_exemplo.json").read_text(encoding="utf-8"))
        response = self.client.post("/v1/flights/evaluate", json=cenario)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["recomendacao"], "voo_a")
        self.assertEqual(response.json()["descartadas"][0]["id"], "voo_b")

    def test_pede_dado_ausente(self):
        response = self.client.post("/v1/flights/evaluate", json={
            "pedido": {"origem": "GRU", "destino": "REC"},
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["acao"], "perguntar_dados")
        self.assertIn("data_ida", response.json()["campos_faltantes"])

    def test_oferta_invalida_retorna_422(self):
        cenario = json.loads((ROOT / "dados" / "viagem_exemplo.json").read_text(encoding="utf-8"))
        cenario["ofertas"][0]["preco_total"] = -1
        response = self.client.post("/v1/flights/evaluate", json=cenario)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
