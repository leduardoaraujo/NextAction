"""Conversão do Schema-Guided Dialogue em exemplos estado,acao."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from opendecision.sgd import converter


def _usuario(texto, preenchidos):
    estado = {"active_intent": "SearchOnewayFlight", "slot_values": {c: ["x"] for c in preenchidos},
              "requested_slots": []}
    return {"speaker": "USER", "utterance": texto,
            "frames": [{"service": "Flights_1", "actions": [{"act": "INFORM"}], "state": estado}]}


def _sistema(texto, atos, chamada=False):
    frame = {"service": "Flights_1", "actions": [{"act": ato} for ato in atos]}
    if chamada:
        frame["service_call"] = {"method": "SearchOnewayFlight"}
    return {"speaker": "SYSTEM", "utterance": texto, "frames": [frame]}


DIALOGO = {"dialogue_id": "1_00000", "services": ["Flights_1"], "turns": [
    _usuario("I need a flight to Seattle", ["destination"]),
    _sistema("Where from?", ["REQUEST"]),
    _usuario("From New York", ["destination", "origin"]),
    _sistema("I found 3 flights", ["INFORM_COUNT", "OFFER"], chamada=True),
    _usuario("Book it", ["destination", "origin"]),
    _sistema("Please confirm", ["CONFIRM"]),
    _usuario("Yes", ["destination", "origin"]),
    _sistema("Booked", ["NOTIFY_SUCCESS"], chamada=True),
]}
SCHEMA = [{"service_name": "Flights_1", "intents": [
    {"name": "SearchOnewayFlight", "required_slots": ["origin", "destination"]},
]}]


class ConverterSgdTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.pasta = Path(self.temp.name)
        (self.pasta / "dialogues_001.json").write_text(json.dumps([DIALOGO]), encoding="utf-8")
        (self.pasta / "schema.json").write_text(json.dumps(SCHEMA), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def _linhas(self, **opcoes):
        destino = self.pasta / "saida.csv"
        converter(self.pasta, destino, **opcoes)
        with destino.open(encoding="utf-8") as entrada:
            return list(csv.DictReader(entrada))

    def test_mapeia_atos_e_chamadas_de_servico(self):
        linhas = self._linhas()
        self.assertEqual([l["acao"] for l in linhas], ["perguntar", "buscar", "confirmar", "executar"])

    def test_estado_mostra_campos_que_faltam(self):
        primeira, segunda = self._linhas()[:2]
        self.assertIn("[faltando origin]", primeira["estado"])
        self.assertIn("[faltando nenhum]", segunda["estado"])
        self.assertTrue(primeira["estado"].endswith("usuario: I need a flight to Seattle"))

    def test_sem_estado_usa_so_a_conversa(self):
        self.assertNotIn("[intencao", self._linhas(com_estado=False)[0]["estado"])

    def test_esquema_passagens_renomeia_e_descarta_execucao(self):
        acoes = [l["acao"] for l in self._linhas(esquema="passagens")]
        self.assertEqual(acoes, ["perguntar_dados", "buscar_voos", "pedir_confirmacao"])


if __name__ == "__main__":
    unittest.main()
