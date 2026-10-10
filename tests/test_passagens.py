"""Contratos do avaliador de ofertas fictícias."""

import json
import unittest
from copy import deepcopy
from pathlib import Path

from opendecision.passagens import avaliar_passagens


EXEMPLO = Path(__file__).resolve().parents[1] / "dados" / "viagem_exemplo.json"


class AvaliarPassagensTest(unittest.TestCase):
    def setUp(self):
        self.cenario = json.loads(EXEMPLO.read_text(encoding="utf-8"))

    def test_pede_campos_obrigatorios_antes_de_buscar(self):
        del self.cenario["pedido"]["data_ida"]
        resposta = avaliar_passagens(self.cenario)
        self.assertEqual(resposta, {"acao": "perguntar_dados", "campos_faltantes": ["data_ida"]})

    def test_busca_quando_ainda_nao_ha_ofertas(self):
        self.cenario["ofertas"] = []
        self.assertEqual(avaliar_passagens(self.cenario)["acao"], "buscar_voos")

    def test_filtra_e_explica_o_ranking(self):
        resposta = avaliar_passagens(self.cenario)
        self.assertEqual(resposta["acao"], "apresentar_opcoes")
        self.assertEqual(resposta["recomendacao"], "voo_a")
        self.assertEqual([oferta["id"] for oferta in resposta["ranking"]], ["voo_a", "voo_c"])
        self.assertEqual(resposta["descartadas"], [
            {"id": "voo_b", "motivos": ["sem_bagagem_exigida"]}
        ])

    def test_escolha_pede_confirmacao(self):
        self.cenario["oferta_escolhida"] = "voo_a"
        resposta = avaliar_passagens(self.cenario)
        self.assertEqual(resposta["acao"], "pedir_confirmacao")
        self.assertEqual(resposta["oferta_escolhida"]["id"], "voo_a")

    def test_nenhuma_oferta_elegivel_pede_revisao(self):
        self.cenario["pedido"]["orcamento_max"] = 500
        self.assertEqual(avaliar_passagens(self.cenario)["acao"], "rever_criterios")

    def test_rejeita_preco_invalido(self):
        cenario = deepcopy(self.cenario)
        cenario["ofertas"][0]["preco_total"] = -1
        with self.assertRaisesRegex(ValueError, "preco_total"):
            avaliar_passagens(cenario)

    def test_rejeita_volta_antes_da_ida(self):
        self.cenario["pedido"]["data_volta"] = "2026-11-01"
        with self.assertRaisesRegex(ValueError, "data_volta"):
            avaliar_passagens(self.cenario)


if __name__ == "__main__":
    unittest.main()
