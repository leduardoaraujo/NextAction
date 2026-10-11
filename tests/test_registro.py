"""Registro de decisões e exportação de rótulos."""

import csv
import tempfile
import unittest
from pathlib import Path

from opendecision.registro import exportar_rotulos, ler, registrar_decisao, registrar_feedback


def _decisao(acao):
    return {"proxima_acao": acao, "probabilidades_estimadas": {acao: 0.6}, "pedir_revisao": True}


class RegistroTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.arquivo = Path(self.temp.name) / "decisoes.jsonl"

    def tearDown(self):
        self.temp.cleanup()

    def test_exporta_so_decisoes_com_feedback_e_usa_o_ultimo(self):
        primeira = registrar_decisao(self.arquivo, "passagens", "falta a data", _decisao("buscar_voos"))
        registrar_decisao(self.arquivo, "passagens", "sem feedback", _decisao("buscar_voos"))
        registrar_decisao(self.arquivo, "pesquisa", "outro cenário", _decisao("pesquisar"))
        registrar_feedback(self.arquivo, primeira, "buscar_voos")
        registrar_feedback(self.arquivo, primeira, "perguntar_dados", resultado="corrigido")

        destino = Path(self.temp.name) / "rotulos.csv"
        resumo = exportar_rotulos(self.arquivo, destino, "passagens")

        self.assertEqual(resumo["decisoes"], 2)
        self.assertEqual(resumo["rotuladas"], 1)
        self.assertEqual(resumo["sugestoes_corrigidas"], 1)
        with destino.open(encoding="utf-8") as entrada:
            linhas = list(csv.DictReader(entrada))
        self.assertEqual([(l["estado"], l["acao"]) for l in linhas], [("falta a data", "perguntar_dados")])

    def test_feedback_exige_decisao_registrada(self):
        with self.assertRaises(KeyError):
            registrar_feedback(self.arquivo, "inexistente", "buscar_voos")
        self.assertEqual(ler(self.arquivo), [])


if __name__ == "__main__":
    unittest.main()
