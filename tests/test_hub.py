"""Exportação do classificador para o Hugging Face Hub."""

import tempfile
import unittest
from pathlib import Path

import skops.io as sio

from opendecision.hub import exportar


class ExportarTest(unittest.TestCase):
    def test_gera_modelo_skops_carregavel_e_model_card(self):
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta) / "passagens"
            resultado = exportar("passagens", destino, repo="usuario/teste", licenca="mit")

            self.assertEqual(resultado["arquivos"], ["README.md", "dados_treino.csv", "model.skops"])
            modelo = sio.load(destino / "model.skops")
            estado = "Tenho origem GRU destino REC ida 12/11 e ainda não consultei voos"
            self.assertEqual(modelo.predict([estado])[0], "buscar_voos")

            card = (destino / "README.md").read_text(encoding="utf-8")
            self.assertTrue(card.startswith("---\nlanguage: pt\n"))
            self.assertIn("license: mit", card)
            self.assertIn('hf_hub_download("usuario/teste", "model.skops")', card)

    def test_rejeita_cenario_desconhecido(self):
        with tempfile.TemporaryDirectory() as pasta, self.assertRaisesRegex(ValueError, "Cenário"):
            exportar("hoteis", Path(pasta))


if __name__ == "__main__":
    unittest.main()
