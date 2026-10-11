"""Relatório de avaliação do classificador."""

import unittest
from pathlib import Path

from opendecision.avaliacao import avaliar, metricas


DADOS = Path(__file__).resolve().parents[1] / "dados" / "passagens.csv"


class AvaliacaoTest(unittest.TestCase):
    def test_validacao_cruzada_avalia_todos_os_exemplos(self):
        resultado = avaliar(DADOS)
        self.assertEqual(resultado["exemplos_avaliados"], 48)
        self.assertEqual(sum(a["suporte"] for a in resultado["por_acao"].values()), 48)
        self.assertIn("aviso", resultado)

    def test_fracao_final_testa_as_ultimas_linhas(self):
        resultado = avaliar(DADOS, fracao_final=0.25)
        self.assertEqual(resultado["exemplos_avaliados"], 12)
        self.assertEqual(resultado["exemplos_de_treino"], 36)

    def test_limiar_separa_automatizadas_de_revisao(self):
        resultado = metricas(
            ["a", "b", "a"], [[0.9, 0.1], [0.6, 0.4], [0.2, 0.8]], ["a", "b"], majoritaria="a"
        )
        no_limiar_07 = next(l for l in resultado["limiares"] if l["limiar"] == 0.7)
        self.assertEqual(no_limiar_07["automatizadas"], round(2 / 3, 3))
        self.assertEqual(no_limiar_07["erros_automatizados"], 1)
        self.assertEqual(no_limiar_07["enviadas_para_revisao"], 1)
        self.assertEqual(resultado["matriz_confusao"]["b"], {"a": 1})

    def test_rejeita_dois_modos_de_teste(self):
        with self.assertRaisesRegex(ValueError, "não os dois"):
            avaliar(DADOS, teste=DADOS, fracao_final=0.2)


if __name__ == "__main__":
    unittest.main()
