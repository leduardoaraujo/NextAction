"""Interface de linha de comando do OpenDecision."""

import argparse
import json
from pathlib import Path

from .modelo import decidir, treinar
from .passagens import avaliar_passagens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    comandos = parser.add_subparsers(dest="comando", required=True)
    comando_treino = comandos.add_parser("treinar", help="Treina a partir de um CSV rotulado")
    comando_treino.add_argument("--dados", type=Path, default=Path("dados/exemplos.csv"))
    comando_treino.add_argument("--modelo", type=Path, default=Path("modelos/decisor.joblib"))
    comando_decisao = comandos.add_parser("decidir", help="Escolhe a próxima ação")
    comando_decisao.add_argument("estado", help="Texto que descreve a situação atual")
    comando_decisao.add_argument("--modelo", type=Path, default=Path("modelos/decisor.joblib"))
    comando_decisao.add_argument("--limiar", type=float, default=0.70)
    comando_passagens = comandos.add_parser(
        "avaliar-passagens", help="Filtra e ordena ofertas de uma viagem descrita em JSON"
    )
    comando_passagens.add_argument("cenario", type=Path)
    argumentos = parser.parse_args()
    try:
        if argumentos.comando == "treinar":
            resultado = treinar(argumentos.dados, argumentos.modelo)
        elif argumentos.comando == "decidir":
            resultado = decidir(argumentos.modelo, argumentos.estado, argumentos.limiar)
        else:
            cenario = json.loads(argumentos.cenario.read_text(encoding="utf-8"))
            resultado = avaliar_passagens(cenario)
    except (ValueError, FileNotFoundError) as erro:
        parser.error(str(erro))
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
