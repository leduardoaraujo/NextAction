"""Interface de linha de comando do OpenDecision."""

import argparse
import json
from pathlib import Path

from .avaliacao import avaliar
from .hub import CENARIOS, exportar
from .modelo import decidir, treinar
from .registro import exportar_rotulos
from .sgd import ESQUEMAS, converter
from .passagens import avaliar_passagens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    comandos = parser.add_subparsers(dest="comando", required=True)
    comando_treino = comandos.add_parser("treinar", help="Treina a partir de um CSV rotulado")
    comando_treino.add_argument("--dados", type=Path, default=Path("dados/exemplos.csv"))
    comando_treino.add_argument("--modelo", type=Path, default=Path("modelos/decisor.joblib"))
    comando_treino.add_argument("--min-df", type=int, default=1, help="Descarta termos mais raros que isso")
    comando_decisao = comandos.add_parser("decidir", help="Escolhe a próxima ação")
    comando_decisao.add_argument("estado", help="Texto que descreve a situação atual")
    comando_decisao.add_argument("--modelo", type=Path, default=Path("modelos/decisor.joblib"))
    comando_decisao.add_argument("--limiar", type=float, default=0.70)
    comando_passagens = comandos.add_parser(
        "avaliar-passagens", help="Filtra e ordena ofertas de uma viagem descrita em JSON"
    )
    comando_passagens.add_argument("cenario", type=Path)
    comando_hf = comandos.add_parser(
        "exportar-hf", help="Gera pasta com modelo skops e model card para o Hugging Face"
    )
    comando_hf.add_argument("cenario", choices=sorted(CENARIOS))
    comando_hf.add_argument("--destino", type=Path, help="Padrão: hf/<cenario>")
    comando_hf.add_argument("--repo", help="Repositório no Hub, por exemplo usuario/opendecision-passagens")
    comando_hf.add_argument("--licenca", help="Identificador de licença do Hub, por exemplo mit")
    comando_avaliar = comandos.add_parser(
        "avaliar", help="Mede erros por ação, calibração e efeito do limiar de revisão"
    )
    comando_avaliar.add_argument("--dados", type=Path, default=Path("dados/exemplos.csv"))
    comando_avaliar.add_argument("--teste", type=Path, help="CSV de teste separado do treino")
    comando_avaliar.add_argument(
        "--fracao-final", type=float, help="Usa as últimas linhas do CSV como teste (ordem cronológica)"
    )
    comando_avaliar.add_argument("--min-df", type=int, default=1, help="Descarta termos mais raros que isso")
    comando_rotulos = comandos.add_parser(
        "exportar-rotulos", help="Converte decisões com feedback em CSV de treino"
    )
    comando_rotulos.add_argument("cenario", choices=sorted(CENARIOS))
    comando_rotulos.add_argument("--registro", type=Path, default=Path("registros/decisoes.jsonl"))
    comando_rotulos.add_argument("--destino", type=Path, help="Padrão: dados/rotulados_<cenario>.csv")
    comando_sgd = comandos.add_parser(
        "importar-sgd", help="Converte um split do Schema-Guided Dialogue em CSV estado,acao"
    )
    comando_sgd.add_argument("split", type=Path, help="Pasta train, dev ou test do SGD")
    comando_sgd.add_argument("--destino", type=Path, required=True)
    comando_sgd.add_argument("--esquema", choices=ESQUEMAS, default="generico")
    comando_sgd.add_argument("--contexto", type=int, default=3, help="Turnos anteriores incluídos")
    comando_sgd.add_argument("--sem-estado", action="store_true", help="Usa só o texto da conversa")
    argumentos = parser.parse_args()
    try:
        if argumentos.comando == "treinar":
            resultado = treinar(argumentos.dados, argumentos.modelo, argumentos.min_df)
        elif argumentos.comando == "decidir":
            resultado = decidir(argumentos.modelo, argumentos.estado, argumentos.limiar)
        elif argumentos.comando == "avaliar":
            resultado = avaliar(
                argumentos.dados, argumentos.teste, argumentos.fracao_final, argumentos.min_df
            )
        elif argumentos.comando == "exportar-rotulos":
            destino = argumentos.destino or Path("dados") / f"rotulados_{argumentos.cenario}.csv"
            resultado = exportar_rotulos(argumentos.registro, destino, argumentos.cenario)
        elif argumentos.comando == "importar-sgd":
            resultado = converter(
                argumentos.split, argumentos.destino, argumentos.esquema,
                argumentos.contexto, not argumentos.sem_estado,
            )
        elif argumentos.comando == "exportar-hf":
            destino = argumentos.destino or Path("hf") / argumentos.cenario
            resultado = exportar(argumentos.cenario, destino, argumentos.repo, argumentos.licenca)
        else:
            cenario = json.loads(argumentos.cenario.read_text(encoding="utf-8"))
            resultado = avaliar_passagens(cenario)
    except (ValueError, FileNotFoundError) as erro:
        parser.error(str(erro))
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
