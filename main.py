"""Protótipo de um modelo que escolhe a próxima ação de um agente."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def carregar_exemplos(caminho: Path) -> tuple[list[str], list[str]]:
    with caminho.open(encoding="utf-8-sig", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)
        if not leitor.fieldnames or not {"estado", "acao"}.issubset(leitor.fieldnames):
            raise ValueError("O CSV precisa das colunas 'estado' e 'acao'.")
        linhas = [(linha["estado"].strip(), linha["acao"].strip()) for linha in leitor]

    if not linhas or any(not estado or not acao for estado, acao in linhas):
        raise ValueError("Todos os exemplos precisam de estado e acao preenchidos.")
    estados, acoes = zip(*linhas)
    contagens = Counter(acoes)
    if len(contagens) < 2 or min(contagens.values()) < 4:
        raise ValueError("Use pelo menos duas ações e quatro exemplos por ação.")
    return list(estados), list(acoes)


def novo_modelo():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode"),
        LogisticRegression(max_iter=1000),
    )


def treinar(dados: Path, destino: Path) -> None:
    estados, acoes = carregar_exemplos(dados)
    total_teste = max(len(set(acoes)), round(len(estados) * 0.25))
    treino_x, teste_x, treino_y, teste_y = train_test_split(
        estados, acoes, test_size=total_teste, random_state=42, stratify=acoes
    )
    modelo_avaliacao = novo_modelo()
    modelo_avaliacao.fit(treino_x, treino_y)
    acuracia = accuracy_score(teste_y, modelo_avaliacao.predict(teste_x))

    modelo = novo_modelo()
    modelo.fit(estados, acoes)
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, destino)
    print(json.dumps({
        "modelo_salvo_em": str(destino),
        "exemplos": len(estados),
        "acoes": sorted(set(acoes)),
        "exemplos_de_teste": len(teste_x),
        "acuracia_no_teste": round(acuracia, 3),
        "aviso": "A base de demonstração é pequena; esta métrica não valida uso real.",
    }, ensure_ascii=False, indent=2))


def decidir(arquivo_modelo: Path, estado: str, limiar: float) -> None:
    if not 0 <= limiar <= 1:
        raise ValueError("O limiar precisa estar entre 0 e 1.")
    # Abra somente modelos que você próprio treinou: arquivos joblib podem executar código.
    modelo = joblib.load(arquivo_modelo)
    probabilidades = modelo.predict_proba([estado])[0]
    distribuicao = {
        acao: round(float(probabilidade), 4)
        for acao, probabilidade in zip(modelo.classes_, probabilidades)
    }
    acao = max(distribuicao, key=distribuicao.get)
    print(json.dumps({
        "tipo": "escolha",
        "proxima_acao": acao,
        "probabilidades_estimadas": distribuicao,
        "pedir_revisao": distribuicao[acao] < limiar,
    }, ensure_ascii=False, indent=2))


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
    argumentos = parser.parse_args()
    try:
        if argumentos.comando == "treinar":
            treinar(argumentos.dados, argumentos.modelo)
        else:
            decidir(argumentos.modelo, argumentos.estado, argumentos.limiar)
    except (ValueError, FileNotFoundError) as erro:
        parser.error(str(erro))


if __name__ == "__main__":
    main()
