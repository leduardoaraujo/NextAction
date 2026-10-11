"""Treino e inferência do classificador de ações."""

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from .dados import carregar_exemplos


def novo_modelo():
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode"),
        LogisticRegression(max_iter=1000),
    )


def ajustar(dados: Path):
    """Avalia em uma divisão de teste e devolve o modelo ajustado em todos os exemplos."""
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
    return modelo, {
        "exemplos": len(estados),
        "acoes": sorted(set(acoes)),
        "exemplos_de_teste": len(teste_x),
        "acuracia_no_teste": round(acuracia, 3),
        "aviso": "A base de demonstração é pequena; esta métrica não valida uso real.",
    }


def treinar(dados: Path, destino: Path) -> dict:
    modelo, resumo = ajustar(dados)
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, destino)
    return {"modelo_salvo_em": str(destino), **resumo}


def decidir(arquivo_modelo: Path, estado: str, limiar: float) -> dict:
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
    return {
        "tipo": "escolha",
        "proxima_acao": acao,
        "probabilidades_estimadas": distribuicao,
        "pedir_revisao": distribuicao[acao] < limiar,
    }
