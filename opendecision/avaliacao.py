"""Avaliação do classificador: erros por ação, calibração e escolha do limiar de revisão."""

from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .dados import carregar_exemplos
from .modelo import novo_modelo


LIMIARES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def avaliar(dados: Path, teste: Path | None = None, fracao_final: float | None = None) -> dict:
    """Avalia por validação cruzada, por um CSV de teste separado ou pelos exemplos mais recentes.

    ``fracao_final`` usa as últimas linhas do CSV como teste, sem embaralhar: com dados
    exportados do registro (em ordem cronológica) isso mede o desempenho em casos futuros.
    """
    estados, acoes = carregar_exemplos(dados)
    if teste is not None and fracao_final is not None:
        raise ValueError("Use --teste ou --fracao-final, não os dois.")
    if teste is not None:
        teste_x, teste_y = carregar_exemplos(teste)
        return _avaliar_divisao(estados, acoes, teste_x, teste_y, f"teste separado: {teste}")
    if fracao_final is not None:
        if not 0 < fracao_final < 1:
            raise ValueError("A fração final precisa estar entre 0 e 1.")
        corte = len(estados) - max(1, round(len(estados) * fracao_final))
        return _avaliar_divisao(
            estados[:corte], acoes[:corte], estados[corte:], acoes[corte:],
            f"últimos {len(estados) - corte} exemplos do arquivo",
        )

    dobras = min(5, min(Counter(acoes).values()))
    validacao = StratifiedKFold(n_splits=dobras, shuffle=True, random_state=42)
    probabilidades = cross_val_predict(
        novo_modelo(), estados, acoes, cv=validacao, method="predict_proba"
    )
    classes = sorted(set(acoes))
    majoritaria = Counter(acoes).most_common(1)[0][0]
    return {
        "metodo": f"validação cruzada estratificada com {dobras} dobras",
        **metricas(acoes, probabilidades, classes, majoritaria),
    }


def _avaliar_divisao(treino_x, treino_y, teste_x, teste_y, metodo: str) -> dict:
    if len(set(treino_y)) < 2:
        raise ValueError("O treino precisa de pelo menos duas ações.")
    modelo = novo_modelo().fit(treino_x, treino_y)
    probabilidades = modelo.predict_proba(teste_x)
    majoritaria = Counter(treino_y).most_common(1)[0][0]
    resultado = {"metodo": metodo, "exemplos_de_treino": len(treino_x)}
    resultado.update(metricas(teste_y, probabilidades, list(modelo.classes_), majoritaria))
    desconhecidas = sorted(set(teste_y) - set(modelo.classes_))
    if desconhecidas:
        resultado["acoes_sem_exemplo_no_treino"] = desconhecidas
    return resultado


def metricas(reais, probabilidades, classes: list[str], majoritaria: str) -> dict:
    reais = np.asarray(reais)
    probabilidades = np.asarray(probabilidades)
    previstas = np.asarray(classes)[probabilidades.argmax(axis=1)]
    confianca = probabilidades.max(axis=1)
    acertos = previstas == reais
    rotulos = sorted(set(classes) | set(reais))

    precisao, revocacao, f1, suporte = precision_recall_fscore_support(
        reais, previstas, labels=rotulos, zero_division=0
    )
    matriz = confusion_matrix(reais, previstas, labels=rotulos)
    resultado = {
        "exemplos_avaliados": len(reais),
        "acuracia": _r(acertos.mean()),
        "acuracia_classe_majoritaria": _r(np.mean(reais == majoritaria)),
        "por_acao": {
            acao: {"precisao": _r(p), "revocacao": _r(r), "f1": _r(f), "suporte": int(s)}
            for acao, p, r, f, s in zip(rotulos, precisao, revocacao, f1, suporte)
        },
        "matriz_confusao": {
            real: {prevista: int(n) for prevista, n in zip(rotulos, linha) if n}
            for real, linha in zip(rotulos, matriz)
        },
        "erro_calibracao_esperado": _r(_erro_calibracao(confianca, acertos)),
        "log_loss": (
            _r(log_loss(reais, probabilidades, labels=classes))
            if set(reais) <= set(classes) else None
        ),
        "limiares": [_limiar(t, confianca, acertos) for t in LIMIARES],
    }
    if len(reais) < 100:
        resultado["aviso"] = (
            f"Só {len(reais)} exemplos avaliados: as métricas variam muito com poucos casos."
        )
    return resultado


def _limiar(limiar: float, confianca, acertos) -> dict:
    automaticas = confianca >= limiar
    return {
        "limiar": limiar,
        "automatizadas": _r(automaticas.mean()),
        "acuracia_automatizadas": _r(acertos[automaticas].mean()) if automaticas.any() else None,
        "erros_automatizados": int((automaticas & ~acertos).sum()),
        "enviadas_para_revisao": int((~automaticas).sum()),
    }


def _erro_calibracao(confianca, acertos, faixas: int = 10) -> float:
    """Diferença média entre confiança e acurácia em faixas de confiança (ECE)."""
    indices = np.minimum((confianca * faixas).astype(int), faixas - 1)
    total = 0.0
    for faixa in range(faixas):
        na_faixa = indices == faixa
        if na_faixa.any():
            total += na_faixa.mean() * abs(confianca[na_faixa].mean() - acertos[na_faixa].mean())
    return total


def _r(valor: float) -> float:
    return round(float(valor), 3)
