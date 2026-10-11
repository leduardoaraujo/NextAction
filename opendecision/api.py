"""API REST para decisões de ação e avaliação de passagens."""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .api_schemas import (
    ActionRequest, ActionResponse, FeedbackRequest, FeedbackResponse, FlightRequest, FlightResponse,
)
from .modelo import decidir
from .passagens import avaliar_passagens
from .registro import registrar_decisao, registrar_feedback


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATHS = {
    "pesquisa": ROOT / "modelos" / "decisor.joblib",
    "passagens": ROOT / "modelos" / "passagens.joblib",
    "agente": ROOT / "modelos" / "agente.joblib",
}
# Defina OPENDECISION_REGISTRO="" para não guardar os estados recebidos.
_registro = os.environ.get("OPENDECISION_REGISTRO", str(ROOT / "registros" / "decisoes.jsonl"))
REGISTRO = Path(_registro) if _registro else None

app = FastAPI(title="OpenDecision API", version="0.1.0")


@app.get("/health", tags=["servico"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models", tags=["servico"])
def models() -> dict[str, dict[str, bool]]:
    return {"modelos": {nome: caminho.is_file() for nome, caminho in MODEL_PATHS.items()}}


@app.post("/v1/actions/decide", response_model=ActionResponse, tags=["decisoes"])
def decide_action(requisicao: ActionRequest) -> dict:
    caminho = MODEL_PATHS[requisicao.cenario]
    if not caminho.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Modelo '{requisicao.cenario}' indisponível. Treine o cenário antes de consultar.",
        )
    try:
        decisao = decidir(caminho, requisicao.estado, requisicao.limiar)
    except ValueError as erro:
        raise HTTPException(status_code=422, detail=str(erro)) from erro
    if REGISTRO is not None:
        decisao["decisao_id"] = registrar_decisao(
            REGISTRO, requisicao.cenario, requisicao.estado, decisao
        )
    return decisao


@app.post(
    "/v1/decisions/{decisao_id}/feedback", response_model=FeedbackResponse, tags=["decisoes"],
)
def decision_feedback(decisao_id: str, requisicao: FeedbackRequest) -> dict:
    """Informa a ação que deveria ter sido tomada; vira exemplo rotulado para o próximo treino."""
    if REGISTRO is None:
        raise HTTPException(status_code=409, detail="O registro de decisões está desativado.")
    try:
        nota = registrar_feedback(REGISTRO, decisao_id, **requisicao.model_dump())
    except KeyError as erro:
        raise HTTPException(status_code=404, detail="Decisão não encontrada.") from erro
    return {"decisao_id": decisao_id, "acao_final": nota["acao_final"], "registrado_em": nota["em"]}


@app.post(
    "/v1/flights/evaluate", response_model=FlightResponse,
    response_model_exclude_none=True, tags=["decisoes"],
)
def evaluate_flights(requisicao: FlightRequest) -> dict:
    try:
        return avaliar_passagens(requisicao.model_dump(exclude_none=True))
    except ValueError as erro:
        raise HTTPException(status_code=422, detail=str(erro)) from erro
