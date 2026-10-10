"""API REST para decisões de ação e avaliação de passagens."""

from pathlib import Path

from fastapi import FastAPI, HTTPException

from .api_schemas import ActionRequest, ActionResponse, FlightRequest, FlightResponse
from .modelo import decidir
from .passagens import avaliar_passagens


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATHS = {
    "pesquisa": ROOT / "modelos" / "decisor.joblib",
    "passagens": ROOT / "modelos" / "passagens.joblib",
}

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
        return decidir(caminho, requisicao.estado, requisicao.limiar)
    except ValueError as erro:
        raise HTTPException(status_code=422, detail=str(erro)) from erro


@app.post(
    "/v1/flights/evaluate", response_model=FlightResponse,
    response_model_exclude_none=True, tags=["decisoes"],
)
def evaluate_flights(requisicao: FlightRequest) -> dict:
    try:
        return avaliar_passagens(requisicao.model_dump(exclude_none=True))
    except ValueError as erro:
        raise HTTPException(status_code=422, detail=str(erro)) from erro
