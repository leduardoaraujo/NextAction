"""REST API for next-action decisions, feedback and flight offer ranking."""

import os
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse

from .decision_log import record_decision, record_feedback
from .flights import rank_flights
from .i18n import LocalizedError, language_from_header, t
from .model import decide
from .progress import describe, list_jobs
from .scenarios import ROOT, SCENARIOS
from .schemas import (
    ActionRequest, ActionResponse, FeedbackRequest, FeedbackResponse, FlightRequest, FlightResponse,
)

MODEL_PATHS = {name: scenario.model for name, scenario in SCENARIOS.items()}
# Set NEXTACTION_LOG="" to stop storing the states received (NEXTACTION_REGISTRO also works).
_log = os.environ.get("NEXTACTION_LOG", os.environ.get("NEXTACTION_REGISTRO", str(ROOT / "logs" / "decisions.jsonl")))
DECISION_LOG = Path(_log) if _log else None

app = FastAPI(title="NextAction API", version="0.2.0")


def language(accept_language: Annotated[str | None, Header()] = None) -> str:
    """Messages follow Accept-Language (en or pt); field names and actions are always English."""
    return language_from_header(accept_language)


Language = Annotated[str, Depends(language)]


@app.get("/health", tags=["service"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models", tags=["service"])
def models() -> dict[str, dict[str, bool]]:
    return {"models": {name: path.is_file() for name, path in MODEL_PATHS.items()}}


@app.get("/v1/jobs", tags=["service"])
def jobs(lang: Language) -> dict[str, list[dict]]:
    """Training, evaluation and import jobs with their progress, most recent first."""
    return {"jobs": [describe(record, lang) for record in list_jobs()]}


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> str:
    return (Path(__file__).parent / "static" / "dashboard.html").read_text(encoding="utf-8")


@app.post("/v1/actions/decide", response_model=ActionResponse, tags=["decisions"])
def decide_action(request: ActionRequest, lang: Language) -> dict:
    path = MODEL_PATHS[request.scenario]
    if not path.is_file():
        raise HTTPException(status_code=503, detail=t("model_unavailable", lang, scenario=request.scenario))
    try:
        decision = decide(path, request.state, request.threshold, SCENARIOS[request.scenario].policy)
    except LocalizedError as error:
        raise HTTPException(status_code=422, detail=error.message(lang)) from error
    if DECISION_LOG is not None:
        decision["decision_id"] = record_decision(DECISION_LOG, request.scenario, request.state, decision)
    return decision


@app.post("/v1/decisions/{decision_id}/feedback", response_model=FeedbackResponse, tags=["decisions"])
def decision_feedback(decision_id: str, request: FeedbackRequest, lang: Language) -> dict:
    """Report the action that should have been taken; it becomes a labeled training example."""
    if DECISION_LOG is None:
        raise HTTPException(status_code=409, detail=t("log_disabled", lang))
    try:
        note = record_feedback(DECISION_LOG, decision_id, **request.model_dump())
    except KeyError as error:
        raise HTTPException(status_code=404, detail=t("decision_not_found", lang)) from error
    return {"decision_id": decision_id, "final_action": note["final_action"], "recorded_at": note["at"]}


@app.post(
    "/v1/flights/evaluate", response_model=FlightResponse,
    response_model_exclude_none=True, tags=["decisions"],
)
def evaluate_flights(request: FlightRequest, lang: Language) -> dict:
    try:
        return rank_flights(request.model_dump(exclude_none=True), lang)
    except LocalizedError as error:
        raise HTTPException(status_code=422, detail=error.message(lang)) from error
