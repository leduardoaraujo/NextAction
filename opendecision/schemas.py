"""Public JSON contracts. Requests accept English or Portuguese field names; responses are English."""

from typing import Annotated, Literal

from pydantic import AfterValidator, AliasChoices, BaseModel, BeforeValidator, ConfigDict, Field

from .vocabulary import normalize_action, normalize_scenario


def field(english: str, portuguese: str, **options):
    return Field(validation_alias=AliasChoices(english, portuguese), **options)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


ScenarioName = Annotated[Literal["agent", "flights", "research"], BeforeValidator(normalize_scenario)]


class ActionRequest(StrictModel):
    scenario: ScenarioName = field("scenario", "cenario")
    state: str = field("state", "estado", min_length=1)
    threshold: float = field("threshold", "limiar", default=0.70, ge=0, le=1)


class ActionResponse(StrictModel):
    type: Literal["choice"]
    action: str
    probabilities: dict[str, float]
    needs_review: bool
    decision_id: str | None = None


class FeedbackRequest(StrictModel):
    final_action: Annotated[str, AfterValidator(normalize_action)] = field(
        "final_action", "acao_final", pattern=r"^[a-z0-9_]{1,64}$"
    )
    outcome: str | None = field("outcome", "resultado", default=None, max_length=500)
    comment: str | None = field("comment", "comentario", default=None, max_length=2000)


class FeedbackResponse(StrictModel):
    decision_id: str
    final_action: str
    recorded_at: str


class Weights(StrictModel):
    price: float = field("price", "preco", ge=0)
    duration: float = field("duration", "duracao", ge=0)
    stops: float = field("stops", "escalas", ge=0)


class TripRequest(StrictModel):
    origin: str | None = field("origin", "origem", default=None)
    destination: str | None = field("destination", "destino", default=None)
    departure_date: str | None = field("departure_date", "data_ida", default=None)
    return_date: str | None = field("return_date", "data_volta", default=None)
    passengers: int | None = field("passengers", "passageiros", default=None, ge=1)
    max_budget: float | None = field("max_budget", "orcamento_max", default=None, gt=0)
    max_stops: int | None = field("max_stops", "max_escalas", default=None, ge=0)
    baggage_required: bool = field("baggage_required", "bagagem_obrigatoria", default=False)
    weights: Weights | None = field("weights", "pesos", default=None)


class FlightOffer(StrictModel):
    id: str = Field(min_length=1)
    total_price: float = field("total_price", "preco_total", gt=0)
    duration_min: float = field("duration_min", "duracao_min", gt=0)
    stops: int = field("stops", "escalas", ge=0)
    baggage_included: bool = field("baggage_included", "bagagem_incluida")


class FlightRequest(StrictModel):
    request: TripRequest = field("request", "pedido")
    offers: list[FlightOffer] = field("offers", "ofertas", default_factory=list)
    selected_offer: str | None = field("selected_offer", "oferta_escolhida", default=None)


class ScoredOffer(FlightOffer):
    score: float
    components: dict[str, float]


class DiscardedOffer(StrictModel):
    id: str
    reasons: list[str]


class FlightResponse(StrictModel):
    action: Literal["ask", "search", "present_options", "revise_criteria", "confirm"]
    missing_fields: list[str] | None = None
    criteria: TripRequest | None = None
    recommendation: str | None = None
    ranking: list[ScoredOffer] | None = None
    discarded: list[DiscardedOffer] | None = None
    applied_weights: dict[str, float] | None = None
    selected_offer: str | ScoredOffer | None = None
    notice: str | None = None
