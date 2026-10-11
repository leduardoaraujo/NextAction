"""Rule-based filtering and ranking of flight offers.

Works on the English field names; ``schemas.FlightRequest`` also accepts the Portuguese ones
and produces the dict this module expects.
"""

from datetime import date
from math import isfinite

from .i18n import LocalizedError, t

DEFAULT_WEIGHTS = {"price": 0.5, "duration": 0.3, "stops": 0.2}
REQUIRED_FIELDS = ("origin", "destination", "departure_date", "passengers")


def _number(value, field: str, *, minimum: float = 0, integer: bool = False) -> float:
    types = (int,) if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, types) or not isfinite(float(value)):
        raise LocalizedError("number_invalid", field=field)
    if value < minimum:
        raise LocalizedError("number_minimum", field=field, minimum=minimum)
    return float(value)


def _date(value, field: str) -> date:
    if not isinstance(value, str):
        raise LocalizedError("date_format", field=field)
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise LocalizedError("date_format", field=field) from error


def rank_flights(trip: dict, lang: str | None = None) -> dict:
    """Choose the next step and rank eligible offers by explicit criteria.

    The score is a relative utility, not a probability or an ML prediction.
    """
    if not isinstance(trip, dict) or not isinstance(trip.get("request"), dict):
        raise LocalizedError("request_required")
    request = trip["request"]
    missing = [field for field in REQUIRED_FIELDS if request.get(field) in (None, "")]
    if missing:
        return {"action": "ask", "missing_fields": missing}

    for field in ("origin", "destination"):
        if not isinstance(request[field], str) or not request[field].strip():
            raise LocalizedError("text_required", field=field)
    if request["origin"].strip().upper() == request["destination"].strip().upper():
        raise LocalizedError("same_origin_destination")
    departure = _date(request["departure_date"], "departure_date")
    if request.get("return_date") is not None and _date(request["return_date"], "return_date") < departure:
        raise LocalizedError("return_before_departure")
    _number(request["passengers"], "passengers", minimum=1, integer=True)

    budget = request.get("max_budget")
    if budget is not None:
        _number(budget, "max_budget", minimum=0.01)
    max_stops = request.get("max_stops")
    if max_stops is not None:
        _number(max_stops, "max_stops", integer=True)
    baggage = request.get("baggage_required", False)
    if not isinstance(baggage, bool):
        raise LocalizedError("boolean_required", field="baggage_required")

    weights = request.get("weights", DEFAULT_WEIGHTS)
    if not isinstance(weights, dict) or set(weights) != set(DEFAULT_WEIGHTS):
        raise LocalizedError("weights_keys")
    weights = {key: _number(weights[key], f"weights.{key}") for key in DEFAULT_WEIGHTS}
    total = sum(weights.values())
    if total == 0:
        raise LocalizedError("weights_zero")
    weights = {key: value / total for key, value in weights.items()}

    offers = trip.get("offers", [])
    if not isinstance(offers, list):
        raise LocalizedError("offers_list")
    if not offers:
        return {"action": "search", "criteria": request}

    eligible, discarded, ids = [], [], set()
    for offer in offers:
        if not isinstance(offer, dict) or not isinstance(offer.get("id"), str) or not offer["id"].strip():
            raise LocalizedError("offer_id_required")
        if offer["id"] in ids:
            raise LocalizedError("offer_id_duplicate", id=offer["id"])
        ids.add(offer["id"])
        for field in ("total_price", "duration_min"):
            _number(offer.get(field), field, minimum=0.01)
        _number(offer.get("stops"), "stops", integer=True)
        if not isinstance(offer.get("baggage_included"), bool):
            raise LocalizedError("boolean_required", field="baggage_included")

        reasons = []
        if budget is not None and offer["total_price"] > budget:
            reasons.append("over_budget")
        if max_stops is not None and offer["stops"] > max_stops:
            reasons.append("too_many_stops")
        if baggage and not offer["baggage_included"]:
            reasons.append("missing_required_baggage")
        if reasons:
            discarded.append({"id": offer["id"], "reasons": reasons})
        else:
            eligible.append(offer)

    if not eligible:
        return {"action": "revise_criteria", "discarded": discarded}

    lowest_price = min(offer["total_price"] for offer in eligible)
    shortest = min(offer["duration_min"] for offer in eligible)
    scored = []
    for offer in eligible:
        components = {
            "price": lowest_price / offer["total_price"],
            "duration": shortest / offer["duration_min"],
            "stops": 1 / (1 + offer["stops"]),
        }
        score = sum(weights[key] * value for key, value in components.items())
        scored.append((score, offer, components))
    scored.sort(key=lambda item: (-item[0], item[1]["total_price"], item[1]["duration_min"], item[1]["id"]))
    ranking = [
        {**offer, "score": round(score, 4),
         "components": {key: round(value, 4) for key, value in components.items()}}
        for score, offer, components in scored
    ]

    chosen = trip.get("selected_offer")
    if chosen is not None:
        if chosen not in ids:
            raise LocalizedError("selected_offer_unknown")
        selected = next((offer for offer in ranking if offer["id"] == chosen), None)
        if selected is None:
            return {"action": "revise_criteria", "selected_offer": chosen, "discarded": discarded}
        return {"action": "confirm", "selected_offer": selected, "notice": t("confirm_notice", lang)}

    return {"action": "present_options", "recommendation": ranking[0]["id"],
            "ranking": ranking, "discarded": discarded,
            "applied_weights": {key: round(value, 4) for key, value in weights.items()}}
