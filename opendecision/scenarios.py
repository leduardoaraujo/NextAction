"""Built-in scenarios: where their data and models live and how they are described."""

from dataclasses import dataclass
from pathlib import Path

from .vocabulary import normalize_scenario

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXTERNAL = DATA / "external"
MODELS = ROOT / "models"
SGD_CITATION = (
    "[Schema-Guided Dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)"
    " (Rastogi et al., 2020, CC BY-SA 4.0)"
)


@dataclass(frozen=True)
class Scenario:
    name: str
    data: Path
    model: Path
    decision: str
    example: str
    data_source: str
    languages: tuple[str, ...]
    test: Path | None = None
    min_df: int = 1
    license: str | None = None
    ship_data: bool = True


SCENARIOS = {
    "agent": Scenario(
        name="agent",
        data=EXTERNAL / "sgd_generic_train.csv",
        test=EXTERNAL / "sgd_generic_test.csv",
        model=MODELS / "agent.joblib",
        min_df=5,
        languages=("en",),
        license="cc-by-sa-4.0",
        ship_data=False,
        decision="what an assistant should do next in a task-oriented conversation",
        example=(
            "[intent SearchOnewayFlight] [filled destination_airport origin_airport] "
            "[missing departure_date] [requested none] || "
            "user: I need a one way flight from New York to Seattle."
        ),
        data_source=(
            f"assistant turns from the {SGD_CITATION}: conversations between people and a virtual "
            "assistant across 20 domains, written by crowd workers from simulated outlines. "
            "Build the CSV with `opendecision import-sgd`"
        ),
    ),
    "flights": Scenario(
        name="flights",
        data=DATA / "flights.csv",
        model=MODELS / "flights.joblib",
        languages=("pt",),
        decision="the next step of an agent that helps find flight tickets",
        example="Tenho origem GRU destino REC ida 12/11 e ainda não consultei voos",
        data_source="hand-written **synthetic** examples in Portuguese; they do not come from real users",
    ),
    "research": Scenario(
        name="research",
        data=DATA / "research.csv",
        model=MODELS / "research.joblib",
        languages=("pt",),
        decision="the next action of a research agent",
        example="Preciso saber a cotação de hoje e ainda não consultei nenhuma fonte",
        data_source="hand-written **synthetic** examples in Portuguese; they do not come from real users",
    ),
}


def get_scenario(name: str) -> Scenario:
    return SCENARIOS[normalize_scenario(name)]
