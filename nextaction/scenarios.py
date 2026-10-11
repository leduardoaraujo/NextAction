"""Scenarios are defined by ``configs/<name>.toml``: data, model parameters, tuning grid and policy.

Adding a scenario means adding a TOML file; see ``docs/policy.md`` for every field.
"""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .i18n import LocalizedError
from .policy import Policy, from_dict

ROOT = Path(__file__).resolve().parent.parent
CALIBRATIONS = ("none", "sigmoid", "isotonic")


def configs_dir() -> Path:
    return Path(os.environ.get("NEXTACTION_CONFIGS_DIR", ROOT / "configs"))


@dataclass(frozen=True)
class ModelParams:
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1
    C: float = 1.0
    class_weight: str | None = None
    max_iter: int = 1000
    calibration: str = "none"


@dataclass(frozen=True)
class Scenario:
    name: str
    alias: str | None
    data: Path
    model: Path
    decision: str
    example: str
    data_source: str
    languages: tuple[str, ...]
    config_path: Path
    test: Path | None = None
    validation: Path | None = None
    license: str | None = None
    ship_data: bool = True
    input_format: str = "text"
    dataset_id: str | None = None
    dataset_name: str | None = None
    citation: str | None = None
    params: ModelParams = ModelParams()
    tuning: dict = field(default_factory=dict)
    policy: Policy = Policy()


def model_params(data: dict) -> ModelParams:
    calibration = data.get("calibration", "none")
    if calibration not in CALIBRATIONS:
        raise LocalizedError("config_invalid_value", field="model.calibration", value=calibration,
                             options=", ".join(CALIBRATIONS))
    class_weight = data.get("class_weight", "none")
    return ModelParams(
        ngram_range=tuple(data.get("ngram_range", (1, 2))),
        min_df=int(data.get("min_df", 1)),
        C=float(data.get("C", 1.0)),
        class_weight=None if class_weight in (None, "none") else class_weight,
        max_iter=int(data.get("max_iter", 1000)),
        calibration=calibration,
    )


def load_scenario(path: Path) -> Scenario:
    with path.open("rb") as file:
        config = tomllib.load(file)
    info = config.get("scenario", {})
    try:
        return Scenario(
            name=info["name"], alias=info.get("alias"),
            data=ROOT / info["data"], model=ROOT / info["model"],
            test=ROOT / info["test"] if info.get("test") else None,
            validation=ROOT / info["validation"] if info.get("validation") else None,
            decision=info["decision"], example=info["example"], data_source=info["data_source"],
            languages=tuple(info.get("languages", ("en",))), license=info.get("license"),
            ship_data=bool(info.get("ship_data", True)), config_path=path,
            input_format=info.get("input", "text"), dataset_id=info.get("dataset_id"),
            dataset_name=info.get("dataset_name"), citation=info.get("citation"),
            params=model_params(config.get("model", {})),
            tuning=config.get("tuning", {}),
            policy=from_dict(config.get("policy", {})),
        )
    except KeyError as missing:
        raise LocalizedError("config_missing_field", path=path, field=f"scenario.{missing.args[0]}") from missing


def load_scenarios() -> dict[str, Scenario]:
    return {s.name: s for s in map(load_scenario, sorted(configs_dir().glob("*.toml")))}


SCENARIOS = load_scenarios()
SCENARIO_ALIASES = {s.alias: s.name for s in SCENARIOS.values() if s.alias}


def normalize_scenario(name: str) -> str:
    key = name.strip().lower()
    scenario = SCENARIO_ALIASES.get(key, key)
    if scenario not in SCENARIOS:
        options = ", ".join(f"{s.name} ({s.alias})" if s.alias else s.name for s in SCENARIOS.values())
        raise LocalizedError("unknown_scenario", name=name, options=options)
    return scenario


def get_scenario(name: str) -> Scenario:
    return SCENARIOS[normalize_scenario(name)]
