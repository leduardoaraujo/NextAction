"""Command line interface. Commands and options accept English names or Portuguese aliases."""

import argparse
import json
from dataclasses import replace
import sys
import time
from pathlib import Path

from pydantic import ValidationError

from .decision_log import export_labels
from .evaluation import evaluate
from .flights import rank_flights
from .hub import export
from .i18n import LocalizedError, resolve_language, t
from .model import decide, train
from .tuning import calibrate_policy, tune
from .progress import Job, describe, list_jobs, set_terminal
from .scenarios import ROOT, SCENARIO_ALIASES, get_scenario
from .schemas import FlightRequest
from .sgd import SCHEMA_ALIASES, SCHEMAS, convert
from .vision import inspect_image
from .vocabulary import normalize_action


def build_parser(lang: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nextaction", description=t("cli_description", lang))
    parser.add_argument("--lang", "--idioma", help=t("cli_lang", lang))
    parser.add_argument("--no-progress", "--sem-progresso", action="store_true", help=t("cli_no_progress", lang))
    commands = parser.add_subparsers(dest="command", required=True)
    scenario_help = t("cli_scenario", lang, options=", ".join(f"{s} ({pt})" for pt, s in SCENARIO_ALIASES.items()))

    def command(name: str, alias: str, help_key: str):
        sub = commands.add_parser(name, aliases=[alias], help=t(help_key, lang))
        sub.set_defaults(command=name)
        return sub

    def scenario_argument(sub):
        sub.add_argument("scenario", type=str, help=scenario_help)

    train_cmd = command("train", "treinar", "cli_train")
    scenario_argument(train_cmd)
    train_cmd.add_argument("--data", "--dados", type=Path, help=t("cli_data", lang))
    train_cmd.add_argument("--model", "--modelo", type=Path, help=t("cli_model", lang))
    train_cmd.add_argument("--min-df", type=int, help=t("cli_min_df", lang))

    decide_cmd = command("decide", "decidir", "cli_decide")
    scenario_argument(decide_cmd)
    decide_cmd.add_argument("state", help=t("cli_state", lang))
    decide_cmd.add_argument("--model", "--modelo", type=Path, help=t("cli_model", lang))
    decide_cmd.add_argument("--threshold", "--limiar", type=float, help=t("cli_threshold", lang))
    decide_cmd.add_argument("--allowed-actions", "--acoes-permitidas", nargs="+",
                            help="actions allowed by the calling application")

    evaluate_cmd = command("evaluate", "avaliar", "cli_evaluate")
    scenario_argument(evaluate_cmd)
    evaluate_cmd.add_argument("--data", "--dados", type=Path, help=t("cli_data", lang))
    evaluate_cmd.add_argument("--test", "--teste", type=Path, help=t("cli_test", lang))
    evaluate_cmd.add_argument(
        "--recent-fraction", "--fracao-final", type=float, help=t("cli_recent_fraction", lang)
    )
    evaluate_cmd.add_argument("--min-df", type=int, help=t("cli_min_df", lang))

    sgd_cmd = command("import-sgd", "importar-sgd", "cli_import_sgd")
    sgd_cmd.add_argument("split", type=Path, help=t("cli_split", lang))
    sgd_cmd.add_argument("--output", "--destino", type=Path, required=True, help=t("cli_output", lang))
    sgd_cmd.add_argument(
        "--schema", "--esquema", choices=SCHEMAS + tuple(SCHEMA_ALIASES), default="generic",
        help=t("cli_schema", lang),
    )
    sgd_cmd.add_argument("--context", "--contexto", type=int, default=3, help=t("cli_context", lang))
    sgd_cmd.add_argument("--no-state", "--sem-estado", action="store_true", help=t("cli_no_state", lang))

    labels_cmd = command("export-labels", "exportar-rotulos", "cli_export_labels")
    scenario_argument(labels_cmd)
    labels_cmd.add_argument(
        "--log", "--registro", type=Path, default=ROOT / "logs" / "decisions.jsonl", help=t("cli_log", lang)
    )
    labels_cmd.add_argument("--output", "--destino", type=Path, help=t("cli_output", lang))

    hf_cmd = command("export-hf", "exportar-hf", "cli_export_hf")
    scenario_argument(hf_cmd)
    hf_cmd.add_argument("--output", "--destino", type=Path, help=t("cli_output", lang))
    hf_cmd.add_argument("--repo", help=t("cli_repo", lang))
    hf_cmd.add_argument("--license", "--licenca", help=t("cli_license", lang))

    flights_cmd = command("rank-flights", "avaliar-passagens", "cli_rank_flights")
    flights_cmd.add_argument("trip", type=Path, help=t("cli_trip", lang))

    image_cmd = commands.add_parser("inspect-image", aliases=["inspecionar-imagem"],
                                    help="detect people, animals and objects in an image")
    image_cmd.set_defaults(command="inspect-image")
    image_cmd.add_argument("image", type=Path)
    image_cmd.add_argument("--output", "--saida", type=Path)
    image_cmd.add_argument("--threshold", "--limiar", type=float, default=0.5)

    tune_cmd = command("tune", "ajustar", "cli_tune")
    scenario_argument(tune_cmd)
    tune_cmd.add_argument("--write", "--gravar", action="store_true", help=t("cli_write", lang))

    calibrate_cmd = command("calibrate-policy", "calibrar-politica", "cli_calibrate")
    scenario_argument(calibrate_cmd)
    calibrate_cmd.add_argument("--write", "--gravar", action="store_true", help=t("cli_write", lang))

    jobs_cmd = command("jobs", "tarefas", "cli_jobs")
    jobs_cmd.add_argument("--watch", "--acompanhar", action="store_true", help=t("cli_watch", lang))
    return parser


LONG_COMMANDS = {"train", "evaluate", "import-sgd", "export-hf", "tune", "calibrate-policy"}


def jobs_table(lang: str):
    from rich.table import Table

    table = Table(box=None, header_style="bold")
    for column in ("id", "job", "status", "stage", "%", "ETA"):
        table.add_column(column)
    records = [describe(record, lang) for record in list_jobs()]
    if not records:
        table.add_row("", t("no_jobs", lang), "", "", "", "")
    for record in records:
        percent = "" if record["percent"] is None else f"{record['percent']:.0f}%"
        eta = "" if not record["eta_s"] else f"{record['eta_s']:.0f}s"
        table.add_row(record["id"], record["name"], record["status_label"], record["stage_label"] or "", percent, eta)
    return table


def show_jobs(watch: bool, lang: str) -> None:
    from rich.console import Console
    from rich.live import Live

    console = Console()
    if not watch:
        console.print(jobs_table(lang))
        return
    with Live(jobs_table(lang), console=console, refresh_per_second=2) as live:
        try:
            while True:
                time.sleep(1)
                live.update(jobs_table(lang))
        except KeyboardInterrupt:
            pass


def run(args: argparse.Namespace, lang: str) -> dict:
    if args.command == "import-sgd":
        return convert(args.split, args.output, args.schema, args.context, not args.no_state)
    if args.command == "rank-flights":
        trip = FlightRequest.model_validate(json.loads(args.trip.read_text(encoding="utf-8")))
        return rank_flights(trip.model_dump(exclude_none=True), lang)
    if args.command == "inspect-image":
        return inspect_image(args.image, args.output, args.threshold)

    scenario = get_scenario(args.scenario)
    params = replace(scenario.params, min_df=args.min_df) if getattr(args, "min_df", None) else scenario.params
    if args.command == "train":
        return train(args.data or scenario.data, args.model or scenario.model, params, lang)
    if args.command == "decide":
        return decide(args.model or scenario.model, args.state, args.threshold, scenario.policy,
                      {normalize_action(action) for action in args.allowed_actions}
                      if args.allowed_actions is not None else None)
    if args.command == "evaluate":
        data = args.data or scenario.data
        test = args.test or (scenario.test if args.data is None and args.recent_fraction is None else None)
        return evaluate(data, test, args.recent_fraction, params, lang, scenario.policy)
    if args.command == "tune":
        return tune(scenario, args.write)
    if args.command == "calibrate-policy":
        return calibrate_policy(scenario, args.write)
    if args.command == "export-labels":
        output = args.output or ROOT / "data" / f"labeled_{scenario.name}.csv"
        return export_labels(args.log, output, scenario.name)
    output = args.output or ROOT / "hf" / scenario.name
    return export(scenario.name, output, args.repo, args.license)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--lang", "--idioma")
    lang = resolve_language(early.parse_known_args(argv)[0].lang)

    parser = build_parser(lang)
    args = parser.parse_args(argv)
    set_terminal(not args.no_progress)
    if args.command == "jobs":
        show_jobs(args.watch, lang)
        return
    try:
        if args.command in LONG_COMMANDS:
            target = getattr(args, "scenario", None) or args.split.name
            with Job(f"{args.command} {target}"):
                result = run(args, lang)
        else:
            result = run(args, lang)
    except LocalizedError as error:
        parser.error(error.message(lang))
    except (ValidationError, FileNotFoundError, ValueError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))

