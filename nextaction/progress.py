"""Progress reporting: a terminal bar for people and one JSON file per job for other tools.

Library code reports through the current job without receiving it as an argument::

    with Job("train agent"):
        with stage("fit", total=1000) as fit:
            fit.update(250)

Outside a job, ``stage`` is a no-op, so instrumented functions work unchanged in tests and
when imported as a library. ``nextaction jobs`` and ``GET /v1/jobs`` read the JSON files.
"""

import contextvars
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .scenarios import ROOT

STALE_AFTER_S = 300
WRITE_INTERVAL_S = 0.5
_current: contextvars.ContextVar["Job | None"] = contextvars.ContextVar("nextaction_job", default=None)
_terminal = {"enabled": True}


def jobs_dir() -> Path:
    return Path(os.environ.get("NEXTACTION_JOBS_DIR", ROOT / "jobs"))


def set_terminal(enabled: bool) -> None:
    """Turn the terminal bar on or off; the JSON files are always written."""
    _terminal["enabled"] = enabled


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Stage:
    def __init__(self, job: "Job | None", label: str, total: int | None, estimate_eta: bool = True):
        self.job, self.label, self.total, self.estimate_eta = job, label, total, estimate_eta
        self.done = 0
        self.started = time.monotonic()

    def update(self, done: int) -> None:
        self.done = done
        if self.job:
            self.job._report(self)

    def advance(self, amount: int = 1) -> None:
        self.update(self.done + amount)

    @property
    def percent(self) -> float | None:
        if not self.total:
            return None
        return round(min(100.0, 100 * self.done / self.total), 1)

    @property
    def eta_s(self) -> float | None:
        if not self.estimate_eta or not self.total or not self.done:
            return None
        elapsed = time.monotonic() - self.started
        return round(elapsed * (self.total - self.done) / self.done, 1)


class Job:
    """A unit of work shown as one bar per stage and stored in ``jobs/<id>.json``."""

    def __init__(self, name: str, directory: Path | None = None):
        self.name = name
        self.id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        self.path = (directory or jobs_dir()) / f"{self.id}.json"
        self.record = {
            "id": self.id, "name": name, "status": "running", "stage": None, "done": 0,
            "total": None, "percent": None, "eta_s": None, "started_at": _now(),
            "updated_at": _now(), "error": None, "pid": os.getpid(),
        }
        self._last_write = 0.0
        self._bar = None
        self._bar_tasks: dict[str, int] = {}
        self._token = None

    def __enter__(self) -> "Job":
        self._token = _current.set(self)
        if _terminal["enabled"] and sys.stderr.isatty():
            from rich.progress import (
                BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn,
            )
            from rich.console import Console

            self._bar = Progress(
                TextColumn("[bold]{task.description}"), BarColumn(), TaskProgressColumn(),
                TimeRemainingColumn(), console=Console(stderr=True),
            )
            self._bar.start()
        self._write(force=True)
        return self

    def __exit__(self, kind, error, traceback) -> None:
        self.record["status"] = "failed" if error else "done"
        if error:
            self.record["error"] = str(error)
        elif self.record["total"]:
            self.record.update(done=self.record["total"], percent=100.0, eta_s=0)
        self._write(force=True)
        if self._bar:
            self._bar.stop()
        _current.reset(self._token)

    def _report(self, stage: Stage) -> None:
        changed = self.record["stage"] != stage.label
        self.record.update(
            stage=stage.label, done=stage.done, total=stage.total,
            percent=stage.percent, eta_s=stage.eta_s,
        )
        if self._bar:
            if stage.label not in self._bar_tasks:
                self._bar_tasks[stage.label] = self._bar.add_task(stage.label, total=stage.total)
            self._bar.update(self._bar_tasks[stage.label], completed=stage.done, total=stage.total)
        self._write(force=changed)

    def _write(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_write < WRITE_INTERVAL_S:
            return
        self._last_write = now
        self.record["updated_at"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.record, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self.path)


@contextmanager
def stage(label: str, total: int | None = None, estimate_eta: bool = True):
    """Report a stage of the current job; does nothing outside a job.

    Pass ``estimate_eta=False`` when progress is not linear in time, such as solver
    iterations that may stop early on convergence.
    """
    current = Stage(_current.get(), label, total, estimate_eta)
    current.update(0)
    yield current
    if total:
        current.update(total)


def list_jobs(directory: Path | None = None, limit: int = 20) -> list[dict]:
    """Most recent jobs first. Running jobs without updates for 5 minutes are marked stale."""
    folder = directory or jobs_dir()
    records = []
    for path in folder.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        updated = datetime.fromisoformat(record["updated_at"])
        if record["status"] == "running" and (datetime.now(timezone.utc) - updated).total_seconds() > STALE_AFTER_S:
            record["status"] = "stale"
        records.append(record)
    return sorted(records, key=lambda r: r["started_at"], reverse=True)[:limit]


def describe(record: dict, lang: str | None = None) -> dict:
    """Add human-readable ``stage_label`` and ``status_label`` in the requested language."""
    from .i18n import MESSAGES, t

    stage_key, status_key = f"stage_{record['stage']}", f"status_{record['status']}"
    return {
        **record,
        "stage_label": t(stage_key, lang) if stage_key in MESSAGES else record["stage"],
        "status_label": t(status_key, lang) if status_key in MESSAGES else record["status"],
    }
