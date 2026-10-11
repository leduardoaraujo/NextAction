"""Progress reporting through job files, the CLI and the API."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from opendecision.api import app
from opendecision.cli import main
from opendecision.progress import Job, describe, list_jobs, stage


class ProgressTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"OPENDECISION_JOBS_DIR": self.temp.name})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_stage_outside_a_job_is_a_no_op(self):
        with stage("fit", total=10) as fitting:
            fitting.advance(3)
        self.assertEqual(list_jobs(), [])

    def test_job_file_tracks_stage_percent_and_completion(self):
        with Job("train flights") as job:
            with stage("fit", total=200) as fitting:
                fitting.update(50)
                self.assertEqual(fitting.percent, 25.0)
            running = json.loads(job.path.read_text(encoding="utf-8"))
            self.assertEqual(running["status"], "running")
            self.assertEqual(running["stage"], "fit")
        finished = list_jobs()[0]
        self.assertEqual((finished["status"], finished["percent"]), ("done", 100.0))
        self.assertFalse(list(self.folder.glob("*.tmp")))

    def test_failed_job_keeps_the_error(self):
        with self.assertRaises(RuntimeError), Job("import sgd"):
            raise RuntimeError("disk full")
        record = list_jobs()[0]
        self.assertEqual((record["status"], record["error"]), ("failed", "disk full"))

    def test_running_job_without_updates_becomes_stale(self):
        with Job("old") as job:
            pass
        record = json.loads(job.path.read_text(encoding="utf-8"))
        old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
        record.update(status="running", updated_at=old)
        job.path.write_text(json.dumps(record), encoding="utf-8")
        self.assertEqual(list_jobs()[0]["status"], "stale")

    def test_labels_are_localized(self):
        record = {"stage": "fit", "status": "running"}
        self.assertEqual(describe(record, "pt")["stage_label"], "Ajustando modelo")
        self.assertEqual(describe(record, "en")["status_label"], "running")

    def test_cli_train_records_a_job_and_lists_it(self):
        model = str(self.folder / "flights.joblib")
        with contextlib.redirect_stdout(io.StringIO()):
            main(["--no-progress", "train", "flights", "--model", model])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(["jobs"])
        self.assertIn("train flights", output.getvalue())
        self.assertEqual(list_jobs()[0]["status"], "done")

    def test_api_lists_jobs_and_serves_dashboard(self):
        with Job("evaluate agent"):
            pass
        with TestClient(app) as client:
            body = client.get("/v1/jobs", headers={"Accept-Language": "pt-BR"}).json()
            page = client.get("/dashboard")
        self.assertEqual(body["jobs"][0]["status_label"], "concluído")
        self.assertEqual(page.status_code, 200)
        self.assertIn("/v1/jobs", page.text)


if __name__ == "__main__":
    unittest.main()
