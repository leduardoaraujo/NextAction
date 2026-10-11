# Tracking progress

Training, evaluation, imports and exports run as **jobs**. Each job draws a progress bar in the terminal and writes `jobs/<id>.json`, so you can follow it from another terminal or from the browser.

## In the terminal

Long commands (`train`, `evaluate`, `import-sgd`, `export-hf`) show one bar per stage:

```text
Vectorizing text  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
Fitting model     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100%
```

`--no-progress` hides the bars; the job file is still written. To follow jobs started elsewhere:

```bash
nextaction jobs            # list recent jobs
nextaction jobs --watch    # refresh every second (Ctrl+C to stop)
nextaction --lang pt tarefas --acompanhar
```

```text
 id                      job          status   stage          %    ETA
 20261010-230805-b5465b  train agent  running  Fitting model  5%
```

## In the browser

With the API running (`uvicorn nextaction.api:app`), open `http://127.0.0.1:8000/dashboard`. The page polls `GET /v1/jobs` every two seconds and shows each job's stage, percentage, ETA, status and error. It follows the browser language (English or Portuguese) and the system light or dark theme.

## Job file

```json
{
  "id": "20261010-230805-b5465b", "name": "train agent", "status": "running",
  "stage": "fit", "done": 50, "total": 1000, "percent": 5.0, "eta_s": null,
  "started_at": "2026-10-11T02:08:05+00:00", "updated_at": "2026-10-11T02:08:25+00:00",
  "error": null, "pid": 13096
}
```

- `status` is `running`, `done` or `failed`. A running job without updates for five minutes is reported as `stale`, which usually means the process was killed.
- `GET /v1/jobs` adds `stage_label` and `status_label` in the language of `Accept-Language`.
- Files are written atomically at most twice per second. The folder is git-ignored; set `NEXTACTION_JOBS_DIR` to move it.

## What the stages measure

| Stage | Progress unit |
| --- | --- |
| `import` | dialogue files read |
| `vectorize` | the TF-IDF fit over all examples (reported at start and end) |
| `fit` | solver iterations out of `max_iter`, in chunks of 50; it jumps to 100% when the solver converges early, so no ETA is shown |
| `cross_validation` | folds completed |
| `predict` | test predictions |

## Reporting progress from your own code

```python
from nextaction.progress import Job, stage

with Job("my import"):
    with stage("import", total=len(files)) as progress:
        for path in files:
            ...
            progress.advance()
```

Outside a `Job`, `stage` does nothing, so library functions can report progress unconditionally.
