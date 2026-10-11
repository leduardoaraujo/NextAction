# AI orchestrator routing demo

This NextAction scenario predicts the **next action of an AI orchestrator**. It is
inspired by the boundary in Luiz's Axis workspace: analytical requests go to a
data analyst agent, ordinary conversation to a conversation agent, low-confidence
intent to a fallback. The demo adds explicit `clarification_needed` and
`error_recovery` actions so that the router's next step can be evaluated, rather
than just the final agent name. It does not modify Axis or use its private data.

## Reproduce locally

```powershell
.venv/Scripts/python.exe demos/orchestrator/generate_data.py
.venv/Scripts/python.exe -m nextaction train orchestrator
.venv/Scripts/python.exe -m nextaction calibrate-policy orchestrator --write
.venv/Scripts/python.exe -m nextaction evaluate orchestrator
.venv/Scripts/python.exe demos/orchestrator/benchmark.py
.venv/Scripts/python.exe demos/orchestrator/analysis.py
.venv/Scripts/python.exe demos/orchestrator/play.py "Compare revenue by region"
```

`orquestrador` is the Portuguese scenario alias. The [benchmark](RESULTS.md)
reports both exact next-action accuracy and the simpler data-agent versus
conversation-route accuracy. [Error analysis](ANALYSIS.md) names every missed
test case and shows confidence by band.

## State and policy

```text
[blocked none] [context datasource_available last_error_none missing_none] || user: Compare monthly revenue by region
```

The environment supplies source availability, a prior SQL error and any missing
metric. `[blocked data_query]` prevents the policy from selecting a data-agent
dispatch without an available data source. The `review` outcome means that
another decision maker must handle a low-confidence route; its outcome is **not**
assumed correct in the benchmark. The four model actions are `data_query`,
`pure_chat`, `clarification_needed` and `error_recovery`. Those map to the two
agent routes, a clarification request and a recovery step.

## Data and comparison

The 159 cases are hand-authored and synthetic: 100 train, 25 validation and 34
test. Phrase families are disjoint across splits. The test set is roughly balanced by
action, so its accuracy is a diagnostic, not a production estimate. No Axis
messages, prompts, schemas or credentials are included. The keyword baseline is
an independent, frozen recreation of the older Axis visualization heuristic. It
offers only data/chat routes; reporting both route and exact-action accuracy
makes the narrower taxonomy visible.

The Axis LLM classifier needs the Axis runtime and credentials, which are not
present in this workspace. Its measured results can be added without modifying
the benchmark:

```powershell
# Run with the Axis backend's Python environment and configured API credentials.
python demos/orchestrator/export_axis_predictions.py --axis-backend PATH/TO/axis/backend
.venv/Scripts/python.exe demos/orchestrator/benchmark.py --axis-predictions data/external/axis_orchestrator_predictions.csv
```

The exporter makes 34 classifier calls and writes only `family`, `action` and
`latency_ms` to a git-ignored CSV. Do not run it without intending those calls.
For a portfolio claim about a real deployment, replace or extend this synthetic
set with representative, manually labeled and permission-safe Axis traces,
keep whole conversation families out of training, and compare all three
routers on the same held-out cases. The current demo shows a reproducible
prototype, not superiority over the Axis LLM classifier.
