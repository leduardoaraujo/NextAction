# AI orchestrator routing benchmark

The test set contains 34 hand-authored request families held out from training and policy calibration. Each family appears once. All examples are synthetic and roughly balanced across four actions; the figures do not estimate production traffic accuracy.

| Router | Next-action accuracy | 95% Wilson interval | Agent-route accuracy | Macro F1 | Wrong data-agent dispatches | Data requests missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Keyword baseline | 29.4% | 16.8%–46.2% | 70.6% | 0.177 | 3 | 7 |
| NextAction model | 85.3% | 69.9%–93.6% | 91.2% | 0.848 | 1 | 2 |

Agent-route accuracy differs by +20.6% for NextAction versus the keyword baseline (95% paired family-bootstrap interval +5.9% to +35.3%). The interval excludes zero on this synthetic set; it does not establish a production advantage.

With calibrated policy, NextAction automates 33/34 cases (97.1%); 29/33 automated next actions are correct. Another decision is needed for 1 case(s). Review accuracy is not assumed.

Local batch inference took 0.36 ms per case for the NextAction model on this machine. This is not a comparable end-to-end latency measurement for the other routers.

The Axis LLM classifier was not run: its service credentials and runtime are not present here. Use `--axis-predictions` with one exported classification per test family to add its measured result.

Agent-route accuracy asks only whether the request reaches the data analyst or the conversation path. Next-action accuracy distinguishes data, chat, clarification and recovery. The keyword baseline is an independent frozen recreation of the legacy Axis visualization heuristic; it has only data/chat outputs. The NextAction policy also reads explicit context tags and blocks data-agent dispatch when no data source is available. This is an end-to-end routing comparison, not a controlled model-only ablation.
