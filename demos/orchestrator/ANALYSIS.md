# AI orchestrator error analysis

All 34 test families are synthetic and held out from training. Counts are case counts, not estimates of production frequency.

![Confusion matrix](analysis/confusion_matrix.png)

## Incorrect next actions

| User message | Expected | Predicted | Confidence | Policy review? | Keyword router |
| --- | --- | --- | ---: | --- | --- |
| Quais segmentos concentram churn? | `data_query` | `pure_chat` | 43.2% | no | `pure_chat` |
| O SQL anterior deu erro, agora mostre o ranking de produtos | `data_query` | `error_recovery` | 64.7% | no | `pure_chat` |
| Me mostre algo interessante | `clarification_needed` | `data_query` | 34.7% | yes | `pure_chat` |
| What should I analyze? | `clarification_needed` | `pure_chat` | 51.9% | no | `pure_chat` |
| Olhe esses valores para mim | `clarification_needed` | `pure_chat` | 41.7% | no | `pure_chat` |

## Confidence

![Confidence](analysis/confidence.png)

| Confidence band | Cases | Mean confidence | Observed accuracy |
| --- | ---: | ---: | ---: |
| 0.3–0.4 | 2 | 37.3% | 50.0% |
| 0.4–0.5 | 12 | 44.4% | 83.3% |
| 0.5–0.6 | 8 | 55.4% | 87.5% |
| 0.6–0.7 | 5 | 64.9% | 80.0% |
| 0.7–0.8 | 5 | 75.5% | 100.0% |
| 0.8–1.0 | 2 | 83.9% | 100.0% |

These bins are too small for a strong calibration claim. The validation thresholds choose when to hand the route to another decision maker; the review outcome is not included in the accuracy score.
