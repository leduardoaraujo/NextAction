# Evaluation

A decision model is only useful if you know **how often it is wrong, on which actions, and how much you can automate safely**. `python -m opendecision avaliar` answers those three questions.

## Choosing the test data

| Option | When to use |
| --- | --- |
| *(default)* stratified cross-validation, up to 5 folds | small datasets, where every example is needed for training |
| `--teste file.csv` | a fixed held-out set, such as the official SGD test split |
| `--fracao-final 0.2` | data exported from the decision log; the most recent 20% become the test set, without shuffling |

`--fracao-final` is the closest to real use: you train on the past and measure on the future. Random splits overestimate performance when traffic changes over time.

## Report fields

| Field | Meaning |
| --- | --- |
| `acuracia` | share of correct predictions |
| `acuracia_classe_majoritaria` | accuracy of always predicting the most common training action; the model must beat this by a wide margin |
| `por_acao` | precision, recall, F1 and support per action. Low recall on a risky action (such as `confirmar`) matters more than overall accuracy |
| `matriz_confusao` | `{actual: {predicted: count}}`, showing which actions get mixed up |
| `erro_calibracao_esperado` | ECE: average gap between stated confidence and actual accuracy, over 10 confidence bins. 0 means "70% confident" is right 70% of the time |
| `log_loss` | penalizes confident mistakes; useful for comparing models |
| `limiares` | for each threshold: share of decisions automated, accuracy on those, errors that would slip through, and cases sent to review |
| `aviso` | shown when fewer than 100 examples were evaluated, because the numbers are unstable |

## Picking a review threshold

Read the `limiares` table as a trade-off curve. Each row says: *if decisions below this confidence go to a human, X% are automated, and Y errors happen without review.*

1. Estimate the cost of an unreviewed error and the cost of a human review.
2. Choose the lowest threshold whose unreviewed error count is acceptable.
3. Re-check on fresh data. Probabilities are not calibrated, so a threshold that works on one dataset can drift on another.

Example on the SGD test split: a threshold of 0.9 automates 50.0% of decisions at 99.6% accuracy, and 0.7 automates 75.1% at 97.8%.

## A cautionary result

On the 48 synthetic flight examples, a single random split reported 0.667 accuracy. Cross-validation showed 0.50, and no prediction ever exceeded 0.5 confidence, so with a 0.70 threshold every decision would go to review. Small datasets make single-split numbers look better than they are; the report says so explicitly.
