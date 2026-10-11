# Evaluation

A decision model is only useful if you know **how often it is wrong, on which actions, and how much you can automate safely**. `nextaction evaluate <scenario>` answers those three questions.

## Choosing the test data

| Option | When to use |
| --- | --- |
| *(default)* stratified cross-validation, up to 5 folds | small datasets, where every example is needed for training |
| `--test file.csv` | a fixed held-out set; the `agent` scenario uses the official SGD test split by default |
| `--recent-fraction 0.2` | data exported from the decision log; the most recent 20% become the test set, without shuffling |

`--recent-fraction` is the closest to real use: you train on the past and measure on the future. Random splits overestimate performance when traffic changes over time.

## Report fields

| Field | Meaning |
| --- | --- |
| `accuracy` | share of correct predictions |
| `majority_baseline_accuracy` | accuracy of always predicting the most common training action; the model must beat this by a wide margin |
| `per_action` | precision, recall, F1 and support per action. Low recall on a risky action (such as `confirm`) matters more than overall accuracy |
| `confusion_matrix` | `{actual: {predicted: count}}`, showing which actions get mixed up |
| `expected_calibration_error` | ECE: average gap between stated confidence and actual accuracy, over 10 confidence bins. 0 means "70% confident" is right 70% of the time |
| `log_loss` | penalizes confident mistakes; useful for comparing models |
| `thresholds` | for each threshold: share of decisions `automated`, `automated_accuracy`, `unreviewed_errors` and cases `sent_to_review` |
| `warning` | shown when fewer than 100 examples were evaluated, because the numbers are unstable |

`method` and `warning` are human-readable and follow `--lang`; every other field is the same in both languages.

## Picking a review threshold

Read the `thresholds` table as a trade-off curve. Each row says: *if decisions below this confidence go to a human, X% are automated, and Y errors happen without review.*

1. Estimate the cost of an unreviewed error and the cost of a human review.
2. Choose the lowest threshold whose unreviewed error count is acceptable.
3. Re-check on fresh data. Probabilities are not calibrated, so a threshold that works on one dataset can drift on another.

Example on the SGD test split: a threshold of 0.9 automates 50.0% of decisions at 99.6% accuracy, and 0.7 automates 75.1% at 97.8%.

## A cautionary result

On the 48 synthetic flight examples, a single random split once reported 0.667 accuracy. Cross-validation showed 0.50, and no prediction ever exceeded 0.5 confidence, so with a 0.70 threshold every decision would go to review. Small datasets make single-split numbers look better than they are, which is why `train` no longer prints a test score and points to `evaluate` instead.
