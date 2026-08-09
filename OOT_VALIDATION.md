# Out-of-Time Validation

## Why this document exists, and why it's separate from SEGMENTATION_POLICY.md

Every model and result up through `SEGMENTATION_POLICY.md` was evaluated on a **random** train/val/test split (`prepare_data.py`: `train_test_split(..., random_state=42)`). Loans from every origination year -- including the 2007-2009 crisis vintages -- are scattered randomly across train, val, and test. That means every prior model has been tested on held-out **loans**, but never on a held-out **time period**: it has already seen what every vintage era looks like during training. A real deployment doesn't work that way -- you train on history and predict forward into a future you haven't seen yet.

This document rebuilds the whole evaluation chain on a **chronological** split instead, to answer directly: does the profitability edge over grade-based baseline found in `SEGMENTATION_POLICY.md` survive when the model is tested on genuinely unseen future vintages? **The prior documents are left as-is, not rewritten** -- they're accurate descriptions of what a same-era holdout shows, which is a real (if more limited) form of evidence. This document adds the stronger test on top, and the honest headline finding is that it changes the conclusion's *magnitude* substantially, though not its *direction*.

## Methodology

`scripts/prepare_data_oot.py` replays the identical feature engineering and leakage drops as `prepare_data.py`, but splits by `issue_dt` instead of randomly:

| Split | Period | n | Bad rate |
|---|---|---|---|
| Train | issue year <= 2013 | 230,706 | 15.65% |
| Val | issue year = 2014 | 175,509 | 14.63% |
| Test | issue year 2015-2016 | 332,261 | 14.85% |

Val strictly precedes test in time, so XGBoost's early stopping can't see anything from the test period. Train-only median imputation is recomputed on the OOT train set (69 columns had missingness, vs. a similar count in the original split).

`scripts/score_models_oot.py` refits the two frozen policy candidates -- **same 24 features, same XGBoost hyperparameters** as the originals -- on this split, and saves them to the model registry as **v2** (v1 = original random split, kept unchanged for comparison):

| Model | Split | Val AUC | Test AUC | Test KS |
|---|---|---|---|---|
| LR-24 | Random (v1) | 0.6906 | 0.6939 | 0.2869 |
| LR-24 | **Out-of-time (v2)** | 0.6877 | **0.6929** | 0.2815 |
| XGBoost | Random (v1) | 0.7081 | 0.7092 | 0.2894 |
| XGBoost | **Out-of-time (v2)** | 0.6971 | **0.6998** | 0.2897 |

**Discrimination barely moves.** LR-24 loses essentially nothing (0.6939 -> 0.6929). XGBoost loses about 1 point of AUC (0.7092 -> 0.6998) -- expected for the higher-capacity model, but still a small, reassuring gap. Both models rank-order risk on genuinely future loans almost as well as on same-era holdout loans. **This part of the story is good news and doesn't need re-litigating.**

`scripts/oot_policy_comparison.py` reruns the grade-cutoff baseline and the model-vs-grade frontier comparison entirely on the OOT test set (2015-2016 vintages) -- the same equal-volume / equal-loss-rate mechanics as `SEGMENTATION_POLICY.md`, just recomputed on data neither model trained on.

## The headline finding: the edge survives, but shrinks about 5x

`images/methodology_comparison.png` puts both evaluations side by side directly:

| Comparison | Model | Random split (same-era) | Out-of-time | Shrinkage |
|---|---|---|---|---|
| Equal volume | LR-24 | +$38,670 | +$7,135 | 5.4x |
| Equal volume | XGBoost | +$71,526 | +$15,021 | 4.8x |
| Equal loss rate | LR-24 | +$55,994 | +$7,198 | 7.8x |
| Equal loss rate | XGBoost | +$96,198 | +$15,174 | 6.3x |

Every cell stays **positive** -- the model never loses to grade-based bucketing out-of-time, on either model, on either axis. But the same-era random-split numbers in `SEGMENTATION_POLICY.md` were overstating the real forward-looking edge by roughly 5-8x. If you were building the business case for a credit committee off the random-split numbers alone, you'd be promising something like 5x too much.

## Why the edge shrinks this much (a real effect, not noise)

Two things compound:

1. **The OOT test population's baseline is already closer to "approve everyone."** In the random-split test, the A-D reference baseline sits at 93.3% volume; in the OOT test (2015-2016 vintages only), it's **96.2%** -- `images/oot_baseline_policy_curve.csv`. LendingClub's own underwriting/approval mix shifted over time (also compounded by the Step 1 seasoning cutoffs: the 2015-2016 test window is 36-month loans only, and 2016 is truncated since the data was pulled before all 2016 vintages finished seasoning -- noted in `SCOPING.md`'s methodology, carried forward here as a caveat, not fixed). With less volume sitting between the baseline cutoff and 100% approval, there's mechanically less room for a model to reorder anyone.
2. **Slightly worse rank-ordering out-of-time** (the AUC gap above) compounds the first effect, especially for XGBoost.

Both are legitimate, explainable effects -- not a bug in the OOT pipeline. `images/oot_frontier_comparison.png` shows the same visual pattern as the original frontier chart, just visibly compressed near the reference point.

## What this changes about the recommendation

Not the direction -- a model-based policy still beats grade-only bucketing, robustly, out-of-time. It does change how the number should be presented: **use the OOT figures ($7K-$15K per 1,000 applicants) as the defensible headline for Step 8's business case, not the random-split figures** ($38K-$96K). The random-split numbers stay in `SEGMENTATION_POLICY.md` as a documented, correctly-labeled same-era holdout result -- a weaker form of evidence, now explicitly identified as such.

## Outputs

`scripts/prepare_data_oot.py`, `scripts/score_models_oot.py`, `scripts/oot_policy_comparison.py`, `scripts/methodology_comparison_chart.py`. `images/oot_baseline_policy_curve.csv`, `images/oot_frontier_results.json`, `images/oot_frontier_comparison.png`, `images/methodology_comparison.png`, `images/val_oot_scores.csv`, `images/test_oot_scores.csv`. Models: `models/lr-forward-selected-24/v2/`, `models/xgboost-full-features-earlystop/v2/`.
