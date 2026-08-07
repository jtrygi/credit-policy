# Segmentation & Policy Design (Steps 3, 6, 7)

Step 3 (baseline) was never actually built as an artifact during the modeling phase (FEATURE_SELECTION.md) -- it's built here first, since Step 7 explicitly requires comparing against it. All dollar figures are **per 1,000 test-population applicants** (declined applicants contribute $0), matching Section 4's definition exactly. No cost of funds is modeled anywhere in this document -- same simplification as `economics.py`, deferred to Step 8's sensitivity analysis.

## Step 3: baseline policy

`scripts/reconstruct_policy_split.py` first replays `prepare_data.py`'s exact pre-split logic (same target definition, same feature engineering, same two-stage `train_test_split(random_state=42)`) but keeps `id`/`grade`/`funded_amnt`/`total_pymnt`/`collection_recovery_fee` -- columns `prepare_data.py` correctly drops as leakage/identifiers before modeling, but which are needed now for realized-$ policy evaluation. **Verified, not assumed:** the reconstructed train/val/test splits were checked for exact positional match against the existing `bad` column in `data/{train,val,test}.csv` before trusting any downstream dollar figure -- all three matched exactly.

`scripts/baseline_policy.py` defines the baseline the design doc asks for ("approve everyone above a single existing risk grade cutoff") and sweeps it across all 7 possible cutoffs on LendingClub's own A-G grade, rather than picking one arbitrary point -- `images/baseline_policy_curve.png` / `.csv`.

| Cutoff | Volume | Bad rate | Net profit / 1,000 |
|---|---|---|---|
| A only | 21.2% | 5.52% | $212,312 |
| A-B | 53.6% | 9.02% | $584,316 |
| A-C | 80.4% | 12.26% | $925,175 |
| **A-D (reference baseline)** | **93.3%** | **13.86%** | **$1,097,981** |
| A-E | 98.2% | 14.68% | $1,185,228 |
| A-F | 99.7% | 14.98% | $1,225,710 |
| A-G (approve all) | 100.0% | 15.05% | $1,232,583 |

**A-D is used as the reference single-cutoff baseline** for the Step 7 comparison (94% approval is a plausible "current practice" volume; grades E-G are historically thin segments anyway, 6.6% of volume combined).

**Non-obvious finding, worth being upfront about:** net profit per 1,000 applicants rises *monotonically* all the way to approve-everyone -- even grade G is net profitable on average in this book. This is a data limitation, not a modeling result: this dataset contains only loans LendingClub already approved and priced (the rejected-applicant population isn't in it), so of course every grade LC actually funded looks profitable in aggregate -- their own pricing already cleared that bar. It means the real value a sharper risk model can add here isn't "bulk-decline a whole risk tier" (there isn't one that's unprofitable on average) -- it's finding pockets of relative unprofitability *within* LC's own coarse grades, or supporting differentiated pricing. That's exactly what Steps 6-7 test below.

## Step 6: segmentation

`scripts/score_models.py` scores every val/test applicant with two models: **LR-24**, the forward-selected 24-feature logistic regression from `FEATURE_SELECTION.md` (refit here for the first time and persisted to the registry -- it was the one surviving candidate from that round never actually saved), and **XGBoost full-features-earlystop**, the best-performing model overall (val AUC 0.708 / test AUC 0.709, loaded from the registry). Both were checked for calibration by decile on val before segmenting on them (predicted PD vs. observed bad rate) -- both track closely, no recalibration needed, which matters because segment boundaries are literally PD cutpoints.

`scripts/segment_policy.py` bins each model's score into **6 quantile segments, boundaries defined on val only**, then applies those same boundaries to test to check stability. Bad rate rises monotonically from segment 1 to 6 on val for both models, and **the same boundaries still produce a monotonic, well-separated bad rate on test** -- `images/segments_lr24.png`, `images/segments_xgb.png`.

| Segment | XGBoost bad rate (test) | XGBoost avg $/applicant (test) | LR-24 bad rate (test) | LR-24 avg $/applicant (test) |
|---|---|---|---|---|
| 1 (lowest risk) | 3.66% | $1,194 | 4.32% | $1,117 |
| 2 | 7.61% | $1,239 | 8.07% | $1,162 |
| 3 | 11.03% | $1,320 | 11.25% | $1,227 |
| 4 | 15.15% | $1,324 | 15.47% | $1,265 |
| 5 | 21.42% | $1,260 | 21.20% | $1,262 |
| 6 (highest risk) | 31.54% | **$1,060** | 30.02% | $1,364 |

**Interesting divergence between the two models' segment 6:** LR-24's riskiest sextile still has the *highest* average dollar profit of any segment (higher interest rates on riskier loans dominate) -- but XGBoost's riskiest sextile has the *lowest* average dollar profit of the six, clearly below segments 3-5 despite carrying the highest bad rate. XGBoost's finer nonlinear discrimination is finding a sub-pocket of thin-margin risk within the loans LR-24's linear score treats as uniformly well-compensated by pricing. That's a concrete, non-obvious example of what the extra AUC actually buys.

## Step 7: policy design and comparison to baseline

**Segment-level approve/decline (val-derived, applied to test):** decision = approve iff the segment's mean realized $ outcome on val is positive. Every one of the 6 segments is profitable on average for both models (consistent with Step 3's finding), so this rule converges to **approve-all for both models** -- $1,232,583/1,000, identical to the A-G baseline point, since approving 100% of applicants gives the same total regardless of which model did the ranking.

That result is honest but not useful for comparison, so the real test is Section 4's actual methodology -- **hold one of volume/loss-rate fixed at the baseline's value, let the model's finer ranking pick who fills that slot instead of the coarse grade cutoff** (`scripts/segment_policy.py`'s `frontier_comparison`, reading fixed points off each model's approve-lowest-PD-first ranking on test):

| Comparison | Baseline (A-D) | LR-24 | XGBoost |
|---|---|---|---|
| **Equal volume** (93.3%) | bad rate 13.86% | bad rate **13.61%** (−0.24pp), profit/1,000 $1,136,650 (+$38,669) | bad rate **13.46%** (−0.39pp), profit/1,000 $1,169,507 (+$71,526) |
| **Equal loss rate** (13.86%) | volume 93.3% | volume **94.7%** (+1.36pp), profit/1,000 $1,153,974 (+$55,993) | volume **95.4%** (+2.13pp), profit/1,000 $1,194,179 (+$96,198) |

Both models clear the design doc's success criterion 1 ("measurable, defensible improvement... on at least one axis") comfortably, **on both axes, at once** -- ranking by either model's score instead of LendingClub's own coarse grade finds a meaningfully lower-loss-rate subset at the same approval volume, or supports meaningfully more volume at the same loss rate.

**The interpretability-vs-lift tradeoff, quantified (the deliverable Step 5 promised):** XGBoost captures roughly 1.6-1.9x the improvement LR-24 does on both axes -- its extra 0.017 AUC (0.708 vs. 0.691) is worth about **$33,000-$40,000 more per 1,000 applicants** at matched-risk operating points than the interpretable model. That's the real number a bank would weigh against the governance cost of getting a non-linear model past model risk/compliance -- not an abstract "black box" objection, an actual dollar figure.

**Recommended policy (operational simplicity, criterion 2):** the 6-segment table above *is* the deployable rule set -- a credit committee can review 6 PD-range buckets with observed bad rates and $ economics per bucket. Given Step 3's finding that every segment is profitable in this survivorship-biased book, the policy recommendation is **approve all 6 segments** at the current volume, but price segment 6 differently (thinnest realized margin, especially under XGBoost) -- a differentiated-pricing lever, not a decline lever, is what this data actually supports. Whether to instead decline segment 6 to match baseline's exact volume/loss-rate (the equal-volume/equal-loss-rate rows above) is a live business choice, not a modeling one, and is exactly the operating point this table makes explicit.

## Outputs

`images/baseline_policy_curve.{png,csv}`, `images/baseline_policy_reference.json`, `images/val_scores.csv`, `images/test_scores.csv`, `images/segments_{lr24,xgb}.png`, `images/segments_{lr24,xgb}_{val,test}.csv`, `images/segment_policy_results.json`. Models: `models/lr-forward-selected-24/v1/` (newly persisted).
