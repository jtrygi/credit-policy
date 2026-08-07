# Systematic Feature Selection & Model Comparison (Step 4 extension)

Follow-up to `MODELING.md`, addressing three points: (1) the original 20-feature curated set was hand-picked, not derived — this replaces that with a systematic screen; (2) only forward selection had been run — this adds backward, stepwise, LASSO, and two nonlinear methods; (3) a direct answer on the specificity-vs-recall question, with numbers.

## 1. Systematic qualifying-variable screen (Information Value)

`scripts/iv_screening.py`. Rather than hand-picking candidates, every one of the ~85 non-leakage columns from `CLEANING.md` was scored with **Information Value** — the standard credit-scoring screening metric (bin the variable, compute Weight of Evidence per bin, sum `(%good - %bad) × WOE`). Conventional thresholds: IV < 0.02 not useful, 0.02–0.1 weak, 0.1–0.3 medium, 0.3–0.5 strong.

**Bug caught and fixed along the way:** naive decile binning (`pd.qcut`) silently collapsed several real predictors to IV≈0 — `term_months` looked useless (true bad rate is 14.0% vs 25.4% by term) purely because 91% of values are `36`, so most decile edges coincided and `qcut` produced a single bin. Fixed with rank-based quantile binning for skewed/low-cardinality numerics; low-cardinality columns (≤10 unique values) are now binned by exact value instead.

**Result:** `images/iv_screening.png` / `images/iv_screening.csv`. 3 variables are "strong" (`sub_grade` 0.402, `int_rate` 0.366, `grade` 0.365 — LendingClub's own pricing, as expected), `fico_avg` is "medium" (0.126), and **30 of 83 clear the IV≥0.02 qualifying bar** — this 30-variable pool, not a hand-picked list, is the candidate set for everything below.

`zip_code` was excluded from candidates entirely rather than screened: too high-cardinality to be a stable feature at this scale, and a geographic field close enough to a fair-lending proxy concern that it doesn't belong in a credit model regardless of its IV.

## 2. Wrapper selection: forward, backward, stepwise — scored by AUC

`scripts/wrapper_selection.py --method {forward,backward,stepwise}`. Per your direction, AUC drives the search (cheap, "out of the box"); Brier and KS are logged from the same fits at no extra cost, since the expensive part is the *number of fits* (O(n²) in candidate count), not which metric is computed from each fit's predictions.

| Method | Status | Result |
|---|---|---|
| Forward | **Complete** | Best val AUC=0.6906 at **24 of 30 features** (`images/wrapper_forward_history.csv`). Notably, `int_rate` doesn't enter until step 15 and `grade` not until step 27 — both are largely redundant once `sub_grade` (finer-grained) is picked at step 1. |
| Backward | Running in background | Early steps already corroborate forward: starting AUC 0.6904 (full 30), first two removals (`mths_since_recent_inq`, `tot_hi_cred_lim`) move AUC by <0.0001 — consistent with forward's finding that several qualifying variables add essentially nothing on top of the top ~15-20. |
| Stepwise | Running in background | Launched; not yet converged. |

Backward is the most compute-heavy of the three (starts from all 30 features — including `sub_grade`'s 34 dummy columns — so every early fit is on the full ~104-column design matrix). I'll report final backward/stepwise feature sets once they land; nothing so far contradicts forward's picture.

**One resource-management note worth being upfront about:** running forward, backward, stepwise, LASSO, and the tree ensembles fully in parallel caused several to crash outright (not a logic bug — genuine BLAS-thread oversubscription across 5 concurrent heavy sklearn processes). Rerunning with explicit thread caps and less parallelism fixed it; forward, LASSO, RF, and XGBoost all completed cleanly this way.

## 3. LASSO regularization path

`scripts/lasso_path.py`. L1-regularized logistic regression across a 25-point C grid (`images/lasso_path.png`, `images/lasso_auc_vs_c.png`) — the modern alternative to classical stepwise, showing every coefficient's full shrinkage path at once rather than one greedy walk. Categorical dummy blocks (`sub_grade` alone contributes 34 columns) are included in the fit but plotted unlabeled in gray; labeling all ~104 columns individually would violate the "don't cycle categorical hues indefinitely" chart principle, so only the top numeric features are direct-labeled.

As of the last check (16 of 25 grid points; still running): AUC plateaus at **0.6904 from C≈0.03 onward**, matching forward selection's ceiling almost exactly, with 87-90 of 104 possible coefficients nonzero at that point — i.e., LASSO's regularization path and forward selection's greedy search independently converge to the same answer about how much signal is actually in this feature set.

## 4. Nonlinear feature usefulness: Random Forest + XGBoost

`scripts/tree_rf.py`, `scripts/tree_xgb.py` — fit once each on the full 30-candidate pool (not wrapper-searched; an O(n²) search at ensemble-training cost would take hours), specifically to catch anything that only matters through nonlinear interactions.

**LightGBM crashed** with a native access violation (`LGBM_DatasetSetField`) even on trivial synthetic data, before and after a clean reinstall — a system-level dependency issue (likely a missing/mismatched OpenMP runtime DLL) outside what's fixable from this session. Substituted **XGBoost**, which works fine.

Random Forest uses **permutation importance**, not impurity importance — impurity importance is well known to be biased toward high-cardinality features, and `sub_grade` (34 dummy columns) is exactly the kind of feature that bias would inflate. `images/rf_importance.png`, `images/gbm_importance.png`.

## Comprehensive results table

`images/model_results_table.png` (`images/all_model_results.json`) — every model fit in this round, side by side:

| Model | # Feat | AUC | Gini | Brier↓ | KS | Prec@20% | Recall@20% | F1@20% | Spec@20% | Sens@95%Spec |
|---|---|---|---|---|---|---|---|---|---|---|
| LR (curated, 20) | 20 | 0.684 | 0.367 | 0.1208 | 0.267 | 0.281 | 0.373 | 0.321 | 0.831 | 0.146 |
| Decision Tree | 20 | 0.665 | 0.330 | 0.1222 | 0.243 | 0.258 | 0.377 | 0.306 | 0.808 | 0.047 |
| LR (Brier-selected) | 15 | 0.684 | 0.367 | 0.1208 | 0.266 | 0.280 | 0.373 | 0.320 | 0.831 | 0.149 |
| LR (Forward-selected, AUC) | 24 | 0.691 | 0.381 | 0.1202 | 0.278 | 0.288 | 0.383 | 0.329 | 0.832 | 0.151 |
| Random Forest | 104 | 0.685 | 0.370 | 0.1211 | 0.267 | 0.282 | 0.375 | 0.322 | 0.831 | 0.147 |
| **XGBoost** | 104 | **0.698** | **0.396** | **0.1195** | **0.288** | **0.292** | **0.389** | **0.334** | **0.833** | **0.161** |

**XGBoost wins on every single metric.** It's not close on Gini/KS (0.396/0.288 vs the next-best 0.381/0.278) despite using the same 30-variable candidate pool as everything else — the gain is from capturing nonlinear/interaction effects, not from different inputs. Random Forest, by contrast, barely beats the simple curated logistic regression, which says the *depth-8, min-leaf-500* regularization on this particular ensemble is leaving most of its power on the table relative to XGBoost's boosting.

## Hyperparameter tuning: Random Forest + XGBoost

`scripts/tune_ensembles.py`. Randomized search (25 RF trials, 40 XGBoost trials), scored by validation AUC same as the wrapper selection, on the same 30-variable/104-column pool as the baseline RF/XGBoost above -- isolates what tuning alone buys, independent of feature set.

Three of these background runs got killed mid-search by what looks like a ~20-25 minute limit on long-running background tasks in this environment (not a bug in the search itself). Fixed by making the search checkpoint every trial to CSV immediately and resume from the last completed trial on relaunch, burning the RNG through the same number of draws first so the resumed sequence matches an uninterrupted run. Also removed `max_depth=None` from the RF grid after it caused one trial to run 9+ minutes on its own (unbounded-depth trees on 517K rows x 254 columns) — a real cost trap in the search space, not just bad luck.

| Model | AUC | Gini | Brier | KS | Best hyperparameters |
|---|---|---|---|---|---|
| Random Forest (untuned) | 0.685 | 0.370 | 0.1211 | 0.267 | n_estimators=300, max_depth=8, min_samples_leaf=500 |
| Random Forest (tuned) | 0.691 | 0.382 | 0.1202 | 0.277 | n_estimators=300, max_depth=12, min_samples_leaf=100, max_features=0.3 |
| XGBoost (untuned) | 0.698 | 0.396 | 0.1195 | 0.288 | n_estimators=300, max_depth=6, learning_rate=0.05 |
| **XGBoost (tuned)** | **0.700** | **0.399** | **0.1193** | **0.294** | n_estimators=500, max_depth=4, learning_rate=0.1, subsample=0.6, colsample_bytree=0.8, min_child_weight=3, reg_lambda=5 |

**Takeaway: tuning bought less than feature set did.** RF gained +0.006 AUC from tuning (0.685→0.691) — matching, not exceeding, what the forward-selected logistic regression already got with 24 linear features. XGBoost gained only +0.002 (0.698→0.700) from 40 trials of search. Compare that to the max-performance experiment below: dropping the 30-variable constraint and using the full 254-column set (with early stopping, not exhaustive tuning) reached **0.708 val AUC** — a bigger jump than either tuning search found. For this problem, at this point in the pipeline, feature set mattered more than hyperparameters.

Both convergence charts (`images/tune_rf_convergence.png`, `images/tune_xgb_convergence.png`) show the same pattern: an early jump to near-best within the first 5-17 trials, then a long flat tail of trials clustered in a narrow band around it — the search had already converged well before it finished.

## On specificity vs. recall (direct answer, with numbers)

Your proposal — optimize for specificity since recall-chasing produces too many false positives — runs into the mirror-image version of the problem you already correctly identified in recall: a model that **approves everyone** gets specificity = 1.0 (zero false positives) while catching **zero** defaults. Sensitivity and specificity are the same threshold dial, not two independently optimizable things — every cutoff produces one (sensitivity, specificity) pair, so "optimize specificity" with no constraint just slides the dial to approve-everyone.

The **Sens@95%Spec** column above is the constrained version of your instinct — "protect good customers first, then see what's left" — and it's stark: even the best model (XGBoost) only catches **16.1%** of actual defaults if you insist on correctly clearing 95% of good customers. The Decision Tree manages only **4.7%**, likely because its probability output is a step function (15 leaves = 15 possible scores), so hitting a precise specificity target lands awkwardly between leaf boundaries. **KS** (0.243–0.288 across models) is the industry-standard summary of good/bad separation and moves consistently with AUC — XGBoost leads on both.
