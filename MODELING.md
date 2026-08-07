# Step 4: Logistic Regression + Decision Tree

Trained on `data/train.csv` (516,933 rows), evaluated on `data/val.csv` (110,771 rows) only — `data/test.csv` stays untouched until a final model is chosen. Produced by `scripts/train_models.py`.

## Feature set

Curated and explicit (not "every column in train.csv" — that file still carries `issue_dt`, `zip_code`, `addr_state`, which are not model inputs):

- **Numeric (16):** `loan_amnt`, `int_rate`, `annual_inc`, `dti`, `delinq_2yrs`, `fico_avg` (derived from `fico_range_low`/`high`), `inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `credit_history_months`, `loan_to_income`, `emp_length_years`, `term_months`
- **Categorical (4, one-hot):** `grade`, `home_ownership`, `verification_status`, `purpose`
- **10 missing-value flags** carried over from `prepare_data.py`

Logistic regression uses train-fit `StandardScaler`'d features (for interpretable, comparable coefficients); the tree uses raw values (doesn't need scaling). Neither model uses `class_weight="balanced"` — that would inflate recall by wrecking calibration, and Step 7's profit math needs the raw predicted probabilities to be trustworthy, not just rank-ordered.

## Model comparison: AUC/Gini + calibration, not F1

| Metric | Logistic Regression | Decision Tree (depth 4) |
|---|---|---|
| ROC-AUC | **0.684** | 0.665 |
| Gini | 0.367 | 0.330 |
| Average Precision (PR-AUC) | 0.266 | 0.238 |
| Brier score (lower is better) | 0.1208 | 0.1222 |

Logistic regression wins on every metric, though the gap is modest — expected, since the tree is intentionally shallow (depth 4) for interpretability rather than tuned for maximum performance.

**Calibration** (`images/calibration_curve.png`): both models track the diagonal closely across deciles — predicted probabilities can be trusted as real probabilities, which matters because Step 7's dollar-profit calculation will multiply these probabilities directly against loss/revenue figures. This is also *why* we avoided `class_weight="balanced"` — that flag would have broken exactly this property to buy a recall number that doesn't mean what it looks like it means.

### Why AUC over F1 for model comparison

F1 requires picking one classification threshold, but the policy this project is building isn't a single yes/no threshold — it's a segmented approve/decline + tiering system (design doc Step 6). AUC measures ranking quality across *all* thresholds at once, which is what actually matters when the end product is a score used to carve out multiple segments. The design doc says the same thing in Step 4: "evaluate discrimination (AUC/Gini) and calibration."

### The 0.5-threshold trap

At a 15% base rate, neither model puts many applicants above predicted probability 0.5 — recall and F1 *at that threshold* would look near-zero for both models and make them look broken when they aren't. Threshold-based metrics below are reported at a **policy-relevant operating point** instead: declining the riskiest 20% by predicted score.

| | Logistic Regression | Decision Tree |
|---|---|---|
| Threshold (80th pct. of score) | 0.220 | 0.208 |
| Precision | 0.281 | 0.258 |
| Recall | 0.373 | 0.377 |
| F1 | 0.321 | 0.306 |
| Bad rate among declined 20% | 28.1% | 25.8% |
| Bad rate among approved 80% | 11.8% | 12.0% |

Even at this reasonable operating point, precision sits around 0.26-0.28 — most declines under either model are still "false alarms" in raw classification terms. That's normal for weak-signal consumer credit risk and is exactly why the *threshold* shouldn't be chosen to maximize a classification metric at all (see below).

## The actual question: is a bad loan more costly, and by how much?

Computed directly from realized outcomes in the scoped population (`scripts/economics.py`), verified against LendingClub's own accounting (`total_pymnt` already includes `recoveries` — confirmed empirically before computing anything, since double-counting recoveries would have overstated losses):

- **Average profit on a Fully Paid loan: $2,323**
- **Average loss on a Charged Off loan: -$4,940**
- **Ratio: 2.13** — missing one default (a false negative) costs about as much as wrongly declining **~2.1 good loans** (false positives)

If that were the whole story, it would argue for weighting recall roughly 2x over precision — closer to F2 than F1, not F1 itself. But it isn't the whole story:

- **Good loans outnumber bad loans 5.7:1** in this population (85% good / 15% bad)

Put together: a policy that chases recall by declining broadly will decline far more good loans than the defaults it prevents, because the volume imbalance (5.7:1) outweighs the per-loan cost imbalance (2.1:1). A generic metric like F1 (or F2) can't see this trade-off — it doesn't know about either number. **The right primary metric isn't a classification metric at all — it's expected dollars per 1,000 applicants**, which is already the design doc's Section 4 framework and directly multiplies these exact two numbers (predicted bad probability × $4,940 loss, vs. predicted good probability × $2,323 profit) rather than approximating them with a 1:1 or 2:1 metric weighting.

**Recommendation:**
1. **Compare models** with AUC/Gini + calibration (done above) — threshold-independent, matches the segmented-policy use case.
2. **Don't use F1** to pick the operating threshold(s) — it implicitly assumes false positives and false negatives cost the same, and we now know that's off by a factor that itself varies with the volume mix.
3. **Set the actual approve/decline/segment cutoffs in Step 7 by maximizing expected $ profit per 1,000 applicants** directly, using each loan's predicted bad probability and the $2,323 / $4,940 figures above (refined further with grade or segment-level averages if warranted). This is a strict improvement over optimizing any classification metric as a proxy, since it's not a proxy — it's the actual objective from Section 4 of the design doc.

## Influential variables

`images/lr_importance.png` / `images/tree_importance.png` (full feature set) show `grade` and `int_rate` dominating both models — unsurprising, since LendingClub's own pricing already encodes a lot of risk information, but it makes the "what actually drives risk" story circular against a grade-based baseline (Step 3).

`images/lr_importance_excl_grade.png` / `images/tree_importance_excl_grade.png` are **separately refit** models (not just a filtered view of the same model — a model that already split on `int_rate` first would make everything else look artificially unimportant) with `grade`/`int_rate` excluded entirely. Excluding them only drops logistic regression AUC from 0.684 to 0.667 — borrower attributes alone retain most of the discriminative power. Top borrower-level drivers: `fico_avg` (protective), `term_months` and `loan_to_income` (risk-raising) for the tree; `fico_avg` (protective), `loan_to_income`, `dti`, and renting vs. owning for logistic regression.

`images/tree_diagram.png` shows the full-feature tree's structure — every split down to depth 3 is on `int_rate`, visually confirming the circularity point above.

## Feature selection by validation Brier score

A mock-up of greedy forward feature selection (`scripts/feature_selection.py`), using validation Brier score — not AUC — as the selection criterion, consistent with the calibration-first philosophy above: Step 7's profit math needs trustworthy probabilities, not just good rank-ordering. Each of the 20 curated candidates (16 numeric + 4 categorical, categorical variables added as whole one-hot blocks) is tried one at a time; the one that most improves validation Brier is kept, repeated until all 20 are ranked in order of marginal contribution.

**This is a mock-up, not a production pipeline**: single train/val split (no cross-validation), greedy rather than exhaustive search, reduced `max_iter` during the search for speed with a full-precision refit of the final selected model for fair comparison against the other two.

`images/feature_selection_brier.png` shows the full trajectory. The result: **validation Brier flatlines exactly at 0.12081 from 15 features onward** — the last 5 candidates (`delinq_2yrs`, `pub_rec`, `credit_history_months`, `annual_inc`, `loan_amnt`) add zero further calibration benefit on top of the other 15. Interesting order-of-entry detail: `grade` is selected *first* here (not `int_rate`, which enters 15th) — for pure calibration (as opposed to discrimination), the categorical grade bucket turns out to be a stronger early signal than the continuous rate.

**Practical takeaway:** a 15-feature logistic regression matches the full 20-feature model on every metric (AUC 0.684 vs 0.684, Brier 0.1208 vs 0.1208) while dropping 5 variables — a genuinely useful result for the design doc's interpretability goal (a model a credit committee can review benefits from fewer inputs at no measured cost).

## Files

- `images/roc_curve.png`, `images/pr_curve.png`, `images/calibration_curve.png`
- `images/lr_importance.png`, `images/lr_importance_excl_grade.png`
- `images/tree_importance.png`, `images/tree_importance_excl_grade.png`
- `images/tree_diagram.png`
- `images/feature_selection_brier.png`, `images/feature_selection_history.csv`
- `images/model_results_table.png` — visual comparison across all three models
- `images/model_results.json` — raw metrics for all three models
