# Point-Level Explainability for XGBoost (SHAP)

## Why this exists

The interpretability-vs-lift tradeoff quantified in `SEGMENTATION_POLICY.md` and `STEP8_BUSINESS_CASE.md` treated XGBoost as effectively a black box, with LR-24 as the only credit-committee-explainable option. That framing is only half right. **ECOA/Regulation B requires specific, accurate reasons for an adverse action** (a denial or worse-terms offer) — not full model transparency. That's a narrower, concrete bar, and **SHAP (TreeSHAP)** clears it directly for tree ensembles: it decomposes any individual prediction into exact per-feature contributions that sum to the model's output, giving a real, defensible reason list for any specific applicant.

This does not fully close the governance gap (see the caveat at the end) — but it means the earlier "XGBoost = ungovernable" framing overstated the cost.

## Method

`scripts/shap_explain.py`, using `shap.TreeExplainer` (exact, not approximate, for tree models) on the **OOT XGBoost v2 model** (the defensible one per `OOT_VALIDATION.md`), scored on a 5,000-applicant sample of the OOT test set (2015-2016 vintages, unseen in training).

## Global picture: does the model behave sensibly?

`images/shap_summary.png` — every top feature's direction matches domain intuition, which matters for the "conceptual soundness" pillar of model risk review, not just adverse-action reasons: higher `int_rate` → higher risk, higher `annual_inc` → lower risk, higher `loan_to_income` → higher risk, higher `fico_avg` → lower risk, more `acc_open_past_24mths` → higher risk. Nothing counterintuitive dominates. `images/shap_bar.png` gives the same ranking as a plain bar chart; `images/shap_dependence_int_rate.png` (and 5 other features) show the shape of each relationship — `int_rate`'s is smoothly monotonic up to ~20%, then flattens, which is itself a useful, reportable finding about the model's behavior.

## Individual reason codes: three examples

`images/shap_waterfall_{safest,riskiest,borderline}.png`, with the underlying numbers in `images/shap_reason_codes.json`.

**Riskiest applicant in sample (predicted PD = 51.1%).** Primary risk-increasing factors: `int_rate` (17.57%), `loan_to_income` (0.44), low `total_bc_limit` ($500), 6 accounts opened in the past 24 months, low `total_rev_hi_lim` ($4,900). This is exactly the form an adverse-action notice needs: specific, ranked, and tied to real applicant data.

**Borderline applicant (predicted PD = 9.7%, near the segment 3/4 boundary from `SEGMENTATION_POLICY.md`).** Risk-increasing: low `annual_inc` ($35,000), 2 recent inquiries, elevated `bc_util` (48.2%). Risk-reducing: moderate `int_rate` (11.53%), 36-month term, low `dti` (6.93%). This is the case where explanations matter most in practice — a clear approve or clear decline barely needs justifying, but a marginal case is exactly where an applicant (or a regulator) will ask why.

**Safest applicant in sample (predicted PD = 0.5%).** Dominated by a low `int_rate` (5.32%, LendingClub's own pricing already recognized this applicant as low-risk) and a high `fico_avg` (842).

## What this does and doesn't fix

**Fixes:** per-decision adverse-action reason generation (ECOA/Reg B), and gives a validator something concrete to interrogate for "effective challenge" (SR 11-7) — they can now ask "why did the model do this" for any applicant and get an exact, auditable answer, not a shrug.

**Doesn't fix:** disparate impact / fair lending testing (a separate, portfolio-level statistical question — this project has already established the data doesn't support it, see the fair-lending data-availability discussion) is completely untouched by this. Nor does it give the same single global statement a linear coefficient does ("higher DTI always increases risk by X") — SHAP contributions vary by applicant due to interactions, so "explainable" here means explainable-per-decision, not simplified-to-a-formula. The governance burden of validating and monitoring a 247-feature, hundreds-of-trees ensemble is still real; SHAP narrows the interpretability objection, it doesn't eliminate the broader model-risk-management cost difference from LR-24.

## Outputs

`scripts/shap_explain.py`. `images/shap_summary.png`, `images/shap_bar.png`, `images/shap_dependence_{int_rate,dti,revol_util,fico_avg,bc_util,mo_sin_old_rev_tl_op}.png`, `images/shap_waterfall_{safest,riskiest,borderline}.png`, `images/shap_reason_codes.json`.
