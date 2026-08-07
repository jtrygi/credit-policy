# Data-Driven Credit Approval Policy: Project Description & Analysis Plan

**Prepared for:** Fifth Third Bank — Business Strategy & Optimization Analyst interview case study
**Dataset:** LendingClub public loan-level data (loan applications with actual repayment outcomes)

---

## 1. Background

Consumer lenders must decide, for every applicant, whether to approve the loan and — if approved — what risk-based tier or rate to assign. That decision is normally made off a single risk score and a static cutoff. This project asks whether a segmented, model-based approval policy can do better than that simple cutoff, and quantifies exactly how much better in dollar terms — mirroring the core loop of the Business Strategy & Optimization Analyst role: predict risk, segment, design policy, quantify P&L impact, and propose how to validate it before rollout.

## 2. Business Problem (framed the way a bank would frame it)

> "We currently approve personal loan applicants above a single risk-score cutoff. Is there a smarter, segmented policy that would improve portfolio profitability — either by cutting losses at the same approval volume, or approving more volume at the same loss rate — and how confident can we be that it would hold up in production?"

## 3. Scoped, Achievable Outcome

This project will **not** attempt to build a production-grade underwriting model or cover every product line in the JD (auto, card, mortgage). It is intentionally scoped to one product (personal installment loans) and one decision (approve/decline + tiering), so it can be completed end-to-end with a defensible, well-documented result rather than left partially finished across several products.

**Deliverable:** A recommended segmented approval policy, benchmarked against a naive single-cutoff baseline, with:
- A quantified expected profitability impact (in dollars per 1,000 applications, so it's scale-independent and portable to a real portfolio)
- An honest accounting of assumptions and where the estimate could be wrong
- A proposed in-market test design to validate the policy before full rollout

**Success criteria for the analysis itself** (how I'll judge whether the project achieved its goal):
1. The segmented policy shows a **measurable, defensible improvement** over the baseline cutoff on at least one axis (expected loss rate at equal approval volume, OR approval volume at equal loss rate) — "measurable" meaning the effect size clearly exceeds reasonable estimation noise, not just directionally positive.
2. The policy is **simple enough to operationalize** — a bank can't act on a policy that takes 15 variables and a black box to explain to a credit committee. The final recommended policy should be expressible as a small number of segments (roughly 4–6) with a clear rule for each.
3. The business case is **stress-tested**, not just point-estimated — i.e., I show how the profitability estimate moves under pessimistic assumptions (losses run hotter, fewer applicants convert, etc.), not just the best-case number.
4. There is a **concrete, statistically grounded plan** for how you'd actually test this policy on real applicants before rolling it out fully, including what sample size you'd need and what would make you stop or scale it.

If the analysis instead showed the segmented policy does *not* beat the baseline, that would still be a successful, honest outcome — the point is a rigorous decision process, not a predetermined answer.

## 4. Decision Criteria (what "better" means, defined up front)

To avoid moving the goalposts after seeing results, the comparison is fixed in advance:

| Criterion | Baseline (current-state proxy) | Candidate (segmented policy) | How compared |
|---|---|---|---|
| Approval volume | Fixed at a reference cutoff (e.g., top X% by risk score) | Whatever volume the segmented policy approves | Hold one constant, compare the other |
| Expected loss rate | Loss rate of the baseline population | Loss rate of the segmented population | $ of expected charge-offs per 1,000 applicants |
| Expected revenue | Interest income on baseline-approved loans | Interest income on segmented-approved loans | $ of expected interest income per 1,000 applicants |
| Net expected profit | Revenue − expected loss − cost of funds, baseline | Same formula, segmented | This is the headline comparison number |
| Operational simplicity | Single cutoff (1 rule) | Segmented policy (target: ≤6 rules) | Judgment call, stated explicitly, not hidden |

The primary decision metric is **net expected profit per 1,000 applicants**. Approval volume and loss rate are reported as supporting detail so the trade-off is visible, not just the bottom line.

## 5. Analysis Plan

### Step 1 — Data acquisition & scoping
- Pull LendingClub loan-level data; restrict to a single product type (personal installment loans) and a cohort of vintages with enough seasoning that final loan status (fully paid / charged off / default) is known — avoids the "loans still open, outcome unknown" problem.
- Document exclusions explicitly (e.g., loans still current with insufficient history) so the sample isn't silently biased.

### Step 2 — Data cleaning & feature preparation
- Handle missing values, standardize categorical fields (employment length, purpose, home ownership, etc.)
- Engineer standard risk features: debt-to-income ratio, credit history length, revolving utilization, delinquency history, loan-to-income ratio
- Define the target variable: binary bad/good, where "bad" = charged off or defaulted, "good" = fully paid
- Split into train/validation/test sets *before* any modeling, to avoid leakage

### Step 3 — Baseline policy definition
- Establish the "current-state proxy": approve everyone above a single existing risk grade/score cutoff (mirrors how many lenders actually operate before adopting more sophisticated policy)
- Calculate baseline approval volume, loss rate, revenue, and net profit per 1,000 applicants — this is the number everything else is compared against

### Step 4 — Risk model: logistic regression (interpretable baseline model)
- Train a logistic regression predicting probability of default
- Chosen first because it's the model type a bank's model risk/compliance function can actually approve and explain to a credit committee or regulator — interpretability is a real business constraint here, not just a technical choice
- Evaluate discrimination (e.g., AUC/Gini) and calibration (do predicted probabilities match actual outcome rates)

### Step 5 — Risk model: gradient-boosted tree (challenger model)
- Train a tree-based ML model on the same features and target
- Compare performance lift vs. the logistic regression
- Explicitly evaluate the trade-off: how much predictive lift does the ML model provide, and is that lift worth the reduced interpretability and added model-governance burden in a regulated lending context — this comparison is itself a deliverable, not just a model-selection footnote

### Step 6 — Segmentation
- Use a decision tree (or binned model output) to split applicants into a small number of risk tiers, chosen for business interpretability, not just statistical purity
- Validate that segments are stable and meaningfully different in bad rate (not just numerically distinct)

### Step 7 — Policy design
- For each segment, define an approve/decline rule and, if in scope, a differentiated pricing tier
- Recompute approval volume, loss rate, revenue, and net profit per 1,000 applicants under the segmented policy
- Compare directly against the Step 3 baseline using the fixed decision criteria from Section 4

### Step 8 — Business case & sensitivity analysis
- Present the net profit comparison as the headline result
- Stress-test under at least two pessimistic scenarios (e.g., realized losses run 20% above modeled, or macro downturn shifts the whole population riskier) to show the recommendation isn't fragile
- State assumptions plainly (cost of funds, no behavioral response from applicants, static risk relationships) so the limits of the estimate are visible, not hidden

### Step 9 — Proposed validation test design
- Since this is historical data, not a live portfolio, close with how the policy would actually be validated in production: a champion/challenger test design, the sample size needed for a statistically credible read, the guardrail metrics to monitor (e.g., complaint rates, early delinquency), and the decision rule for scaling or killing the test

### Step 10 — Monitoring plan (brief, forward-looking)
- Outline, at a high level, what a vintage/roll-rate monitoring view would look like post-launch to catch drift early — this doesn't need to be built out fully, just specified, to show the analysis doesn't stop at launch

## 6. Deliverables

1. A written analysis (this becomes the interview walkthrough document)
2. Supporting charts: risk distribution, model performance comparison, segment bad-rate table, profit comparison, sensitivity table
3. A one-page executive summary written the way you'd present to senior leadership: recommendation, headline number, key risk, and next step — this is arguably the most important artifact, since "communicate effectively with senior leadership" and "create convincing business cases" are explicit requirements in the JD

## 7. Explicit Limitations (stated up front, not discovered later)

- LendingClub is peer-to-peer lending, not a bank's on-balance-sheet portfolio — underwriting standards, funding costs, and regulatory context differ from Fifth Third's actual book. The project is a demonstration of method, not a literal transferable policy.
- Historical data reflects one economic environment; the sensitivity analysis is a partial substitute for testing across cycles, not a full replacement.
- No live A/B test is possible on historical data — Step 9 is a design proposal, not an executed test.

---

*Next step: begin with Step 1 (data acquisition) and Step 2 (cleaning), sharing intermediate results before moving to modeling, so each stage can be reviewed rather than delivered as a single finished output.*
