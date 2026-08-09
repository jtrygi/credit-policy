# Step 9: Proposed Validation Test Design

Everything so far is retrospective. This closes the design doc's Step 9 requirement: how would you actually validate the segmented policy on real, live applicants before rolling it out fully — a champion/challenger design, the sample size needed for a credible read, the guardrails to watch, and the rule for scaling or killing it.

## The design: randomize only the disagreement region

**Champion** = current policy (baseline grade A-D cutoff). **Challenger** = model-ranked policy at the equal-loss-rate operating point (`SEGMENTATION_POLICY.md`/`STEP8_BUSINESS_CASE.md`).

A naive test randomizes the whole applicant population into champion vs. challenger arms and compares overall bad rates. That wastes almost all of its power: `scripts/validation_test_design.py` shows champion and challenger only actually disagree on a small minority of applicants —

| | XGBoost vs. champion | LR-24 vs. champion |
|---|---|---|
| Both approve / both decline (no test needed) | 95.5% | 95.1% |
| **Disagreement region** | **4.5%** (14,834 applicants) | **4.9%** (16,275 applicants) |

`images/validation_disagreement_region.png`. **The efficient design randomizes (or at minimum, analyzes) only this disagreement region** — applicants where the champion and challenger reach different accept/decline decisions. Where both policies already agree, there's nothing to learn from randomizing, and doing so only dilutes the signal and adds unnecessary approvals of loans nobody is questioning.

Concretely: for applicants who fall in the disagreement region, randomly assign each one to be treated according to champion's decision or challenger's decision. The "champion declines, challenger approves" applicants who land in the challenger arm are the real treatment group — loans that wouldn't exist today, actually originated, with real repayment behavior observed. This also **limits downside exposure**: you're testing on ~4.5% of volume, not gambling the whole book on an unproven policy.

## Sample size

Using the historical (OOT, 2015-2016) bad rates actually realized within the disagreement region as the effect-size estimate — a real strength of having retrospective ground truth for a group a live test would otherwise have to guess about:

| | XGBoost | LR-24 |
|---|---|---|
| "Champion approves, challenger declines" bad rate | 35.7% | 32.8% |
| "Champion declines, challenger approves" bad rate | 30.4% | 31.2% |
| Bad-rate gap within disagreement region | **5.24pp** | **1.61pp** |
| n required per arm (α=0.05, power=0.80) | **1,263** | **13,106** |
| Total test size | **2,526** | **26,213** |

**This is a genuinely important finding, not just a formality: XGBoost's disagreement-region decisions are far better risk-differentiated than LR-24's (5.24pp gap vs. 1.61pp), which means XGBoost's edge isn't just larger in dollar terms — it's roughly 10x easier to statistically prove in a live test.** At LendingClub's own historical origination pace in this window (~1,060 disagreement-eligible applicants/month for XGBoost, ~1,163/month for LR-24, from 14,834/14,275 respectively over the 14-month OOT test window), accumulating the needed sample would take roughly **2-3 months for XGBoost vs. nearly 2 years for LR-24**. That gap is itself a real input to the interpretability-vs-lift decision: a policy that takes 2 years to validate is a much harder sell than one that takes a quarter, independent of the dollar figures already discussed.

(These are scale estimates from this project's own historical applicant flow, not a commitment about a real institution's actual volume — use them as a reference point, not a forecast.)

## The timing tension: fast enough sample, slow outcome maturation

Accumulating the required sample size is fast (months). **Knowing the true outcome is slow** — this project's own seasoning logic (`SCOPING.md`) required 36-month loans to season ~3 years and 60-month loans ~5 years before charge-off status is reliably known. A test that reads out its primary metric (full charge-off rate) can't fully conclude until loans season, regardless of how fast the sample accumulates. This is why guardrail metrics matter — not as a substitute for the primary read, but as an early-warning system while it matures.

## Guardrail metrics (fast signals, monitored continuously)

- **Early delinquency / first-payment default.** Whether a loan misses its first payment or hits 30+ days past due within the first 90-120 days is known almost immediately and is a well-established leading indicator of eventual charge-off — the fastest real signal available.
- **Approval rate drift.** The challenger arm's actual approval rate should track its designed rate; a silent drift signals an implementation bug or population shift, not a policy effect.
- **Adverse-action reason-code stability.** Per `SHAP_EXPLAINABILITY.md`, the top reason codes generated for challenger-arm declines should stay consistent with what the model actually learned — a sudden shift suggests a pipeline or data problem, not a real change in the applicant pool.
- **Complaint rate.** Consumer complaints (CFPB complaint database categories, or internal equivalents) for challenger-arm applicants, checked against the champion arm — a rise here can outrun the credit-performance read entirely and is a real, if noisier, signal.

## Decision rule

- **Any guardrail breach triggers an immediate pause**, independent of whether the primary metric has matured — early delinquency spiking, reason codes drifting, or complaints rising are all reasons to stop before waiting for full charge-off data.
- **Interim futility check**: once early-delinquency data is available (a few months in), compare its trend between arms as a leading proxy. If early delinquency in the "champion declines, challenger approves" cohort is running dramatically worse than the historical estimate that powered the test, stop early rather than waiting out the full seasoning window.
- **Scale decision**: requires the primary metric (charge-off rate difference in the disagreement region) to reach statistical significance at the pre-registered sample size, in the direction consistent with `STEP8_BUSINESS_CASE.md`'s stress-tested estimate, with no unresolved guardrail breach.
- **Kill decision**: any of (a) a guardrail breach that isn't a data/pipeline artifact, (b) the primary metric reads significantly opposite to the retrospective estimate, or (c) the interim futility check fails.

## Outputs

`scripts/validation_test_design.py`, `scripts/validation_test_chart.py`. `images/validation_test_design.json`, `images/validation_disagreement_region.png`.
