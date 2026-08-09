# Step 8: Business Case & Sensitivity Analysis

## Headline result

This uses the **out-of-time (OOT) numbers** from `OOT_VALIDATION.md`, not the original random-split numbers in `SEGMENTATION_POLICY.md` — the OOT figures are the defensible estimate of what the edge actually looks like on genuinely unseen future loans, and that document explicitly recommends using them here.

**Recommended policy: equal-loss-rate framing.** Hold today's baseline loss rate fixed (grade A-D, 14.06% bad rate) and let the model's finer ranking approve more volume at that same risk tolerance, rather than holding volume fixed. On the OOT test set (332,261 loans from 2015-2016, never seen in training):

| | Baseline (grade A-D) | LR-24 policy | XGBoost policy |
|---|---|---|---|
| Volume | 96.2% | 96.4% | 96.8% |
| Bad rate | 14.06% | 14.06% (matched) | 14.06% (matched) |
| Net profit / 1,000 applicants | $800,946 | $808,144 | $816,120 |
| **Improvement over baseline** | — | **+$7,198** | **+$15,174** |

That's the headline: switching from grade-only bucketing to a model-ranked policy is worth an estimated **$7,200 (interpretable model) to $15,200 (XGBoost) per 1,000 applicants**, at the same risk tolerance the business already accepts today. Modest, real, and — per `OOT_VALIDATION.md` — the honestly-scoped version of this number, not the inflated same-era estimate.

## Stress-tested under two pessimistic scenarios

`scripts/stress_test.py`. For each pool (baseline's approved loans, each policy's approved loans), the realized $ economics are decomposed into three numbers — bad rate, average profit on a performing loan, average loss on a charged-off loan — and two of them are shocked independently:

- **Severity shock (+20% loss per charged-off loan).** The design doc's own suggested magnitude. I checked whether the vintage data could size this empirically instead (parallel to how the macro shock below was sized) — it can't: average realized loss per bad loan ranges $4,000-$5,800 across origination years with no clean macro-correlated pattern (documented in `SEGMENTATION_POLICY.md`'s stress-scenario planning). So this one stays a stated assumption, not a derived one.
- **Macro shock (bad rate x1.39, ~+39% relative).** This one *is* data-driven: this dataset contains LendingClub's actual 2007-2009 vintages, and 2008 — the worst full crisis year with enough volume to be meaningful (n=2,393) — had a realized bad rate of 20.7% against this project's OOT test population's 14.85%. Rather than guessing a downturn severity, this uses what LendingClub's own book actually did in a real crisis.
- **Combined**: both shocks applied together, as the worst case.

Mechanics: `avg_profit_good` is held fixed (a performing loan's payoff doesn't change under either shock); `bad_rate` is scaled by the macro multiplier and `avg_loss_bad` by the severity multiplier, independently, then net profit is recomputed from those three numbers. This is standard PD/LGD stress-test mechanics — the same idea banks use in CCAR-style stress testing — not a resimulation of individual loans.

| Scenario | Baseline | LR-24 policy | XGBoost policy | LR-24 edge | XGBoost edge |
|---|---|---|---|---|---|
| Base case | $800,946 | $808,144 | $816,120 | +$7,198 | +$15,174 |
| Severity shock (+20%) | $667,836 | $675,829 | $682,233 | +$7,993 | +$14,397 |
| Macro shock (x1.39) | $444,152 | $452,710 | $456,567 | +$8,558 | +$12,414 |
| Combined | $258,597 | $268,264 | $269,928 | **+$9,667** | **+$11,332** |

`images/stress_test_profit.png` (absolute profit) and `images/stress_test_delta.png` (edge over baseline).

## The actual robustness finding

Absolute profit collapses hard under stress — the combined scenario is a 68% drop from base case for everyone, baseline included. That's expected; a worse book is a worse book regardless of who's picking the loans. **The question that matters is whether the edge survives, and it does, robustly**: XGBoost's advantage over baseline stays in a tight $11,300-$15,200 band across every scenario, and LR-24's advantage actually *grows* under stress ($7,200 to $9,700). That's a mathematically sensible result, not a fluke — baseline and each policy approve almost the same set of loans (they differ only in how the riskiest few percent at the margin are chosen), so a proportional shock to bad rate or severity hits both pools by roughly the same proportion, leaving the absolute dollar gap between them fairly stable. **This is the "not fragile" answer Step 8's success criterion asks for**: the recommendation doesn't just hold up under stress, it holds up almost exactly as well in dollar terms as it does in the base case.

## Assumptions & limitations, stated plainly

- **Cost of funds is not modeled anywhere in this analysis.** This is a real omission, not a neutral one: the recommended policy approves slightly more volume than baseline (96.4-96.8% vs. 96.2%), so ignoring funding cost is marginally *more* generous to the recommended policy than to baseline. The effect is small given how close the volumes are, but it means the $7K-$15K edge is very slightly overstated, not understated.
- **No behavioral response modeled.** Approving more or fewer applicants, or at different terms, doesn't change who applies or how they behave in this analysis — no adverse selection, no reaction to a visibly different underwriting policy. Real deployment could see this drift.
- **Static risk relationships assumed.** Every number here assumes the PD relationships learned from 2007-2014 data continue to hold going forward. `OOT_VALIDATION.md` is direct evidence this holds reasonably well from 2014 into 2015-2016 — it is not evidence it holds indefinitely. This is exactly why Steps 9-10 (validation test design, monitoring) exist.
- **XGBoost's governance cost is real but narrower than "black box" implies.** `SHAP_EXPLAINABILITY.md` shows per-decision reason codes (the concrete ECOA/Reg B adverse-action requirement) are available for the XGBoost model — so the $7K-$15K/1,000 framed above as "the cost of interpretability" partly overstates the governance gap. It doesn't close it: disparate impact testing and full global auditability remain harder for the ensemble than for LR-24.

## Outputs

`scripts/stress_test.py`. `images/stress_test_results.{csv,json}`, `images/stress_test_profit.png`, `images/stress_test_delta.png`.
