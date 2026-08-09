# Step 10: Monitoring Plan (Post-Launch)

Brief and forward-looking, per the design doc — specified, not built. The goal is to catch drift early enough to act on it, using signals available well before full charge-off outcomes mature (the same timing tension `STEP9_VALIDATION_TEST.md` raised for the validation test applies permanently in production).

## 1. Population & feature drift — Population Stability Index (PSI)

Every month, compare the live scored population's distribution — both the final PD score and each of the 24 (LR-24) or top ~20 (XGBoost, by SHAP importance) input features — against the development population (OOT train, `data/train_oot.csv`) using PSI, the standard metric for this:

PSI = Σ (actual% − expected%) × ln(actual% / expected%), computed over score/feature deciles.

Conventional thresholds: **<0.10** no meaningful shift, **0.10–0.25** moderate shift (investigate, no action yet), **>0.25** significant shift (escalate — the population the model now sees is no longer the population it was built on). Feature-level PSI catches upstream data problems specifically — e.g., a bureau-data vendor change that silently shifts `revol_util`'s distribution would show up here before it shows up in performance.

## 2. Vintage / roll-rate tracking

Track each origination month's cohort through delinquency stages over time (current → 30dpd → 60dpd → 90dpd → charge-off), the classic roll-rate view. Compare each new vintage's early roll-rate curve (first 90-120 days, available fast) against the roll-rate curves of the vintages the model was validated on (2015-2016, `OOT_VALIDATION.md`). A new vintage rolling faster into early delinquency than any validated vintage did is the earliest possible signal of either model degradation or a genuine macro shift (`STEP8_BUSINESS_CASE.md`'s macro-shock scenario, arriving for real rather than hypothetically).

## 3. Calibration monitoring

Re-run the same predicted-PD-vs-observed-bad-rate-by-decile check from `scripts/score_models.py` on a recurring quarterly cadence, restricted to cohorts old enough to have matured (mirrors the seasoning-cutoff logic from `SCOPING.md`). Both models were well-calibrated at launch (`score_models.py`'s decile check) — this monitors whether that holds up, not just whether it held at one point in time. Systematic over- or under-prediction by decile is a direct, actionable signal (unlike a raw AUC drop, which doesn't say which direction the model is wrong).

## 4. Discrimination tracking (AUC / KS / Gini by vintage)

Once each vintage matures enough for outcomes to be known, recompute AUC/KS (`scripts/metrics_extra.py`) for that vintage specifically and plot the trend across vintages. `OOT_VALIDATION.md` already establishes a real vintage-to-vintage baseline to compare against: AUC moved by only +0.003 to +0.005 between the 2014 validation vintage and the 2015-2016 test vintage for both models — a live vintage's discrimination dropping meaningfully below that established range (not just any decline) is the trigger, not an arbitrary absolute AUC floor.

## 5. Reason-code stability

Per `SHAP_EXPLAINABILITY.md`, track the distribution of top adverse-action reason codes issued each month. A sudden shift in which features dominate declines (e.g., `int_rate` suddenly losing its top-reason share to a feature that was previously minor) usually indicates a pipeline or data-quality problem before it indicates a real change in applicant risk — this is a fast, cheap check to run continuously.

## 6. Trigger thresholds and actions

| Signal | Watch | Escalate | Likely action |
|---|---|---|---|
| Score/feature PSI | 0.10-0.25 | >0.25 | Investigate source; if population genuinely shifted, consider redevelopment |
| Early roll-rate (90-day) | Above validated range | Above validated range for 2+ consecutive vintages | Pause volume growth; investigate before scaling further |
| Calibration (decile) | Predicted vs. actual gap >2pp in any decile | Systematic gap across most deciles | Recalibrate (simple) or redevelop (if driver has shifted) |
| Vintage AUC/KS | Below established range | Below established range for 2+ consecutive vintages | Formal model review |
| Reason-code mix | Notable shift, no data explanation found | Persists after investigation | Pipeline audit; possible redevelopment |

## 7. Redevelopment cadence

Independent of drift signals, a full model redevelopment (repeat Steps 1-9 on current data) on a fixed schedule — annually is a common baseline for consumer credit — is standard hygiene, not just a drift response. Waiting for a trigger before ever refreshing the model risks always being reactive; a scheduled refresh, cross-checked against everything above, keeps improvements moving even when nothing looks obviously broken.

## Why this is only specified, not built

Every check above is a light extension of infrastructure that already exists in this project (`scripts/score_models.py`'s calibration check, `scripts/metrics_extra.py`'s AUC/KS, `OOT_VALIDATION.md`'s vintage comparison, `SHAP_EXPLAINABILITY.md`'s reason codes) — the point of Step 10 is showing the analysis doesn't stop at launch, not standing up a production monitoring system against data that doesn't exist yet (there is no live portfolio to monitor).
