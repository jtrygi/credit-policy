"""Point-level explainability for the XGBoost model via SHAP (TreeSHAP,
exact and fast for tree ensembles) -- the concrete answer to ECOA/Reg B's
adverse-action-reasons requirement (see the interpretability-hurdle
discussion this follows): a declined applicant is owed specific reasons,
and SHAP gives an exact per-applicant decomposition of the score into
feature contributions that sum to the model's output.

Uses the OOT (out-of-time) XGBoost v2 model -- the defensible one per
OOT_VALIDATION.md -- scored on a sample of the OOT test set (2015-2016
vintages, genuinely unseen in training).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from model_registry import load_model
from score_models_oot import build_full_matrix_oot, load_oot

SAMPLE_N = 5000
TOP_DEPENDENCE_FEATURES = ["int_rate", "dti", "revol_util", "fico_avg", "bc_util", "mo_sin_old_rev_tl_op"]


def main():
    model, _, meta = load_model("xgboost-full-features-earlystop", "v2")
    train, val, test = load_oot("train"), load_oot("val"), load_oot("test")
    _, _, _, _, X_test, y_test, cols = build_full_matrix_oot(train, val, test)
    X_test = X_test.reindex(columns=meta["feature_cols"], fill_value=0)

    p_test = model.predict_proba(X_test)[:, 1]

    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_test), size=SAMPLE_N, replace=False)
    X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
    p_sample = p_test[sample_idx]

    print(f"Computing TreeSHAP values for {SAMPLE_N:,} OOT test applicants "
          f"({X_sample.shape[1]} features)...", flush=True)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_sample)
    print("Done.", flush=True)

    # --- Global summary (beeswarm): which features matter, and which direction ---
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.title("XGBoost (OOT model): global feature impact on predicted default risk", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("images/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Wrote images/shap_summary.png")

    # --- Global bar (mean |SHAP|): a plainer ranked-importance view ---
    plt.figure(figsize=(10, 8))
    shap.plots.bar(explanation, max_display=20, show=False)
    plt.title("XGBoost (OOT model): mean |SHAP value| by feature", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("images/shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Wrote images/shap_bar.png")

    # --- Dependence plots for top numeric features ---
    for feat in TOP_DEPENDENCE_FEATURES:
        if feat not in X_sample.columns:
            continue
        plt.figure(figsize=(8, 5.5))
        shap.plots.scatter(explanation[:, feat], color=explanation, show=False)
        plt.title(f"XGBoost (OOT model): SHAP dependence -- {feat}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"images/shap_dependence_{feat}.png", dpi=150)
        plt.close()
    print(f"Wrote images/shap_dependence_*.png for {TOP_DEPENDENCE_FEATURES}")

    # --- Individual waterfall plots: safest, riskiest, borderline applicant ---
    order = np.argsort(p_sample)
    safest_i = order[0]
    riskiest_i = order[-1]
    borderline_i = order[len(order) // 2]  # median-PD applicant as the "borderline" example

    examples = [("safest", safest_i, "Safest applicant in sample"),
                ("riskiest", riskiest_i, "Riskiest applicant in sample"),
                ("borderline", borderline_i, "Median-risk (borderline) applicant")]

    for tag, i, title in examples:
        plt.figure(figsize=(12, 7))
        shap.plots.waterfall(explanation[i], max_display=15, show=False)
        plt.title(f"{title} -- predicted PD={p_sample[i]:.1%}", fontsize=11, fontweight="bold", pad=40)
        plt.tight_layout()
        plt.savefig(f"images/shap_waterfall_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Wrote images/shap_waterfall_{tag}.png (predicted PD={p_sample[i]:.2%})")

    # --- Save top reason codes for the three examples as a JSON record ---
    reason_codes = {}
    for tag, i, title in examples:
        contribs = pd.Series(explanation.values[i], index=X_sample.columns)
        top_pos = contribs.sort_values(ascending=False).head(5)
        top_neg = contribs.sort_values().head(5)
        reason_codes[tag] = dict(
            predicted_pd=float(p_sample[i]),
            base_value=float(explanation.base_values[i]),
            top_risk_increasing=[{"feature": k, "shap": float(v), "value": float(X_sample.iloc[i][k])} for k, v in top_pos.items()],
            top_risk_reducing=[{"feature": k, "shap": float(v), "value": float(X_sample.iloc[i][k])} for k, v in top_neg.items()],
        )
    with open("images/shap_reason_codes.json", "w") as f:
        json.dump(reason_codes, f, indent=2)
    print("Wrote images/shap_reason_codes.json")


if __name__ == "__main__":
    main()
