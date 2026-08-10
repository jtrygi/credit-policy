"""Builds the compact JSON dataset embedded in the published interactive
model explorer artifact. Two parts:

  1. Row-level sample: the EXACT SAME 5,000-row seed-42 sample from
     shap_explain.py (same rng call, same order) so SHAP reason codes can
     be computed for these specific rows without a second random draw
     that could drift out of alignment with a differently-sampled set.
  2. Vintage aggregates: full 2007-2016 origination-year bad rate/volume,
     for the monitoring-style panel -- computed directly from
     scoped_accepted.csv, not from the 5,000-row sample (this doesn't need
     row-level data, and 5,000 rows would be a noisy, non-representative
     slice for a 10-year trend anyway).

Both models' predicted PD are included (a model toggle in the explorer),
plus the top-5 positive/negative SHAP contributors per row for the
XGBoost model specifically (the LR-24 coefficients are already a direct,
literal explanation -- SHAP's marginal value is for the ensemble).
"""
import json

import numpy as np
import pandas as pd
import shap

from model_registry import load_model
from prepare_data import TARGET_BAD, TARGET_GOOD
from score_models_oot import build_full_matrix_oot, load_oot

SAMPLE_N = 5000
DISPLAY_FEATURES = ["int_rate", "dti", "revol_util", "fico_avg", "annual_inc",
                     "loan_to_income", "term_months", "purpose", "home_ownership",
                     "verification_status", "emp_length_years"]


def build_row_sample():
    lr_model, lr_scaler, lr_meta = load_model("lr-forward-selected-24", "v2")
    xgb_model, _, xgb_meta = load_model("xgboost-full-features-earlystop", "v2")

    train, val, test = load_oot("train"), load_oot("val"), load_oot("test")
    policy = pd.read_csv("data/test_oot_policy.csv")
    assert len(policy) == len(test) and (policy["bad"].values == test["bad"].values).all()

    _, _, _, _, X_test_full, y_test, cols = build_full_matrix_oot(train, val, test)
    X_test_full = X_test_full.reindex(columns=xgb_meta["feature_cols"], fill_value=0)
    p_xgb = xgb_model.predict_proba(X_test_full)[:, 1]

    import wrapper_selection as ws
    from score_models import LR24_FEATURES
    train_cols = set(train.columns)
    numeric_all, categorical_all = ws.get_candidate_types(train, LR24_FEATURES)
    numeric_sel = [f for f in LR24_FEATURES if f in numeric_all]
    categorical_sel = [f for f in LR24_FEATURES if f not in numeric_all]
    X_test_lr = ws.build_columns(test, numeric_sel, categorical_sel, train_cols, one_hot_cols=lr_meta["feature_cols"])
    p_lr = lr_model.predict_proba(lr_scaler.transform(X_test_lr))[:, 1]

    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(X_test_full), size=SAMPLE_N, replace=False)

    print(f"Computing TreeSHAP for {SAMPLE_N:,} rows (explorer sample)...", flush=True)
    explainer = shap.TreeExplainer(xgb_model)
    X_sample = X_test_full.iloc[sample_idx].reset_index(drop=True)
    explanation = explainer(X_sample)
    print("Done.", flush=True)

    test_display = test.iloc[sample_idx].reset_index(drop=True)
    test_display["fico_avg"] = (test_display["fico_range_low"] + test_display["fico_range_high"]) / 2
    policy_display = policy.iloc[sample_idx].reset_index(drop=True)

    rows = []
    for i in range(SAMPLE_N):
        contribs = pd.Series(explanation.values[i], index=X_sample.columns)
        top_pos = contribs.sort_values(ascending=False).head(4)
        top_neg = contribs.sort_values().head(4)
        row = dict(
            id=int(i),
            bad=int(test_display["bad"].iloc[i]),
            grade=str(policy_display["grade"].iloc[i]),
            net_realized=round(float(policy_display["net_realized"].iloc[i]), 2),
            lr24_pd=round(float(p_lr[sample_idx[i]]), 4),
            xgb_pd=round(float(p_xgb[sample_idx[i]]), 4),
            reasons_up=[{"f": k, "v": round(float(v), 3)} for k, v in top_pos.items()],
            reasons_down=[{"f": k, "v": round(float(v), 3)} for k, v in top_neg.items()],
        )
        for feat in DISPLAY_FEATURES:
            val = test_display[feat].iloc[i]
            row[feat] = round(float(val), 2) if isinstance(val, (int, float, np.floating, np.integer)) else str(val)
        rows.append(row)

    return rows


def build_vintage_aggregates():
    df = pd.read_csv("data/scoped_accepted.csv", usecols=["issue_d", "loan_status"], low_memory=False)
    df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
    df["year"] = df["issue_dt"].dt.year
    df = df[df["loan_status"].isin(TARGET_BAD | TARGET_GOOD)].copy()
    df["bad"] = df["loan_status"].isin(TARGET_BAD).astype(int)
    tbl = df.groupby("year").agg(n=("bad", "size"), bad_rate=("bad", "mean")).reset_index()
    return [dict(year=int(r.year), n=int(r.n), bad_rate=round(float(r.bad_rate), 4)) for r in tbl.itertuples()]


def main():
    rows = build_row_sample()
    vintages = build_vintage_aggregates()

    payload = dict(
        rows=rows,
        vintages=vintages,
        meta=dict(
            n_rows=len(rows),
            sample_note="5,000-row seed-42 sample of the OOT test set (2015-2016 vintages, unseen in training)",
            overall_bad_rate=round(sum(r["bad"] for r in rows) / len(rows), 4),
        ),
    )
    with open("images/explorer_data.json", "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    import os
    size_kb = os.path.getsize("images/explorer_data.json") / 1024
    print(f"\nWrote images/explorer_data.json ({size_kb:.0f} KB, {len(rows):,} rows, {len(vintages)} vintage years)")


if __name__ == "__main__":
    main()
