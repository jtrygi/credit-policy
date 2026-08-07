"""Mock-up of greedy forward feature selection using validation Brier score
as the selection criterion -- consistent with this project's calibration-first
philosophy (Step 7's profit math consumes predicted probabilities directly,
so a well-calibrated model matters more than a maximally discriminative one).

Candidates are the same 16 numeric + 4 categorical variables used in
train_models.py's curated feature set (152 raw LendingClub columns were
already narrowed down there -- this is a second-stage selection on top of
that curation, not a search over everything). Categorical variables are
added/removed as whole blocks (all their dummy columns together), and each
numeric variable brings its paired missing-flag along automatically.

This is a mock-up, not a production feature-selection pipeline: no cross-
validation, single train/val split, greedy (not exhaustive) search. Search
steps use a lower max_iter for speed; the final selected model is refit at
full precision on the same data as the other models for fair comparison.
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from train_models import (
    CATEGORICAL_FEATURES, GRID, HUE_LR, HUE_TREE, INK, MUTED,
    NUMERIC_FEATURES, load,
)

FEATURE_TO_MISSING = {
    "annual_inc": "annual_inc_missing", "delinq_2yrs": "delinq_2yrs_missing",
    "inq_last_6mths": "inq_last_6mths_missing", "open_acc": "open_acc_missing",
    "pub_rec": "pub_rec_missing", "revol_util": "revol_util_missing",
    "total_acc": "total_acc_missing",
    "credit_history_months": "credit_history_months_missing",
    "loan_to_income": "loan_to_income_missing", "emp_length_years": "emp_length_years_missing",
}
CANDIDATES = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # 20 candidate variable blocks


def build_block_columns(df, numeric_selected, categorical_selected, one_hot_cols=None):
    num_cols = []
    for f in numeric_selected:
        num_cols.append(f)
        if f in FEATURE_TO_MISSING:
            num_cols.append(FEATURE_TO_MISSING[f])
    num = df[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)
    if categorical_selected:
        cat = pd.get_dummies(df[categorical_selected].astype(str), drop_first=True)
    else:
        cat = pd.DataFrame(index=df.index)
    X = pd.concat([num, cat], axis=1)
    if one_hot_cols is not None:
        X = X.reindex(columns=one_hot_cols, fill_value=0)
    return X


def fit_eval_brier(train, val, feature_set, max_iter=300):
    numeric_sel = [f for f in feature_set if f in NUMERIC_FEATURES]
    categorical_sel = [f for f in feature_set if f in CATEGORICAL_FEATURES]
    X_train = build_block_columns(train, numeric_sel, categorical_sel)
    cols = X_train.columns.tolist()
    X_val = build_block_columns(val, numeric_sel, categorical_sel, one_hot_cols=cols)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_val)
    lr = LogisticRegression(max_iter=max_iter, random_state=42)
    lr.fit(Xtr, train["bad"].values)
    p = lr.predict_proba(Xv)[:, 1]
    return brier_score_loss(val["bad"].values, p), roc_auc_score(val["bad"].values, p)


def main():
    train = load("train")
    val = load("val")

    remaining = list(CANDIDATES)
    selected = []
    history = []

    t0 = time.time()
    while remaining:
        best = None
        for cand in remaining:
            brier, auc = fit_eval_brier(train, val, selected + [cand])
            if best is None or brier < best[1]:
                best = (cand, brier, auc)
        cand, brier, auc = best
        selected.append(cand)
        remaining.remove(cand)
        history.append(dict(step=len(selected), feature_added=cand, brier=brier, auc=auc))
        print(f"Step {len(selected):2d}: +{cand:26s} val Brier={brier:.5f}  AUC={auc:.4f}  "
              f"({time.time() - t0:.0f}s elapsed)")

    hist_df = pd.DataFrame(history)
    hist_df.to_csv("images/feature_selection_history.csv", index=False)

    best_row = hist_df.loc[hist_df["brier"].idxmin()]
    best_n = int(best_row["step"])
    selected_final = selected[:best_n]
    print(f"\nBest val Brier at step {best_n}: {best_row['brier']:.5f}")
    print(f"Selected features: {selected_final}")

    # --- Trajectory chart ---
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(hist_df["step"], hist_df["brier"], color=HUE_LR, marker="o", linewidth=2, zorder=3)
    ax.axvline(best_n, color=MUTED, linestyle="--", linewidth=1)
    ax.scatter([best_n], [best_row["brier"]], color=HUE_TREE, s=100, zorder=4,
               label=f"Best: {best_n} features (val Brier={best_row['brier']:.5f})")
    ax.set_xlabel("Feature added (greedy forward order)")
    ax.set_ylabel("Validation Brier score (lower is better)")
    ax.set_title("Forward feature selection by validation Brier score", fontsize=13, fontweight="bold")
    ax.set_xticks(hist_df["step"])
    ax.set_xticklabels(hist_df["feature_added"], rotation=60, ha="right", fontsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("images/feature_selection_brier.png", dpi=150)
    plt.close(fig)

    # --- Final selected-feature model, refit at full precision for fair comparison ---
    numeric_sel = [f for f in selected_final if f in NUMERIC_FEATURES]
    categorical_sel = [f for f in selected_final if f in CATEGORICAL_FEATURES]
    X_train = build_block_columns(train, numeric_sel, categorical_sel)
    cols = X_train.columns.tolist()
    X_val = build_block_columns(val, numeric_sel, categorical_sel, one_hot_cols=cols)
    y_train, y_val = train["bad"].values, val["bad"].values

    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(scaler.fit_transform(X_train), y_train)
    p = lr.predict_proba(scaler.transform(X_val))[:, 1]

    auc = roc_auc_score(y_val, p)
    gini = 2 * auc - 1
    ap = average_precision_score(y_val, p)
    brier = brier_score_loss(y_val, p)
    thresh = np.quantile(p, 0.80)
    pred = (p >= thresh).astype(int)
    prec, rec, f1 = precision_score(y_val, pred), recall_score(y_val, pred), f1_score(y_val, pred)
    cm = confusion_matrix(y_val, pred)

    result = dict(
        auc=auc, gini=gini, avg_precision=ap, brier=brier, threshold=thresh,
        precision=prec, recall=rec, f1=f1, confusion_matrix=cm.tolist(),
        approval_rate=1 - pred.mean(), bad_rate_declined=y_val[pred == 1].mean(),
        bad_rate_approved=y_val[pred == 0].mean(), n_features=len(selected_final),
        features=selected_final,
    )

    with open("images/model_results.json") as f:
        all_results = json.load(f)
    all_results["Logistic Regression (Brier-selected)"] = result
    with open("images/model_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\n=== Logistic Regression (Brier-selected, {len(selected_final)} features) ===")
    print(f"AUC={auc:.4f}  Gini={gini:.4f}  AP={ap:.4f}  Brier={brier:.4f}")


if __name__ == "__main__":
    main()
