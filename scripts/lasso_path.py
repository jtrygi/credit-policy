"""LASSO (L1-regularized logistic regression) path on the same IV-qualified
candidate pool as wrapper_selection.py -- the modern alternative to classical
stepwise selection: instead of a single greedy walk, shows every feature's
coefficient shrinking to exactly zero as regularization strengthens, across
the whole path at once.

Categorical dummy blocks (grade, sub_grade especially -- 34 dummy columns on
its own) are included in the fit for a fair comparison against the wrapper
methods, but plotted as an unlabeled backdrop -- 83 individual spaghetti
lines with a full legend would be unreadable. Numeric features get direct
labels; categorical block survival is reported in a companion table instead.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from metrics_extra import ks_statistic

HUE = "#2563EB"
CAT_HUE = "#D1D5DB"
ACCENT = "#DC2626"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"


def load(name):
    df = pd.read_csv(f"data/{name}.csv", low_memory=False)
    df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    return df


def main():
    with open("images/qualified_features.json") as f:
        candidates = json.load(f)

    train = load("train")
    val = load("val")
    numeric = [c for c in candidates if c == "fico_avg" or train[c].dtype != object]
    categorical = [c for c in candidates if c not in numeric]

    num_cols = []
    for f in numeric:
        num_cols.append(f)
        flag = f"{f}_missing"
        if flag in train.columns:
            num_cols.append(flag)

    X_train_num = train[num_cols].copy()
    X_train_cat = pd.get_dummies(train[categorical].astype(str), drop_first=True)
    X_train = pd.concat([X_train_num, X_train_cat], axis=1)
    cols = X_train.columns.tolist()
    cat_dummy_cols = X_train_cat.columns.tolist()
    numeric_cols_only = [c for c in cols if c not in cat_dummy_cols]

    X_val_num = val[num_cols].copy()
    X_val_cat = pd.get_dummies(val[categorical].astype(str), drop_first=True).reindex(columns=cat_dummy_cols, fill_value=0)
    X_val = pd.concat([X_val_num, X_val_cat], axis=1).reindex(columns=cols, fill_value=0)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_val)
    y_train, y_val = train["bad"].values, val["bad"].values

    C_grid = np.logspace(-4, 1, 25)
    coef_paths = []
    metrics = []
    for C in C_grid:
        lr = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=1000, random_state=42)
        lr.fit(Xtr, y_train)
        coef_paths.append(lr.coef_[0].copy())
        p = lr.predict_proba(Xv)[:, 1]
        auc = roc_auc_score(y_val, p)
        brier = brier_score_loss(y_val, p)
        ks, _ = ks_statistic(y_val, p)
        n_nonzero = int((lr.coef_[0] != 0).sum())
        metrics.append(dict(C=C, auc=auc, brier=brier, ks=ks, n_nonzero=n_nonzero))
        print(f"C={C:.5f}  AUC={auc:.4f}  Brier={brier:.5f}  KS={ks:.4f}  nonzero={n_nonzero}/{len(cols)}", flush=True)

    coef_paths = np.array(coef_paths)  # shape (n_C, n_features)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("images/lasso_path_metrics.csv", index=False)

    best_idx = metrics_df["auc"].idxmax()
    best_C = metrics_df.loc[best_idx, "C"]
    best_coefs = coef_paths[best_idx]
    nonzero_at_best = [c for c, v in zip(cols, best_coefs) if v != 0]
    nonzero_numeric = [c for c in nonzero_at_best if c in numeric_cols_only]
    print(f"\nBest val AUC={metrics_df.loc[best_idx,'auc']:.4f} at C={best_C:.5f}, "
          f"{len(nonzero_at_best)}/{len(cols)} nonzero coefficients")

    # Categorical block survival at best C
    cat_survival = {}
    for cvar in categorical:
        block_cols = [c for c in cat_dummy_cols if c.startswith(cvar + "_")]
        nonzero_in_block = [c for c in block_cols if cols.index(c) < len(best_coefs) and best_coefs[cols.index(c)] != 0]
        cat_survival[cvar] = f"{len(nonzero_in_block)}/{len(block_cols)} levels nonzero"

    with open("images/lasso_selected.json", "w") as f:
        json.dump({
            "best_C": float(best_C), "auc": float(metrics_df.loc[best_idx, "auc"]),
            "brier": float(metrics_df.loc[best_idx, "brier"]), "ks": float(metrics_df.loc[best_idx, "ks"]),
            "nonzero_numeric_features": nonzero_numeric,
            "categorical_survival": cat_survival,
            "n_nonzero_total": len(nonzero_at_best),
        }, f, indent=2)

    # --- Coefficient path chart ---
    fig, ax = plt.subplots(figsize=(11, 7))
    log_C = np.log10(C_grid)
    for j, c in enumerate(cols):
        if c in numeric_cols_only:
            continue
        ax.plot(log_C, coef_paths[:, j], color=CAT_HUE, linewidth=0.8, zorder=1)

    # Distinct colors for numeric features via a repeating small palette (dataviz: don't
    # cycle indefinitely for many series -- direct-label only the most significant ones)
    numeric_final_abs = {c: abs(best_coefs[cols.index(c)]) for c in numeric_cols_only}
    top_numeric = sorted(numeric_final_abs, key=numeric_final_abs.get, reverse=True)[:8]

    for c in numeric_cols_only:
        j = cols.index(c)
        color = HUE if c in top_numeric else "#93C5FD"
        lw = 2 if c in top_numeric else 1
        ax.plot(log_C, coef_paths[:, j], color=color, linewidth=lw, zorder=2)
        if c in top_numeric:
            ax.annotate(c, (log_C[-1], coef_paths[-1, j]), fontsize=8, color=INK,
                        xytext=(4, 0), textcoords="offset points", va="center")

    ax.axvline(np.log10(best_C), color=ACCENT, linewidth=1.5, linestyle="--",
               label=f"Selected C={best_C:.4f} (best val AUC)")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("log10(C)  (stronger regularization →  ←  weaker regularization)")
    ax.set_ylabel("Standardized coefficient")
    ax.set_title("LASSO regularization path (numeric features labeled; categorical\n"
                  "dummy blocks shown unlabeled in gray -- see companion table)",
                  fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/lasso_path.png", dpi=150)
    plt.close(fig)

    # --- AUC / n_nonzero vs C chart ---
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(log_C, metrics_df["auc"], color=HUE, linewidth=2, marker="o", markersize=3)
    ax1.axvline(np.log10(best_C), color=ACCENT, linewidth=1.5, linestyle="--")
    ax1.set_xlabel("log10(C)")
    ax1.set_ylabel("Validation AUC")
    ax1.set_title("LASSO: validation AUC vs regularization strength", fontsize=12, fontweight="bold")
    ax1.grid(color=GRID, linewidth=0.8)
    ax1.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/lasso_auc_vs_c.png", dpi=150)
    plt.close(fig)

    print("\nWrote images/lasso_path.png, images/lasso_auc_vs_c.png, images/lasso_selected.json")


if __name__ == "__main__":
    main()
