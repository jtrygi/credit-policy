"""Gradient-boosted trees (XGBoost) on the IV-qualified pool. Originally
written for LightGBM, but LightGBM 4.7.0 crashes with a native access
violation on this machine even on trivial synthetic data and after a clean
reinstall -- a system-level dependency issue (likely a missing/mismatched
OpenMP runtime DLL), not something fixable from within this session.
XGBoost is a different native backend and works fine here."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score

from ensemble_common import build_ensemble_matrix
from metrics_extra import ks_statistic

HUE_GBM = "#7C3AED"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    X_train, y_train, X_val, y_val, cols = build_ensemble_matrix()

    print("Fitting XGBoost...", flush=True)
    gbm = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                             random_state=42, n_jobs=8)
    gbm.fit(X_train, y_train)
    p_gbm = gbm.predict_proba(X_val)[:, 1]
    auc_gbm, brier_gbm = roc_auc_score(y_val, p_gbm), brier_score_loss(y_val, p_gbm)
    ks_gbm, _ = ks_statistic(y_val, p_gbm)
    print(f"XGBoost: AUC={auc_gbm:.4f} Brier={brier_gbm:.5f} KS={ks_gbm:.4f}", flush=True)

    with open("images/gbm_results.json", "w") as f:
        json.dump(dict(auc=auc_gbm, brier=brier_gbm, ks=ks_gbm, n_features=len(cols)), f, indent=2, default=float)

    gbm_importance = pd.Series(gbm.feature_importances_, index=cols).sort_values(ascending=False)
    gbm_importance = gbm_importance / gbm_importance.sum()
    gbm_importance.to_csv("images/gbm_importance.csv")

    top_gbm = gbm_importance.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_gbm.index, top_gbm.values, color=HUE_GBM, zorder=3)
    ax.set_title("XGBoost: feature importance (gain, normalized)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Share of total gain")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/gbm_importance.png", dpi=150)
    plt.close(fig)

    print("\nWrote images/gbm_results.json, images/gbm_importance.csv, images/gbm_importance.png")


if __name__ == "__main__":
    main()
