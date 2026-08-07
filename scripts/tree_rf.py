"""Random Forest on the IV-qualified pool, with permutation importance
(corrects impurity importance's known bias toward high-cardinality features
like sub_grade's 34 dummy columns). Saves results immediately after each
stage so a downstream failure can't lose completed work."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, roc_auc_score

from ensemble_common import build_ensemble_matrix
from metrics_extra import ks_statistic

HUE_RF = "#059669"
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

    print("Fitting Random Forest...", flush=True)
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=500,
                                 n_jobs=8, random_state=42)
    rf.fit(X_train, y_train)
    p_rf = rf.predict_proba(X_val)[:, 1]
    auc_rf, brier_rf = roc_auc_score(y_val, p_rf), brier_score_loss(y_val, p_rf)
    ks_rf, _ = ks_statistic(y_val, p_rf)
    print(f"Random Forest: AUC={auc_rf:.4f} Brier={brier_rf:.5f} KS={ks_rf:.4f}", flush=True)

    with open("images/rf_results.json", "w") as f:
        json.dump(dict(auc=auc_rf, brier=brier_rf, ks=ks_rf, n_features=len(cols)), f, indent=2, default=float)

    print("Computing permutation importance...", flush=True)
    perm = permutation_importance(rf, X_val, y_val, n_repeats=5, random_state=42,
                                    scoring="roc_auc", n_jobs=8)
    rf_importance = pd.Series(perm.importances_mean, index=cols).sort_values(ascending=False)
    rf_importance.to_csv("images/rf_permutation_importance.csv")

    top_rf = rf_importance.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_rf.index, top_rf.values, color=HUE_RF, zorder=3)
    ax.set_title("Random Forest: permutation importance (drop in val AUC)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean AUC decrease when feature is shuffled")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/rf_importance.png", dpi=150)
    plt.close(fig)

    print("\nWrote images/rf_results.json, images/rf_permutation_importance.csv, images/rf_importance.png")


if __name__ == "__main__":
    main()
