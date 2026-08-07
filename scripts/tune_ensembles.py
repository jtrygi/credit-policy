"""Hyperparameter tuning for the nonparametric models (Random Forest,
XGBoost) via randomized search, scored by validation AUC -- consistent with
the wrapper-selection methodology (AUC drives the search; Brier/KS are
logged from the same fits at no extra cost).

Deliberately NOT k-fold cross-validated: this project already committed to
a single train/val/test split (CLEANING.md), and re-deriving hyperparameters
under k-fold CV while every other model comparison in this project uses the
same fixed val set would make the two incomparable. The tradeoff is a
noisier per-trial score than CV would give; with 110K validation rows the
noise is small relative to the differences we're searching over.

Decision Tree is intentionally excluded -- its max_depth=4 is a deliberate
interpretability constraint (design doc: "credit committee can review it"),
not an oversight to tune away.
"""
import json
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from ensemble_common import build_ensemble_matrix
from extend_metrics import full_metrics

HUE_RF = "#059669"
HUE_GBM = "#7C3AED"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"

RF_GRID = dict(
    n_estimators=[100, 200, 300, 500],
    max_depth=[4, 6, 8, 10, 12, None],
    min_samples_leaf=[50, 100, 300, 500, 1000],
    max_features=["sqrt", "log2", 0.3, 0.5, 0.7],
)

XGB_GRID = dict(
    n_estimators=[100, 200, 300, 500, 800],
    max_depth=[3, 4, 5, 6, 8],
    learning_rate=[0.01, 0.02, 0.05, 0.1, 0.2],
    subsample=[0.6, 0.7, 0.8, 0.9, 1.0],
    colsample_bytree=[0.6, 0.7, 0.8, 0.9, 1.0],
    min_child_weight=[1, 3, 5, 10],
    reg_lambda=[0, 0.5, 1, 2, 5],
    reg_alpha=[0, 0.1, 0.5, 1],
)


def sample_params(grid, rng):
    return {k: rng.choice(v) for k, v in grid.items()}


def random_search(model_name, fit_fn, grid, n_trials, X_train, y_train, X_val, y_val, seed=42):
    rng = random.Random(seed)
    history = []
    best = None
    t0 = time.time()
    for trial in range(1, n_trials + 1):
        params = sample_params(grid, rng)
        model = fit_fn(params)
        model.fit(X_train, y_train)
        p = model.predict_proba(X_val)[:, 1]
        m = full_metrics(y_val, p, X_train.shape[1])
        row = dict(trial=trial, **params, **m)
        history.append(row)
        if best is None or m["auc"] > best[1]["auc"]:
            best = (params, m, model)
        print(f"[{model_name}] Trial {trial:2d}/{n_trials}: AUC={m['auc']:.4f} Brier={m['brier']:.5f} "
              f"KS={m['ks']:.4f}  best_so_far={best[1]['auc']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return best, pd.DataFrame(history)


def plot_convergence(history_df, model_name, color, filename):
    history_df = history_df.copy()
    history_df["best_so_far"] = history_df["auc"].cummax()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(history_df["trial"], history_df["auc"], color=MUTED, s=20, alpha=0.6, zorder=2, label="Trial AUC")
    ax.step(history_df["trial"], history_df["best_so_far"], color=color, linewidth=2, where="post",
            zorder=3, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Validation AUC")
    ax.set_title(f"{model_name}: randomized hyperparameter search", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def main():
    X_train, y_train, X_val, y_val, cols = build_ensemble_matrix()

    def make_rf(params):
        p = dict(params)
        return RandomForestClassifier(n_jobs=8, random_state=42, **p)

    def make_xgb(params):
        p = dict(params)
        return xgb.XGBClassifier(random_state=42, n_jobs=8, **p)

    print("=== Random Forest random search (25 trials) ===", flush=True)
    rf_best, rf_history = random_search("RF", make_rf, RF_GRID, 25, X_train, y_train, X_val, y_val)
    rf_history.to_csv("images/tune_rf_history.csv", index=False)
    plot_convergence(rf_history, "Random Forest", HUE_RF, "images/tune_rf_convergence.png")

    print("\n=== XGBoost random search (40 trials) ===", flush=True)
    xgb_best, xgb_history = random_search("XGBoost", make_xgb, XGB_GRID, 40, X_train, y_train, X_val, y_val)
    xgb_history.to_csv("images/tune_xgb_history.csv", index=False)
    plot_convergence(xgb_history, "XGBoost", HUE_GBM, "images/tune_xgb_convergence.png")

    with open("images/all_model_results.json") as f:
        all_results = json.load(f)

    all_results["Random Forest (tuned)"] = rf_best[1]
    all_results["Random Forest (tuned)"]["best_params"] = {k: (v if not isinstance(v, type(None)) else None)
                                                              for k, v in rf_best[0].items()}
    all_results["XGBoost (tuned)"] = xgb_best[1]
    all_results["XGBoost (tuned)"]["best_params"] = xgb_best[0]

    with open("images/all_model_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\nBest RF: AUC={rf_best[1]['auc']:.4f}  params={rf_best[0]}")
    print(f"Best XGBoost: AUC={xgb_best[1]['auc']:.4f}  params={xgb_best[0]}")
    print("\nWrote images/tune_rf_*, images/tune_xgb_*, updated images/all_model_results.json")


if __name__ == "__main__":
    main()
