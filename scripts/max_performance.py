"""Max-performance experiment: interpretability constraint dropped, full
~87-column feature set (not the 30 IV-qualified ones), and a deliberately
overfit neural net to characterize (a) how far training performance can be
pushed vs (b) what actually survives out-of-sample -- the gap between the
two IS the answer to "how much of that is real."

Everything below is scored on val.csv until the very last step, which
spends test.csv for the first time in this project on the single winning
candidate, since val has been reused many times now for model/feature
selection.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from full_feature_matrix import build_full_matrix, load
from metrics_extra import ks_statistic

HUE_TRAIN = "#93C5FD"
HUE_VAL = "#2563EB"
HUE_TEST = "#DC2626"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    train = load("train")
    val = load("val")
    test = load("test")
    X_train, y_train, X_val, y_val, X_test, y_test, cols = build_full_matrix(train, val, test)
    print(f"Full feature matrix: {len(cols)} columns (vs 104 for the IV-qualified pool)", flush=True)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xv_s = scaler.transform(X_val)

    results = {}

    # --- 1. Deliberately overfit MLP: small subsample, high capacity, no regularization ---
    print("\n=== Overfit MLP (50K subsample, no regularization, no early stopping) ===", flush=True)
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(X_train), size=50_000, replace=False)
    Xtr_sub, ytr_sub = Xtr_s[sub_idx], y_train[sub_idx]

    mlp_overfit = MLPClassifier(hidden_layer_sizes=(300, 150, 75), alpha=1e-8, max_iter=400,
                                 early_stopping=False, solver="adam", random_state=42)
    mlp_overfit.fit(Xtr_sub, ytr_sub)
    p_train_of = mlp_overfit.predict_proba(Xtr_sub)[:, 1]
    p_val_of = mlp_overfit.predict_proba(Xv_s)[:, 1]
    auc_train_of, auc_val_of = roc_auc_score(ytr_sub, p_train_of), roc_auc_score(y_val, p_val_of)
    results["MLP (overfit, 50K subsample)"] = dict(train_auc=auc_train_of, val_auc=auc_val_of,
                                                     gap=auc_train_of - auc_val_of)
    print(f"Train AUC={auc_train_of:.4f}  Val AUC={auc_val_of:.4f}  Gap={auc_train_of - auc_val_of:.4f}", flush=True)

    # --- 2. Regularized MLP with early stopping, full train set ---
    print("\n=== Regularized MLP (full train, early stopping) ===", flush=True)
    mlp_reg = MLPClassifier(hidden_layer_sizes=(100, 50), alpha=1e-2, max_iter=200,
                             early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                             solver="adam", random_state=42)
    mlp_reg.fit(Xtr_s, y_train)
    p_train_reg = mlp_reg.predict_proba(Xtr_s)[:, 1]
    p_val_reg = mlp_reg.predict_proba(Xv_s)[:, 1]
    auc_train_reg, auc_val_reg = roc_auc_score(y_train, p_train_reg), roc_auc_score(y_val, p_val_reg)
    ks_reg, _ = ks_statistic(y_val, p_val_reg)
    results["MLP (regularized, full train)"] = dict(train_auc=auc_train_reg, val_auc=auc_val_reg,
                                                      gap=auc_train_reg - auc_val_reg, ks=ks_reg,
                                                      n_iter=mlp_reg.n_iter_)
    print(f"Train AUC={auc_train_reg:.4f}  Val AUC={auc_val_reg:.4f}  Gap={auc_train_reg - auc_val_reg:.4f}  "
          f"(stopped at iter {mlp_reg.n_iter_})", flush=True)

    # --- 3. XGBoost, deliberately unregularized, full feature set ---
    print("\n=== XGBoost (overfit: deep, many rounds, no regularization) ===", flush=True)
    xgb_overfit = xgb.XGBClassifier(n_estimators=500, max_depth=12, learning_rate=0.1,
                                     min_child_weight=1, reg_lambda=0, reg_alpha=0,
                                     subsample=1.0, colsample_bytree=1.0, random_state=42, n_jobs=8)
    xgb_overfit.fit(X_train, y_train)
    p_train_xof = xgb_overfit.predict_proba(X_train)[:, 1]
    p_val_xof = xgb_overfit.predict_proba(X_val)[:, 1]
    auc_train_xof, auc_val_xof = roc_auc_score(y_train, p_train_xof), roc_auc_score(y_val, p_val_xof)
    results["XGBoost (overfit)"] = dict(train_auc=auc_train_xof, val_auc=auc_val_xof,
                                         gap=auc_train_xof - auc_val_xof)
    print(f"Train AUC={auc_train_xof:.4f}  Val AUC={auc_val_xof:.4f}  Gap={auc_train_xof - auc_val_xof:.4f}", flush=True)

    # --- 4. XGBoost, full features, early-stopped against val (properly regularized) ---
    print("\n=== XGBoost (full features, early-stopped) ===", flush=True)
    xgb_full = xgb.XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.03,
                                  min_child_weight=5, reg_lambda=2, subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, n_jobs=8, early_stopping_rounds=50, eval_metric="auc")
    xgb_full.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    p_train_xf = xgb_full.predict_proba(X_train)[:, 1]
    p_val_xf = xgb_full.predict_proba(X_val)[:, 1]
    auc_train_xf, auc_val_xf = roc_auc_score(y_train, p_train_xf), roc_auc_score(y_val, p_val_xf)
    ks_xf, _ = ks_statistic(y_val, p_val_xf)
    results["XGBoost (full features, early-stopped)"] = dict(
        train_auc=auc_train_xf, val_auc=auc_val_xf, gap=auc_train_xf - auc_val_xf,
        ks=ks_xf, best_iteration=int(xgb_full.best_iteration))
    print(f"Train AUC={auc_train_xf:.4f}  Val AUC={auc_val_xf:.4f}  Gap={auc_train_xf - auc_val_xf:.4f}  "
          f"(best round {xgb_full.best_iteration})", flush=True)

    with open("images/max_performance_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    # --- Chart: train vs val AUC, grouped bars ---
    names = list(results.keys())
    train_aucs = [results[n]["train_auc"] for n in names]
    val_aucs = [results[n]["val_auc"] for n in names]
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, train_aucs, width, color=HUE_TRAIN, label="Train AUC", zorder=3)
    ax.bar(x + width / 2, val_aucs, width, color=HUE_VAL, label="Val AUC", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("AUC")
    ax.set_title("Train vs. validation AUC: the overfitting gap, made visible", fontsize=13, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/overfit_gap.png", dpi=150)
    plt.close(fig)

    # --- Final: spend test.csv ONCE on the best val performer ---
    best_name = max(results, key=lambda n: results[n]["val_auc"])
    print(f"\n=== Best val performer: {best_name} -- evaluating on TEST (first use of test.csv) ===", flush=True)
    if best_name == "XGBoost (full features, early-stopped)":
        p_test = xgb_full.predict_proba(X_test)[:, 1]
    elif best_name == "XGBoost (overfit)":
        p_test = xgb_overfit.predict_proba(X_test)[:, 1]
    elif best_name == "MLP (regularized, full train)":
        p_test = mlp_reg.predict_proba(scaler.transform(X_test))[:, 1]
    else:
        p_test = mlp_overfit.predict_proba(scaler.transform(X_test))[:, 1]

    auc_test = roc_auc_score(y_test, p_test)
    ks_test, _ = ks_statistic(y_test, p_test)
    print(f"TEST AUC={auc_test:.4f}  TEST KS={ks_test:.4f}  "
          f"(val AUC was {results[best_name]['val_auc']:.4f} -- gap of {results[best_name]['val_auc'] - auc_test:+.4f})", flush=True)

    with open("images/final_test_evaluation.json", "w") as f:
        json.dump(dict(best_model=best_name, val_auc=results[best_name]["val_auc"],
                        test_auc=auc_test, test_ks=ks_test), f, indent=2, default=float)

    print("\nWrote images/overfit_gap.png, images/max_performance_results.json, images/final_test_evaluation.json")


if __name__ == "__main__":
    main()
