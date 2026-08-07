"""Step 4: logistic regression + a shallow decision tree, compared on val only
(test stays untouched). Curated, explicit feature list -- not "all columns"
(train.csv still carries issue_dt/zip_code/addr_state, which are not features).

Metric philosophy (see MODELING.md for the full writeup):
  - Model comparison: ROC-AUC / Gini + calibration (design doc Step 4), not F1.
  - No class_weight='balanced' -- would wreck calibration, and Step 7's profit
    math consumes these probabilities directly.
  - Threshold-based metrics (precision/recall/F1/confusion matrix) reported at
    a policy-relevant operating point (decline riskiest 20% by score), not the
    default 0.5 -- at a 15% base rate, 0.5 is degenerate for both models.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

NUMERIC_FEATURES = [
    "loan_amnt", "int_rate", "annual_inc", "dti", "delinq_2yrs",
    "fico_avg", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc",
    "credit_history_months", "loan_to_income", "emp_length_years", "term_months",
]
MISSING_FLAGS = [
    "annual_inc_missing", "delinq_2yrs_missing", "inq_last_6mths_missing",
    "open_acc_missing", "pub_rec_missing", "revol_util_missing", "total_acc_missing",
    "credit_history_months_missing", "loan_to_income_missing", "emp_length_years_missing",
]
CATEGORICAL_FEATURES = ["grade", "home_ownership", "verification_status", "purpose"]

HUE_LR = "#2563EB"
HUE_TREE = "#DC2626"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(name):
    df = pd.read_csv(f"data/{name}.csv", low_memory=False)
    df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    return df


def build_xy(df, one_hot_cols=None, exclude_pricing=False):
    numeric_features = NUMERIC_FEATURES
    categorical_features = CATEGORICAL_FEATURES
    if exclude_pricing:
        numeric_features = [c for c in numeric_features if c != "int_rate"]
        categorical_features = [c for c in categorical_features if c != "grade"]
    num = df[numeric_features + MISSING_FLAGS].copy()
    cat = pd.get_dummies(df[categorical_features].astype(str), drop_first=True)
    X = pd.concat([num, cat], axis=1)
    if one_hot_cols is not None:
        X = X.reindex(columns=one_hot_cols, fill_value=0)
    y = df["bad"].values
    return X, y


def fit_and_importance(train, val, exclude_pricing, suffix):
    """Fit a fresh LR + tree on a given feature set and save importance charts."""
    X_train, y_train = build_xy(train, exclude_pricing=exclude_pricing)
    cols = X_train.columns.tolist()
    X_val, y_val = build_xy(val, one_hot_cols=cols, exclude_pricing=exclude_pricing)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=1000, random_state=42)
    tree.fit(X_train, y_train)

    label = " (excl. grade/int_rate)" if exclude_pricing else ""

    coefs = pd.Series(lr.coef_[0], index=cols)
    top = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(15)[::-1]
    colors = [HUE_LR if v > 0 else HUE_TREE for v in top.values]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top.values, color=colors, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_title(f"Logistic Regression: top standardized coefficients{label}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Standardized coefficient (blue = raises risk, red = lowers risk)")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"images/lr_importance{suffix}.png", dpi=150)
    plt.close(fig)

    imp = pd.Series(tree.feature_importances_, index=cols)
    top_imp = imp[imp > 0].sort_values(ascending=False).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_imp.index, top_imp.values, color=HUE_TREE, zorder=3)
    ax.set_title(f"Decision Tree: feature importance{label}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"images/tree_importance{suffix}.png", dpi=150)
    plt.close(fig)

    auc = roc_auc_score(y_val, lr.predict_proba(scaler.transform(X_val))[:, 1])
    print(f"  [importance refit{label}] LR val AUC = {auc:.4f}")
    return auc


def main():
    train = load("train")
    val = load("val")

    X_train, y_train = build_xy(train)
    feature_cols = X_train.columns.tolist()
    X_val, y_val = build_xy(val, one_hot_cols=feature_cols)

    # --- Logistic regression (scaled) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    p_lr = lr.predict_proba(X_val_scaled)[:, 1]

    # --- Decision tree (unscaled, shallow for calibration + interpretability) ---
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=1000, random_state=42)
    tree.fit(X_train, y_train)
    p_tree = tree.predict_proba(X_val)[:, 1]

    results = {}
    for name, p in [("Logistic Regression", p_lr), ("Decision Tree", p_tree)]:
        auc = roc_auc_score(y_val, p)
        gini = 2 * auc - 1
        ap = average_precision_score(y_val, p)
        brier = brier_score_loss(y_val, p)

        thresh = np.quantile(p, 0.80)  # decline riskiest 20% by score
        pred = (p >= thresh).astype(int)
        prec = precision_score(y_val, pred)
        rec = recall_score(y_val, pred)
        f1 = f1_score(y_val, pred)
        cm = confusion_matrix(y_val, pred)

        results[name] = dict(
            auc=auc, gini=gini, avg_precision=ap, brier=brier,
            threshold=thresh, precision=prec, recall=rec, f1=f1,
            confusion_matrix=cm.tolist(),
            approval_rate=1 - pred.mean(),
            bad_rate_declined=y_val[pred == 1].mean(),
            bad_rate_approved=y_val[pred == 0].mean(),
        )
        print(f"\n=== {name} ===")
        print(f"AUC={auc:.4f}  Gini={gini:.4f}  Avg Precision={ap:.4f}  Brier={brier:.4f}")
        print(f"At 'decline riskiest 20%' threshold ({thresh:.3f}): "
              f"precision={prec:.3f} recall={rec:.3f} f1={f1:.3f}")
        print(f"Confusion matrix [[TN,FP],[FN,TP]]:\n{cm}")
        print(f"Bad rate among declined: {results[name]['bad_rate_declined']:.1%}  "
              f"among approved: {results[name]['bad_rate_approved']:.1%}")

    with open("images/model_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    # --- ROC curve overlay ---
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, p, color in [("Logistic Regression", p_lr, HUE_LR), ("Decision Tree", p_tree, HUE_TREE)]:
        fpr, tpr, _ = roc_curve(y_val, p)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={roc_auc_score(y_val, p):.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve: Logistic Regression vs Decision Tree", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/roc_curve.png", dpi=150)
    plt.close(fig)

    # --- PR curve overlay ---
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, p, color in [("Logistic Regression", p_lr, HUE_LR), ("Decision Tree", p_tree, HUE_TREE)]:
        prec_c, rec_c, _ = precision_recall_curve(y_val, p)
        ax.plot(rec_c, prec_c, color=color, linewidth=2,
                 label=f"{name} (AP={average_precision_score(y_val, p):.3f})")
    ax.axhline(y_val.mean(), color=MUTED, linewidth=1, linestyle="--", label=f"Base rate ({y_val.mean():.1%})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/pr_curve.png", dpi=150)
    plt.close(fig)

    # --- Calibration curve ---
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, p, color in [("Logistic Regression", p_lr, HUE_LR), ("Decision Tree", p_tree, HUE_TREE)]:
        bins = pd.qcut(p, 10, duplicates="drop")
        cal = pd.DataFrame({"p": p, "y": y_val, "bin": bins}).groupby("bin", observed=True).agg(
            mean_pred=("p", "mean"), mean_actual=("y", "mean")
        )
        ax.plot(cal["mean_pred"], cal["mean_actual"], marker="o", color=color, linewidth=2, label=name)
    lims = [0, max(p_lr.max(), p_tree.max()) * 1.05]
    ax.plot(lims, lims, color=MUTED, linewidth=1, linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability (decile bin)")
    ax.set_ylabel("Actual bad rate (decile bin)")
    ax.set_title("Calibration: predicted vs actual bad rate", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/calibration_curve.png", dpi=150)
    plt.close(fig)

    # --- Feature importance: full model, and a separately-refit model with
    # grade/int_rate excluded (not just filtered from the display -- a model
    # that already split on int_rate first will make everything else look
    # artificially unimportant by leftover-impurity accounting) ---
    fit_and_importance(train, val, exclude_pricing=False, suffix="")
    fit_and_importance(train, val, exclude_pricing=True, suffix="_excl_grade")

    # --- Decision tree diagram ---
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(tree, feature_names=feature_cols, class_names=["Good", "Bad"],
              filled=True, rounded=True, fontsize=8, ax=ax, max_depth=4,
              impurity=False, proportion=True)
    ax.set_title("Decision tree structure (max_depth=4)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("images/tree_diagram.png", dpi=150)
    plt.close(fig)

    print("\nWrote all charts to images/")


if __name__ == "__main__":
    main()
