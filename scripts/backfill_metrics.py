"""Add KS statistic, specificity@20%-threshold, and sensitivity@95%-specificity
to the three models already in images/model_results.json, without re-running
the expensive searches -- just quick refits of the same, already-known
feature sets.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from feature_selection import build_block_columns
from metrics_extra import ks_statistic, sensitivity_at_specificity, specificity_at_threshold
from train_models import build_xy, load

BRIER_SELECTED_FEATURES = [
    "grade", "loan_to_income", "home_ownership", "dti", "inq_last_6mths",
    "fico_avg", "emp_length_years", "term_months", "purpose", "revol_util",
    "open_acc", "revol_bal", "total_acc", "verification_status", "int_rate",
]


def add_metrics(results, name, y_val, p, threshold):
    ks, ks_thresh = ks_statistic(y_val, p)
    spec_at_20pct = specificity_at_threshold(y_val, p, threshold)
    sens_at_95spec, _ = sensitivity_at_specificity(y_val, p, target_specificity=0.95)
    results[name]["ks"] = ks
    results[name]["specificity"] = spec_at_20pct
    results[name]["sensitivity_at_95_specificity"] = sens_at_95spec
    print(f"{name}: KS={ks:.4f} (at threshold {ks_thresh:.3f})  "
          f"Specificity@20%={spec_at_20pct:.3f}  Sensitivity@95%Spec={sens_at_95spec:.3f}")


def main():
    train = load("train")
    val = load("val")

    with open("images/model_results.json") as f:
        results = json.load(f)

    # --- Logistic Regression (full curated 20) ---
    X_train, y_train = build_xy(train)
    cols = X_train.columns.tolist()
    X_val, y_val = build_xy(val, one_hot_cols=cols)
    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(scaler.fit_transform(X_train), y_train)
    p_lr = lr.predict_proba(scaler.transform(X_val))[:, 1]
    add_metrics(results, "Logistic Regression", y_val, p_lr, results["Logistic Regression"]["threshold"])

    # --- Decision Tree ---
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=1000, random_state=42)
    tree.fit(X_train, y_train)
    p_tree = tree.predict_proba(X_val)[:, 1]
    add_metrics(results, "Decision Tree", y_val, p_tree, results["Decision Tree"]["threshold"])

    # --- LR (Brier-selected) ---
    numeric_sel = [f for f in BRIER_SELECTED_FEATURES if f != "grade" and f != "home_ownership"
                   and f != "verification_status" and f != "purpose"]
    categorical_sel = [f for f in BRIER_SELECTED_FEATURES if f in ("grade", "home_ownership", "verification_status", "purpose")]
    X_train2 = build_block_columns(train, numeric_sel, categorical_sel)
    cols2 = X_train2.columns.tolist()
    X_val2 = build_block_columns(val, numeric_sel, categorical_sel, one_hot_cols=cols2)
    scaler2 = StandardScaler()
    lr2 = LogisticRegression(max_iter=2000, random_state=42)
    lr2.fit(scaler2.fit_transform(X_train2), y_train)
    p_lr2 = lr2.predict_proba(scaler2.transform(X_val2))[:, 1]
    add_metrics(results, "Logistic Regression (Brier-selected)", y_val,
                p_lr2, results["Logistic Regression (Brier-selected)"]["threshold"])

    with open("images/model_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nUpdated images/model_results.json")


if __name__ == "__main__":
    main()
