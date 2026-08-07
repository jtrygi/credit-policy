"""Compute the full metric suite (AUC, Gini, Brier, KS, precision/recall/F1
at the 'decline riskiest 20%' threshold, specificity, sensitivity@95%-
specificity) for models whose scripts only logged AUC/Brier/KS during
search: the forward-selected LR, Random Forest, and XGBoost. Writes a
unified images/all_model_results.json merging these with the three models
already in images/model_results.json.
"""
import json

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss,
)
from sklearn.preprocessing import StandardScaler

from ensemble_common import build_ensemble_matrix
from metrics_extra import ks_statistic, sensitivity_at_specificity, specificity_at_threshold
from wrapper_selection import build_columns, get_candidate_types, load


def full_metrics(y_val, p, n_features):
    auc = roc_auc_score(y_val, p)
    gini = 2 * auc - 1
    brier = brier_score_loss(y_val, p)
    ks, _ = ks_statistic(y_val, p)
    thresh = np.quantile(p, 0.80)
    pred = (p >= thresh).astype(int)
    prec, rec, f1 = precision_score(y_val, pred), recall_score(y_val, pred), f1_score(y_val, pred)
    spec = specificity_at_threshold(y_val, p, thresh)
    sens95, _ = sensitivity_at_specificity(y_val, p, 0.95)
    return dict(auc=auc, gini=gini, brier=brier, ks=ks, threshold=float(thresh),
                precision=prec, recall=rec, f1=f1, specificity=spec,
                sensitivity_at_95_specificity=sens95, n_features=n_features)


def main():
    with open("images/model_results.json") as f:
        all_results = json.load(f)

    # --- Forward-selected LR ---
    with open("images/wrapper_forward_selected.json") as f:
        fwd = json.load(f)
    selected = fwd["selected"]
    train = load("train")
    val = load("val")
    numeric_all, _ = get_candidate_types(train, selected)
    train_cols = set(train.columns)
    numeric_sel = [f for f in selected if f in numeric_all]
    categorical_sel = [f for f in selected if f not in numeric_all]
    X_train = build_columns(train, numeric_sel, categorical_sel, train_cols)
    cols = X_train.columns.tolist()
    X_val = build_columns(val, numeric_sel, categorical_sel, train_cols, one_hot_cols=cols)
    scaler = StandardScaler()
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(scaler.fit_transform(X_train), train["bad"].values)
    p = lr.predict_proba(scaler.transform(X_val))[:, 1]
    all_results["LR (Forward-selected)"] = full_metrics(val["bad"].values, p, len(selected))
    print("LR (Forward-selected):", all_results["LR (Forward-selected)"])

    # --- Random Forest, XGBoost (refit on the qualified pool for predictions) ---
    X_train2, y_train2, X_val2, y_val2, cols2 = build_ensemble_matrix()

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=500,
                                 n_jobs=8, random_state=42)
    rf.fit(X_train2, y_train2)
    p_rf = rf.predict_proba(X_val2)[:, 1]
    all_results["Random Forest"] = full_metrics(y_val2, p_rf, len(cols2))
    print("Random Forest:", all_results["Random Forest"])

    gbm = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=8)
    gbm.fit(X_train2, y_train2)
    p_gbm = gbm.predict_proba(X_val2)[:, 1]
    all_results["XGBoost"] = full_metrics(y_val2, p_gbm, len(cols2))
    print("XGBoost:", all_results["XGBoost"])

    with open("images/all_model_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print("\nWrote images/all_model_results.json")


if __name__ == "__main__":
    main()
