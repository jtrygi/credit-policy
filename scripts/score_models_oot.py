"""OOT counterpart to score_models.py: refit the same two frozen candidates
(LR-24, XGBoost full-features-earlystop) on the chronological split from
prepare_data_oot.py -- same features, same hyperparameters, only the split
changed from random to time-based. Saved to the registry as v2 with notes
distinguishing them from the v1 (random-split) versions.
"""
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

import wrapper_selection as ws
from metrics_extra import ks_statistic
from model_registry import save_model
from score_models import LR24_FEATURES

EXCLUDE = {"bad", "issue_dt", "zip_code", "fico_range_low", "fico_range_high"}


def load_oot(name):
    df = pd.read_csv(f"data/{name}_oot.csv", low_memory=False)
    df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    return df


def build_full_matrix_oot(train, val, test):
    candidates = [c for c in train.columns if c not in EXCLUDE]
    numeric = [c for c in candidates if train[c].dtype != object]
    categorical = [c for c in candidates if train[c].dtype == object]

    def build(df, one_hot_cols=None):
        num = df[numeric].copy()
        cat = pd.get_dummies(df[categorical].astype(str), drop_first=True)
        X = pd.concat([num, cat], axis=1)
        if one_hot_cols is not None:
            X = X.reindex(columns=one_hot_cols, fill_value=0)
        return X

    X_train = build(train)
    cols = X_train.columns.tolist()
    X_val = build(val, one_hot_cols=cols)
    X_test = build(test, one_hot_cols=cols)
    return X_train, train["bad"].values, X_val, val["bad"].values, X_test, test["bad"].values, cols


def fit_lr24_oot():
    train, val, test = load_oot("train"), load_oot("val"), load_oot("test")
    train_cols = set(train.columns)
    numeric_all, categorical_all = ws.get_candidate_types(train, LR24_FEATURES)
    numeric_sel = [f for f in LR24_FEATURES if f in numeric_all]
    categorical_sel = [f for f in LR24_FEATURES if f not in numeric_all]

    X_train = ws.build_columns(train, numeric_sel, categorical_sel, train_cols)
    cols = X_train.columns.tolist()
    X_val = ws.build_columns(val, numeric_sel, categorical_sel, train_cols, one_hot_cols=cols)
    X_test = ws.build_columns(test, numeric_sel, categorical_sel, train_cols, one_hot_cols=cols)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_val)
    Xt = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=300, random_state=42)
    lr.fit(Xtr, train["bad"].values)

    p_val = lr.predict_proba(Xv)[:, 1]
    p_test = lr.predict_proba(Xt)[:, 1]
    y_val, y_test = val["bad"].values, test["bad"].values

    auc_val, auc_test = roc_auc_score(y_val, p_val), roc_auc_score(y_test, p_test)
    ks_val, _ = ks_statistic(y_val, p_val)
    ks_test, _ = ks_statistic(y_test, p_test)
    print(f"LR-24 (OOT): val({2014}) AUC={auc_val:.4f} KS={ks_val:.4f}  |  "
          f"test(2015-16) AUC={auc_test:.4f} KS={ks_test:.4f}")

    save_model(
        "lr-forward-selected-24", "v2", lr, cols,
        metrics=dict(val_auc=auc_val, test_auc=auc_test, val_ks=ks_val, test_ks=ks_test,
                     val_brier=brier_score_loss(y_val, p_val), test_brier=brier_score_loss(y_test, p_test)),
        hyperparameters=dict(max_iter=300),
        scaler=scaler,
        notes=("OOT (out-of-time) counterpart to v1 -- same 24 frozen features, refit on a "
               "chronological split (train<=2013, val=2014, test=2015-2016) instead of the "
               "random split v1 used. Tests whether the model generalizes to a genuinely "
               "unseen future time period, not just unseen loans from the same era."),
    )
    return p_val, p_test, y_val, y_test


def score_xgb_oot():
    train, val, test = load_oot("train"), load_oot("val"), load_oot("test")
    X_train, y_train, X_val, y_val, X_test, y_test, cols = build_full_matrix_oot(train, val, test)

    xgb_full = xgb.XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.03,
                                  min_child_weight=5, reg_lambda=2, subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, n_jobs=8, early_stopping_rounds=50, eval_metric="auc")
    xgb_full.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    p_val = xgb_full.predict_proba(X_val)[:, 1]
    p_test = xgb_full.predict_proba(X_test)[:, 1]
    auc_val, auc_test = roc_auc_score(y_val, p_val), roc_auc_score(y_test, p_test)
    ks_val, _ = ks_statistic(y_val, p_val)
    ks_test, _ = ks_statistic(y_test, p_test)
    print(f"XGBoost (OOT): val(2014) AUC={auc_val:.4f} KS={ks_val:.4f}  |  "
          f"test(2015-16) AUC={auc_test:.4f} KS={ks_test:.4f}  best_iteration={xgb_full.best_iteration}")

    save_model(
        "xgboost-full-features-earlystop", "v2", xgb_full, cols,
        metrics=dict(val_auc=auc_val, test_auc=auc_test, val_ks=ks_val, test_ks=ks_test,
                     best_iteration=int(xgb_full.best_iteration)),
        hyperparameters=dict(n_estimators=2000, max_depth=5, learning_rate=0.03, min_child_weight=5,
                              reg_lambda=2, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50),
        notes=("OOT (out-of-time) counterpart to v1 -- same hyperparameters and full 254-column "
               "feature set, refit on a chronological split (train<=2013, val=2014, "
               "test=2015-2016) instead of the random split v1 used."),
    )
    return p_val, p_test, y_val, y_test


def main():
    p_val_lr, p_test_lr, y_val_lr, y_test_lr = fit_lr24_oot()
    p_val_xgb, p_test_xgb, y_val_xgb, y_test_xgb = score_xgb_oot()

    assert (y_val_lr == y_val_xgb).all() and (y_test_lr == y_test_xgb).all()

    val_out = pd.DataFrame({"bad": y_val_lr, "lr24_pd": p_val_lr, "xgb_pd": p_val_xgb})
    test_out = pd.DataFrame({"bad": y_test_lr, "lr24_pd": p_test_lr, "xgb_pd": p_test_xgb})

    val_policy = pd.read_csv("data/val_oot_policy.csv")
    test_policy = pd.read_csv("data/test_oot_policy.csv")
    assert (val_out["bad"].values == val_policy["bad"].values).all()
    assert (test_out["bad"].values == test_policy["bad"].values).all()

    val_out.to_csv("images/val_oot_scores.csv", index=False)
    test_out.to_csv("images/test_oot_scores.csv", index=False)
    print("\nWrote images/val_oot_scores.csv, images/test_oot_scores.csv")


if __name__ == "__main__":
    main()
