"""Score val/test with the two candidate policy models:
  - LR-24: the forward-selected 24-feature logistic regression
    (FEATURE_SELECTION.md) -- the interpretable candidate, refit here for
    the first time on train and PERSISTED to the model registry (it was
    the one surviving model from that round never actually saved).
  - XGBoost full-features-earlystop -- the best-performing model overall
    (val AUC 0.708 / test AUC 0.709), loaded from the registry.

Outputs images/val_scores.csv and images/test_scores.csv: bad + both
models' predicted PD, row-order-aligned with data/val_policy.csv /
data/test_policy.csv (verified by reconstruct_policy_split.py) so they can
be concatenated positionally with the realized-economics columns needed
for Step 6/7.

Also checks calibration (predicted PD vs. observed bad rate by decile) for
both models on val -- the segmentation step bins directly on these PD
values (segment 3 = "PD 8-12%"), so miscalibration would silently corrupt
every downstream $ figure.
"""
import json

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import wrapper_selection as ws
from full_feature_matrix import build_full_matrix, load as load_full
from model_registry import load_model, save_model
from metrics_extra import ks_statistic
from sklearn.metrics import roc_auc_score, brier_score_loss

LR24_FEATURES = [
    "sub_grade", "loan_to_income", "acc_open_past_24mths", "home_ownership",
    "dti", "fico_avg", "term_months", "mort_acc", "inq_last_6mths", "purpose",
    "total_rev_hi_lim", "mths_since_recent_bc", "percent_bc_gt_75", "bc_util",
    "int_rate", "num_actv_rev_tl", "num_tl_op_past_12m", "mo_sin_old_rev_tl_op",
    "verification_status", "total_bc_limit", "revol_util", "tot_cur_bal",
    "mo_sin_rcnt_tl", "mo_sin_rcnt_rev_tl_op",
]


def fit_lr24():
    train = ws.load("train")
    val = ws.load("val")
    test = ws.load("test")
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
    print(f"LR-24: val AUC={auc_val:.4f} KS={ks_val:.4f}  |  test AUC={auc_test:.4f} KS={ks_test:.4f}")

    save_model(
        "lr-forward-selected-24", "v1", lr, cols,
        metrics=dict(val_auc=auc_val, test_auc=auc_test, val_ks=ks_val, test_ks=ks_test,
                     val_brier=brier_score_loss(y_val, p_val), test_brier=brier_score_loss(y_test, p_test)),
        hyperparameters=dict(max_iter=300),
        scaler=scaler,
        notes=("Forward-selected 24-feature logistic regression (FEATURE_SELECTION.md) -- the "
               "interpretable candidate for the deployed segmentation/policy. Never persisted "
               "during the original wrapper-selection search; refit here identically (same "
               "features, same train split) for that purpose."),
    )
    return p_val, p_test, y_val, y_test


def score_xgb():
    train, val, test = load_full("train"), load_full("val"), load_full("test")
    X_train, y_train, X_val, y_val, X_test, y_test, cols = build_full_matrix(train, val, test)

    model, _, meta = load_model("xgboost-full-features-earlystop", "v1")
    X_val = X_val.reindex(columns=meta["feature_cols"], fill_value=0)
    X_test = X_test.reindex(columns=meta["feature_cols"], fill_value=0)

    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    auc_val, auc_test = roc_auc_score(y_val, p_val), roc_auc_score(y_test, p_test)
    print(f"XGBoost: val AUC={auc_val:.4f} (registry says {meta['metrics']['val_auc']:.4f})  |  "
          f"test AUC={auc_test:.4f} (registry says {meta['metrics']['test_auc']:.4f})")
    return p_val, p_test, y_val, y_test


def calibration_check(name, y, p, n_bins=10):
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    tbl = df.groupby("bin", observed=True).agg(n=("y", "size"), mean_pred=("p", "mean"), obs_rate=("y", "mean"))
    print(f"\n{name} calibration by decile (val):")
    print(tbl.to_string())
    return tbl


def main():
    p_val_lr, p_test_lr, y_val_lr, y_test_lr = fit_lr24()
    p_val_xgb, p_test_xgb, y_val_xgb, y_test_xgb = score_xgb()

    assert (y_val_lr == y_val_xgb).all() and (y_test_lr == y_test_xgb).all(), \
        "val/test bad labels must align positionally between the two scoring paths"

    calibration_check("LR-24", y_val_lr, p_val_lr)
    calibration_check("XGBoost", y_val_xgb, p_val_xgb)

    val_out = pd.DataFrame({"bad": y_val_lr, "lr24_pd": p_val_lr, "xgb_pd": p_val_xgb})
    test_out = pd.DataFrame({"bad": y_test_lr, "lr24_pd": p_test_lr, "xgb_pd": p_test_xgb})

    val_policy = pd.read_csv("data/val_policy.csv")
    test_policy = pd.read_csv("data/test_policy.csv")
    assert (val_out["bad"].values == val_policy["bad"].values).all()
    assert (test_out["bad"].values == test_policy["bad"].values).all()

    val_out.to_csv("images/val_scores.csv", index=False)
    test_out.to_csv("images/test_scores.csv", index=False)
    print("\nWrote images/val_scores.csv, images/test_scores.csv")


if __name__ == "__main__":
    main()
