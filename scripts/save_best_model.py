"""Refit and PERSIST the winning model from max_performance.py's experiment.
That script found the best model of this entire project (XGBoost, full 254-
column feature set, early-stopped) but exited without saving the fitted
object -- only metrics survived. This refits the identical config and
actually stores it via model_registry, plus the regularized MLP as a
reference point for the overfitting-gap story.
"""
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from full_feature_matrix import build_full_matrix, load
from metrics_extra import ks_statistic
from model_registry import save_model

train = load("train")
val = load("val")
test = load("test")
X_train, y_train, X_val, y_val, X_test, y_test, cols = build_full_matrix(train, val, test)

print("Refitting XGBoost (full features, early-stopped)...", flush=True)
xgb_full = xgb.XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.03,
                              min_child_weight=5, reg_lambda=2, subsample=0.8, colsample_bytree=0.8,
                              random_state=42, n_jobs=8, early_stopping_rounds=50, eval_metric="auc")
xgb_full.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

p_val = xgb_full.predict_proba(X_val)[:, 1]
p_test = xgb_full.predict_proba(X_test)[:, 1]
auc_val, auc_test = roc_auc_score(y_val, p_val), roc_auc_score(y_test, p_test)
ks_val, _ = ks_statistic(y_val, p_val)
ks_test, _ = ks_statistic(y_test, p_test)
print(f"Val AUC={auc_val:.4f}  Test AUC={auc_test:.4f}", flush=True)

save_model(
    "xgboost-full-features-earlystop", "v1", xgb_full, cols,
    metrics=dict(val_auc=auc_val, test_auc=auc_test, val_ks=ks_val, test_ks=ks_test,
                 best_iteration=int(xgb_full.best_iteration)),
    hyperparameters=dict(n_estimators=2000, max_depth=5, learning_rate=0.03, min_child_weight=5,
                          reg_lambda=2, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50),
    notes=("Best-performing model found in the max-performance experiment (FEATURE_SELECTION.md). "
           "Full 254-column feature set, interpretability not a design goal for this candidate -- "
           "use the forward-selected 24-feature logistic regression instead if a credit-committee-"
           "explainable model is required. First model version saved via model_registry."),
)

print("\nRefitting regularized MLP (reference point for the overfitting-gap story)...", flush=True)
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(X_train)
Xv_s = scaler.transform(X_val)
Xt_s = scaler.transform(X_test)
mlp_reg = MLPClassifier(hidden_layer_sizes=(100, 50), alpha=1e-2, max_iter=200,
                         early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                         solver="adam", random_state=42)
mlp_reg.fit(Xtr_s, y_train)
p_val_m = mlp_reg.predict_proba(Xv_s)[:, 1]
p_test_m = mlp_reg.predict_proba(Xt_s)[:, 1]
auc_val_m, auc_test_m = roc_auc_score(y_val, p_val_m), roc_auc_score(y_test, p_test_m)
print(f"Val AUC={auc_val_m:.4f}  Test AUC={auc_test_m:.4f}", flush=True)

save_model(
    "mlp-regularized-full-features", "v1", mlp_reg, cols,
    metrics=dict(val_auc=auc_val_m, test_auc=auc_test_m, n_iter=mlp_reg.n_iter_),
    hyperparameters=dict(hidden_layer_sizes=(100, 50), alpha=1e-2, early_stopping=True,
                          validation_fraction=0.1, n_iter_no_change=10),
    scaler=scaler,
    notes="Reference point for the overfitting-gap demonstration -- not a recommended candidate.",
)

print("\nDone. Run `python scripts/model_registry.py` to list all saved versions.")
