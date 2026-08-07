"""Full (non-interpretable-constrained) feature matrix: every cleaned column
from CLEANING.md, not just the 30 IV-qualified ones -- for the max-
performance experiment where individual-variable interpretability no longer
matters. Missing-value flag columns are included as real features here
(unlike the IV screen, which folded missingness into each parent variable's
WOE bins instead)."""
import pandas as pd

EXCLUDE = {"bad", "issue_dt", "zip_code", "fico_range_low", "fico_range_high"}


def load(name):
    df = pd.read_csv(f"data/{name}.csv", low_memory=False)
    df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    return df


def build_full_matrix(train, val, test=None):
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
    y_train, y_val = train["bad"].values, val["bad"].values
    if test is not None:
        X_test = build(test, one_hot_cols=cols)
        y_test = test["bad"].values
        return X_train, y_train, X_val, y_val, X_test, y_test, cols
    return X_train, y_train, X_val, y_val, cols
