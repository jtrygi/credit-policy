"""Shared data loading/feature-building for the RF and LightGBM scripts,
which run as separate processes (a same-process RF-then-LightGBM run hit a
Windows-specific access violation, likely a threading conflict between
sklearn's joblib/BLAS threads and LightGBM's own thread pool)."""
import json

import pandas as pd


def load(name):
    df = pd.read_csv(f"data/{name}.csv", low_memory=False)
    df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    return df


def build_ensemble_matrix():
    with open("images/qualified_features.json") as f:
        candidates = json.load(f)

    train = load("train")
    val = load("val")
    numeric = [c for c in candidates if c == "fico_avg" or train[c].dtype != object]
    categorical = [c for c in candidates if c not in numeric]

    num_cols = []
    for f in numeric:
        num_cols.append(f)
        flag = f"{f}_missing"
        if flag in train.columns:
            num_cols.append(flag)

    X_train = pd.concat([train[num_cols], pd.get_dummies(train[categorical].astype(str), drop_first=True)], axis=1)
    cols = X_train.columns.tolist()
    X_val = pd.concat([val[num_cols], pd.get_dummies(val[categorical].astype(str), drop_first=True)], axis=1)
    X_val = X_val.reindex(columns=cols, fill_value=0)
    y_train, y_val = train["bad"].values, val["bad"].values
    return X_train, y_train, X_val, y_val, cols
