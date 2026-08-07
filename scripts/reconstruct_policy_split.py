"""Step 6/7 needs realized $ economics (funded_amnt, total_pymnt,
collection_recovery_fee) and grade attached to the val/test rows -- but
prepare_data.py correctly drops those as leakage/identifiers before saving
train.csv/val.csv/test.csv. This replays prepare_data.py's exact
pre-split logic (same target definition, same feature engineering, same
row order) so the identical train_test_split(random_state=42) calls
reproduce the same split, this time keeping id + economics columns
alongside.

Verified, not assumed: after reconstructing, we check the reconstructed
val/test splits line up POSITIONALLY with the existing val.csv/test.csv
(same n, same bad values in the same order) before trusting a single
dollar figure downstream.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from prepare_data import LEAKAGE_COLS, JOINT_ONLY_COLS, DROP_OTHER, TARGET_BAD, TARGET_GOOD, parse_emp_length

SRC = "data/scoped_accepted.csv"
ECON_KEEP = ["id", "grade", "funded_amnt", "total_pymnt", "collection_recovery_fee"]


def main():
    df = pd.read_csv(SRC, low_memory=False)
    df["issue_dt"] = pd.to_datetime(df["issue_dt"])

    df = df[df["loan_status"].isin(TARGET_BAD | TARGET_GOOD)].copy()
    df["bad"] = df["loan_status"].isin(TARGET_BAD).astype(int)

    df["earliest_cr_line_dt"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    df["credit_history_months"] = (
        (df["issue_dt"] - df["earliest_cr_line_dt"]).dt.days / 30.44
    ).round(1)
    df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)
    df["emp_length_years"] = df["emp_length"].apply(parse_emp_length)
    df["home_ownership"] = df["home_ownership"].replace(
        {"OTHER": "OTHER", "NONE": "OTHER", "ANY": "OTHER"}
    )
    df["term_months"] = df["term"].str.extract(r"(\d+)").astype(int)

    # Same drop_cols as prepare_data.py EXCEPT we keep id/econ cols for now
    # and drop them only after the split.
    drop_cols = LEAKAGE_COLS + JOINT_ONLY_COLS + DROP_OTHER + [
        "loan_status", "issue_d", "earliest_cr_line",
        "earliest_cr_line_dt", "emp_length", "term",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns and c not in ECON_KEEP]
    df = df.drop(columns=drop_cols)

    # Identical two-stage split as prepare_data.py
    train, temp = train_test_split(df, test_size=0.30, stratify=df["bad"], random_state=42)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["bad"], random_state=42)

    # --- Verify positional match against the existing train/val/test.csv ---
    for name, part in [("train", train), ("val", val), ("test", test)]:
        existing = pd.read_csv(f"data/{name}.csv", usecols=["bad"], low_memory=False)
        same_n = len(existing) == len(part)
        same_bad = same_n and (existing["bad"].values == part["bad"].values).all()
        print(f"{name}: reconstructed n={len(part):,} vs existing n={len(existing):,}  "
              f"positional bad-match={'OK' if same_bad else 'MISMATCH'}")
        if not same_bad:
            raise RuntimeError(f"{name}: reconstructed split does not match existing {name}.csv positionally")

    for name, part in [("val", val), ("test", test)]:
        econ = part[ECON_KEEP + ["bad"]].copy()
        econ["net_realized"] = econ["total_pymnt"] - econ["collection_recovery_fee"] - econ["funded_amnt"]
        econ.to_csv(f"data/{name}_policy.csv", index=False)
        print(f"Wrote data/{name}_policy.csv ({len(econ):,} rows)")


if __name__ == "__main__":
    main()
