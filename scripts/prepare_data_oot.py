"""Out-of-time (OOT) counterpart to prepare_data.py. The original train/val/
test split is RANDOM (train_test_split(..., random_state=42)) -- loans from
every origination year, including the 2007-2009 crisis vintages, are
scattered across all three splits. That means every model in this project
has been evaluated on held-out LOANS but never on a held-out TIME PERIOD --
it has effectively already "seen" what every vintage era looks like during
training. This script fixes that: split chronologically by issue_dt instead,
so val and test are strictly later in time than everything the model
trained on -- the actual deployment scenario (train on history, predict
forward).

  train: issue year <= 2013  (230,706 loans)
  val:   issue year == 2014  (175,509 loans)
  test:  issue year 2015-2016 (332,261 loans)

Same feature engineering / leakage drops / train-only median imputation as
prepare_data.py, applied identically -- only the split mechanism changes.
Also writes data/{val,test}_oot_policy.csv (id + realized economics) for
the OOT frontier comparison, same approach as reconstruct_policy_split.py.

Caveat carried through to the writeup, not fixed here: the Step 1 seasoning
cutoffs (36mo <= Feb 2016, 60mo <= Mar 2014) mean the 2015-2016 test window
is 36-month loans only, and 2016 is truncated (data pulled before all 2016
vintages finished seasoning). The OOT baseline is recomputed on this same
population, so the baseline-vs-model comparison stays internally fair --
but the test window's loan-term mix differs from train/val's.
"""
import numpy as np
import pandas as pd

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

    year = df["issue_dt"].dt.year
    df["split"] = np.select(
        [year <= 2013, year == 2014, year.between(2015, 2016)],
        ["train", "val", "test"],
        default="excluded",
    )
    print(df["split"].value_counts())
    assert (df["split"] != "excluded").all(), "issue_dt outside expected 2007-2016 range"

    drop_cols = LEAKAGE_COLS + JOINT_ONLY_COLS + DROP_OTHER + [
        "loan_status", "issue_d", "earliest_cr_line",
        "earliest_cr_line_dt", "emp_length", "term",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns and c not in ECON_KEEP]
    modeling_df = df.drop(columns=drop_cols)

    train = modeling_df[modeling_df["split"] == "train"].drop(columns="split").copy()
    val = modeling_df[modeling_df["split"] == "val"].drop(columns="split").copy()
    test = modeling_df[modeling_df["split"] == "test"].drop(columns="split").copy()
    print(f"\nSplit sizes: train={len(train):,}  val={len(val):,}  test={len(test):,}")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name} bad rate: {part['bad'].mean():.3%}")

    # Train-only median imputation, exactly as prepare_data.py.
    numeric_cols = train.select_dtypes(include=[np.number]).columns.drop("bad")
    numeric_cols = [c for c in numeric_cols if c not in ECON_KEEP]
    missing_rates = train[numeric_cols].isna().mean()
    cols_with_missing = missing_rates[missing_rates > 0].index.tolist()
    medians = train[cols_with_missing].median()
    print(f"\n{len(cols_with_missing)} numeric columns have missing values in OOT train")

    for name, part in [("train", train), ("val", val), ("test", test)]:
        for col in cols_with_missing:
            part[f"{col}_missing"] = part[col].isna().astype(int)
            part[col] = part[col].fillna(medians[col])

    for name, part in [("train", train), ("val", val), ("test", test)]:
        econ = part[ECON_KEEP].copy()
        modeling_part = part.drop(columns=[c for c in ECON_KEEP if c != "id"])
        modeling_part = modeling_part.drop(columns="id")
        modeling_part.to_csv(f"data/{name}_oot.csv", index=False)
        print(f"Wrote data/{name}_oot.csv ({modeling_part.shape[0]:,} rows, {modeling_part.shape[1]} cols)")
        if name != "train":
            econ["bad"] = part["bad"].values
            econ["net_realized"] = econ["total_pymnt"] - econ["collection_recovery_fee"] - econ["funded_amnt"]
            econ.to_csv(f"data/{name}_oot_policy.csv", index=False)
            print(f"Wrote data/{name}_oot_policy.csv ({len(econ):,} rows)")


if __name__ == "__main__":
    main()
