"""Step 2: cleaning, feature engineering, target definition, and a leakage-safe
train/val/test split. See CLEANING.md for the rationale behind each decision.

Ordering matters for leakage:
  1. Deterministic feature engineering (date math, ratios) - safe pre-split,
     since it uses no dataset-wide statistics.
  2. Split into train/val/test.
  3. Any statistic fit on data (imputation medians, rare-category buckets)
     is fit on train only and applied to val/test.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SRC = "data/scoped_accepted.csv"

# --- Columns that are post-origination performance data (leakage) ---
LEAKAGE_COLS = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date", "hardship_end_date",
    "payment_plan_start_date", "hardship_length", "hardship_dpd",
    "hardship_loan_status", "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage", "settlement_term",
]

# --- Joint-application-only columns: 100% missing after scoping to Individual ---
JOINT_ONLY_COLS = [
    "annual_inc_joint", "dti_joint", "verification_status_joint", "revol_bal_joint",
    "sec_app_fico_range_low", "sec_app_fico_range_high", "sec_app_earliest_cr_line",
    "sec_app_inq_last_6mths", "sec_app_mort_acc", "sec_app_open_acc",
    "sec_app_revol_util", "sec_app_open_act_il", "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog",
]

# --- Identifiers / free text / zero-variance-after-scoping / not usable as-is ---
DROP_OTHER = [
    "member_id", "url", "desc", "title", "emp_title",  # free text / no signal without NLP
    "pymnt_plan", "policy_code", "application_type", "disbursement_method",  # constant post-scoping
    "funded_amnt", "funded_amnt_inv",  # redundant with loan_amnt for approved loans
]

TARGET_BAD = {"Charged Off", "Does not meet the credit policy. Status:Charged Off"}
TARGET_GOOD = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}


def parse_emp_length(s):
    if pd.isna(s):
        return np.nan
    if s == "< 1 year":
        return 0
    if s == "10+ years":
        return 10
    return int(s.split()[0])


def main():
    df = pd.read_csv(SRC, low_memory=False)
    df["issue_dt"] = pd.to_datetime(df["issue_dt"])
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    # --- Target ---
    df = df[df["loan_status"].isin(TARGET_BAD | TARGET_GOOD)].copy()
    df["bad"] = df["loan_status"].isin(TARGET_BAD).astype(int)
    print(f"\nTarget distribution:\n{df['bad'].value_counts(normalize=True)}")

    # --- Deterministic feature engineering (pre-split safe) ---
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

    # --- Drop leakage / joint-only / unusable columns ---
    drop_cols = LEAKAGE_COLS + JOINT_ONLY_COLS + DROP_OTHER + [
        "loan_status", "id", "issue_d", "earliest_cr_line",
        "earliest_cr_line_dt", "emp_length", "term",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"\nDropped {len(drop_cols)} columns (leakage/joint-only/unusable)")
    print(f"Remaining: {df.shape[1]} columns")

    # --- Split (stratified on target, before any train-fit statistics) ---
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["bad"], random_state=42
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["bad"], random_state=42
    )
    print(f"\nSplit sizes: train={len(train):,}  val={len(val):,}  test={len(test):,}")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name} bad rate: {part['bad'].mean():.3%}")

    # --- Missing-value handling: numeric -> train-median impute + missing flag ---
    numeric_cols = train.select_dtypes(include=[np.number]).columns.drop("bad")
    missing_rates = train[numeric_cols].isna().mean()
    cols_with_missing = missing_rates[missing_rates > 0].index.tolist()
    print(f"\n{len(cols_with_missing)} numeric columns have missing values in train")

    medians = train[cols_with_missing].median()
    for name, part in [("train", train), ("val", val), ("test", test)]:
        for col in cols_with_missing:
            part[f"{col}_missing"] = part[col].isna().astype(int)
            part[col] = part[col].fillna(medians[col])

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(f"data/{name}.csv", index=False)
        print(f"Wrote data/{name}.csv ({part.shape[0]:,} rows, {part.shape[1]} cols)")


if __name__ == "__main__":
    main()
