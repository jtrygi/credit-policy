"""Step 1 scoping EDA: find a data-driven seasoning cutoff and check whether
application_type / other candidate "product" dimensions warrant exclusion,
inclusion as a feature, or use as a selection criterion.
"""
import pandas as pd

COLS = [
    "issue_d", "term", "loan_status", "application_type",
    "loan_amnt", "purpose", "disbursement_method", "initial_list_status",
]

df = pd.read_csv(
    "data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv",
    usecols=COLS,
)

print("=== Rows ===")
print(len(df))

print("\n=== loan_status value counts ===")
print(df["loan_status"].value_counts(dropna=False))

print("\n=== term value counts ===")
print(df["term"].value_counts(dropna=False))

print("\n=== application_type value counts ===")
print(df["application_type"].value_counts(dropna=False))

print("\n=== disbursement_method value counts ===")
print(df["disbursement_method"].value_counts(dropna=False))

print("\n=== purpose value counts ===")
print(df["purpose"].value_counts(dropna=False))

# --- Seasoning: % of loans with a TERMINAL status, by issue month and term ---
TERMINAL = {"Fully Paid", "Charged Off", "Default",
            "Does not meet the credit policy. Status:Fully Paid",
            "Does not meet the credit policy. Status:Charged Off"}

df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
df["is_terminal"] = df["loan_status"].isin(TERMINAL)

seasoning = (
    df.groupby(["term", "issue_dt"])["is_terminal"]
    .agg(["mean", "count"])
    .reset_index()
    .sort_values(["term", "issue_dt"])
)

print("\n=== % terminal by issue month, term=36 months (last 24 rows) ===")
print(seasoning[seasoning["term"] == " 36 months"].tail(24).to_string(index=False))

print("\n=== % terminal by issue month, term=60 months (last 24 rows) ===")
print(seasoning[seasoning["term"] == " 60 months"].tail(24).to_string(index=False))

print("\n=== application_type x loan_status (row %) ===")
print(pd.crosstab(df["application_type"], df["loan_status"], normalize="index").round(3))
