"""Apply Step 1 scoping decisions (see SCOPING.md) to the raw LendingClub
accepted-loans file and write the scoped population to data/scoped_accepted.csv.
"""
import pandas as pd

SRC = "data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"
DEST = "data/scoped_accepted.csv"

SEASONING_CUTOFF = {
    " 36 months": "2016-02-01",
    " 60 months": "2014-03-01",
}

TERMINAL_STATUSES = {
    "Fully Paid",
    "Charged Off",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",
}

df = pd.read_csv(SRC, low_memory=False)
print(f"Loaded {len(df):,} rows")

df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y")

seasoned_mask = pd.Series(False, index=df.index)
for term, cutoff in SEASONING_CUTOFF.items():
    seasoned_mask |= (df["term"] == term) & (df["issue_dt"] <= cutoff)

scoped = df[
    seasoned_mask
    & df["loan_status"].isin(TERMINAL_STATUSES)
    & (df["application_type"] == "Individual")
    & (df["disbursement_method"] == "Cash")
].copy()

print(f"Scoped rows: {len(scoped):,} ({len(scoped) / len(df):.1%} of total)")

scoped.to_csv(DEST, index=False)
print(f"Wrote {DEST}")
