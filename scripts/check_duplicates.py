"""Check for duplicate records in the raw and scoped data, and specifically
whether any duplicate ends up split across train/val/test (a leakage risk
the current pipeline does not explicitly guard against).
"""
import pandas as pd
from sklearn.model_selection import train_test_split

RAW = "data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"
SCOPED = "data/scoped_accepted.csv"

print("=== RAW file ===")
raw = pd.read_csv(RAW, low_memory=False)
print(f"Rows: {len(raw):,}")
print(f"Unique id: {raw['id'].nunique():,}  (dup ids: {raw['id'].duplicated().sum():,})")

content_cols = [c for c in raw.columns if c not in ("id", "member_id")]
full_dupes = raw.duplicated(subset=content_cols).sum()
print(f"Full-content duplicate rows (excluding id/member_id): {full_dupes:,}")

print("\n=== SCOPED file ===")
scoped = pd.read_csv(SCOPED, low_memory=False)
print(f"Rows: {len(scoped):,}")
print(f"Unique id: {scoped['id'].nunique():,}  (dup ids: {scoped['id'].duplicated().sum():,})")
content_cols_s = [c for c in scoped.columns if c not in ("id", "member_id")]
full_dupes_s = scoped.duplicated(subset=content_cols_s).sum()
print(f"Full-content duplicate rows (excluding id/member_id): {full_dupes_s:,}")

# --- Reproduce the exact split from prepare_data.py, but keep `id`, to check
# whether any duplicate content spans train/val/test ---
print("\n=== Split leakage check (reproducing prepare_data.py's split) ===")
TARGET_BAD = {"Charged Off", "Does not meet the credit policy. Status:Charged Off"}
TARGET_GOOD = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
scoped = scoped[scoped["loan_status"].isin(TARGET_BAD | TARGET_GOOD)].copy()
scoped["bad"] = scoped["loan_status"].isin(TARGET_BAD).astype(int)

train, temp = train_test_split(scoped, test_size=0.30, stratify=scoped["bad"], random_state=42)
val, test = train_test_split(temp, test_size=0.50, stratify=temp["bad"], random_state=42)

if full_dupes_s > 0:
    dupe_mask = scoped.duplicated(subset=content_cols_s, keep=False)
    dupe_ids = set(scoped.loc[dupe_mask, "id"])
    in_train = dupe_ids & set(train["id"])
    in_val = dupe_ids & set(val["id"])
    in_test = dupe_ids & set(test["id"])
    print(f"IDs involved in a full-content duplicate: {len(dupe_ids):,}")
    print(f"  present in train: {len(in_train):,}, val: {len(in_val):,}, test: {len(in_test):,}")
    spans_splits = (bool(in_train) + bool(in_val) + bool(in_test)) > 1
    print(f"Any duplicate content spans more than one split? {spans_splits}")
else:
    print("No full-content duplicates in scoped data -> nothing to check for split leakage.")
