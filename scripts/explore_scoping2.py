"""Follow-up: find the empirical seasoning cutoff per term, and re-check
application_type risk differential on a seasoning-matched (not vintage-confounded)
basis.
"""
import pandas as pd

COLS = ["issue_d", "term", "loan_status", "application_type"]

df = pd.read_csv(
    "data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv",
    usecols=COLS,
)
df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y")

TERMINAL = {"Fully Paid", "Charged Off", "Default",
            "Does not meet the credit policy. Status:Fully Paid",
            "Does not meet the credit policy. Status:Charged Off"}
df["is_terminal"] = df["loan_status"].isin(TERMINAL)

print("=== issue_d range ===")
print(df["issue_dt"].min(), "to", df["issue_dt"].max())

print("\n=== application_type first/last issue_dt ===")
print(df.groupby("application_type")["issue_dt"].agg(["min", "max", "count"]))

# Find, per term, the most recent issue month where terminal% >= 0.90
seasoning = (
    df.groupby(["term", "issue_dt"])["is_terminal"]
    .mean()
    .reset_index()
    .sort_values(["term", "issue_dt"])
)

for t in [" 36 months", " 60 months"]:
    sub = seasoning[seasoning["term"] == t]
    seasoned_enough = sub[sub["is_terminal"] >= 0.90]
    cutoff = seasoned_enough["issue_dt"].max()
    print(f"\n{t}: last issue month with >=90% terminal = {cutoff}")

# --- Fair application_type comparison: restrict to a common, seasoned vintage window ---
# Joint App only exists from 2017 on, and needs time to season -> compare on loans
# issued in a window where BOTH types are present and reasonably seasoned relative
# to their own term, using bad-rate-among-resolved instead of raw status mix.
BAD = {"Charged Off", "Default"}
GOOD = {"Fully Paid"}
resolved = df[df["loan_status"].isin(BAD | GOOD)].copy()
resolved["is_bad"] = resolved["loan_status"].isin(BAD)

print("\n=== application_type: bad rate AMONG RESOLVED loans only (all vintages) ===")
print(resolved.groupby("application_type")["is_bad"].agg(["mean", "count"]))

# Now restrict to loans issued in the window where Joint App exists (2017-01 onward)
# and where the term has had a chance to resolve one way or another, for a fairer read
window = resolved[resolved["issue_dt"] >= "2017-01-01"]
print("\n=== Same, restricted to issue_dt >= 2017-01 (where Joint App exists) ===")
print(window.groupby(["application_type", "term"])["is_bad"].agg(["mean", "count"]))
