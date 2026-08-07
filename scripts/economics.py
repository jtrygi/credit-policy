"""Realized $ economics per loan outcome, computed from the scoped population's
post-origination fields (used here for economic analysis only, never as model
inputs -- those were correctly dropped as leakage in prepare_data.py).

Verified accounting: total_pymnt == total_rec_prncp + total_rec_int +
total_rec_late_fee + recoveries (confirmed empirically, diff ~0 across a
Charged Off sample). So recoveries must NOT be added again on top of
total_pymnt. collection_recovery_fee is netted out separately since it is
not already subtracted from total_pymnt.

Net realized $ per loan = total_pymnt - collection_recovery_fee - funded_amnt

Simplifications stated explicitly (left for Step 8's sensitivity analysis,
not modeled here): no cost of funds, no discounting/time value of money,
no operating cost per loan.
"""
import pandas as pd

SCOPED = "data/scoped_accepted.csv"

ECON_COLS = [
    "loan_status", "grade", "funded_amnt", "total_pymnt",
    "collection_recovery_fee", "int_rate", "term",
]


def load_economics(path=SCOPED, usecols=None):
    df = pd.read_csv(path, usecols=usecols or ECON_COLS, low_memory=False)
    df["net_realized"] = df["total_pymnt"] - df["collection_recovery_fee"] - df["funded_amnt"]
    return df


if __name__ == "__main__":
    df = load_economics()

    good = df[df["loan_status"] == "Fully Paid"]
    bad = df[df["loan_status"] == "Charged Off"]

    avg_profit_good = good["net_realized"].mean()
    avg_loss_bad = bad["net_realized"].mean()  # negative number

    print(f"Good loans (Fully Paid): n={len(good):,}, avg net realized = ${avg_profit_good:,.2f}")
    print(f"Bad loans (Charged Off): n={len(bad):,}, avg net realized = ${avg_loss_bad:,.2f}")
    print()

    ratio = abs(avg_loss_bad) / avg_profit_good
    print(f"|avg loss on a bad loan| / avg profit on a good loan = {ratio:.2f}")
    print(f"-> One missed default (false negative) costs about as much as declining ~{ratio:.1f} good loans (false positives)")

    print()
    base_rate = len(bad) / (len(good) + len(bad))
    print(f"Base rate check: bad = {base_rate:.1%} of terminal loans (goods outnumber bads {(1-base_rate)/base_rate:.1f}:1)")
