"""Step 3: baseline policy definition -- "approve everyone above a single
existing risk grade cutoff" (design doc Section 5, Step 3), evaluated on the
test set's realized economics (data/test_policy.csv, built by
reconstruct_policy_split.py).

Swept across all 7 possible grade cutoffs (A-only through approve-all,
i.e. LendingClub's own pre-assigned A-G grade) rather than picking one
point -- this gives the full baseline frontier the design doc's Section 4
comparison needs ("hold volume constant and compare loss rate" OR "hold
loss rate constant and compare volume"), not just a single number.

All dollar figures are per 1,000 TEST-POPULATION applicants (declined
applicants contribute $0), not per approved loan -- per Section 4's
explicit definition. Revenue/loss are reported as two lines (not just net
profit) by splitting each cutoff's approved pool into realized outcomes:
  revenue per 1,000  = sum(net_realized | Fully Paid)  / n_total * 1000
  loss per 1,000     = -sum(net_realized | Charged Off) / n_total * 1000
  net profit         = revenue - loss  (identically sum(net_realized) over
                        the whole approved pool / n_total * 1000)
No cost of funds is modeled (same simplification as economics.py, deferred
to Step 8's sensitivity analysis).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HUE = "#2563EB"
ACCENT = "#DC2626"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

GRADES = ["A", "B", "C", "D", "E", "F", "G"]
REFERENCE_CUTOFF = "D"  # headline single baseline: approve grades A-D


def sweep(df):
    n_total = len(df)
    rows = []
    for cutoff in GRADES:
        approved = df[df["grade"] <= cutoff]
        n_approved = len(approved)
        good = approved[approved["bad"] == 0]
        bad = approved[approved["bad"] == 1]
        revenue_per_1000 = good["net_realized"].sum() / n_total * 1000
        loss_per_1000 = -bad["net_realized"].sum() / n_total * 1000
        net_profit_per_1000 = approved["net_realized"].sum() / n_total * 1000
        assert abs((revenue_per_1000 - loss_per_1000) - net_profit_per_1000) < 1e-6
        rows.append(dict(
            cutoff=f"A-{cutoff}" if cutoff != "A" else "A",
            n_approved=n_approved,
            volume_pct=n_approved / n_total,
            bad_rate=approved["bad"].mean() if n_approved else float("nan"),
            revenue_per_1000=revenue_per_1000,
            loss_per_1000=loss_per_1000,
            net_profit_per_1000=net_profit_per_1000,
        ))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv("data/test_policy.csv")
    curve = sweep(df)
    curve.to_csv("images/baseline_policy_curve.csv", index=False)

    ref = curve[curve["cutoff"] == (f"A-{REFERENCE_CUTOFF}" if REFERENCE_CUTOFF != "A" else "A")].iloc[0]
    print("Baseline frontier (grade-cutoff sweep, test set, per 1,000 applicants):")
    print(curve.to_string(index=False))
    print(f"\nReference baseline (headline single-cutoff policy): approve grades A-{REFERENCE_CUTOFF}")
    print(f"  Volume: {ref['volume_pct']:.1%}   Bad rate: {ref['bad_rate']:.2%}   "
          f"Revenue/1000: ${ref['revenue_per_1000']:,.0f}   Loss/1000: ${ref['loss_per_1000']:,.0f}   "
          f"Net profit/1000: ${ref['net_profit_per_1000']:,.0f}")

    with open("images/baseline_policy_reference.json", "w") as f:
        json.dump(dict(cutoff=ref["cutoff"], volume_pct=float(ref["volume_pct"]),
                        bad_rate=float(ref["bad_rate"]),
                        revenue_per_1000=float(ref["revenue_per_1000"]),
                        loss_per_1000=float(ref["loss_per_1000"]),
                        net_profit_per_1000=float(ref["net_profit_per_1000"])), f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.plot(curve["volume_pct"] * 100, curve["net_profit_per_1000"], color=HUE, linewidth=2, marker="o", markersize=6)
    for _, r in curve.iterrows():
        ax1.annotate(r["cutoff"], (r["volume_pct"] * 100, r["net_profit_per_1000"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=8.5, color=MUTED)
    ax1.scatter([ref["volume_pct"] * 100], [ref["net_profit_per_1000"]], color=ACCENT, s=110, zorder=5,
                label=f"Reference baseline (A-{REFERENCE_CUTOFF})")
    ax1.set_xlabel("Approval volume (% of applicants)")
    ax1.set_ylabel("Net profit per 1,000 applicants ($)")
    ax1.set_title("Baseline frontier: profit vs. approval volume", fontsize=12, fontweight="bold")
    ax1.legend(frameon=False, loc="lower right")
    ax1.grid(color=GRID, linewidth=0.8)
    ax1.set_axisbelow(True)

    ax2.plot(curve["volume_pct"] * 100, curve["bad_rate"] * 100, color=HUE, linewidth=2, marker="o", markersize=6)
    for _, r in curve.iterrows():
        ax2.annotate(r["cutoff"], (r["volume_pct"] * 100, r["bad_rate"] * 100),
                     textcoords="offset points", xytext=(6, 4), fontsize=8.5, color=MUTED)
    ax2.scatter([ref["volume_pct"] * 100], [ref["bad_rate"] * 100], color=ACCENT, s=110, zorder=5,
                label=f"Reference baseline (A-{REFERENCE_CUTOFF})")
    ax2.set_xlabel("Approval volume (% of applicants)")
    ax2.set_ylabel("Bad rate of approved pool (%)")
    ax2.set_title("Baseline frontier: loss rate vs. approval volume", fontsize=12, fontweight="bold")
    ax2.legend(frameon=False, loc="upper left")
    ax2.grid(color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig("images/baseline_policy_curve.png", dpi=150)
    plt.close(fig)
    print("\nWrote images/baseline_policy_curve.{csv,png}, images/baseline_policy_reference.json")


if __name__ == "__main__":
    main()
