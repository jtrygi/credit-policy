"""Does the model-vs-grade profitability edge from SEGMENTATION_POLICY.md
survive out-of-time? Rebuilds both halves of that comparison on the
chronological split: the grade-cutoff baseline recomputed on the OOT test
set (2015-2016 vintages, never trained on), and each model's full
approve-lowest-PD-first frontier using the v2 (OOT-refit) models' scores.
Same mechanics as baseline_policy.py + segment_policy.py's
frontier_comparison, applied to data/test_oot_policy.csv /
images/test_oot_scores.csv instead of the random-split files.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINE_COLOR = "#6B7280"
LR24_COLOR = "#2563EB"
XGB_COLOR = "#7C3AED"
GUIDE_COLOR = "#DC2626"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

GRADES = ["A", "B", "C", "D", "E", "F", "G"]
REFERENCE_CUTOFF = "D"


def baseline_sweep(df):
    n_total = len(df)
    rows = []
    for cutoff in GRADES:
        approved = df[df["grade"] <= cutoff]
        good = approved[approved["bad"] == 0]
        bad = approved[approved["bad"] == 1]
        revenue = good["net_realized"].sum() / n_total * 1000
        loss = -bad["net_realized"].sum() / n_total * 1000
        rows.append(dict(cutoff=f"A-{cutoff}" if cutoff != "A" else "A",
                          n_approved=len(approved), volume_pct=len(approved) / n_total,
                          bad_rate=approved["bad"].mean() if len(approved) else float("nan"),
                          revenue_per_1000=revenue, loss_per_1000=loss,
                          net_profit_per_1000=revenue - loss))
    return pd.DataFrame(rows)


def model_frontier(test_df, score_col):
    df = test_df.sort_values(score_col).reset_index(drop=True)
    n = len(df)
    cum_n = np.arange(1, n + 1)
    volume_pct = cum_n / n
    bad_rate = df["bad"].cumsum().values / cum_n
    profit_per_1000 = df["net_realized"].cumsum().values / n * 1000
    return volume_pct, bad_rate, profit_per_1000


def frontier_points(volume_pct, bad_rate, profit_per_1000, baseline_ref, n):
    idx_vol = int(round(baseline_ref["volume_pct"] * n)) - 1
    idx_vol = min(max(idx_vol, 0), n - 1)
    equal_volume = dict(volume_pct=float(volume_pct[idx_vol]), bad_rate=float(bad_rate[idx_vol]),
                         net_profit_per_1000=float(profit_per_1000[idx_vol]),
                         bad_rate_delta_vs_baseline=float(bad_rate[idx_vol] - baseline_ref["bad_rate"]))
    eligible = np.where(bad_rate <= baseline_ref["bad_rate"])[0]
    idx_loss = int(eligible.max()) if len(eligible) else 0
    equal_loss_rate = dict(volume_pct=float(volume_pct[idx_loss]), bad_rate=float(bad_rate[idx_loss]),
                            net_profit_per_1000=float(profit_per_1000[idx_loss]),
                            volume_delta_vs_baseline=float(volume_pct[idx_loss] - baseline_ref["volume_pct"]))
    return dict(equal_volume=equal_volume, equal_loss_rate=equal_loss_rate)


def main():
    policy = pd.read_csv("data/test_oot_policy.csv")
    scores = pd.read_csv("images/test_oot_scores.csv")
    assert len(policy) == len(scores) and (policy["bad"].values == scores["bad"].values).all()
    test_df = pd.concat([scores.reset_index(drop=True), policy.reset_index(drop=True)[["net_realized"]]], axis=1)
    n = len(test_df)

    curve = baseline_sweep(policy)
    curve.to_csv("images/oot_baseline_policy_curve.csv", index=False)
    ref_row = curve[curve["cutoff"] == f"A-{REFERENCE_CUTOFF}"].iloc[0]
    baseline_ref = dict(cutoff=ref_row["cutoff"], volume_pct=float(ref_row["volume_pct"]),
                         bad_rate=float(ref_row["bad_rate"]),
                         net_profit_per_1000=float(ref_row["net_profit_per_1000"]))
    print("OOT baseline frontier (test = 2015-2016 vintages):")
    print(curve.to_string(index=False))
    print(f"\nReference baseline (A-{REFERENCE_CUTOFF}): volume={baseline_ref['volume_pct']:.1%}  "
          f"bad_rate={baseline_ref['bad_rate']:.2%}  net_profit/1000=${baseline_ref['net_profit_per_1000']:,.0f}")

    results = {"baseline": baseline_ref}
    frontiers = {}
    for score_col, name in [("lr24_pd", "LR-24"), ("xgb_pd", "XGBoost")]:
        vol, bad, profit = model_frontier(test_df, score_col)
        frontiers[score_col] = (vol, bad, profit)
        pts = frontier_points(vol, bad, profit, baseline_ref, n)
        results[score_col] = pts
        print(f"\n{name} (OOT) frontier vs. baseline:")
        ev, el = pts["equal_volume"], pts["equal_loss_rate"]
        print(f"  Equal volume ({ev['volume_pct']:.1%}): bad_rate={ev['bad_rate']:.2%} "
              f"(baseline {baseline_ref['bad_rate']:.2%}, delta {ev['bad_rate_delta_vs_baseline']:+.2%})  "
              f"net_profit/1000=${ev['net_profit_per_1000']:,.0f} "
              f"(delta ${ev['net_profit_per_1000'] - baseline_ref['net_profit_per_1000']:+,.0f})")
        print(f"  Equal loss rate ({el['bad_rate']:.2%}): volume={el['volume_pct']:.1%} "
              f"(baseline {baseline_ref['volume_pct']:.1%}, delta {el['volume_delta_vs_baseline']:+.2%})  "
              f"net_profit/1000=${el['net_profit_per_1000']:,.0f} "
              f"(delta ${el['net_profit_per_1000'] - baseline_ref['net_profit_per_1000']:+,.0f})")

    with open("images/oot_frontier_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    # --- Overlay chart, same design as frontier_chart.py ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(curve["volume_pct"] * 100, curve["bad_rate"] * 100, color=BASELINE_COLOR,
              linewidth=2, linestyle="--", marker="o", markersize=6, label="Baseline (LendingClub grade)")
    ax2.plot(curve["volume_pct"] * 100, curve["net_profit_per_1000"], color=BASELINE_COLOR,
              linewidth=2, linestyle="--", marker="o", markersize=6, label="Baseline (LendingClub grade)")
    for score_col, color, name in [("lr24_pd", LR24_COLOR, "LR-24 (interpretable model)"),
                                     ("xgb_pd", XGB_COLOR, "XGBoost (full-features)")]:
        vol, bad, profit = frontiers[score_col]
        idx = np.unique(np.linspace(0, n - 1, 400).astype(int))
        ax1.plot(vol[idx] * 100, bad[idx] * 100, color=color, linewidth=2.2, label=name)
        ax2.plot(vol[idx] * 100, profit[idx], color=color, linewidth=2.2, label=name)

    ax1.axvline(baseline_ref["volume_pct"] * 100, color=GUIDE_COLOR, linewidth=1, linestyle=":")
    ax1.scatter([baseline_ref["volume_pct"] * 100], [baseline_ref["bad_rate"] * 100], color=GUIDE_COLOR, s=90, zorder=5)
    ax1.set_xlabel("Approval volume (% of applicants)")
    ax1.set_ylabel("Bad rate of approved pool (%)")
    ax1.set_title("Out-of-time (2015-16 test): loss rate", fontsize=12, fontweight="bold")
    ax1.legend(frameon=False, loc="upper left")
    ax1.grid(color=GRID, linewidth=0.8)
    ax1.set_axisbelow(True)

    ax2.axvline(baseline_ref["volume_pct"] * 100, color=GUIDE_COLOR, linewidth=1, linestyle=":")
    ax2.scatter([baseline_ref["volume_pct"] * 100], [baseline_ref["net_profit_per_1000"]], color=GUIDE_COLOR, s=90,
                zorder=5, label="Reference baseline (A-D)")
    ax2.set_xlabel("Approval volume (% of applicants)")
    ax2.set_ylabel("Net profit per 1,000 applicants ($)")
    ax2.set_title("Out-of-time (2015-16 test): profit", fontsize=12, fontweight="bold")
    ax2.legend(frameon=False, loc="upper left")
    ax2.grid(color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)

    fig.suptitle("Does the model-vs-grade edge survive out-of-time? (trained <=2013, tested on 2015-16)",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig("images/oot_frontier_comparison.png", dpi=150)
    plt.close(fig)
    print("\nWrote images/oot_baseline_policy_curve.csv, images/oot_frontier_results.json, "
          "images/oot_frontier_comparison.png")


if __name__ == "__main__":
    main()
