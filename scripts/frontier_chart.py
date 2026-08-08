"""One chart that shows the whole dynamic SEGMENTATION_POLICY.md describes:
does ranking applicants by model score beat ranking by LendingClub's own
grade? Overlays the baseline grade frontier (7 discrete cutoffs, already
computed by baseline_policy.py) against each model's full continuous
approve-lowest-PD-first frontier on test, with guide lines at the
reference baseline's volume (93.3%) and loss rate (13.86%) so the
equal-volume / equal-loss-rate comparisons in segment_policy.py are
visible directly, not just tabulated.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASELINE_COLOR = "#6B7280"   # grey -- "current state" reference, not a highlighted series
LR24_COLOR = "#2563EB"       # blue -- matches HUE used elsewhere for the interpretable model
XGB_COLOR = "#7C3AED"        # purple -- matches HUE_GBM used in tune_xgb_convergence.png
GUIDE_COLOR = "#DC2626"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def model_frontier(score_col):
    scores = pd.read_csv("images/test_scores.csv")
    policy = pd.read_csv("data/test_policy.csv")
    assert len(scores) == len(policy) and (scores["bad"].values == policy["bad"].values).all()
    df = pd.concat([scores.reset_index(drop=True), policy.reset_index(drop=True)[["net_realized"]]], axis=1)
    df = df.sort_values(score_col).reset_index(drop=True)
    n = len(df)
    cum_n = np.arange(1, n + 1)
    volume_pct = cum_n / n * 100
    bad_rate = df["bad"].cumsum().values / cum_n * 100
    profit_per_1000 = df["net_realized"].cumsum().values / n * 1000
    # Downsample for a lighter file -- the curve is smooth at this scale.
    idx = np.unique(np.linspace(0, n - 1, 400).astype(int))
    return volume_pct[idx], bad_rate[idx], profit_per_1000[idx]


def main():
    baseline = pd.read_csv("images/baseline_policy_curve.csv")
    with open("images/baseline_policy_reference.json") as f:
        ref = json.load(f)

    lr24_vol, lr24_bad, lr24_profit = model_frontier("lr24_pd")
    xgb_vol, xgb_bad, xgb_profit = model_frontier("xgb_pd")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: loss rate vs. volume ---
    ax1.plot(baseline["volume_pct"] * 100, baseline["bad_rate"] * 100, color=BASELINE_COLOR,
              linewidth=2, linestyle="--", marker="o", markersize=6, label="Baseline (LendingClub grade)")
    ax1.plot(lr24_vol, lr24_bad, color=LR24_COLOR, linewidth=2.2, label="LR-24 (interpretable model)")
    ax1.plot(xgb_vol, xgb_bad, color=XGB_COLOR, linewidth=2.2, label="XGBoost (full-features)")
    ax1.axvline(ref["volume_pct"] * 100, color=GUIDE_COLOR, linewidth=1, linestyle=":")
    ax1.scatter([ref["volume_pct"] * 100], [ref["bad_rate"] * 100], color=GUIDE_COLOR, s=90, zorder=5)
    ax1.annotate("Equal volume:\nmodels cut loss rate\nbelow baseline",
                 xy=(ref["volume_pct"] * 100, ref["bad_rate"] * 100), xytext=(-155, -55),
                 textcoords="offset points", fontsize=9, color=MUTED, ha="left",
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax1.set_xlabel("Approval volume (% of applicants)")
    ax1.set_ylabel("Bad rate of approved pool (%)")
    ax1.set_title("Ranking by model score vs. ranking by grade: loss rate", fontsize=12, fontweight="bold")
    ax1.legend(frameon=False, loc="upper left")
    ax1.grid(color=GRID, linewidth=0.8)
    ax1.set_axisbelow(True)

    # --- Panel 2: profit vs. volume ---
    ax2.plot(baseline["volume_pct"] * 100, baseline["net_profit_per_1000"], color=BASELINE_COLOR,
              linewidth=2, linestyle="--", marker="o", markersize=6, label="Baseline (LendingClub grade)")
    ax2.plot(lr24_vol, lr24_profit, color=LR24_COLOR, linewidth=2.2, label="LR-24 (interpretable model)")
    ax2.plot(xgb_vol, xgb_profit, color=XGB_COLOR, linewidth=2.2, label="XGBoost (full-features)")
    ax2.axvline(ref["volume_pct"] * 100, color=GUIDE_COLOR, linewidth=1, linestyle=":")
    ax2.scatter([ref["volume_pct"] * 100], [ref["net_profit_per_1000"]], color=GUIDE_COLOR, s=90, zorder=5,
                label="Reference baseline (A-D)")
    ax2.annotate("Equal volume:\nmodels earn more\nprofit than baseline",
                 xy=(ref["volume_pct"] * 100, ref["net_profit_per_1000"]),
                 xytext=(0.55, 0.18), textcoords="axes fraction",
                 fontsize=9, color=MUTED, ha="left",
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax2.set_xlabel("Approval volume (% of applicants)")
    ax2.set_ylabel("Net profit per 1,000 applicants ($)")
    ax2.set_title("Ranking by model score vs. ranking by grade: profit", fontsize=12, fontweight="bold")
    ax2.legend(frameon=False, loc="upper left")
    ax2.grid(color=GRID, linewidth=0.8)
    ax2.set_axisbelow(True)

    fig.suptitle("Does a multi-variable model beat LendingClub's single-grade cutoff?", fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig("images/frontier_comparison.png", dpi=150)
    plt.close(fig)
    print("Wrote images/frontier_comparison.png")


if __name__ == "__main__":
    main()
