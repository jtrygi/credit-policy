"""Step 6 (segmentation) + Step 7 (policy design), for both scoring models
(LR-24 and XGBoost) so the interpretability-vs-lift dollar tradeoff Step 5
promised is actually computable.

Methodology (per design doc Section 5, Steps 6-7):
  1. Bin each model's PD score into 6 quantile segments -- defined on VAL,
     not test, so the segment boundaries and the approve/decline decision
     for each segment are chosen without ever looking at test. Verify
     bad rate rises monotonically from segment 1 (lowest PD) to 6
     (highest PD) on val, then confirm the SAME boundaries still produce
     a monotonic, meaningfully-differentiated bad rate on test (stability
     check, not re-fit).
  2. Per-segment decision rule: approve iff the segment's mean realized
     $ outcome on VAL is positive (each segment's own realized economics,
     not a single blended $2,323/$4,940 figure -- loss severity and
     revenue both vary by risk tier, and this book's own history is the
     best estimate of each tier's economics).
  3. Apply that val-derived decision to the TEST segments (same
     boundaries) and recompute portfolio approval volume / bad rate /
     revenue / loss / net profit per 1,000 test applicants -- comparable
     to baseline_policy.py's numbers on the same test set.

No cost of funds modeled (same simplification as economics.py / Step 3,
deferred to Step 8's sensitivity analysis).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HUE = "#2563EB"
HUE2 = "#7C3AED"
ACCENT = "#DC2626"
GOOD = "#059669"
BAD = "#DC2626"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

N_SEGMENTS = 6


def load_scored(split, score_col):
    scores = pd.read_csv(f"images/{split}_scores.csv")
    policy = pd.read_csv(f"data/{split}_policy.csv")
    assert len(scores) == len(policy) and (scores["bad"].values == policy["bad"].values).all()
    df = pd.concat([scores.reset_index(drop=True), policy.reset_index(drop=True)[["net_realized"]]], axis=1)
    return df


def define_bins(val_df, score_col, n_segments=N_SEGMENTS):
    _, bins = pd.qcut(val_df[score_col], n_segments, retbins=True, duplicates="drop")
    bins = bins.copy()
    bins[0], bins[-1] = -np.inf, np.inf
    return bins


def segment_table(df, score_col, bins, n_total_for_rate):
    df = df.copy()
    df["segment"] = pd.cut(df[score_col], bins=bins, labels=False, include_lowest=True) + 1
    rows = []
    for seg, g in df.groupby("segment"):
        good = g[g["bad"] == 0]
        bad = g[g["bad"] == 1]
        rows.append(dict(
            segment=int(seg),
            n=len(g),
            pd_range=f"{g[score_col].min():.3f}-{g[score_col].max():.3f}",
            bad_rate=g["bad"].mean(),
            avg_net_realized=g["net_realized"].mean(),
            revenue_per_1000=good["net_realized"].sum() / n_total_for_rate * 1000,
            loss_per_1000=-bad["net_realized"].sum() / n_total_for_rate * 1000,
            net_profit_per_1000=g["net_realized"].sum() / n_total_for_rate * 1000,
        ))
    return pd.DataFrame(rows).sort_values("segment").reset_index(drop=True)


def portfolio_from_decisions(df, score_col, bins, decisions, n_total):
    df = df.copy()
    df["segment"] = pd.cut(df[score_col], bins=bins, labels=False, include_lowest=True) + 1
    df["approve"] = df["segment"].map(decisions)
    approved = df[df["approve"]]
    good = approved[approved["bad"] == 0]
    bad = approved[approved["bad"] == 1]
    revenue = good["net_realized"].sum() / n_total * 1000
    loss = -bad["net_realized"].sum() / n_total * 1000
    return dict(
        n_approved=len(approved), volume_pct=len(approved) / n_total,
        bad_rate=approved["bad"].mean() if len(approved) else float("nan"),
        revenue_per_1000=revenue, loss_per_1000=loss,
        net_profit_per_1000=approved["net_realized"].sum() / n_total * 1000,
    )


def plot_segments(val_tbl, test_tbl, model_name, filename):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = val_tbl["segment"]
    w = 0.35
    axes[0].bar(x - w / 2, val_tbl["bad_rate"] * 100, width=w, color=HUE, label="Val")
    axes[0].bar(x + w / 2, test_tbl["bad_rate"] * 100, width=w, color=HUE2, label="Test")
    axes[0].set_xlabel("Segment (1 = lowest risk, 6 = highest risk)")
    axes[0].set_ylabel("Bad rate (%)")
    axes[0].set_title(f"{model_name}: bad rate by segment", fontsize=12, fontweight="bold")
    axes[0].set_xticks(list(x))
    axes[0].legend(frameon=False)
    axes[0].grid(color=GRID, linewidth=0.8, axis="y")
    axes[0].set_axisbelow(True)

    colors = [GOOD if v > 0 else BAD for v in test_tbl["avg_net_realized"]]
    axes[1].bar(test_tbl["segment"], test_tbl["avg_net_realized"], color=colors)
    axes[1].axhline(0, color=INK, linewidth=1)
    axes[1].set_xlabel("Segment (1 = lowest risk, 6 = highest risk)")
    axes[1].set_ylabel("Avg realized $ per applicant (test)")
    axes[1].set_title(f"{model_name}: avg profit/loss by segment (test)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(list(test_tbl["segment"]))
    axes[1].grid(color=GRID, linewidth=0.8, axis="y")
    axes[1].set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def run_for_model(score_col, model_name, slug):
    val_df = load_scored("val", score_col)
    test_df = load_scored("test", score_col)
    n_val, n_test = len(val_df), len(test_df)

    bins = define_bins(val_df, score_col)
    val_tbl = segment_table(val_df, score_col, bins, n_val)
    test_tbl = segment_table(test_df, score_col, bins, n_test)

    val_monotonic = val_tbl["bad_rate"].is_monotonic_increasing
    test_monotonic = test_tbl["bad_rate"].is_monotonic_increasing
    print(f"\n=== {model_name} ===")
    print("Segment boundaries (val-derived PD cutpoints):", [round(b, 4) for b in bins])
    print("\nVal segment table:")
    print(val_tbl.to_string(index=False))
    print(f"Val bad-rate monotonic across segments: {val_monotonic}")
    print("\nTest segment table (same boundaries):")
    print(test_tbl.to_string(index=False))
    print(f"Test bad-rate monotonic across segments: {test_monotonic}")

    decisions = {int(r.segment): bool(r.avg_net_realized > 0) for r in val_tbl.itertuples()}
    print(f"\nApprove/decline decision per segment (based on VAL avg realized $): {decisions}")

    test_portfolio = portfolio_from_decisions(test_df, score_col, bins, decisions, n_test)
    print(f"Segmented policy on TEST: volume={test_portfolio['volume_pct']:.1%}  "
          f"bad_rate={test_portfolio['bad_rate']:.2%}  "
          f"net_profit/1000=${test_portfolio['net_profit_per_1000']:,.0f}")

    plot_segments(val_tbl, test_tbl, model_name, f"images/segments_{slug}.png")
    val_tbl.to_csv(f"images/segments_{slug}_val.csv", index=False)
    test_tbl.to_csv(f"images/segments_{slug}_test.csv", index=False)

    return dict(
        model=model_name, bins=[float(b) for b in bins],
        val_monotonic=bool(val_monotonic), test_monotonic=bool(test_monotonic),
        decisions=decisions, test_portfolio=test_portfolio,
    )


def frontier_comparison(score_col, model_name, baseline):
    """Every segment nets positive on this already-approved, already-priced
    book (see run_for_model's decisions dict -- always all-True), so a
    blanket per-segment approve/decline decision converges to approve-all
    and can't show the equal-volume / equal-loss-rate improvement Section 4
    asks for. The score still ranks risk far better than the coarse 7-grade
    baseline (AUC ~0.69-0.71 vs. grade alone) -- so the real comparison is:
    at the SAME approval volume as baseline, does ranking by score instead
    of by grade produce a lower loss rate? And at the SAME loss rate, does
    it permit more volume? This reads both points off the model's full
    approve-lowest-PD-first frontier on test, fixed volume/loss-rate targets
    taken from the already-computed Step 3 baseline (not fit to test).
    """
    test_df = load_scored("test", score_col).sort_values(score_col).reset_index(drop=True)
    n_total = len(test_df)
    cum_n = np.arange(1, n_total + 1)
    cum_bad_rate = test_df["bad"].cumsum().values / cum_n
    cum_profit_per_1000 = test_df["net_realized"].cumsum().values / n_total * 1000
    cum_volume_pct = cum_n / n_total

    # Equal volume: approve the same % of applicants as baseline, but pick
    # the lowest-PD ones instead of baseline's grade cutoff.
    idx_vol = int(round(baseline["volume_pct"] * n_total)) - 1
    equal_volume = dict(
        volume_pct=float(cum_volume_pct[idx_vol]), bad_rate=float(cum_bad_rate[idx_vol]),
        net_profit_per_1000=float(cum_profit_per_1000[idx_vol]),
        bad_rate_delta_vs_baseline=float(cum_bad_rate[idx_vol] - baseline["bad_rate"]),
    )

    # Equal loss rate: approve as many as possible while keeping the
    # approved pool's bad rate at or below baseline's.
    eligible = np.where(cum_bad_rate <= baseline["bad_rate"])[0]
    idx_loss = int(eligible.max()) if len(eligible) else 0
    equal_loss_rate = dict(
        volume_pct=float(cum_volume_pct[idx_loss]), bad_rate=float(cum_bad_rate[idx_loss]),
        net_profit_per_1000=float(cum_profit_per_1000[idx_loss]),
        volume_delta_vs_baseline=float(cum_volume_pct[idx_loss] - baseline["volume_pct"]),
    )

    print(f"\n{model_name} frontier vs. baseline (test, score-ranked approve-lowest-PD-first):")
    print(f"  Equal volume ({equal_volume['volume_pct']:.1%}): bad_rate={equal_volume['bad_rate']:.2%} "
          f"(baseline {baseline['bad_rate']:.2%}, delta {equal_volume['bad_rate_delta_vs_baseline']:+.2%})  "
          f"net_profit/1000=${equal_volume['net_profit_per_1000']:,.0f}")
    print(f"  Equal loss rate ({equal_loss_rate['bad_rate']:.2%}): volume={equal_loss_rate['volume_pct']:.1%} "
          f"(baseline {baseline['volume_pct']:.1%}, delta {equal_loss_rate['volume_delta_vs_baseline']:+.2%})  "
          f"net_profit/1000=${equal_loss_rate['net_profit_per_1000']:,.0f}")

    return dict(equal_volume=equal_volume, equal_loss_rate=equal_loss_rate)


def main():
    results = {}
    results["lr24"] = run_for_model("lr24_pd", "LR-24 (interpretable)", "lr24")
    results["xgb"] = run_for_model("xgb_pd", "XGBoost (full-features)", "xgb")

    with open("images/baseline_policy_reference.json") as f:
        baseline = json.load(f)
    print("\n=== Comparison vs. Step 3 baseline (approve A-D) ===")
    print(f"Baseline:  volume={baseline['volume_pct']:.1%}  bad_rate={baseline['bad_rate']:.2%}  "
          f"net_profit/1000=${baseline['net_profit_per_1000']:,.0f}")
    for key, score_col in [("lr24", "lr24_pd"), ("xgb", "xgb_pd")]:
        tp = results[key]["test_portfolio"]
        delta = tp["net_profit_per_1000"] - baseline["net_profit_per_1000"]
        print(f"{results[key]['model']:28s}: volume={tp['volume_pct']:.1%}  bad_rate={tp['bad_rate']:.2%}  "
              f"net_profit/1000=${tp['net_profit_per_1000']:,.0f}  (delta vs baseline: ${delta:+,.0f})")
        results[key]["frontier"] = frontier_comparison(score_col, results[key]["model"], baseline)

    with open("images/segment_policy_results.json", "w") as f:
        json.dump(dict(results=results, baseline=baseline), f, indent=2, default=float)
    print("\nWrote images/segments_{lr24,xgb}_{val,test}.csv, images/segments_{lr24,xgb}.png, "
          "images/segment_policy_results.json")


if __name__ == "__main__":
    main()
