"""The single most important chart in this validation exercise: does the
profitability edge over grade-based baseline hold up once you stop
evaluating on a random split and evaluate out-of-time instead? Reads both
already-computed results (images/segment_policy_results.json for the
original random split, images/oot_frontier_results.json for the
chronological split) and puts the $ improvement over baseline side by
side, same models, same two comparison axes (equal volume / equal loss
rate), so the shrinkage is visible directly rather than requiring the
reader to cross-reference two separate documents' numbers.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LR24_COLOR = "#2563EB"
XGB_COLOR = "#7C3AED"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    with open("images/segment_policy_results.json") as f:
        random_split = json.load(f)
    with open("images/oot_frontier_results.json") as f:
        oot_split = json.load(f)

    random_baseline = random_split["baseline"]["net_profit_per_1000"]
    oot_baseline = oot_split["baseline"]["net_profit_per_1000"]

    models = [("lr24", "lr24_pd", "LR-24\n(interpretable)", LR24_COLOR),
              ("xgb", "xgb_pd", "XGBoost\n(full-features)", XGB_COLOR)]
    comparisons = [("equal_volume", "Equal volume vs. baseline"), ("equal_loss_rate", "Equal loss rate vs. baseline")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=False)

    for ax, (comp_key, comp_title) in zip(axes, comparisons):
        x = np.arange(len(models))
        w = 0.32
        random_vals, oot_vals = [], []
        for key, oot_key, _, _ in models:
            r_profit = random_split["results"][key]["frontier"][comp_key]["net_profit_per_1000"]
            o_profit = oot_split[oot_key][comp_key]["net_profit_per_1000"]
            random_vals.append(r_profit - random_baseline)
            oot_vals.append(o_profit - oot_baseline)

        bars_r = ax.bar(x - w / 2, random_vals, width=w, color=[c for _, _, _, c in models],
                         alpha=0.45, edgecolor=[c for _, _, _, c in models], linewidth=1.5,
                         hatch="///", label="Random split (same-era holdout)")
        bars_o = ax.bar(x + w / 2, oot_vals, width=w, color=[c for _, _, _, c in models],
                         label="Out-of-time split (train<=2013, test 2015-16)")

        for bars, vals in [(bars_r, random_vals), (bars_o, oot_vals)]:
            for rect, v in zip(bars, vals):
                ax.annotate(f"${v:,.0f}", (rect.get_x() + rect.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 5 if v >= 0 else -14),
                            ha="center", fontsize=9.5, color=INK, fontweight="bold")

        ax.axhline(0, color=INK, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([m[2] for m in models])
        ax.set_ylabel("Net profit improvement over baseline\n($ per 1,000 applicants)")
        ax.set_title(comp_title, fontsize=12, fontweight="bold")
        ax.grid(color=GRID, linewidth=0.8, axis="y")
        ax.set_axisbelow(True)

    # Single shared legend for hatched-vs-solid meaning (color already explained by x-tick labels)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=INK, hatch="///", label="Random split (same-era holdout)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=INK, label="Out-of-time split (genuinely unseen future vintages)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("The methodology matters: random-split evaluation overstates the real edge ~5x",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig("images/methodology_comparison.png", dpi=150)
    plt.close(fig)
    print("Wrote images/methodology_comparison.png")


if __name__ == "__main__":
    main()
