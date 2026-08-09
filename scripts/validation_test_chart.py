"""Visualize the champion/challenger disagreement region for XGBoost (the
recommended test candidate) -- the four quadrants of agree/disagree
between champion (baseline grade A-D) and challenger (model policy), sized
by applicant count and colored by realized bad rate.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"
GOOD = "#059669"
BAD = "#DC2626"
NEUTRAL = "#6B7280"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

with open("images/validation_test_design.json") as f:
    results = json.load(f)

r = results["xgb_pd"]
quadrants = [
    ("Both approve\n(no test needed)", r["both_approve_n"], None, NEUTRAL),
    ("Both decline\n(no test needed)", r["both_decline_n"], None, NEUTRAL),
    ("Champion approves,\nchallenger declines", r["only_champion_n"], r["bad_rate_only_champion"], BAD),
    ("Champion declines,\nchallenger approves", r["only_challenger_n"], r["bad_rate_only_challenger"], GOOD),
]

fig, ax = plt.subplots(figsize=(10, 6))
labels = [q[0] for q in quadrants]
sizes = [q[1] for q in quadrants]
colors = [q[3] for q in quadrants]
bars = ax.bar(labels, sizes, color=colors)
alphas = [0.35, 0.35, 0.85, 0.85]
for bar, a in zip(bars, alphas):
    bar.set_alpha(a)
for bar, (label, n, bad_rate, color) in zip(bars, quadrants):
    text = f"n={n:,}" if bad_rate is None else f"n={n:,}\nbad rate={bad_rate:.1%}"
    ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points", xytext=(0, 6), ha="center", fontsize=10, fontweight="bold")

ax.set_ylabel("Applicants (OOT test population)")
ax.set_title("XGBoost vs. baseline: where do they actually disagree?", fontsize=13, fontweight="bold")
ax.annotate(f"Disagreement region: {r['disagreement_n']:,} applicants ({r['disagreement_pct']:.1%} of total) --\n"
            f"this is the only part of the population a live test needs to randomize.",
            xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top", fontsize=10, color=MUTED)
ax.grid(color=GRID, linewidth=0.8, axis="y")
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("images/validation_disagreement_region.png", dpi=150)
plt.close(fig)
print("Wrote images/validation_disagreement_region.png")
