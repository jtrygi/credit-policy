"""Generate overview charts of the raw LendingClub dataset into images/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

RAW = "data/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv"
HUE = "#2563EB"  # single consistent hue: these are single-series magnitude charts, not category comparisons
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

df = pd.read_csv(RAW, usecols=["term", "issue_d"])
df["term"] = df["term"].str.strip()
df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y")

# --- Chart 1: loan count by term length ---
term_counts = df["term"].value_counts().reindex(["36 months", "60 months"])

fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(term_counts.index, term_counts.values, color=HUE, width=0.5, zorder=3)
ax.set_title("Loan records by term length", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Number of loans")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for b, v in zip(bars, term_counts.values):
    ax.annotate(f"{v:,}", (b.get_x() + b.get_width() / 2, v), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig("images/loans_by_term.png", dpi=150)
plt.close(fig)

# --- Chart 2: loan volume by issue month, over time ---
monthly = df.groupby(df["issue_dt"].dt.to_period("M")).size()
monthly.index = monthly.index.to_timestamp()

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.bar(monthly.index, monthly.values, color=HUE, width=20, zorder=3)
ax.set_title("Loan origination volume by month (2007-2018)", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Number of loans")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))
ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("images/loans_by_month.png", dpi=150)
plt.close(fig)

print("Wrote images/loans_by_term.png")
print("Wrote images/loans_by_month.png")
