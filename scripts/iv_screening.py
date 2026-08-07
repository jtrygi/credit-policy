"""Systematic qualifying-variable screen: compute Information Value (IV) for
every candidate column (not just the 20 hand-curated ones from train_models.py),
rank them, and derive the candidate pool for wrapper selection from the data
rather than from judgment.

IV is the standard credit-scoring screening metric: bin the variable, compute
Weight of Evidence (WOE = ln(%good / %bad)) per bin, sum (%good - %bad) * WOE
across bins. Conventional thresholds (Siddiqi, "Credit Risk Scorecards"):
  < 0.02  not useful       0.02-0.1  weak       0.1-0.3  medium
  0.3-0.5  strong          > 0.5     suspiciously good (check for leakage)

Missing values are treated as their own bin -- standard practice, since
missingness itself often carries signal (and we already saw this: several
extended bureau fields are 90%+ missing because of when LendingClub started
collecting them, which is itself informative about vintage/product changes).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXCLUDE = {
    "bad", "issue_dt", "zip_code",  # target / not a feature / high-cardinality fair-lending proxy
    "fico_range_low", "fico_range_high",  # replaced by derived fico_avg
}

HUE = "#2563EB"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"


def compute_iv(x, y, is_categorical, n_bins=10):
    df = pd.DataFrame({"x": x, "y": y})
    if is_categorical:
        df["bin"] = df["x"].astype(str).fillna("MISSING")
    else:
        non_null = df["x"].notna()
        if non_null.sum() == 0:
            return 0.0
        nunique = df.loc[non_null, "x"].nunique()
        if nunique <= n_bins:
            # Low-cardinality numeric (e.g. term_months, count fields): bin by
            # exact value. qcut would collapse a skewed low-cardinality column
            # (e.g. 91% one value) into a single bin and silently hide signal.
            df.loc[non_null, "bin"] = df.loc[non_null, "x"].astype(str)
        else:
            # Rank-based qcut: ties broken by row order gives every row a
            # unique rank, so quantile cuts on the rank always produce n_bins
            # groups even when the raw values are heavily zero-inflated/skewed
            # (plain qcut on skewed values collapses to far fewer bins).
            ranks = df.loc[non_null, "x"].rank(method="first")
            df.loc[non_null, "bin"] = pd.qcut(ranks, n_bins, duplicates="drop").astype(str)
        df["bin"] = df["bin"].fillna("MISSING")

    total_good = (df["y"] == 0).sum()
    total_bad = (df["y"] == 1).sum()
    if total_good == 0 or total_bad == 0:
        return 0.0

    grp = df.groupby("bin")["y"].agg(["sum", "count"])
    grp["bad"] = grp["sum"]
    grp["good"] = grp["count"] - grp["sum"]
    # Laplace smoothing to avoid log(0) in sparse bins
    pct_good = (grp["good"] + 0.5) / (total_good + 0.5 * len(grp))
    pct_bad = (grp["bad"] + 0.5) / (total_bad + 0.5 * len(grp))
    woe = np.log(pct_good / pct_bad)
    iv = ((pct_good - pct_bad) * woe).sum()
    return iv


def main():
    train = pd.read_csv("data/train.csv", low_memory=False)
    train["fico_avg"] = (train["fico_range_low"] + train["fico_range_high"]) / 2

    candidates = [c for c in train.columns if c not in EXCLUDE and not c.endswith("_missing")]
    y = train["bad"].values

    results = []
    for col in candidates:
        is_cat = train[col].dtype == object
        iv = compute_iv(train[col].values, y, is_categorical=is_cat)
        results.append(dict(variable=col, iv=iv, type="categorical" if is_cat else "numeric"))

    iv_df = pd.DataFrame(results).sort_values("iv", ascending=False).reset_index(drop=True)
    iv_df.to_csv("images/iv_screening.csv", index=False)

    print(f"Screened {len(iv_df)} candidate variables\n")
    print(iv_df.to_string(index=False))

    qualified = iv_df[iv_df["iv"] >= 0.02]
    print(f"\nQualifying (IV >= 0.02): {len(qualified)} of {len(iv_df)}")
    print(f"Medium-or-better (IV >= 0.1): {(iv_df['iv'] >= 0.1).sum()}")
    print(f"Strong (IV >= 0.3): {(iv_df['iv'] >= 0.3).sum()}")

    qualified["variable"].to_json("images/qualified_features.json", orient="values")

    # --- Chart: IV ranking, all candidates, with qualifying threshold marked ---
    fig, ax = plt.subplots(figsize=(9, max(10, len(iv_df) * 0.22)))
    colors = [HUE if v >= 0.02 else MUTED for v in iv_df["iv"]]
    ax.barh(iv_df["variable"][::-1], iv_df["iv"][::-1], color=colors[::-1], zorder=3)
    ax.axvline(0.02, color=INK, linewidth=1, linestyle="--")
    ax.text(0.02, len(iv_df) - 0.5, " qualifying threshold (IV=0.02)", fontsize=9, color=INK, va="top")
    ax.set_xlabel("Information Value")
    ax.set_title("Information Value screening -- all candidate variables", fontsize=13, fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/iv_screening.png", dpi=150)
    plt.close(fig)

    print("\nWrote images/iv_screening.csv, images/iv_screening.png, images/qualified_features.json")


if __name__ == "__main__":
    main()
