"""Step 8: business case & sensitivity analysis.

Anchored on the OUT-OF-TIME numbers (OOT_VALIDATION.md), not the original
random-split numbers in SEGMENTATION_POLICY.md -- the OOT figures are the
defensible, less-overstated estimate of the real edge.

Two pessimistic scenarios, applied to both the baseline (grade A-D) and
each recommended policy's approved pool on OOT test:

  Severity shock (+20% loss per charged-off loan): the design doc's own
  suggested magnitude. Checked whether vintage data could size this one
  empirically instead -- it can't (avg $ loss per bad loan has no clean
  macro-correlated pattern across origination years, see
  SEGMENTATION_POLICY.md's investigation) -- so this stays a stated
  judgment call, not a derived one.

  Macro/frequency shock (bad rate up ~39% relative): sized from this
  project's OWN data. LendingClub's 2008 vintage (financial crisis) had a
  realized bad rate of 20.7% vs. this project's OOT test population's
  14.85% -- a real historical data point, not a guessed percentage.

Mechanics: for a given approved pool, hold avg_profit_good fixed (a
performing loan's payoff doesn't change under either shock) and reshock
bad_rate (frequency) and avg_loss_bad (severity) independently -- standard
PD/LGD stress-test mechanics, three numbers and one formula, fully
auditable. Not a resimulation of individual loans.
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
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

SEVERITY_SHOCK = 0.20     # +20% loss per charged-off loan
FREQUENCY_SHOCK = 0.394   # 2008 vintage bad rate (20.7%) / OOT test bad rate (14.85%) - 1


def pool_stats(df, n_total):
    good = df[df["bad"] == 0]
    bad = df[df["bad"] == 1]
    return dict(
        n=len(df), volume_pct=len(df) / n_total, bad_rate=df["bad"].mean(),
        avg_profit_good=good["net_realized"].mean(), avg_loss_bad=bad["net_realized"].mean(),
    )


def net_profit_per_1000(stats, freq_mult=1.0, sev_mult=1.0):
    bad_rate = min(stats["bad_rate"] * freq_mult, 1.0)
    avg_loss_bad = stats["avg_loss_bad"] * sev_mult
    per_applicant = (1 - bad_rate) * stats["avg_profit_good"] + bad_rate * avg_loss_bad
    return stats["volume_pct"] * 1000 * per_applicant


def main():
    policy = pd.read_csv("data/test_oot_policy.csv")
    scores = pd.read_csv("images/test_oot_scores.csv")
    assert len(policy) == len(scores) and (policy["bad"].values == scores["bad"].values).all()
    df = pd.concat([scores.reset_index(drop=True), policy.reset_index(drop=True)[["net_realized", "grade"]]], axis=1)
    n_total = len(df)

    baseline_pool = df[df["grade"] <= "D"]
    baseline_stats = pool_stats(baseline_pool, n_total)

    with open("images/oot_frontier_results.json") as f:
        frontier = json.load(f)

    policies = {"baseline": baseline_stats}
    for score_col, name in [("lr24_pd", "LR-24"), ("xgb_pd", "XGBoost")]:
        vol_pct = frontier[score_col]["equal_loss_rate"]["volume_pct"]
        n_approve = int(round(vol_pct * n_total))
        pool = df.sort_values(score_col).iloc[:n_approve]
        policies[score_col] = pool_stats(pool, n_total)
        print(f"{name} recommended policy (equal-loss-rate cutoff): volume={vol_pct:.1%}  "
              f"bad_rate={policies[score_col]['bad_rate']:.2%}")

    scenarios = [
        ("Base case", 1.0, 1.0),
        (f"Severity shock (+{SEVERITY_SHOCK:.0%} loss/bad loan)", 1.0, 1 + SEVERITY_SHOCK),
        (f"Macro shock (bad rate x{1+FREQUENCY_SHOCK:.2f}, ~2008 vintage)", 1 + FREQUENCY_SHOCK, 1.0),
        ("Combined (severity + macro)", 1 + FREQUENCY_SHOCK, 1 + SEVERITY_SHOCK),
    ]

    rows = []
    for scenario_name, freq_mult, sev_mult in scenarios:
        row = dict(scenario=scenario_name)
        base_profit = net_profit_per_1000(baseline_stats, freq_mult, sev_mult)
        row["baseline"] = base_profit
        for key, name in [("lr24_pd", "lr24"), ("xgb_pd", "xgb")]:
            p = net_profit_per_1000(policies[key], freq_mult, sev_mult)
            row[name] = p
            row[f"{name}_delta"] = p - base_profit
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df.to_csv("images/stress_test_results.csv", index=False)
    print("\n" + results_df.to_string(index=False))

    with open("images/stress_test_results.json", "w") as f:
        json.dump(dict(scenarios=rows, baseline_stats={k: float(v) for k, v in baseline_stats.items()},
                        lr24_stats={k: float(v) for k, v in policies["lr24_pd"].items()},
                        xgb_stats={k: float(v) for k, v in policies["xgb_pd"].items()},
                        severity_shock=SEVERITY_SHOCK, frequency_shock=FREQUENCY_SHOCK),
                  f, indent=2, default=float)

    # --- Chart: net profit per scenario, baseline vs. both policies ---
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(scenarios))
    w = 0.26
    ax.bar(x - w, results_df["baseline"], width=w, color=BASELINE_COLOR, label="Baseline (grade A-D)")
    ax.bar(x, results_df["lr24"], width=w, color=LR24_COLOR, label="LR-24 policy")
    ax.bar(x + w, results_df["xgb"], width=w, color=XGB_COLOR, label="XGBoost policy")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0].replace(" (", "\n(") for s in scenarios], fontsize=9.5)
    ax.set_ylabel("Net profit per 1,000 applicants ($)")
    ax.set_title("Step 8: net profit under stress (OOT test, 2015-16 vintages)", fontsize=13, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(color=GRID, linewidth=0.8, axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/stress_test_profit.png", dpi=150)
    plt.close(fig)

    # --- Chart: delta over baseline per scenario (the actual robustness question) ---
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.bar(x - w / 2, results_df["lr24_delta"], width=w, color=LR24_COLOR, label="LR-24 vs. baseline")
    ax.bar(x + w / 2, results_df["xgb_delta"], width=w, color=XGB_COLOR, label="XGBoost vs. baseline")
    for i, (lr_d, xgb_d) in enumerate(zip(results_df["lr24_delta"], results_df["xgb_delta"])):
        ax.annotate(f"${lr_d:,.0f}", (i - w / 2, lr_d), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9, fontweight="bold", color=INK)
        ax.annotate(f"${xgb_d:,.0f}", (i + w / 2, xgb_d), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9, fontweight="bold", color=INK)
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0].replace(" (", "\n(") for s in scenarios], fontsize=9.5)
    ax.set_ylabel("Net profit improvement over baseline\n($ per 1,000 applicants)")
    ax.set_title("Does the edge survive stress? (OOT test, equal-loss-rate policy)", fontsize=13, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(color=GRID, linewidth=0.8, axis="y")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/stress_test_delta.png", dpi=150)
    plt.close(fig)

    print("\nWrote images/stress_test_results.{csv,json}, images/stress_test_profit.png, images/stress_test_delta.png")


if __name__ == "__main__":
    main()
