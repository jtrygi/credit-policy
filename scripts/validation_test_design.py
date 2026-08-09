"""Step 9: proposed validation test design.

Two things worth computing from real data rather than assuming:
  1. The disagreement region -- champion (baseline grade A-D) and
     challenger (model policy) actually only disagree on a minority of
     applicants. A test that randomizes the WHOLE population wastes power
     on the majority where both policies already agree; restricting
     randomization (and the analysis) to the disagreement region gives a
     concentrated, efficient read with far less exposure.
  2. Sample size for a two-proportion test powered to detect the bad-rate
     difference WITHIN that disagreement region specifically (not the
     diluted population-level effect from STEP8_BUSINESS_CASE.md).
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

policy = pd.read_csv("data/test_oot_policy.csv")
scores = pd.read_csv("images/test_oot_scores.csv")
assert len(policy) == len(scores) and (policy["bad"].values == scores["bad"].values).all()
df = pd.concat([scores.reset_index(drop=True), policy.reset_index(drop=True)[["net_realized", "grade"]]], axis=1)
n_total = len(df)

with open("images/oot_frontier_results.json") as f:
    frontier = json.load(f)

df["champion_approve"] = df["grade"] <= "D"

results = {}
for score_col, name in [("lr24_pd", "LR-24"), ("xgb_pd", "XGBoost")]:
    vol_pct = frontier[score_col]["equal_loss_rate"]["volume_pct"]
    n_approve = int(round(vol_pct * n_total))
    threshold = df.sort_values(score_col)[score_col].iloc[n_approve - 1]
    df["challenger_approve"] = df[score_col] <= threshold

    both_approve = df["champion_approve"] & df["challenger_approve"]
    only_champion = df["champion_approve"] & ~df["challenger_approve"]   # champion approves, challenger would decline
    only_challenger = ~df["champion_approve"] & df["challenger_approve"]  # champion declines, challenger would approve
    both_decline = ~df["champion_approve"] & ~df["challenger_approve"]

    disagreement_n = only_champion.sum() + only_challenger.sum()
    disagreement_pct = disagreement_n / n_total

    # Bad rate for each side of the disagreement region (real historical
    # outcomes -- in a live test, only_challenger's true bad rate is
    # exactly what's unobserved and needs the randomized approval).
    bad_rate_only_champion = df.loc[only_champion, "bad"].mean() if only_champion.sum() else float("nan")
    bad_rate_only_challenger = df.loc[only_challenger, "bad"].mean() if only_challenger.sum() else float("nan")

    print(f"\n=== {name} vs. champion (baseline A-D) ===")
    print(f"Both approve: {both_approve.sum():,} ({both_approve.mean():.1%})")
    print(f"Both decline: {both_decline.sum():,} ({both_decline.mean():.1%})")
    print(f"Champion approves, challenger would decline: {only_champion.sum():,} ({only_champion.mean():.1%}), "
          f"bad_rate={bad_rate_only_champion:.2%}")
    print(f"Champion declines, challenger would approve: {only_challenger.sum():,} ({only_challenger.mean():.1%}), "
          f"bad_rate={bad_rate_only_challenger:.2%}")
    print(f"Total disagreement region: {disagreement_n:,} ({disagreement_pct:.1%} of applicants)")

    # --- Sample size for a two-proportion z-test within the disagreement region ---
    p1, p2 = bad_rate_only_champion, bad_rate_only_challenger
    p_bar = (p1 + p2) / 2
    alpha, power = 0.05, 0.80
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    effect = abs(p1 - p2)
    n_per_arm = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_beta *
                  np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (effect ** 2)

    print(f"Two-proportion test (alpha=0.05, power=0.80) to detect {effect:.2%} bad-rate gap "
          f"within the disagreement region: n={n_per_arm:.0f} per arm ({2 * n_per_arm:.0f} total)")

    results[score_col] = dict(
        model=name, vol_pct=float(vol_pct),
        both_approve_n=int(both_approve.sum()), both_decline_n=int(both_decline.sum()),
        only_champion_n=int(only_champion.sum()), only_challenger_n=int(only_challenger.sum()),
        disagreement_n=int(disagreement_n), disagreement_pct=float(disagreement_pct),
        bad_rate_only_champion=float(bad_rate_only_champion), bad_rate_only_challenger=float(bad_rate_only_challenger),
        n_per_arm_required=float(n_per_arm),
    )

with open("images/validation_test_design.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nWrote images/validation_test_design.json")
