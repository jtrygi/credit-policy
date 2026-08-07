"""Forward, backward, and stepwise wrapper selection on the IV-qualified
candidate pool (images/qualified_features.json), scored by validation AUC
(per project direction: AUC is cheap "out of the box" and drives the search;
Brier/KS are logged alongside every fit for free and compared afterward,
since they cost nothing extra -- same fit, different summary of the same
predicted probabilities).

Usage: python scripts/wrapper_selection.py --method forward|backward|stepwise
"""
import argparse
import json
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from metrics_extra import ks_statistic


def load(name):
    df = pd.read_csv(f"data/{name}.csv", low_memory=False)
    df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    return df


def get_candidate_types(train, candidates):
    numeric, categorical = [], []
    for c in candidates:
        if c == "fico_avg" or train[c].dtype != object:
            numeric.append(c)
        else:
            categorical.append(c)
    return numeric, categorical


def build_columns(df, numeric_selected, categorical_selected, train_cols, one_hot_cols=None):
    num_cols = []
    for f in numeric_selected:
        num_cols.append(f)
        flag = f"{f}_missing"
        if flag in train_cols:
            num_cols.append(flag)
    num = df[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)
    if categorical_selected:
        cat = pd.get_dummies(df[categorical_selected].astype(str), drop_first=True)
    else:
        cat = pd.DataFrame(index=df.index)
    X = pd.concat([num, cat], axis=1)
    if one_hot_cols is not None:
        X = X.reindex(columns=one_hot_cols, fill_value=0)
    return X


def fit_eval(train, val, feature_set, numeric_all, train_cols, max_iter=300):
    numeric_sel = [f for f in feature_set if f in numeric_all]
    categorical_sel = [f for f in feature_set if f not in numeric_all]
    X_train = build_columns(train, numeric_sel, categorical_sel, train_cols)
    cols = X_train.columns.tolist()
    X_val = build_columns(val, numeric_sel, categorical_sel, train_cols, one_hot_cols=cols)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_val)
    lr = LogisticRegression(max_iter=max_iter, random_state=42)
    lr.fit(Xtr, train["bad"].values)
    p = lr.predict_proba(Xv)[:, 1]
    y_val = val["bad"].values
    auc = roc_auc_score(y_val, p)
    brier = brier_score_loss(y_val, p)
    ks, _ = ks_statistic(y_val, p)
    return auc, brier, ks


def forward_selection(train, val, candidates, numeric_all, train_cols, t0):
    remaining = list(candidates)
    selected = []
    history = []
    while remaining:
        best = None
        for cand in remaining:
            auc, brier, ks = fit_eval(train, val, selected + [cand], numeric_all, train_cols)
            if best is None or auc > best[1]:
                best = (cand, auc, brier, ks)
        cand, auc, brier, ks = best
        selected.append(cand)
        remaining.remove(cand)
        history.append(dict(step=len(selected), action=f"+{cand}", n_features=len(selected),
                             auc=auc, brier=brier, ks=ks))
        print(f"[forward] Step {len(selected):2d}: +{cand:26s} AUC={auc:.4f} Brier={brier:.5f} "
              f"KS={ks:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return selected, history


def backward_elimination(train, val, candidates, numeric_all, train_cols, t0):
    selected = list(candidates)
    history = []
    step = 0
    # record the starting full-set point
    auc, brier, ks = fit_eval(train, val, selected, numeric_all, train_cols)
    history.append(dict(step=0, action="(full set)", n_features=len(selected), auc=auc, brier=brier, ks=ks))
    print(f"[backward] Step  0: (full set, n={len(selected)}) AUC={auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
    while len(selected) > 1:
        best = None
        for cand in selected:
            trial = [f for f in selected if f != cand]
            auc, brier, ks = fit_eval(train, val, trial, numeric_all, train_cols)
            if best is None or auc > best[1]:
                best = (cand, auc, brier, ks)
        cand, auc, brier, ks = best
        selected.remove(cand)
        step += 1
        history.append(dict(step=step, action=f"-{cand}", n_features=len(selected),
                             auc=auc, brier=brier, ks=ks))
        print(f"[backward] Step {step:2d}: -{cand:26s} AUC={auc:.4f} Brier={brier:.5f} "
              f"KS={ks:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return selected, history


def stepwise_selection(train, val, candidates, numeric_all, train_cols, t0):
    selected = []
    remaining = list(candidates)
    history = []
    seen = set()
    cur_auc, cur_brier, cur_ks = 0.5, 0.25, 0.0
    step = 0
    max_steps = 3 * len(candidates)
    while step < max_steps:
        step += 1
        add_best, remove_best = None, None
        for cand in remaining:
            auc, brier, ks = fit_eval(train, val, selected + [cand], numeric_all, train_cols)
            if add_best is None or auc > add_best[1]:
                add_best = (cand, auc, brier, ks)
        if len(selected) > 1:
            for cand in selected:
                trial = [f for f in selected if f != cand]
                auc, brier, ks = fit_eval(train, val, trial, numeric_all, train_cols)
                if remove_best is None or auc > remove_best[1]:
                    remove_best = (cand, auc, brier, ks)

        add_auc = add_best[1] if add_best else -1
        remove_auc = remove_best[1] if remove_best else -1

        if add_auc <= cur_auc and remove_auc <= cur_auc:
            print(f"[stepwise] Step {step:2d}: no move improves AUC beyond {cur_auc:.4f} -- stopping", flush=True)
            break

        if add_auc >= remove_auc:
            cand, cur_auc, cur_brier, cur_ks = add_best
            selected.append(cand)
            remaining.remove(cand)
            action = f"+{cand}"
        else:
            cand, cur_auc, cur_brier, cur_ks = remove_best
            selected.remove(cand)
            remaining.append(cand)
            action = f"-{cand}"

        key = frozenset(selected)
        if key in seen:
            print(f"[stepwise] Step {step:2d}: cycle detected -- stopping", flush=True)
            break
        seen.add(key)

        history.append(dict(step=step, action=action, n_features=len(selected),
                             auc=cur_auc, brier=cur_brier, ks=cur_ks))
        print(f"[stepwise] Step {step:2d}: {action:28s} AUC={cur_auc:.4f} Brier={cur_brier:.5f} "
              f"KS={cur_ks:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return selected, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["forward", "backward", "stepwise"])
    args = parser.parse_args()

    with open("images/qualified_features.json") as f:
        candidates = json.load(f)

    train = load("train")
    val = load("val")
    numeric_all, categorical_all = get_candidate_types(train, candidates)
    train_cols = set(train.columns)

    print(f"Running {args.method} selection on {len(candidates)} qualified candidates "
          f"({len(numeric_all)} numeric, {len(categorical_all)} categorical)", flush=True)

    t0 = time.time()
    fn = {"forward": forward_selection, "backward": backward_elimination, "stepwise": stepwise_selection}[args.method]
    selected, history = fn(train, val, candidates, numeric_all, train_cols, t0)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(f"images/wrapper_{args.method}_history.csv", index=False)

    if args.method == "backward":
        best_row = hist_df.loc[hist_df["auc"].idxmax()]
        n_at_best = int(best_row["n_features"])
        # reconstruct the selected set at that point: features NOT yet removed by that step
        removed_by_then = [h["action"][1:] for h in history if h["step"] <= best_row["step"] and h["action"].startswith("-")]
        final_selected = [c for c in candidates if c not in removed_by_then]
    elif args.method == "forward":
        best_row = hist_df.loc[hist_df["auc"].idxmax()]
        n_at_best = int(best_row["n_features"])
        final_selected = selected[:n_at_best]
    else:
        best_row = hist_df.loc[hist_df["auc"].idxmax()]
        final_selected = selected  # stepwise already stops near its best point

    print(f"\n[{args.method}] Best val AUC: {best_row['auc']:.4f} at step {int(best_row['step'])}")
    print(f"[{args.method}] Final selected features ({len(final_selected)}): {final_selected}")

    with open(f"images/wrapper_{args.method}_selected.json", "w") as f:
        json.dump({"method": args.method, "selected": final_selected,
                    "auc": float(best_row["auc"]), "brier": float(best_row["brier"]),
                    "ks": float(best_row["ks"])}, f, indent=2)

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
