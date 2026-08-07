"""Render images/all_model_results.json as a clean visual comparison table
across every model fit in this round: the original curated-feature LR/Tree,
the Brier-selected and Forward-AUC-selected LRs, and the tree ensembles.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK = "#1F2937"
MUTED = "#6B7280"
HEADER_BG = "#1F2937"
BEST_COLOR = "#2563EB"
ROW_BAND = "#F3F4F6"
GRID = "#E5E7EB"

COLUMNS = [
    ("n_features", "# Feat", "{:d}", False),
    ("auc", "AUC", "{:.3f}", True),
    ("gini", "Gini", "{:.3f}", True),
    ("brier", "Brier ↓", "{:.4f}", False),
    ("ks", "KS", "{:.3f}", True),
    ("precision", "Prec@20%", "{:.3f}", True),
    ("recall", "Recall@20%", "{:.3f}", True),
    ("f1", "F1@20%", "{:.3f}", True),
    ("specificity", "Spec@20%", "{:.3f}", True),
    ("sensitivity_at_95_specificity", "Sens@95%Spec", "{:.3f}", True),
]

DISPLAY_NAMES = {
    "Logistic Regression": "LR (curated, 20 feat.)",
    "Decision Tree": "Decision Tree",
    "Logistic Regression (Brier-selected)": "LR (Brier-selected)",
    "LR (Forward-selected)": "LR (Forward-selected, AUC)",
    "Random Forest": "Random Forest",
    "XGBoost": "XGBoost",
}


def main():
    with open("images/all_model_results.json") as f:
        results = json.load(f)

    order = [n for n in DISPLAY_NAMES if n in results]

    best_per_col = {}
    for key, _, _, higher_is_better in COLUMNS:
        vals = {name: results[name][key] for name in order if key in results[name] and results[name][key] is not None}
        if not vals:
            continue
        best_per_col[key] = (max if higher_is_better else min)(vals, key=vals.get)

    NAME_COL_WIDTH = 3.0
    n_rows = len(order) + 1
    n_cols = len(COLUMNS) + NAME_COL_WIDTH
    fig, ax = plt.subplots(figsize=(1.35 * n_cols, 0.9 * n_rows))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.axis("off")
    ax.invert_yaxis()

    ax.add_patch(plt.Rectangle((0, 0), n_cols, 1, color=HEADER_BG, zorder=1))
    ax.text(0.15, 0.5, "Model", va="center", ha="left", color="white",
             fontweight="bold", fontsize=11, zorder=2)
    for j, (_, label, _, _) in enumerate(COLUMNS):
        ax.text(j + NAME_COL_WIDTH + 0.85, 0.5, label, va="center", ha="right", color="white",
                 fontweight="bold", fontsize=9.5, zorder=2)

    for i, name in enumerate(order):
        row_y = i + 1
        display_name = DISPLAY_NAMES[name]
        if i % 2 == 1:
            ax.add_patch(plt.Rectangle((0, row_y), n_cols, 1, color=ROW_BAND, zorder=1))
        ax.text(0.15, row_y + 0.5, display_name, va="center", ha="left", color=INK,
                 fontsize=10, fontweight="normal", zorder=2)
        for j, (key, _, fmt, _) in enumerate(COLUMNS):
            val = results[name].get(key)
            text = fmt.format(val) if val is not None else "—"
            is_best = best_per_col.get(key) == name
            ax.text(j + NAME_COL_WIDTH + 0.85, row_y + 0.5, text, va="center", ha="right",
                     color=BEST_COLOR if is_best else INK,
                     fontweight="bold" if is_best else "normal", fontsize=10, zorder=2)

    for y in range(n_rows + 1):
        ax.plot([0, n_cols], [y, y], color=GRID, linewidth=0.8, zorder=0)

    ax.set_title("Model comparison (validation set)", fontsize=14, fontweight="bold",
                  loc="left", pad=14, color=INK)
    fig.tight_layout()
    fig.savefig("images/model_results_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Wrote images/model_results_table.png")


if __name__ == "__main__":
    main()
