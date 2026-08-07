"""The classic train-vs-validation-per-epoch curve (as in Chollet's Deep
Learning with Python -- e.g. the IMDB/Reuters chapters): train loss/AUC
keeps improving indefinitely while validation AUC improves, plateaus, then
degrades as the network starts memorizing training-set noise instead of
learning generalizable signal. The epoch where val AUC peaks -- not where
train AUC is best, and not a fixed max_iter guess -- is the "pull back"
point: where we'd stop training (or restore weights from) if deploying
this architecture.

Uses warm_start=True and calls .fit() repeatedly, one epoch (max_iter=1)
at a time, evaluating REAL val.csv after every epoch -- unlike sklearn's
built-in early_stopping=True, which decides using a slice carved out of
train, not the actual held-out validation set this project has been
scoring every other model against.

Deliberately uses weak regularization (alpha=1e-5) so overfitting is
visible within a reasonable number of epochs -- the point is to SEE the
degradation, not to avoid it by regularizing it away before the chart
exists.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from full_feature_matrix import build_full_matrix, load

HUE_TRAIN = "#93C5FD"
HUE_VAL = "#2563EB"
ACCENT = "#DC2626"
INK = "#1F2937"
MUTED = "#6B7280"
GRID = "#E5E7EB"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

N_EPOCHS = 120


def main():
    train = load("train")
    val = load("val")
    X_train, y_train, X_val, y_val, cols = build_full_matrix(train, val)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xv = scaler.transform(X_val)

    # Weak regularization on purpose -- we want to SEE overfitting happen
    # over epochs, not prevent it before the chart exists.
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), alpha=1e-5, max_iter=1,
                         warm_start=True, solver="adam", random_state=42)

    train_aucs, val_aucs = [], []
    for epoch in range(1, N_EPOCHS + 1):
        mlp.fit(Xtr, y_train)
        p_train = mlp.predict_proba(Xtr)[:, 1]
        p_val = mlp.predict_proba(Xv)[:, 1]
        auc_train = roc_auc_score(y_train, p_train)
        auc_val = roc_auc_score(y_val, p_val)
        train_aucs.append(auc_train)
        val_aucs.append(auc_val)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: train AUC={auc_train:.4f}  val AUC={auc_val:.4f}", flush=True)

    val_aucs = np.array(val_aucs)
    train_aucs = np.array(train_aucs)
    peak_epoch = int(np.argmax(val_aucs)) + 1
    peak_val_auc = val_aucs[peak_epoch - 1]
    final_train_auc = train_aucs[-1]
    final_val_auc = val_aucs[-1]

    print(f"\nPeak val AUC: {peak_val_auc:.4f} at epoch {peak_epoch}")
    print(f"By epoch {N_EPOCHS}: train AUC={final_train_auc:.4f}, val AUC={final_val_auc:.4f} "
          f"(degraded {peak_val_auc - final_val_auc:+.4f} from peak while train kept climbing "
          f"{final_train_auc - train_aucs[peak_epoch - 1]:+.4f})")

    with open("images/learning_curve.json", "w") as f:
        json.dump(dict(train_aucs=train_aucs.tolist(), val_aucs=val_aucs.tolist(),
                        peak_epoch=peak_epoch, peak_val_auc=float(peak_val_auc)), f, indent=2)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    epochs = np.arange(1, N_EPOCHS + 1)
    ax.plot(epochs, train_aucs, color=HUE_TRAIN, linewidth=2, label="Train AUC")
    ax.plot(epochs, val_aucs, color=HUE_VAL, linewidth=2, label="Val AUC")
    ax.axvline(peak_epoch, color=ACCENT, linewidth=1.5, linestyle="--")
    ax.scatter([peak_epoch], [peak_val_auc], color=ACCENT, s=100, zorder=5,
               label=f"Pull back here: epoch {peak_epoch} (val AUC={peak_val_auc:.4f})")
    ax.annotate("train keeps climbing\n(memorizing noise)", xy=(N_EPOCHS, final_train_auc),
                xytext=(-140, 10), textcoords="offset points", fontsize=9, color=MUTED)
    ax.annotate("val degrades past the peak\n(generalization getting worse)",
                xy=(N_EPOCHS, final_val_auc), xytext=(-190, -30), textcoords="offset points",
                fontsize=9, color=MUTED)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Where to pull back: train vs. validation AUC by epoch", fontsize=13, fontweight="bold")
    ax.legend(loc="center right", frameon=False)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("images/learning_curve.png", dpi=150)
    plt.close(fig)

    print("\nWrote images/learning_curve.png, images/learning_curve.json")


if __name__ == "__main__":
    main()
