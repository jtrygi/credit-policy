"""Shared metrics: KS statistic and specificity-based measures, used across
selection scripts and the results table. Kept separate from train_models.py
so every script (linear, tree, ensemble) can import without pulling in the
curated NUMERIC_FEATURES/CATEGORICAL_FEATURES constants.
"""
import numpy as np
from sklearn.metrics import roc_curve


def ks_statistic(y_true, y_score):
    """Kolmogorov-Smirnov statistic: max separation between the cumulative
    good and cumulative bad distributions across the score. Standard
    credit-scoring separation metric -- closely related to AUC (KS is the
    max vertical gap on the ROC curve, i.e. max(TPR - FPR)) but reported in
    the industry as its own number because it identifies the single
    threshold where good/bad separation is starkest.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ks = np.max(tpr - fpr)
    ks_threshold = thresholds[np.argmax(tpr - fpr)]
    return ks, ks_threshold


def specificity_at_threshold(y_true, y_score, threshold):
    """True negative rate at a given score threshold (predict positive/bad
    when y_score >= threshold)."""
    y_true = np.asarray(y_true)
    pred_bad = (y_score >= threshold).astype(int)
    tn = np.sum((pred_bad == 0) & (y_true == 0))
    fp = np.sum((pred_bad == 1) & (y_true == 0))
    return tn / (tn + fp)


def sensitivity_at_specificity(y_true, y_score, target_specificity=0.95):
    """Recall (sensitivity) achieved at the threshold where specificity is
    closest to (>=) the target. Answers: "if we insist on correctly
    clearing 95% of good customers, what fraction of defaults do we still
    catch?" -- encodes a 'protect good customers first' priority without
    the degenerate approve-everyone solution of optimizing specificity alone.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificity = 1 - fpr
    # thresholds are in descending order; find points meeting the target
    valid = specificity >= target_specificity
    if not valid.any():
        return 0.0, None
    idx = np.where(valid)[0]
    best_idx = idx[np.argmax(tpr[idx])]
    return tpr[best_idx], thresholds[best_idx]
