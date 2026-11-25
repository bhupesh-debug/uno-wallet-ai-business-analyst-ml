"""Evaluation helpers — small, pure, unit-tested functions."""
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def auc(y_true, y_score):
    """ROC-AUC; returns 0.5 when only one class is present."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def mape(y_true, y_pred):
    """Mean absolute percentage error, %; zero targets excluded."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])) * 100)


def precision_recall_at_k(y_true, y_score, k=0.20):
    """Precision/recall among the top-k scored rows (churn triage)."""
    order = np.argsort(-np.asarray(y_score))
    n = max(1, int(len(order) * k))
    top = order[:n]
    tp = int(np.asarray(y_true)[top].sum())
    precision = tp / n
    recall = tp / max(int(np.asarray(y_true).sum()), 1)
    return float(precision), float(recall)


def lift_vs_naive(actual_points, optimal_points):
    """Extra points from optimal routing, as % of actual points."""
    return float(optimal_points / max(actual_points, 1) * 100 - 100)
