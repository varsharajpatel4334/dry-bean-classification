"""Shared helper: one function that computes all 6 required metrics."""
import numpy as np
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)          # probabilities are needed for AUC
    n_classes = len(np.unique(y))

    if n_classes == 2:                         # binary problem
        auc = roc_auc_score(y, y_proba[:, 1])
        avg = "binary"
    else:                                      # multi-class problem
        auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")
        avg = "weighted"

    return {
        "Accuracy":  accuracy_score(y, y_pred),
        "AUC":       auc,
        "Precision": precision_score(y, y_pred, average=avg, zero_division=0),
        "Recall":    recall_score(y, y_pred, average=avg, zero_division=0),
        "F1":        f1_score(y, y_pred, average=avg, zero_division=0),
        "MCC":       matthews_corrcoef(y, y_pred),
    }
