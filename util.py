import numpy as np
from sklearn.metrics import roc_curve

def compute_eer(labels, scores):
    """
    labels: [N] (0 or 1)
    scores: [N] (probabilities)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr

    # find point where FPR ≈ FNR
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_idx]

    return eer * 100