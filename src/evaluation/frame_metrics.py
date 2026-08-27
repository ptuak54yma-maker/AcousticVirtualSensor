"""
===============================================================================
Frame-Level Evaluation Metrics Module
===============================================================================
"""

from typing import Dict
import numpy as np


def compute_frame_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate Precision, Recall, F1, and Accuracy at the frame level."""
    y_t = (np.asarray(y_true) > 0.5).astype(int)
    y_p = (np.asarray(y_pred) > 0.5).astype(int)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-8, precision + recall)
    acc = (tp + tn) / max(1, len(y_t))

    return {
        "frame_precision": float(precision),
        "frame_recall": float(recall),
        "frame_f1": float(f1),
        "frame_accuracy": float(acc)
    }