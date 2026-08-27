"""
===============================================================================
Counting-Level Evaluation Metrics Module
===============================================================================
"""

from typing import Dict, List, Union
import numpy as np


def compute_count_metrics(
    true_counts: Union[int, List[int], np.ndarray],
    pred_counts: Union[int, List[int], np.ndarray]
) -> Dict[str, float]:
    """
    Calculate paper-standard counting metrics: MAE, MAPE, and Accuracy.
    Formula from paper: Acc = (1 - |N_pred - N_true| / N_true) * 100%
    """
    t_counts = np.atleast_1d(true_counts).astype(float)
    p_counts = np.atleast_1d(pred_counts).astype(float)

    abs_errors = np.abs(p_counts - t_counts)
    mae = float(np.mean(abs_errors))
    mape = float(np.mean(abs_errors / np.maximum(1.0, t_counts))) * 100.0
    
    accuracies = np.maximum(0.0, 1.0 - (abs_errors / np.maximum(1.0, t_counts))) * 100.0
    mean_accuracy = float(np.mean(accuracies))

    return {
        "mae": mae,
        "mape": mape,
        "mean_accuracy": mean_accuracy,
        "total_true": int(np.sum(t_counts)),
        "total_pred": int(np.sum(p_counts))
    }