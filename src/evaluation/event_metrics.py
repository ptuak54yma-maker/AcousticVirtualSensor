"""
===============================================================================
Event-Level Evaluation Metrics Module
===============================================================================
"""

from typing import Dict, List, Union
import numpy as np
import config


def compute_event_metrics(
    true_event_frames: Union[List[int], np.ndarray],
    pred_event_frames: Union[List[int], np.ndarray],
    tolerance_frames: int = config.EVENT_MATCH_TOLERANCE_FRAMES
) -> Dict[str, float]:
    """
    Match predicted events against true event frames within a tolerance window.
    """
    true_frames = list(sorted(true_event_frames))
    pred_frames = list(sorted(pred_event_frames))

    matched_true = set()
    tp = 0
    fp = 0

    for pf in pred_frames:
        match_found = False
        for tf in true_frames:
            if tf not in matched_true and abs(pf - tf) <= tolerance_frames:
                matched_true.add(tf)
                tp += 1
                match_found = True
                break
        if not match_found:
            fp += 1

    fn = len(true_frames) - len(matched_true)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-8, precision + recall)

    return {
        "event_tp": tp,
        "event_fp": fp,
        "event_fn": fn,
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "tolerance_frames": tolerance_frames
    }