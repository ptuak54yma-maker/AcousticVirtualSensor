"""
===============================================================================
Binary Sequence to Event Grouping Module
===============================================================================

Converts continuous frame-wise probabilities/binary sequences into discrete,
individual feeding events to prevent over-counting on wide activation pulses.
"""

from typing import List, Dict, Any, Optional  # <-- Bổ sung Optional
import numpy as np
import config


def probability_to_binary(
    probabilities: np.ndarray,
    threshold: float = config.CRNN_THRESHOLD
) -> np.ndarray:
    """Threshold probability sequence into binary 0/1 array."""
    return (probabilities >= threshold).astype(np.uint8)


def group_binary_events(
    binary_seq: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    min_distance_frames: int = config.MIN_EVENT_DISTANCE_FRAMES,
    representative_method: str = config.EVENT_REPRESENTATIVE_METHOD
) -> List[Dict[str, Any]]:
    """
    Group consecutive 1-frames into contiguous events and select the representative peak frame.

    Parameters
    ----------
    binary_seq : np.ndarray
        1D binary array with values in {0, 1}.
    probabilities : np.ndarray, optional
        1D probability array used to find the maximum probability frame within an event.
    min_distance_frames : int
        Minimum gap in frames to separate two distinct events.
    representative_method : str
        Method to choose the event center ('maximum_probability', 'center', 'first').

    Returns
    -------
    events : list of dict
        List containing dictionaries for each detected event:
        {'event_id', 'start_frame', 'end_frame', 'representative_frame', 'peak_prob'}
    """
    events: List[Dict[str, Any]] = []
    in_event = False
    start_frame = 0

    total_frames = len(binary_seq)

    for t in range(total_frames):
        if binary_seq[t] == 1 and not in_event:
            in_event = True
            start_frame = t
        elif binary_seq[t] == 0 and in_event:
            in_event = False
            end_frame = t
            
            # Extract representative frame
            if representative_method == "maximum_probability" and probabilities is not None:
                rep_frame = start_frame + int(np.argmax(probabilities[start_frame:end_frame]))
                peak_p = float(probabilities[rep_frame])
            elif representative_method == "center":
                rep_frame = start_frame + (end_frame - start_frame) // 2
                peak_p = float(probabilities[rep_frame]) if probabilities is not None else 1.0
            else:
                rep_frame = start_frame
                peak_p = float(probabilities[rep_frame]) if probabilities is not None else 1.0

            events.append({
                "event_id": len(events) + 1,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "representative_frame": rep_frame,
                "peak_prob": peak_p
            })

    # Close trailing event if active at the end
    if in_event:
        end_frame = total_frames
        rep_frame = start_frame + (end_frame - start_frame) // 2
        events.append({
            "event_id": len(events) + 1,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "representative_frame": rep_frame,
            "peak_prob": 1.0
        })

    # Post-filtering by minimum event distance
    if min_distance_frames > 0 and len(events) > 1:
        filtered_events = [events[0]]
        for curr in events[1:]:
            prev = filtered_events[-1]
            if (curr["representative_frame"] - prev["representative_frame"]) >= min_distance_frames:
                filtered_events.append(curr)
            else:
                # Merge: Keep the one with higher probability
                if curr.get("peak_prob", 0) > prev.get("peak_prob", 0):
                    filtered_events[-1] = curr
        events = filtered_events

    return events