"""
Event Processing & Counting Package
"""
from .peak_detection import detect_peaks_envelope
from .binary_to_event import probability_to_binary, group_binary_events
from .counting import count_parts_from_events, compute_feed_rate

__all__ = [
    "detect_peaks_envelope",
    "probability_to_binary",
    "group_binary_events",
    "count_parts_from_events",
    "compute_feed_rate"
]