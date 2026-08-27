"""
Annotation Package
"""
from .peak_reference import detect_reference_peaks
from .annotation_io import create_gaussian_label, save_annotation, load_annotation
from .manual_label import InteractiveAnnotator

__all__ = [
    "detect_reference_peaks",
    "create_gaussian_label",
    "save_annotation",
    "load_annotation",
    "InteractiveAnnotator"
]