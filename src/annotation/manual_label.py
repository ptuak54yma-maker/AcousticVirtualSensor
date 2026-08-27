"""
===============================================================================
Manual Frame-Level Annotation Tool (Matplotlib Interactive GUI)
===============================================================================

Provides an interactive GUI to visually verify and manually edit part-impact events
on the NMF activation envelope H_event_sum.

Interactive Controls:
- Left-Click: Add / Remove peak label at the selected frame.
- Key 's': Save current labels and close window.
- Key 'r': Reset to baseline reference peaks.
- Key 'c': Clear all labels.
"""

from typing import List, Optional, Set
import numpy as np
import matplotlib.pyplot as plt

import config
from src.audio.framing import frame_to_time


class InteractiveAnnotator:
    """
    Interactive Matplotlib GUI for frame-by-frame event labeling.
    """

    def __init__(
        self,
        audio_filename: str,
        h_event_sum: np.ndarray,
        reference_peaks: Optional[np.ndarray] = None,
        prominence_threshold: Optional[float] = config.PEAK_PROMINENCE,
        click_tolerance_frames: int = 4
    ):
        """
        Parameters
        ----------
        audio_filename : str
            Name of the audio file being annotated.
        h_event_sum : np.ndarray
            1D float32 array representing the impact activation envelope.
        reference_peaks : np.ndarray, optional
            Initial peak indices suggested by peak detection.
        prominence_threshold : float, optional
            Reference baseline threshold line.
        click_tolerance_frames : int
            Radius in frames within which a click will delete an existing marker.
        """
        self.filename = audio_filename
        self.h_event_sum = h_event_sum
        self.total_frames = len(h_event_sum)
        self.click_tolerance = click_tolerance_frames
        self.prominence_threshold = prominence_threshold

        # Store confirmed frame indices as a set
        self.initial_ref_peaks = set(reference_peaks) if reference_peaks is not None else set()
        self.confirmed_peaks: Set[int] = set(self.initial_ref_peaks)

        # Time axis for physical representation
        self.frame_indices = np.arange(self.total_frames)
        self.time_axis = frame_to_time(self.frame_indices)

        # Plot elements
        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        self.peak_lines: List[plt.Line2D] = []
        self.peak_points = None
        self.saved = False

        self._setup_plot()
        self._connect_events()

    def _setup_plot(self):
        """Configure Matplotlib layout and visual curves."""
        self.ax.plot(
            self.time_axis,
            self.h_event_sum,
            color="#1f77b4",
            linewidth=1.2,
            label=r"$H_{\mathrm{event\_sum}}(t)$"
        )

        if self.prominence_threshold is not None:
            self.ax.axhline(
                y=self.prominence_threshold,
                color="orange",
                linestyle="--",
                alpha=0.6,
                label=f"Prominence Baseline ({self.prominence_threshold})"
            )

        self.ax.set_title(
            f"Annotation Tool: {self.filename}\n"
            f"[Left-Click]: Add/Remove Label | [S]: Save & Next | [R]: Reset | [C]: Clear All",
            fontsize=12,
            fontweight="bold"
        )
        self.ax.set_xlabel("Time (seconds)", fontsize=11)
        self.ax.set_ylabel("Activation Energy", fontsize=11)
        self.ax.grid(True, linestyle=":", alpha=0.6)
        self.ax.set_xlim(self.time_axis[0], self.time_axis[-1])

        self._redraw_peaks()
        self.ax.legend(loc="upper right")

    def _redraw_peaks(self):
        """Redraw all peak markers according to self.confirmed_peaks."""
        # Remove old markers
        if self.peak_points is not None:
            self.peak_points.remove()
            self.peak_points = None

        if len(self.confirmed_peaks) > 0:
            peak_list = sorted(list(self.confirmed_peaks))
            times = self.time_axis[peak_list]
            amplitudes = self.h_event_sum[peak_list]

            self.peak_points = self.ax.scatter(
                times,
                amplitudes,
                color="crimson",
                s=60,
                zorder=5,
                label="Ground Truth Peaks"
            )

        current_count = len(self.confirmed_peaks)
        self.ax.set_title(
            f"Annotation: {self.filename} | Confirmed Count: {current_count}\n"
            f"[Left-Click]: Add/Remove | [S]: Save & Close | [R]: Reset | [C]: Clear",
            fontsize=11
        )
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax or event.button != 1:
            return

        click_time = event.xdata
        if click_time is None:
            return

        # Find nearest frame index
        clicked_frame = int(np.argmin(np.abs(self.time_axis - click_time)))

        # Check if clicking near an existing peak to delete
        existing_peak_to_remove = None
        for p in self.confirmed_peaks:
            if abs(p - clicked_frame) <= self.click_tolerance:
                existing_peak_to_remove = p
                break

        if existing_peak_to_remove is not None:
            self.confirmed_peaks.remove(existing_peak_to_remove)
        else:
            # Snap to local maximum in a local search window of +/- 3 frames
            search_start = max(0, clicked_frame - 3)
            search_end = min(self.total_frames, clicked_frame + 4)
            local_max_frame = search_start + int(np.argmax(self.h_event_sum[search_start:search_end]))
            self.confirmed_peaks.add(local_max_frame)

        self._redraw_peaks()

    def _on_key(self, event):
        """Handle keyboard shortcut commands."""
        key = event.key.lower() if event.key else ""

        if key == "s":
            self.saved = True
            plt.close(self.fig)
        elif key == "r":
            self.confirmed_peaks = set(self.initial_ref_peaks)
            self._redraw_peaks()
        elif key == "c":
            self.confirmed_peaks.clear()
            self._redraw_peaks()

    def _connect_events(self):
        """Connect interactive Matplotlib event handlers."""
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def show(self) -> np.ndarray:
        """
        Display GUI and block execution until closed.

        Returns
        -------
        peak_array : np.ndarray
            Sorted 1D array of confirmed ground truth peak frame indices.
        """
        plt.tight_layout()
        plt.show()
        return np.array(sorted(list(self.confirmed_peaks)), dtype=np.int64)