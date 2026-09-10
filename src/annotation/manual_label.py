"""
===============================================================================
Manual Frame-Level Annotation Tool (Matplotlib Interactive GUI)
===============================================================================

Provides an interactive GUI with audio playback, real-time vertical cursor tracking,
and Tri-Count monitoring (Peak Detection vs Ground Truth vs Confirmed).

Interactive Controls:
- Left-Click: Add / Remove peak label at the selected frame.
- Key 's': Save current labels and close window.
- Key 'r': Reset to baseline reference peaks (P*, D*).
- Key 'c': Clear all labels.
- Key 'Space': Toggle Play / Pause audio.
- Button [Play]: Start / Resume audio playback with 60 FPS smooth cursor tracking.
- Button [Pause]: Pause audio playback (cursor remains at current timestamp).
- Button [Stop]: Stop playback and reset cursor to t = 0.0s (cursor stays visible).
"""

from typing import List, Optional, Set
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.animation as animation

import config
from src.audio.framing import frame_to_time

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


class InteractiveAnnotator:
    """
    Interactive Matplotlib GUI with Audio Playback and Tri-Count Monitoring.
    Optimized for high-FPS, continuous visible cursor tracking.
    """

    def __init__(
        self,
        audio_filename: str,
        h_event_sum: np.ndarray,
        reference_peaks: Optional[np.ndarray] = None,
        prominence_threshold: Optional[float] = config.DEFAULT_PEAK_PROMINENCE,
        ground_truth_count: Optional[int] = None,
        audio_samples: Optional[np.ndarray] = None,
        sample_rate: int = config.SAMPLE_RATE,
        click_tolerance_frames: int = config.ANNOTATION_CLICK_TOLERANCE_FRAMES,
        search_radius_frames: int = config.ANNOTATION_LOCAL_SEARCH_RADIUS_FRAMES
    ):
        self.filename = audio_filename
        self.h_event_sum = h_event_sum
        self.total_frames = len(h_event_sum)
        self.click_tolerance = click_tolerance_frames
        self.search_radius = search_radius_frames
        self.prominence_threshold = prominence_threshold
        self.ground_truth_count = ground_truth_count
        self.audio_samples = audio_samples
        self.sample_rate = sample_rate

        self.initial_ref_peaks = set(reference_peaks) if reference_peaks is not None else set()
        self.peak_detect_count = len(self.initial_ref_peaks)
        self.confirmed_peaks: Set[int] = set(self.initial_ref_peaks)

        self.frame_indices = np.arange(self.total_frames)
        self.time_axis = frame_to_time(self.frame_indices)
        self.max_time = float(self.time_axis[-1]) if len(self.time_axis) > 0 else 0.0

        self.is_playing = False
        self.play_start_time = 0.0
        self.current_cursor_time = 0.0

        self.fig, self.ax = plt.subplots(figsize=(15, 7))
        plt.subplots_adjust(bottom=0.20, top=0.88)

        self.peak_points = None
        self.cursor_line = None
        self.anim = None

        self._setup_plot()
        self._setup_buttons()
        self._connect_events()

    def _setup_plot(self):
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
                label=f"Prominence P* ({self.prominence_threshold:.2f})"
            )

        self.cursor_line = self.ax.axvline(
            x=self.current_cursor_time,
            color="#2ca02c",
            linestyle="-",
            linewidth=1.8,
            alpha=0.9,
            animated=True,
            label="Playback Cursor"
        )

        self.ax.set_xlabel("Time (seconds)", fontsize=11)
        self.ax.set_ylabel("Activation Energy", fontsize=11)
        self.ax.grid(True, linestyle=":", alpha=0.6)
        self.ax.set_xlim(self.time_axis[0], self.time_axis[-1])

        self._update_title()
        self._redraw_peaks()
        self.ax.legend(loc="upper right")

    def _setup_buttons(self):
        ax_play = plt.axes([0.15, 0.05, 0.10, 0.06])
        ax_pause = plt.axes([0.27, 0.05, 0.10, 0.06])
        ax_stop = plt.axes([0.39, 0.05, 0.10, 0.06])

        self.btn_play = Button(ax_play, "▶ Play", color="#d4edda", hovercolor="#c3e6cb")
        self.btn_pause = Button(ax_pause, "⏸ Pause", color="#fff3cd", hovercolor="#ffeeba")
        self.btn_stop = Button(ax_stop, "■ Stop", color="#f8d7da", hovercolor="#f5c6cb")

        self.btn_play.on_clicked(self._on_play)
        self.btn_pause.on_clicked(self._on_pause)
        self.btn_stop.on_clicked(self._on_stop)

    def _update_title(self):
        gt_str = str(self.ground_truth_count) if self.ground_truth_count is not None else "N/A"
        cur_count = len(self.confirmed_peaks)

        title_text = (
            f"File: {self.filename}  |  "
            f"[Peak Detect (P*, D*): {self.peak_detect_count}]   "
            f"[Ground Truth: {gt_str}]   "
            f"[Confirmed: {cur_count}]\n"
            f"[Left-Click]: Add/Remove  |  [Space]: Play/Pause  |  [S]: Save & Next  |  [R]: Reset  |  [C]: Clear"
        )
        self.ax.set_title(title_text, fontsize=11, fontweight="bold")
        self.fig.canvas.draw_idle()

    def _redraw_peaks(self):
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
                label="Confirmed Peaks"
            )

        self._update_title()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return

        click_time = event.xdata
        if click_time is None:
            return

        clicked_frame = int(np.argmin(np.abs(self.time_axis - click_time)))

        existing_peak_to_remove = None
        for p in self.confirmed_peaks:
            if abs(p - clicked_frame) <= self.click_tolerance:
                existing_peak_to_remove = p
                break

        if existing_peak_to_remove is not None:
            self.confirmed_peaks.remove(existing_peak_to_remove)
        else:
            search_start = max(0, clicked_frame - self.search_radius)
            search_end = min(self.total_frames, clicked_frame + self.search_radius + 1)
            local_max_frame = search_start + int(np.argmax(self.h_event_sum[search_start:search_end]))
            self.confirmed_peaks.add(local_max_frame)

        self._redraw_peaks()

    def _on_key(self, event):
        key = event.key.lower() if event.key else ""

        if key == "s":
            self._on_stop(None)
            plt.close(self.fig)
        elif key == "r":
            self.confirmed_peaks = set(self.initial_ref_peaks)
            self._redraw_peaks()
        elif key == "c":
            self.confirmed_peaks.clear()
            self._redraw_peaks()
        elif key == " ":
            if self.is_playing:
                self._on_pause(None)
            else:
                self._on_play(None)

    def _on_play(self, _):
        if not HAS_SOUNDDEVICE or self.audio_samples is None:
            print("[!] Cần cài đặt sounddevice để phát audio: pip install sounddevice")
            return

        if self.is_playing:
            return

        if self.current_cursor_time >= self.max_time:
            self.current_cursor_time = 0.0

        start_sample = int(self.current_cursor_time * self.sample_rate)
        start_sample = min(start_sample, len(self.audio_samples) - 1)

        sd.stop()
        sd.play(self.audio_samples[start_sample:], self.sample_rate)
        self.play_start_time = time.time() - self.current_cursor_time
        self.is_playing = True

    def _on_pause(self, _):
        if self.is_playing:
            if HAS_SOUNDDEVICE:
                sd.stop()
            self.current_cursor_time = time.time() - self.play_start_time
            self.is_playing = False
            self.cursor_line.set_xdata([self.current_cursor_time, self.current_cursor_time])
            self.fig.canvas.draw_idle()

    def _on_stop(self, _):
        if HAS_SOUNDDEVICE:
            sd.stop()
        self.is_playing = False
        self.current_cursor_time = 0.0
        self.cursor_line.set_xdata([0.0, 0.0])
        self.fig.canvas.draw_idle()

    def _init_cursor_anim(self):
        self.cursor_line.set_xdata([self.current_cursor_time, self.current_cursor_time])
        return (self.cursor_line,)

    def _update_cursor(self, _):
        if self.is_playing:
            elapsed = time.time() - self.play_start_time
            if elapsed >= self.max_time:
                self._on_stop(None)
            else:
                self.current_cursor_time = elapsed

        self.cursor_line.set_xdata([self.current_cursor_time, self.current_cursor_time])
        return (self.cursor_line,)

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.anim = animation.FuncAnimation(
            self.fig,
            self._update_cursor,
            init_func=self._init_cursor_anim,
            interval=15,
            blit=True,
            cache_frame_data=False
        )

    def show(self) -> np.ndarray:
        plt.show()
        if HAS_SOUNDDEVICE:
            sd.stop()
        return np.array(sorted(list(self.confirmed_peaks)), dtype=np.int64)