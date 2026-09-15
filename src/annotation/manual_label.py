"""
===============================================================================
Manual Frame-Level Annotation Tool (Matplotlib Interactive GUI)
===============================================================================

Provides a dedicated annotation GUI separating visual references from user labels:
- Reference Peaks (P*, D*): Visual guide only (non-interactive, immutable).
- Confirmed Peaks: User ground truth (interactive, editable).
- Audio Playback & Seeking: Synchronized using monotonic timeline and frame snap.

Interaction Controls:
- Left-Click: Seek playback position (snaps to nearest frame; does NOT edit peaks).
- Right-Click: Add / Remove User Peak (snaps to local maximum in search radius).
- Key 's': Save current labels and close window.
- Key 'r': Reset confirmed peaks to initial reference proposal.
- Key 'c': Clear all confirmed user peaks (reference markers remain visible).
- Key 'Space': Toggle Play / Pause audio playback.
- Button [Play]: Start / Resume playback from current cursor position.
- Button [Pause]: Pause playback (cursor stays at current timestamp).
- Button [Stop]: Stop playback and reset cursor to t = 0.0s.
"""

from pathlib import Path
import sys
from typing import Optional, Set
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.animation as animation

# Allow this module to be imported/run from the repository without relying on
# the current working directory to locate the root-level config.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.audio.framing import frame_to_time, time_to_frame

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


class InteractiveAnnotator:
    """
    Interactive Matplotlib GUI with separated Reference vs User annotation layers,
    audio seeking via Left-Click, and peak editing via Right-Click.
    """

    def __init__(
        self,
        audio_filename: str,
        h_event_sum: np.ndarray,
        reference_peaks: Optional[np.ndarray] = None,
        prominence_threshold: Optional[float] = getattr(config, "DEFAULT_PEAK_PROMINENCE", 0.5),
        ground_truth_count: Optional[int] = None,
        audio_samples: Optional[np.ndarray] = None,
        sample_rate: int = config.SAMPLE_RATE,
        click_tolerance_frames: int = getattr(config, "ANNOTATION_CLICK_TOLERANCE_FRAMES", 4),
        search_radius_frames: int = getattr(config, "ANNOTATION_LOCAL_SEARCH_RADIUS_FRAMES", 3),
        sync_offset: float = getattr(config, "ANNOTATION_AUDIO_SYNC_OFFSET", 0.0)
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
        self.sync_offset = sync_offset

        # Reference peaks are immutable visual guides. Confirmed peaks are the
        # editable copy used as the final annotation output.
        self.reference_peaks: Set[int] = set(reference_peaks) if reference_peaks is not None else set()
        self.peak_detect_count = len(self.reference_peaks)
        self.confirmed_peaks: Set[int] = set(self.reference_peaks)

        # Normalized time axis on the STFT frame grid.
        self.frame_indices = np.arange(self.total_frames)
        self.time_axis = frame_to_time(
            self.frame_indices,
            hop_length=config.HOP_LENGTH,
            sr=self.sample_rate,
        )
        self.max_time = float(self.time_axis[-1]) if len(self.time_axis) > 0 else 0.0

        # Playback state uses a monotonic clock and keeps the audio start time
        # separate from the displayed cursor time.
        self.is_playing = False
        self.audio_start_time = 0.0
        self.playback_start_time = 0.0
        self.current_cursor_time = 0.0

        self.fig, self.ax = plt.subplots(figsize=(15, 7))
        plt.subplots_adjust(bottom=0.20, top=0.88)

        self.ref_points = None
        self.confirmed_points = None
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
                alpha=0.5,
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

        self._draw_reference_peaks()
        self._redraw_confirmed_peaks()
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
            f"[Ref Detect (P*, D*): {self.peak_detect_count}]   "
            f"[Ground Truth: {gt_str}]   "
            f"[Confirmed: {cur_count}]\n"
            f"[Left-Click]: Seek  |  [Right-Click]: Add/Remove Peak  |  "
            f"[Space]: Play/Pause  |  [S]: Save  |  [R]: Reset  |  [C]: Clear"
        )
        self.ax.set_title(title_text, fontsize=11, fontweight="bold")
        self.fig.canvas.draw_idle()

    def _draw_reference_peaks(self):
        """Draw immutable P*, D* reference peaks as a separate visual layer."""
        if len(self.reference_peaks) > 0:
            ref_list = sorted(self.reference_peaks)
            times = self.time_axis[ref_list]
            amps = self.h_event_sum[ref_list]

            self.ref_points = self.ax.scatter(
                times,
                amps,
                color="#7f7f7f",
                marker="^",
                s=45,
                alpha=0.6,
                zorder=4,
                label=r"Ref Peaks ($P^*, D^*$)"
            )

    def _redraw_confirmed_peaks(self):
        """Draw editable user-confirmed peaks as prominent markers."""
        if self.confirmed_points is not None:
            self.confirmed_points.remove()
            self.confirmed_points = None

        if len(self.confirmed_peaks) > 0:
            peak_list = sorted(self.confirmed_peaks)
            times = self.time_axis[peak_list]
            amps = self.h_event_sum[peak_list]

            self.confirmed_points = self.ax.scatter(
                times,
                amps,
                color="crimson",
                marker="o",
                s=70,
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
                label="Confirmed Peaks"
            )

        self._update_title()

    def _on_mouse_down(self, event):
        if event.inaxes != self.ax:
            return

        click_time = event.xdata
        if click_time is None or self.total_frames == 0:
            return

        snapped_frame = int(
            time_to_frame(
                click_time,
                hop_length=config.HOP_LENGTH,
                sr=self.sample_rate,
            )
        )
        snapped_frame = int(np.clip(snapped_frame, 0, self.total_frames - 1))
        snapped_time = float(self.time_axis[snapped_frame])

        # LEFT CLICK: seek audio/cursor only.
        if event.button == 1:
            self._seek_to(snapped_time)

        # RIGHT CLICK: add/remove only user-confirmed peaks.
        elif event.button == 3:
            existing_peak_to_remove = None
            for p in self.confirmed_peaks:
                if abs(p - snapped_frame) <= self.click_tolerance:
                    existing_peak_to_remove = p
                    break

            if existing_peak_to_remove is not None:
                self.confirmed_peaks.remove(existing_peak_to_remove)
            else:
                start = max(0, snapped_frame - self.search_radius)
                end = min(self.total_frames, snapped_frame + self.search_radius + 1)
                local_max = start + int(np.argmax(self.h_event_sum[start:end]))
                self.confirmed_peaks.add(local_max)

            self._redraw_confirmed_peaks()

    def _seek_to(self, target_time: float):
        """Move cursor immediately and seek audio to the same timestamp."""
        was_playing = self.is_playing
        if was_playing and HAS_SOUNDDEVICE:
            sd.stop()

        self.current_cursor_time = target_time
        self.audio_start_time = target_time
        self.cursor_line.set_xdata([self.current_cursor_time, self.current_cursor_time])
        self.fig.canvas.draw_idle()

        if was_playing:
            self._start_audio_stream()

    def _start_audio_stream(self):
        if not HAS_SOUNDDEVICE or self.audio_samples is None or len(self.audio_samples) == 0:
            return

        start_sample = int(self.audio_start_time * self.sample_rate)
        start_sample = int(np.clip(start_sample, 0, len(self.audio_samples) - 1))

        sd.stop()
        sd.play(self.audio_samples[start_sample:], self.sample_rate)
        self.playback_start_time = time.monotonic()
        self.is_playing = True

    def _on_play(self, _):
        if not HAS_SOUNDDEVICE or self.audio_samples is None:
            print("[!] Cần thư viện sounddevice và audio_samples để phát âm thanh.")
            return

        if self.is_playing:
            return

        if self.current_cursor_time >= self.max_time:
            self.current_cursor_time = 0.0
            self.audio_start_time = 0.0

        self.audio_start_time = self.current_cursor_time
        self._start_audio_stream()

    def _on_pause(self, _):
        if self.is_playing:
            if HAS_SOUNDDEVICE:
                sd.stop()

            elapsed = time.monotonic() - self.playback_start_time
            self.current_cursor_time = min(
                self.audio_start_time + elapsed + self.sync_offset,
                self.max_time,
            )
            self.audio_start_time = max(
                0.0,
                self.current_cursor_time - self.sync_offset,
            )
            self.is_playing = False
            self.cursor_line.set_xdata([self.current_cursor_time, self.current_cursor_time])
            self.fig.canvas.draw_idle()

    def _on_stop(self, _):
        if HAS_SOUNDDEVICE:
            sd.stop()
        self.is_playing = False
        self.current_cursor_time = 0.0
        self.audio_start_time = 0.0
        self.cursor_line.set_xdata([0.0, 0.0])
        self.fig.canvas.draw_idle()

    def _init_cursor_anim(self):
        self.cursor_line.set_xdata([self.current_cursor_time, self.current_cursor_time])
        return (self.cursor_line,)

    def _update_cursor(self, _):
        if self.is_playing:
            elapsed = time.monotonic() - self.playback_start_time
            current = self.audio_start_time + elapsed + self.sync_offset
            if current >= self.max_time:
                self._on_stop(None)
            else:
                self.current_cursor_time = current

        self.cursor_line.set_xdata([self.current_cursor_time, self.current_cursor_time])
        return (self.cursor_line,)

    def _on_key(self, event):
        key = event.key.lower() if event.key else ""

        if key == "s":
            self._on_stop(None)
            plt.close(self.fig)
        elif key == "r":
            self.confirmed_peaks = set(self.reference_peaks)
            self._redraw_confirmed_peaks()
        elif key == "c":
            self.confirmed_peaks.clear()
            self._redraw_confirmed_peaks()
        elif key == " ":
            if self.is_playing:
                self._on_pause(None)
            else:
                self._on_play(None)

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse_down)
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
