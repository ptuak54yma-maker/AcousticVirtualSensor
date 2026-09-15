"""
===============================================================================
Manual Frame-Level Annotation Tool (Matplotlib Interactive GUI)
===============================================================================

Provides a dedicated annotation GUI separating visual references from user labels:
- Reference Peaks (P*, D*): Visual guide (grey triangles) - immutable, non-interactive.
- Confirmed Peaks: User ground truth (red circles) - dynamic blit, toggles instantly.
- Audio Playback: Frame-synchronized cursor tracking with hardware latency calibration.

Interaction Controls:
- Left-Click: Seek audio playback position (snaps to frame; does NOT edit peaks).
- Right-Click: Toggle Peak (Add / Remove) - shows or hides red circle INSTANTLY.
  * Click close to an existing red circle (<= 2 frames) -> Removes/Hides the peak.
  * Click anywhere else -> Snaps to the local maximum (+/- 6 frames) and creates a new peak.
- Key 's': Save confirmed labels and close window.
- Key 'r': Reset confirmed peaks to initial reference proposal.
- Key 'c': Clear all confirmed peaks (hide all red circles; reference triangles remain).
- Key 'Space': Toggle Play / Pause audio.
- Button [Play]: Start / Resume playback from current cursor position.
- Button [Pause]: Pause playback (cursor stays at current timestamp).
- Button [Stop]: Stop playback and reset cursor to t = 0.0s.
"""

from pathlib import Path
import sys

# Lớp bảo vệ nạp root directory vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Optional, Set
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.animation as animation

import config
from src.audio.framing import frame_to_time, time_to_frame

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


class InteractiveAnnotator:
    """
    Interactive Matplotlib GUI with instant marker toggling (Zero-lag Blit refresh).
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
        snap_radius_frames: int = 10,
        click_tolerance_frames: int = 2,
        search_radius_frames: int = 6,
        sync_offset: float = getattr(config, "ANNOTATION_AUDIO_SYNC_OFFSET", 0.0)
    ):
        self.filename = audio_filename
        self.h_event_sum = h_event_sum
        self.total_frames = len(h_event_sum)
        self.snap_radius = snap_radius_frames
        self.click_tolerance = click_tolerance_frames
        self.search_radius = search_radius_frames
        self.prominence_threshold = prominence_threshold
        self.ground_truth_count = ground_truth_count
        self.audio_samples = audio_samples
        self.sample_rate = sample_rate
        self.sync_offset = sync_offset

        # 1. Đo độ trễ phần cứng âm thanh
        self.hardware_latency = 0.0
        if HAS_SOUNDDEVICE:
            try:
                device_info = sd.query_devices(kind='output')
                self.hardware_latency = float(device_info.get('default_low_output_latency', 0.05))
            except Exception:
                self.hardware_latency = 0.05

        self.fft_center_offset = (config.N_FFT / 2.0) / float(self.sample_rate)

        # 2. Hai tập mốc riêng biệt
        self.reference_peaks: Set[int] = set(reference_peaks) if reference_peaks is not None else set()
        self.peak_detect_count = len(self.reference_peaks)
        self.confirmed_peaks: Set[int] = set(self.reference_peaks)

        # Trục thời gian chuẩn
        self.frame_indices = np.arange(self.total_frames)
        self.time_axis = frame_to_time(self.frame_indices, hop_length=config.HOP_LENGTH, sr=self.sample_rate)
        self.max_time = float(self.time_axis[-1]) if len(self.time_axis) > 0 else 0.0

        # Trạng thái phát âm thanh
        self.is_playing = False
        self.audio_start_time = 0.0
        self.playback_start_time = 0.0
        self.current_cursor_time = 0.0

        # Cửa sổ hiển thị
        self.fig, self.ax = plt.subplots(figsize=(15, 7))
        plt.subplots_adjust(bottom=0.20, top=0.88)

        self.ref_line = None
        self.confirmed_line = None
        self.cursor_line = None
        self.anim = None

        self._setup_plot()
        self._setup_buttons()
        self._connect_events()

    def _setup_plot(self):
        # Đường kích hoạt NMF (nền tĩnh)
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

        # Con chạy dọc phát lại (đối tượng động blit)
        self.cursor_line = self.ax.axvline(
            x=self.current_cursor_time,
            color="#2ca02c",
            linestyle="-",
            linewidth=1.8,
            alpha=0.9,
            animated=True,
            label="Playback Cursor"
        )

        # Lớp 1: Reference Peaks (Tam giác xám cố định ở nền tĩnh)
        ref_times = []
        ref_amps = []
        if len(self.reference_peaks) > 0:
            ref_list = sorted(list(self.reference_peaks))
            ref_times = self.time_axis[ref_list]
            ref_amps = self.h_event_sum[ref_list]

        (self.ref_line,) = self.ax.plot(
            ref_times,
            ref_amps,
            linestyle="",
            marker="^",
            color="#7f7f7f",
            markersize=7,
            alpha=0.7,
            zorder=4,
            label=r"Ref Peaks ($P^*, D^*$)"
        )

        # Lớp 2: Confirmed Peaks (animated=True kết hợp Line2D để Blit làm mới tức thì)
        (self.confirmed_line,) = self.ax.plot(
            [],
            [],
            linestyle="",
            marker="o",
            color="crimson",
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=5,
            animated=True,
            label="Confirmed Peaks"
        )

        self.ax.set_xlabel("Time (seconds)", fontsize=11)
        self.ax.set_ylabel("Activation Energy", fontsize=11)
        self.ax.grid(True, linestyle=":", alpha=0.6)
        self.ax.set_xlim(self.time_axis[0], self.time_axis[-1])

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
            f"[Ref (P*, D*): {self.peak_detect_count}]   "
            f"[GT: {gt_str}]   "
            f"[Confirmed: {cur_count}]   "
            f"[Audio Latency: {self.hardware_latency*1000:.1f}ms]\n"
            f"[Left-Click]: Seek  |  [Right-Click]: Toggle Peak (Hide/Show)  |  "
            f"[Space]: Play/Pause  |  [S]: Save  |  [R]: Reset  |  [C]: Clear"
        )
        self.ax.set_title(title_text, fontsize=11, fontweight="bold")
        self.fig.canvas.draw_idle()

    def _redraw_confirmed_peaks(self):
        """Cập nhật dữ liệu tọa độ cho confirmed_line."""
        if len(self.confirmed_peaks) > 0:
            peak_list = sorted(list(self.confirmed_peaks))
            times = self.time_axis[peak_list]
            amps = self.h_event_sum[peak_list]
            self.confirmed_line.set_data(times, amps)
        else:
            self.confirmed_line.set_data([], [])

        self._update_title()

    def _on_mouse_down(self, event):
        if event.inaxes != self.ax:
            return

        click_time = event.xdata
        if click_time is None:
            return

        snapped_frame = int(time_to_frame(click_time, hop_length=config.HOP_LENGTH, sr=self.sample_rate))
        snapped_frame = int(np.clip(snapped_frame, 0, self.total_frames - 1))
        snapped_time = float(self.time_axis[snapped_frame])

        # CHUỘT TRÁI: Seek Audio & Con chạy
        if event.button == 1:
            self._seek_to(snapped_time)

        # CHUỘT PHẢI: BẬT / TẮT HOẶC TẠO ĐỈNH MỚI
        elif event.button == 3:
            # 1. Nếu click thực sự sát một đỉnh đã chọn (<= click_tolerance) -> Xóa đỉnh
            existing_peak_to_remove = None
            for p in self.confirmed_peaks:
                if abs(p - snapped_frame) <= self.click_tolerance:
                    existing_peak_to_remove = p
                    break

            if existing_peak_to_remove is not None:
                self.confirmed_peaks.remove(existing_peak_to_remove)
            else:
                # 2. Nếu click gần một Reference Peak (trong snap_radius) -> Snap vào Reference Peak
                candidates_in_range = [
                    p for p in self.reference_peaks
                    if abs(p - snapped_frame) <= self.snap_radius
                ]

                if len(candidates_in_range) > 0:
                    target_peak = min(candidates_in_range, key=lambda p: abs(p - snapped_frame))
                    self.confirmed_peaks.add(target_peak)
                else:
                    # 3. Tạo đỉnh tự do: Tìm cực đại cục bộ trong phạm vi +/- search_radius quanh chỗ click
                    start = max(0, snapped_frame - self.search_radius)
                    end = min(self.total_frames, snapped_frame + self.search_radius + 1)
                    local_max = start + int(np.argmax(self.h_event_sum[start:end]))
                    self.confirmed_peaks.add(local_max)

            # Cập nhật dữ liệu ngay lập tức
            self._redraw_confirmed_peaks()

    def _seek_to(self, target_time: float):
        was_playing = self.is_playing
        if was_playing and HAS_SOUNDDEVICE:
            sd.stop()

        self.current_cursor_time = target_time
        self.audio_start_time = target_time
        self.cursor_line.set_xdata([self.current_cursor_time, self.current_cursor_time])

        if was_playing:
            self._start_audio_stream()

    def _start_audio_stream(self):
        if not HAS_SOUNDDEVICE or self.audio_samples is None:
            return

        start_sample = int(self.audio_start_time * self.sample_rate)
        start_sample = min(start_sample, len(self.audio_samples) - 1)

        sd.stop()
        sd.play(self.audio_samples[start_sample:], self.sample_rate)
        self.playback_start_time = time.monotonic()
        self.is_playing = True

    def _on_play(self, _):
        if not HAS_SOUNDDEVICE or self.audio_samples is None:
            print("[!] Cần sounddevice và audio_samples để phát âm thanh.")
            return

        if self.is_playing:
            return

        if self.current_cursor_time >= self.max_time:
            self.current_cursor_time = 0.0
            self.audio_start_time = 0.0

        self.audio_start_time = self.current_cursor_time
        self._start_audio_stream()

    def _get_current_playback_time(self) -> float:
        elapsed_clock = time.monotonic() - self.playback_start_time
        actual_audio_elapsed = max(0.0, elapsed_clock - self.hardware_latency)
        pos = self.audio_start_time + actual_audio_elapsed - self.fft_center_offset + self.sync_offset
        return max(0.0, min(pos, self.max_time))

    def _on_pause(self, _):
        if self.is_playing:
            if HAS_SOUNDDEVICE:
                sd.stop()
            self.current_cursor_time = self._get_current_playback_time()
            self.audio_start_time = self.current_cursor_time
            self.is_playing = False
            self.cursor_line.set_xdata([self.current_cursor_time, self.current_cursor_time])

    def _on_stop(self, _):
        if HAS_SOUNDDEVICE:
            sd.stop()
        self.is_playing = False
        self.current_cursor_time = 0.0
        self.audio_start_time = 0.0
        self.cursor_line.set_xdata([0.0, 0.0])

    def _init_cursor_anim(self):
        """Khởi tạo các đối tượng động trong Blit."""
        self.cursor_line.set_xdata([self.current_cursor_time, self.current_cursor_time])
        return (self.cursor_line, self.confirmed_line)

    def _update_cursor(self, _):
        """Vòng lặp animation 15ms: vẽ đè cả vạch chạy dọc và các chấm tròn đỏ."""
        if self.is_playing:
            current = self._get_current_playback_time()
            if current >= self.max_time:
                self._on_stop(None)
            else:
                self.current_cursor_time = current

        self.cursor_line.set_xdata([self.current_cursor_time, self.current_cursor_time])
        return (self.cursor_line, self.confirmed_line)

    def _on_key(self, event):
        key = event.key.lower() if event.key else ""

        if key == "s":
            self._on_stop(None)
            plt.close(self.fig)
        elif key == "r":
            self.confirmed_peaks = set(self.reference_peaks)
            self._redraw_confirmed_peaks()
        elif key == "c":
            # Ẩn toàn bộ chấm đỏ
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