"""
===============================================================================
Offline Runtime Adapter (src/runtime/offline.py)
===============================================================================

Handles file-based batch processing for offline testing and evaluation.
"""

from pathlib import Path
from typing import Dict, Any, Union
import numpy as np

import config
from src.audio.loader import load_audio
from src.runtime.sensor import AcousticVirtualSensorCore


class OfflineAudioProcessor:
    """
    Offline adapter that wraps AcousticVirtualSensorCore to process WAV files.
    """

    def __init__(self, mode: str = "crnn"):
        self.sensor = AcousticVirtualSensorCore(mode=mode)

    def process_file(self, wav_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load an audio file and execute full virtual sensor pipeline.
        """
        path = Path(wav_path).resolve()
        audio, sr = load_audio(path, target_sr=config.SAMPLE_RATE)
        
        self.sensor.reset()
        result = self.sensor.process_offline_file(audio)
        result["filename"] = path.name
        result["duration_sec"] = len(audio) / float(sr)
        return result