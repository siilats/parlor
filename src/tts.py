"""Qwen3 TTS via mlx-audio (Apple Silicon)."""

import platform
import sys

import numpy as np


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


class TTSBackend:
    """Unified TTS interface."""

    sample_rate: int = 24000

    def generate(
        self, text: str, voice: str = "Vivian", speed: float = 1.1
    ) -> np.ndarray:
        raise NotImplementedError


class Qwen3TTSBackend(TTSBackend):
    """mlx-audio backend (Apple Silicon GPU via MLX)."""

    TTS_REPO = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"

    def __init__(self):
        from mlx_audio.tts.generate import load_model

        self._model = load_model(self.TTS_REPO)
        self.sample_rate = self._model.sample_rate
        # Warmup: triggers pipeline init (phonemizer, spacy, etc.)
        list(self._model.generate(text="你好", voice="Vivian", language="Auto"))

    def generate(
        self, text: str, voice: str = "Vivian", speed: float = 1.1
    ) -> np.ndarray:
        results = list(self._model.generate(text=text, voice=voice, language="Auto"))
        return np.concatenate([np.array(r.audio) for r in results])


def load() -> TTSBackend:
    """Load the TTS backend for this platform."""
    if not _is_apple_silicon():
        raise RuntimeError(
            "TTS requires Apple Silicon. Install on a supported Mac to use this feature."
        )

    backend = Qwen3TTSBackend()
    print(f"TTS: Qwen3-TTS mlx-audio (Apple GPU, sample_rate={backend.sample_rate})")
    return backend
