"""ASR backend using mlx-audio for speech transcription."""

import platform
import sys


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


class MLXASRBackend:
    def __init__(self):
        from mlx_audio.stt import load as load_stt

        self._model = load_stt("mlx-community/Qwen3-ASR-0.6B-8bit")
        print("ASR: mlx-audio loaded (Apple GPU)")

    def transcribe(self, audio_path: str) -> str:
        result = self._model.generate(audio_path)
        return result.text


def load():
    if _is_apple_silicon():
        return MLXASRBackend()

    raise RuntimeError(
        "ASR requires Apple Silicon. Install on a supported Mac to use this feature."
    )
