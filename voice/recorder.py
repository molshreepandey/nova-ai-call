"""Microphone capture with sounddevice: push-to-talk or fixed duration.

Clips are returned as in-memory 16-bit mono WAV, ready to send to an STT API.
"""

from __future__ import annotations

import io
import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)

MAX_RECORDING_SECONDS = 60
# RMS (on a 0-1 scale) below which a clip is treated as silence. Whisper tends to
# hallucinate phrases like "Thank you." on silent audio, so those clips are dropped.
SILENCE_RMS = 0.004


class MicrophoneError(RuntimeError):
    """Raised when no usable input device is available."""


@dataclass(frozen=True)
class AudioClip:
    """A recorded utterance."""

    wav_bytes: bytes
    duration_s: float
    rms: float

    @property
    def is_silent(self) -> bool:
        """True if the clip is too quiet to contain speech."""
        return self.rms < SILENCE_RMS or self.duration_s < 0.3


class Recorder:
    """Records mono audio from the default (or a chosen) input device."""

    def __init__(self, sample_rate: int = 16000, device: int | str | None = None) -> None:
        self.sample_rate = sample_rate
        self.device = device

    def check_device(self) -> str:
        """Return the input device name, or raise :class:`MicrophoneError`."""
        try:
            info = sd.query_devices(self.device, kind="input")
        except (ValueError, sd.PortAudioError) as exc:
            raise MicrophoneError(f"no microphone found: {exc}") from exc
        return str(info["name"])

    def record_fixed(self, seconds: float) -> AudioClip:
        """Record for exactly ``seconds`` seconds."""
        frames = sd.rec(
            int(seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
        )
        sd.wait()
        return self._to_clip(frames[:, 0])

    def record_until(self, stop: Callable[[], None]) -> AudioClip:
        """Record until ``stop()`` returns (e.g. it blocks waiting for Enter).

        Recording also ends automatically after ``MAX_RECORDING_SECONDS``.
        """
        chunks: queue.Queue[np.ndarray] = queue.Queue()
        done = threading.Event()

        def callback(indata: np.ndarray, _frames: int, _time: object, status: sd.CallbackFlags) -> None:
            if status:
                logger.debug("[MIC] %s", status)
            chunks.put(indata[:, 0].copy())

        stopper = threading.Thread(target=lambda: (stop(), done.set()), daemon=True)
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=callback,
        ):
            stopper.start()
            if not done.wait(MAX_RECORDING_SECONDS):
                logger.warning("[MIC] reached %ds limit; stopping", MAX_RECORDING_SECONDS)

        samples = [chunks.get_nowait() for _ in range(chunks.qsize())]
        audio = np.concatenate(samples) if samples else np.zeros(0, dtype="float32")
        return self._to_clip(audio)

    def _to_clip(self, audio: np.ndarray) -> AudioClip:
        if audio.size and not np.any(audio):
            # A real mic always has some noise floor; exact zeros mean the OS is muting us.
            logger.warning(
                "[MIC] microphone returned pure digital silence. On macOS, allow microphone "
                "access for your terminal app in System Settings > Privacy & Security > "
                "Microphone, then restart the terminal. On Windows, check Settings > Privacy > "
                "Microphone. Or use --text mode."
            )
        buffer = io.BytesIO()
        sf.write(buffer, audio, self.sample_rate, format="WAV", subtype="PCM_16")
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        clip = AudioClip(wav_bytes=buffer.getvalue(), duration_s=audio.size / self.sample_rate, rms=rms)
        logger.info("[MIC] recorded %.1fs (rms=%.4f)", clip.duration_s, clip.rms)
        return clip
