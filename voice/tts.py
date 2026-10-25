"""Text-to-speech interface, the edge-tts implementation and a silent fallback.

edge-tts returns MP3, which libsndfile (bundled with the soundfile wheel) decodes
on Windows, macOS and Linux without ffmpeg. Playback uses sounddevice.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import time
from typing import Protocol

import edge_tts
import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")  # "।" ends Hindi sentences


class TextToSpeech(Protocol):
    """Anything that can speak a piece of text aloud."""

    async def speak(self, text: str, voice: str | None = None) -> None:
        """Synthesize ``text`` (optionally in a specific ``voice``) and play it,
        returning when playback ends."""
        ...


class SilentTTS:
    """No-op TTS for text mode without ``--tts``."""

    async def speak(self, text: str, voice: str | None = None) -> None:
        """Do nothing."""


class EdgeTTS:
    """Microsoft Edge neural voices (free, no key) played through the default speaker."""

    def __init__(self, voice: str = "en-IN-NeerjaNeural", rate: str = "+0%") -> None:
        self.voice = voice
        self._rate = rate

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """Return MP3 audio for ``text`` in ``voice`` (default: the instance voice)."""
        audio = bytearray()
        async for chunk in edge_tts.Communicate(text, voice or self.voice, rate=self._rate).stream():
            if chunk["type"] == "audio":
                audio += chunk["data"]
        if not audio:
            raise RuntimeError("edge-tts returned no audio")
        return bytes(audio)

    async def synthesize_parallel(self, text: str, voice: str | None = None) -> bytes:
        """MP3 for ``text``, synthesizing all sentences concurrently.

        edge-tts returns raw MP3 frames, so per-sentence clips concatenate into one
        playable file. Latency is roughly the slowest sentence instead of the sum.
        """
        sentences = split_sentences(text) or [text]
        clips = await asyncio.gather(*(self.synthesize(s, voice) for s in sentences))
        return b"".join(clips)

    async def speak(self, text: str, voice: str | None = None) -> None:
        """Synthesize and play ``text`` sentence by sentence.

        The next sentence is synthesized while the current one plays, so speech
        starts after one sentence's synthesis instead of the whole answer's.
        Failures are logged, never raised: the answer is already on screen, so a
        missing speaker shouldn't end the call.
        """
        sentences = split_sentences(text)
        if not sentences:
            return
        start = time.perf_counter()
        pending = asyncio.create_task(self.synthesize(sentences[0], voice))
        try:
            for i in range(len(sentences)):
                mp3 = await pending
                if i + 1 < len(sentences):
                    pending = asyncio.create_task(self.synthesize(sentences[i + 1], voice))
                samples, sample_rate = sf.read(io.BytesIO(mp3), dtype="float32")
                if i == 0:
                    logger.info("[TTS] first audio after %.0f ms", (time.perf_counter() - start) * 1000)
                await asyncio.to_thread(_play_blocking, samples, sample_rate)
        except Exception as exc:  # noqa: BLE001 - TTS is best-effort
            pending.cancel()
            logger.warning("[TTS] could not speak (%s: %s); answer is shown on screen", type(exc).__name__, exc)


def split_sentences(text: str, min_chars: int = 20) -> list[str]:
    """Split text into sentences, merging very short ones to avoid choppy pauses."""
    parts = [p.strip() for p in _SENTENCE_END.split(text.strip()) if p.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged


def _play_blocking(samples: np.ndarray, sample_rate: int) -> None:
    sd.play(samples, sample_rate)
    sd.wait()
