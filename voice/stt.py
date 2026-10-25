"""Speech-to-text interface and the Groq Whisper implementation."""

from __future__ import annotations

import logging
import time
from typing import Protocol

from groq import AsyncGroq

logger = logging.getLogger(__name__)

# Whisper's prompt biases recognition toward these spellings, which matters for
# made-up product names like "NovaBand X200".
VOCABULARY_HINT = "NovaTech support call about the NovaBand X200, NovaHub, NovaFit app, NovaCare+ warranty."


class SpeechToText(Protocol):
    """Anything that turns a WAV clip into text."""

    async def transcribe(self, wav_bytes: bytes) -> str:
        """Return the transcript of ``wav_bytes`` (may be empty)."""
        ...


class GroqWhisperSTT:
    """Transcription via Groq's hosted Whisper models."""

    def __init__(self, api_key: str, model: str = "whisper-large-v3-turbo", language: str = "en") -> None:
        self._client = AsyncGroq(api_key=api_key, max_retries=2)
        self.model = model
        self._language = language

    async def transcribe(self, wav_bytes: bytes) -> str:
        """Send the clip to Groq Whisper and return the stripped transcript."""
        start = time.perf_counter()
        result = await self._client.audio.transcriptions.create(
            file=("speech.wav", wav_bytes),
            model=self.model,
            language=self._language,
            prompt=VOCABULARY_HINT,
            temperature=0.0,
            response_format="json",
        )
        text = (result.text or "").strip()
        logger.info("[STT] %r (%.0f ms)", text, (time.perf_counter() - start) * 1000)
        return text
