"""Speech-to-text interface and the Groq Whisper implementation."""

from __future__ import annotations

import logging
import time
import unicodedata
from typing import Protocol

from groq import AsyncGroq

logger = logging.getLogger(__name__)

# Whisper's prompt biases recognition toward these spellings and terms. With no
# language set, the bilingual hint helps it pick up "English" / "हिंदी" answers.
VOCABULARY_HINTS: dict[str | None, str] = {
    "en": "NovaWealth investor support call about SIP, NAV, ELSS, mutual funds, expense ratio, exit load, KYC.",
    "hi": "नोवावेल्थ निवेश सहायता कॉल: SIP, म्यूचुअल फंड, NAV, ELSS, KYC, रिडेम्पशन, एक्सपेंस रेशियो।",
    None: "English or Hindi? अंग्रेज़ी या हिंदी? NovaWealth investor support.",
}

# Phrases Whisper is known to produce on noise or silence (training-data artefacts).
_HALLUCINATIONS = {
    "thank you", "thanks for watching", "thank you for watching", "you", "subtitles by",
    "please subscribe", "धन्यवाद", "शुक्रिया", "सब्सक्राइब करें",
}


def is_probable_hallucination(text: str) -> bool:
    """True for a transcript that is most likely Whisper hallucinating on noise."""
    # Strip punctuation by Unicode category: a regex like [^\w\s] would also strip
    # Devanagari vowel signs, which are combining marks rather than word characters.
    without_punct = "".join(ch for ch in text.lower() if not unicodedata.category(ch).startswith("P"))
    normalised = " ".join(without_punct.split())
    return not normalised or normalised in _HALLUCINATIONS


class SpeechToText(Protocol):
    """Anything that turns recorded audio into text."""

    async def transcribe(self, audio: bytes, filename: str = "speech.wav", language: str | None = None) -> str:
        """Return the transcript of ``audio`` (may be empty).

        ``filename``'s extension tells the provider the container format, and
        ``language`` is an ISO code such as "en" or "hi" (None = auto-detect).
        """
        ...


class GroqWhisperSTT:
    """Transcription via Groq's hosted Whisper models."""

    def __init__(
        self, api_key: str, model: str = "whisper-large-v3-turbo", models_by_language: dict[str, str] | None = None
    ) -> None:
        """
        Args:
            api_key: Groq API key.
            model: Default Whisper model.
            models_by_language: Per-language overrides, e.g. {"hi": "whisper-large-v3"}.
        """
        self._client = AsyncGroq(api_key=api_key, max_retries=2)
        self.model = model
        self._models_by_language = models_by_language or {}

    async def transcribe(self, audio: bytes, filename: str = "speech.wav", language: str | None = None) -> str:
        """Send the clip to Groq Whisper and return the stripped transcript."""
        start = time.perf_counter()
        kwargs = {"language": language} if language else {}
        result = await self._client.audio.transcriptions.create(
            file=(filename, audio),
            model=self._models_by_language.get(language or "", self.model),
            prompt=VOCABULARY_HINTS.get(language, VOCABULARY_HINTS[None]),
            temperature=0.0,
            response_format="json",
            **kwargs,
        )
        text = (result.text or "").strip()
        logger.info("[STT] %r (lang=%s, %.0f ms)", text, language or "auto", (time.perf_counter() - start) * 1000)
        return text
