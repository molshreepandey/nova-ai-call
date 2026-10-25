"""One phone call's flow, shared by the CLI and the web UI.

A call starts by asking for a language. Once the caller picks English or Hindi,
the system prompt, STT language and TTS voice all follow that choice for the
rest of the call. Every other utterance goes to the orchestrator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from groq import RateLimitError

from agent.language import LANGUAGE_QUESTION, LANGUAGE_REPROMPT, LANGUAGES, Language, detect_language_choice
from agent.orchestrator import Orchestrator, TurnResult
from agent.prompts import build_system_prompt, is_hangup

logger = logging.getLogger(__name__)


@dataclass
class CallReply:
    """What the agent says back for one caller utterance."""

    text: str
    language: Language | None
    ended: bool = False
    turn: TurnResult | None = None


class CallSession:
    """Language selection, hang-up handling and error recovery around an Orchestrator."""

    def __init__(self, orchestrator: Orchestrator, language: Language | None = None) -> None:
        self.orchestrator = orchestrator
        self.language: Language | None = None
        if language:
            self.choose_language(language)

    def opening(self) -> str:
        """First thing the agent says: the language question, or the greeting if preset."""
        return LANGUAGES[self.language].greeting if self.language else LANGUAGE_QUESTION

    def choose_language(self, language: Language) -> None:
        """Lock the call to ``language`` and tell the LLM to reply in it."""
        self.language = language
        self.orchestrator.set_system_prompt(build_system_prompt(language))
        logger.info("[CALL] language set to %s", LANGUAGES[language].name)

    @property
    def goodbye(self) -> str:
        """Closing line in the call language (English if none was chosen)."""
        return LANGUAGES[self.language or "en"].goodbye

    @property
    def not_heard(self) -> str:
        """Re-prompt for when nothing usable was heard."""
        return LANGUAGES[self.language].not_heard if self.language else LANGUAGE_REPROMPT

    async def handle(self, transcript: str) -> CallReply:
        """Respond to one caller utterance. Never raises: failures become spoken apologies."""
        if is_hangup(transcript):
            return CallReply(self.goodbye, self.language, ended=True)

        if self.language is None:
            choice = detect_language_choice(transcript)
            if choice is None:
                return CallReply(LANGUAGE_REPROMPT, None)
            self.choose_language(choice)
            return CallReply(LANGUAGES[choice].greeting, choice)

        pack = LANGUAGES[self.language]
        try:
            turn = await self.orchestrator.run_turn(transcript)
        except RateLimitError as exc:
            logger.warning("[CALL] Groq rate limit persisted (fallback included): %s", str(exc)[:300])
            return CallReply(pack.rate_limited, self.language)
        except Exception:  # noqa: BLE001 - the call must survive any turn failure
            logger.exception("[CALL] turn failed")
            return CallReply(pack.turn_failed, self.language)
        return CallReply(turn.answer, self.language, turn=turn)
