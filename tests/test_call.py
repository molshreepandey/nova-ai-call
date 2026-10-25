"""Call flow: language selection, hang-up, error recovery (fake orchestrator, no network)."""

import pytest
from groq import RateLimitError
from httpx import Request, Response

from agent.call import CallSession
from agent.language import LANGUAGE_QUESTION, LANGUAGE_REPROMPT, LANGUAGES, detect_language_choice
from agent.orchestrator import TurnResult
from agent.prompts import SYSTEM_PROMPT, is_hangup
from voice.stt import is_probable_hallucination


class FakeOrchestrator:
    def __init__(self, error: Exception | None = None) -> None:
        self.system_prompt = SYSTEM_PROMPT
        self.turns: list[str] = []
        self.error = error

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    async def run_turn(self, text: str) -> TurnResult:
        self.turns.append(text)
        if self.error:
            raise self.error
        return TurnResult(answer=f"answer to {text}")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("English please", "en"),
        ("I'd like to talk in english.", "en"),
        ("Hindi", "hi"),
        ("हिंदी में", "hi"),
        ("हिन्दी", "hi"),
        ("अंग्रेज़ी", "en"),
        ("मुझे इंग्लिश में बात करनी है", "en"),
        ("नमस्ते, मेरा सवाल है", "hi"),  # spoke Hindi without naming a language
        ("hello there", None),
        ("Hindi or English, either is fine", None),
    ],
)
def test_detect_language_choice(text: str, expected: str | None) -> None:
    assert detect_language_choice(text) == expected


async def test_call_asks_language_then_locks_it() -> None:
    orch = FakeOrchestrator()
    call = CallSession(orch)  # type: ignore[arg-type]
    assert call.opening() == LANGUAGE_QUESTION

    unclear = await call.handle("what's up")
    assert unclear.text == LANGUAGE_REPROMPT and call.language is None
    assert orch.turns == []  # nothing reaches the LLM before a language is chosen

    chosen = await call.handle("हिंदी")
    assert chosen.text == LANGUAGES["hi"].greeting and chosen.language == "hi"
    assert "Devanagari" in orch.system_prompt

    answer = await call.handle("SIP क्या है?")
    assert answer.text == "answer to SIP क्या है?" and answer.turn is not None


async def test_preset_language_skips_question() -> None:
    orch = FakeOrchestrator()
    call = CallSession(orch, language="en")  # type: ignore[arg-type]
    assert call.opening() == LANGUAGES["en"].greeting
    assert "Reply only in English" in orch.system_prompt


@pytest.mark.parametrize("phrase", ["goodbye", "Okay, end call.", "अलविदा", "ठीक है, धन्यवाद, अल्विदा!", "ठीक है, कॉल खत्म करो।"])
async def test_hangup_in_either_language(phrase: str) -> None:
    call = CallSession(FakeOrchestrator(), language="hi")  # type: ignore[arg-type]
    reply = await call.handle(phrase)
    assert reply.ended and reply.text == LANGUAGES["hi"].goodbye


def test_normal_sentences_are_not_hangups() -> None:
    assert not is_hangup("What is the exit load on the liquid fund?")
    assert not is_hangup("ELSS में लॉक-इन कितना है?")


async def test_turn_errors_become_apologies_in_call_language() -> None:
    rate_limit = RateLimitError("slow down", response=Response(429, request=Request("POST", "http://x")), body=None)
    call = CallSession(FakeOrchestrator(error=rate_limit), language="hi")  # type: ignore[arg-type]
    assert (await call.handle("सवाल")).text == LANGUAGES["hi"].rate_limited

    call = CallSession(FakeOrchestrator(error=RuntimeError("boom")), language="en")  # type: ignore[arg-type]
    assert (await call.handle("question")).text == LANGUAGES["en"].turn_failed


@pytest.mark.parametrize(
    ("text", "hallucination"),
    [("Thank you.", True), ("", True), ("  ...  ", True), ("धन्यवाद।", True),
     ("Thank you, what is NAV?", False), ("English", False)],
)
def test_whisper_hallucination_filter(text: str, hallucination: bool) -> None:
    assert is_probable_hallucination(text) is hallucination
