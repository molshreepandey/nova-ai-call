"""Small pure helpers: LLM output parsing, hang-up detection, TTS splitting, audio clips."""

import numpy as np
import pytest

from agent.llm_client import _parse_arguments, _strip_reasoning
from agent.prompts import is_hangup
from voice.recorder import Recorder
from voice.tts import split_sentences


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("<think>let me reason</think>The answer is 147.", "The answer is 147."),
        ("some leaked reasoning\n</think>\n\nThe answer is 147.", "The answer is 147."),
        ("  Plain answer.  ", "Plain answer."),
    ],
)
def test_reasoning_is_never_spoken(raw: str, clean: str) -> None:
    assert _strip_reasoning(raw) == clean


def test_tool_argument_parsing() -> None:
    assert _parse_arguments('{"city": "Mumbai"}') == ({"city": "Mumbai"}, None)
    assert _parse_arguments(None) == ({}, None)
    assert _parse_arguments("{not json")[1] is not None
    assert _parse_arguments("[1, 2]")[1] is not None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Goodbye!", True),
        ("Okay thanks, goodbye.", True),
        ("Please end call", True),
        ("Can you hang up now", True),
        ("What's the battery life?", False),
        ("Is the X200 good?", False),
    ],
)
def test_hangup_detection(text: str, expected: bool) -> None:
    assert is_hangup(text) is expected


def test_split_sentences_merges_fragments_only() -> None:
    assert split_sentences("Hi! I'm Nova, how can I help you today?") == ["Hi! I'm Nova, how can I help you today?"]
    assert split_sentences("Yes, it's still covered. It's been 238 of 365 days.") == [
        "Yes, it's still covered.", "It's been 238 of 365 days."
    ]
    assert split_sentences("   ") == []


def test_audio_clip_silence_detection() -> None:
    recorder = Recorder(sample_rate=16000)
    speech = recorder._to_clip((0.1 * np.sin(np.linspace(0, 2000, 16000))).astype("float32"))
    quiet = recorder._to_clip(np.random.default_rng(0).normal(0, 0.0005, 16000).astype("float32"))
    blip = recorder._to_clip((0.1 * np.ones(1600)).astype("float32"))
    assert not speech.is_silent
    assert quiet.is_silent
    assert blip.is_silent  # too short to be speech
    assert speech.wav_bytes[:4] == b"RIFF"


def test_digital_silence_warns_about_mic_permission(caplog: pytest.LogCaptureFixture) -> None:
    Recorder()._to_clip(np.zeros(16000, dtype="float32"))
    assert "Microphone" in caplog.text


def test_repetition_guard_cuts_looping_answers() -> None:
    from agent.llm_client import _trim_repetition

    looping = "Total is 12 lakh. No wait. Total is 12 lakh. Sorry. Total is 12 lakh."
    assert _trim_repetition(looping) == "Total is 12 lakh. No wait."
    clean = "ठीक है। कुल निवेश ₹12 लाख है। अनुमानित मूल्य ₹23.23 लाख है।"
    assert _trim_repetition(clean) == clean


@pytest.mark.parametrize(
    ("amount", "spoken"),
    [(2_323_390.76, "₹23.23 lakh"), (1_200_000, "₹12 lakh"), (15_000_000, "₹1.5 crore"),
     (45_000, "₹45,000"), (500, "₹500"), (123_456, "₹1.23 lakh")],
)
def test_format_inr_uses_lakh_and_crore(amount: float, spoken: str) -> None:
    from mcp_server.finance import format_inr

    assert format_inr(amount) == spoken


async def test_llm_falls_back_to_second_model_when_rate_limited() -> None:
    from types import SimpleNamespace

    from groq import RateLimitError
    from httpx import Request, Response

    from agent.llm_client import GroqLLMClient

    llm = GroqLLMClient("key", "primary-model", reasoning_effort="none", fallback_model="backup-model")
    calls: list[dict] = []

    async def fake_create(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs)
        if kwargs["model"] == "primary-model":
            raise RateLimitError(
                "Rate limit reached ... on tokens per day (TPD): Limit 200000",
                response=Response(429, request=Request("POST", "http://x")), body=None,
            )
        message = SimpleNamespace(content="Answer from backup.", tool_calls=None)
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=message)])

    llm._client.chat.completions.create = fake_create  # type: ignore[method-assign]
    first = await llm.complete([{"role": "user", "content": "hi"}])
    second = await llm.complete([{"role": "user", "content": "hi again"}])

    assert first.content == second.content == "Answer from backup."
    # Primary tried once, then skipped during the cool-down; model-specific settings dropped.
    assert [c["model"] for c in calls] == ["primary-model", "backup-model", "backup-model"]
    assert "reasoning_effort" not in calls[1]
