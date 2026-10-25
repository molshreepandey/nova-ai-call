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
