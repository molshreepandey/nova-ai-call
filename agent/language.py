"""Call language selection (English / Hindi) and the fixed phrases for each language."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

Language = Literal["en", "hi"]

# Spoken at the start of every call, before a language is chosen.
LANGUAGE_QUESTION = (
    "Hello, welcome to NovaWealth. Would you like to talk in English or Hindi? "
    "नमस्ते, नोवावेल्थ में आपका स्वागत है। आप अंग्रेज़ी में बात करना चाहेंगे या हिंदी में?"
)
LANGUAGE_REPROMPT = "Please say English or Hindi. कृपया अंग्रेज़ी या हिंदी बोलिए।"

_HINDI_WORDS = re.compile(r"hindi|hindee|हिंदी|हिन्दी|हिंदि")
_ENGLISH_WORDS = re.compile(r"english|inglish|इंग्लिश|अंग्रेज़ी|अंग्रेजी|अंग्रेज़ी")


@dataclass(frozen=True)
class LanguagePack:
    """Everything that differs per call language."""

    code: Language
    name: str
    greeting: str
    goodbye: str
    not_heard: str
    turn_failed: str
    rate_limited: str
    prompt_instruction: str


LANGUAGES: dict[Language, LanguagePack] = {
    "en": LanguagePack(
        code="en",
        name="English",
        greeting="Great, let's continue in English. How can I help you with your investments today?",
        goodbye="Thank you for calling NovaWealth. Have a great day, goodbye!",
        not_heard="Sorry, I didn't catch that. Could you say it again?",
        turn_failed="Sorry, I ran into a problem answering that. Could you try again?",
        rate_limited="Sorry, I'm handling a lot of requests right now. Please ask me again in a few seconds.",
        prompt_instruction="The caller chose English. Reply only in English.",
    ),
    "hi": LanguagePack(
        code="hi",
        name="Hindi",
        greeting="ठीक है, हम हिंदी में बात करेंगे। बताइए, मैं आपके निवेश से जुड़ी किस बात में मदद कर सकती हूँ?",
        goodbye="नोवावेल्थ को कॉल करने के लिए धन्यवाद। आपका दिन शुभ हो, नमस्ते!",
        not_heard="माफ़ कीजिए, मैं सुन नहीं पाई। क्या आप दोबारा बोल सकते हैं?",
        turn_failed="माफ़ कीजिए, जवाब देने में कोई दिक्कत आ गई। कृपया फिर से पूछिए।",
        rate_limited="माफ़ कीजिए, अभी बहुत सारे अनुरोध आ रहे हैं। कृपया कुछ सेकंड बाद फिर से पूछिए।",
        prompt_instruction=(
            "The caller chose Hindi. Reply only in natural, simple spoken Hindi written in "
            "Devanagari script. Common financial terms like SIP, NAV, mutual fund and "
            "expense ratio may stay in English. Write numbers and amounts as digits, e.g. "
            "₹23.2 लाख or 12%, never spelled out in words. Replying in Hindi does not change the tool "
            "rules: for every NovaWealth question you must still call search_knowledge_base "
            "first and answer only from its results. Write tool arguments, including "
            "search queries, in English."
        ),
    ),
}


def detect_language_choice(text: str) -> Language | None:
    """Return the language the caller asked for, or None if the answer is unclear."""
    lowered = text.lower()
    wants_hindi = bool(_HINDI_WORDS.search(lowered))
    wants_english = bool(_ENGLISH_WORDS.search(lowered))
    if wants_hindi == wants_english:
        # Neither or both named: fall back to the script the caller spoke in.
        if wants_hindi:
            return None
        if re.search(r"[ऀ-ॿ]", text):
            return "hi"
        return None
    return "hi" if wants_hindi else "en"
