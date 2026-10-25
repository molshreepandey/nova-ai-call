"""Central configuration: environment variables, paths and logging setup.

Every tunable lives here so the rest of the code never reads ``os.environ``
directly. Only ``GROQ_API_KEY`` is required.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

ROOT_DIR = Path(__file__).resolve().parent
DOCS_DIR = ROOT_DIR / "data" / "docs"
CHROMA_DIR = ROOT_DIR / "storage" / "chroma"
LOG_DIR = ROOT_DIR / "logs"
LOG_FILE = LOG_DIR / "agent.log"

load_dotenv(ROOT_DIR / ".env")

# Quiet third-party startup noise (progress bars, HF token hints) before those libs load.
for _var, _value in {
    "TRANSFORMERS_VERBOSITY": "error",
    "HF_HUB_VERBOSITY": "error",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TORCH_CPP_LOG_LEVEL": "ERROR",
}.items():
    os.environ.setdefault(_var, _value)


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable application settings resolved from the environment."""

    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"))
    # Empty = model default. Qwen with "none" skips tools and hallucinates, so it is not
    # the default; "low" is a reasonable choice for openai/gpt-oss-* models.
    reasoning_effort: str = field(default_factory=lambda: os.getenv("GROQ_REASONING_EFFORT", ""))
    stt_model: str = field(default_factory=lambda: os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo"))
    tts_voice: str = field(default_factory=lambda: os.getenv("TTS_VOICE", "en-US-AriaNeural"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    collection_name: str = "novatech_kb"
    max_tool_iterations: int = field(default_factory=lambda: _int_env("MAX_TOOL_ITERATIONS", 5))
    max_history_turns: int = field(default_factory=lambda: _int_env("MAX_HISTORY_TURNS", 8))
    rag_top_k: int = field(default_factory=lambda: _int_env("RAG_TOP_K", 4))
    sample_rate: int = field(default_factory=lambda: _int_env("RECORD_SAMPLE_RATE", 16000))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())

    def require_groq_key(self) -> str:
        """Return the Groq API key or raise a helpful error if it is missing."""
        if not self.groq_api_key:
            raise ConfigError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
                "(https://console.groq.com/keys)."
            )
        return self.groq_api_key


settings = Settings()


def setup_logging(level: str | None = None, console: Console | None = None) -> None:
    """Configure rich console logging plus a plain-text file log at logs/agent.log.

    The file log is what the MCP ``read_system_log`` tool exposes. Pass the app's
    ``console`` so log lines and live spinners share one Rich console and don't garble.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    console_handler = RichHandler(console=console, rich_tracebacks=True, show_path=False, markup=True)
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level or settings.log_level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Third-party libraries are chatty at INFO.
    noisy = ("httpx", "httpx2", "httpcore", "chromadb", "sentence_transformers", "mcp", "urllib3")
    for name in noisy:
        logging.getLogger(name).setLevel(logging.WARNING)
    for name in ("huggingface_hub", "transformers", "filelock", "torch.distributed.elastic"):
        logging.getLogger(name).setLevel(logging.ERROR)
