"""Central configuration, resolved from environment variables with safe defaults.

Loading .env is optional: the deterministic parts of the system (ingestion,
the TF-IDF retriever, and the order-lookup tool) run with no API key at all,
which is what lets the bulk of the evaluation suite run offline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass


# Repository layout. config.py lives at src/aster_agent/config.py, so the repo
# root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE_DIR = REPO_ROOT / "knowledge-base"
ORDERS_PATH = REPO_ROOT / "data" / "orders.json"
EVAL_DIR = REPO_ROOT / "evaluation"


@dataclass(frozen=True)
class Config:
    gemini_api_key: str | None
    gemini_model: str
    gemini_embed_model: str
    kb_top_k: int
    embed_backend: str  # "gemini" or "tfidf"
    gemini_min_interval: float  # seconds between generate calls (free-tier pacing)

    @property
    def has_api_key(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def effective_embed_backend(self) -> str:
        """Fall back to the offline backend when no key is present."""
        if self.embed_backend == "gemini" and not self.has_api_key:
            return "tfidf"
        return self.embed_backend


def load_config() -> Config:
    return Config(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        gemini_embed_model=os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
        kb_top_k=int(os.getenv("KB_TOP_K", "5")),
        embed_backend=os.getenv("EMBED_BACKEND", "gemini").strip().lower(),
        gemini_min_interval=float(os.getenv("GEMINI_MIN_INTERVAL_S", "13")),
    )
