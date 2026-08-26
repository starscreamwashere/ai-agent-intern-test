"""Structured, JSON-lines tracing for the agent.

Every turn emits one JSON record capturing what the README's observability
section asks for: the user message, relevant conversation history, retrieved
passages with metadata and scores, tool calls and sanitized results, the final
response, and any fallback/handoff. Records go to a stream (stderr by default) or
a file, one JSON object per line, so they are easy to grep or pipe into `jq`.

Safety: the trace is assembled only from already-sanitized data — tool results
come from `order_lookup`, which never returns PII or internal fields, and
retrieved passages are logged as metadata (source, status, scores), not full
text. As defense in depth, `scrub()` drops any key that looks internal before a
record is written, and secrets are never logged.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from typing import Any, Protocol, TextIO


class Tracer(Protocol):
    def log(self, record: dict[str, Any]) -> None: ...


_FORBIDDEN_KEYS = {"customer", "internal", "email", "shipping_address", "risk_score", "warehouse_note"}


def scrub(obj: Any) -> Any:
    """Recursively drop keys that should never appear in a log record."""
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items() if k not in _FORBIDDEN_KEYS}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


class NullTracer:
    """No-op tracer used when tracing is disabled."""

    def log(self, record: dict[str, Any]) -> None:  # noqa: D401
        pass


class JsonTracer:
    def __init__(self, stream: TextIO | None = None, *, enabled: bool = True) -> None:
        self.stream = stream or sys.stderr
        self.enabled = enabled

    def log(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        record = scrub(record)
        record.setdefault("ts", time.time())
        self.stream.write(json.dumps(record, default=str) + "\n")
        self.stream.flush()


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]
