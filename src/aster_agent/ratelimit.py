"""Client-side rate limiting and retry for the Gemini free tier.

The free tier caps `generate_content` at a few requests per minute, so a full
eval run must (a) pace requests and (b) retry on 429 RESOURCE_EXHAUSTED, honoring
the server's suggested retry delay. This keeps the suite completing end-to-end
instead of dying partway through.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Callable, TypeVar

T = TypeVar("T")

_RETRY_DELAY_RE = re.compile(r"retry(?:Delay|\s+in)['\":\s]*([\d.]+)s", re.IGNORECASE)


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc)
    return "429" in s or "RESOURCE_EXHAUSTED" in s


def _parse_retry_delay(exc: Exception, default: float) -> float:
    m = _RETRY_DELAY_RE.search(str(exc))
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return default


class RateLimiter:
    """Enforces a minimum interval between calls, shared across threads/clients."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            gap = now - self._last
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self._last = time.monotonic()


def call_with_retry(
    fn: Callable[[], T],
    *,
    limiter: RateLimiter | None = None,
    max_retries: int = 2,
    default_delay: float = 20.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call `fn`, pacing via `limiter` and retrying on rate-limit errors."""
    attempt = 0
    while True:
        if limiter is not None:
            limiter.wait()
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise non-rate-limit errors
            if not _is_rate_limit(exc) or attempt >= max_retries:
                raise
            delay = _parse_retry_delay(exc, default_delay) + 1.0
            attempt += 1
            sleep(delay)


# Shared limiter for content-generation calls (the constrained quota).
_GENERATE_LIMITER: RateLimiter | None = None


def get_generate_limiter(min_interval: float) -> RateLimiter:
    """Return the process-wide generate limiter, creating it on first use."""
    global _GENERATE_LIMITER
    if _GENERATE_LIMITER is None:
        _GENERATE_LIMITER = RateLimiter(min_interval)
    else:
        # Respect the most conservative interval requested.
        _GENERATE_LIMITER.min_interval = max(_GENERATE_LIMITER.min_interval, min_interval)
    return _GENERATE_LIMITER
