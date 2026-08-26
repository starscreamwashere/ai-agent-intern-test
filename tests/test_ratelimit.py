"""Rate-limit/retry tests (offline, using a fake sleep and fake errors)."""

import pytest

from aster_agent.ratelimit import RateLimiter, call_with_retry


class FakeRateLimitError(Exception):
    pass


def test_retries_on_rate_limit_then_succeeds():
    slept: list[float] = []
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeRateLimitError(
                "429 RESOURCE_EXHAUSTED ... 'retryDelay': '7s'"
            )
        return "ok"

    result = call_with_retry(fn, max_retries=5, sleep=slept.append)
    assert result == "ok"
    assert calls["n"] == 3
    # Two retries, each honoring the parsed 7s delay (+1s buffer).
    assert slept == [8.0, 8.0]


def test_non_rate_limit_error_is_not_retried():
    def fn():
        raise ValueError("something else")

    with pytest.raises(ValueError):
        call_with_retry(fn, sleep=lambda s: None)


def test_gives_up_after_max_retries():
    def fn():
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    with pytest.raises(RuntimeError):
        call_with_retry(fn, max_retries=2, sleep=lambda s: None)


def test_rate_limiter_paces_calls():
    # wait() reads monotonic twice per call; keep the clock frozen at 0 so the
    # gap is always 0 and a full interval must be slept.
    slept: list[float] = []

    import aster_agent.ratelimit as rl

    orig_monotonic = rl.time.monotonic
    orig_sleep = rl.time.sleep
    rl.time.monotonic = lambda: 0.0
    rl.time.sleep = slept.append
    try:
        limiter = RateLimiter(min_interval=10.0)
        limiter.wait()
        limiter.wait()
        assert slept == [pytest.approx(10.0), pytest.approx(10.0)]
    finally:
        rl.time.monotonic = orig_monotonic
        rl.time.sleep = orig_sleep
