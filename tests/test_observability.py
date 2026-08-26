"""Observability tests: trace completeness and secret safety (offline)."""

import io
import json

import pytest

from aster_agent.agent import SupportAgent
from aster_agent.embeddings import TfidfEmbedder
from aster_agent.llm import LLMResult, MockLLM, ToolCall
from aster_agent.observability import JsonTracer, scrub
from aster_agent.order_lookup import OrderStore
from aster_agent.retrieval import KnowledgeBase


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return KnowledgeBase.build(TfidfEmbedder(), use_cache=False)


def _agent(kb, responder, stream):
    return SupportAgent(
        kb, MockLLM(responder=responder), OrderStore(), top_k=5,
        tracer=JsonTracer(stream, enabled=True),
    )


def test_scrub_drops_internal_keys():
    dirty = {"status": "shipped", "internal": {"risk_score": 82}, "customer": {"email": "x@y.z"}}
    clean = scrub(dirty)
    assert "internal" not in clean
    assert "customer" not in clean
    assert clean["status"] == "shipped"


def test_turn_trace_has_required_fields(kb):
    stream = io.StringIO()

    def responder(turns, tools):
        return LLMResult(text="30 calendar days.\nSources: 01-returns-policy-current.md\nHandoff: no")

    agent = _agent(kb, responder, stream)
    agent.chat("return window?")
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(records) == 1
    rec = records[0]
    for key in (
        "session_id", "turn", "user_message", "history", "retrieval_query",
        "retrieved", "tool_calls", "final_answer", "sources", "handoff", "fallback",
    ):
        assert key in rec, f"missing trace field: {key}"
    assert rec["retrieved"][0]["score"] is not None


def test_trace_never_logs_pii_or_internal_on_order_lookup(kb):
    stream = io.StringIO()

    def responder(turns, tools):
        if any(t.role == "tool" for t in turns):
            return LLMResult(text="Shipped with UPS.\nSources: none\nHandoff: no")
        return LLMResult(tool_call=ToolCall(name="order_lookup", args={"order_id": "ORD-1007"}))

    agent = _agent(kb, responder, stream)
    agent.chat("where is ORD-1007?")
    blob = stream.getvalue().lower()
    assert "ava.morgan@example.test" not in blob
    assert "220 king street" not in blob
    assert "fraud" not in blob
    assert "82" not in blob


def test_history_accumulates_across_turns(kb):
    stream = io.StringIO()

    def responder(turns, tools):
        return LLMResult(text="ok\nSources: none\nHandoff: no")

    agent = _agent(kb, responder, stream)
    agent.chat("first")
    agent.chat("second")
    records = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert records[0]["turn"] == 0
    assert records[0]["history"] == []
    assert records[1]["turn"] == 1
    assert len(records[1]["history"]) == 2  # prior user + model


def test_disabled_tracer_is_silent(kb):
    stream = io.StringIO()

    def responder(turns, tools):
        return LLMResult(text="ok\nSources: none\nHandoff: no")

    agent = SupportAgent(
        kb, MockLLM(responder=responder), OrderStore(), top_k=5,
        tracer=JsonTracer(stream, enabled=False),
    )
    agent.chat("hi")
    assert stream.getvalue() == ""
