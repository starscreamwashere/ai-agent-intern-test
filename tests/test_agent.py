"""Agent-core tests using a scripted mock LLM (no API key needed).

These verify the agent's control flow and post-processing: the tool loop,
conversation-aware retrieval, per-session memory, source/handoff parsing, and
forced handoff on tool results that require human review.
"""

import pytest

from aster_agent.agent import SupportAgent
from aster_agent.embeddings import TfidfEmbedder
from aster_agent.llm import LLMResult, MockLLM, ToolCall, Turn
from aster_agent.order_lookup import OrderStore
from aster_agent.retrieval import KnowledgeBase


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return KnowledgeBase.build(TfidfEmbedder(), use_cache=False)


def make_agent(kb, responder) -> SupportAgent:
    return SupportAgent(kb, MockLLM(responder=responder), OrderStore(), top_k=5)


def _last_user_text(turns: list[Turn]) -> str:
    for t in reversed(turns):
        if t.role == "user":
            return t.text
    return ""


def _has_tool_result(turns: list[Turn]) -> bool:
    return any(t.role == "tool" for t in turns)


def test_text_answer_parses_sources_and_handoff(kb):
    def responder(turns, tools):
        return LLMResult(
            text="Standard returns are within 30 calendar days of delivery.\n"
            "Sources: 01-returns-policy-current.md\nHandoff: no"
        )

    agent = make_agent(kb, responder)
    resp = agent.chat("What is the return window?")
    assert "30 calendar days" in resp.answer
    assert "Sources:" not in resp.answer  # markers stripped from user-facing text
    assert resp.sources == ["01-returns-policy-current.md"]
    assert resp.handoff is False
    assert resp.tool_called is False


def test_tool_loop_executes_and_feeds_result_back(kb):
    def responder(turns, tools):
        if not _has_tool_result(turns):
            return LLMResult(tool_call=ToolCall(name="order_lookup", args={"order_id": "ORD-1007"}))
        return LLMResult(text="Your order shipped with UPS.\nSources: none\nHandoff: no")

    agent = make_agent(kb, responder)
    resp = agent.chat("Where is ORD-1007?")
    assert resp.tool_called is True
    assert resp.tool_calls[0].name == "order_lookup"
    assert resp.tool_calls[0].args == {"order_id": "ORD-1007"}
    assert resp.tool_calls[0].result["status"] == "shipped"
    assert "UPS" in resp.answer


def test_exception_order_forces_handoff_even_if_model_says_no(kb):
    def responder(turns, tools):
        if not _has_tool_result(turns):
            return LLMResult(tool_call=ToolCall(name="order_lookup", args={"order_id": "ORD-1010"}))
        # Model wrongly says Handoff: no; agent must override from tool result.
        return LLMResult(text="There is an exception.\nSources: none\nHandoff: no")

    agent = make_agent(kb, responder)
    resp = agent.chat("What's happening with ORD-1010?")
    assert resp.handoff is True


def test_unknown_order_forces_handoff(kb):
    def responder(turns, tools):
        if not _has_tool_result(turns):
            return LLMResult(tool_call=ToolCall(name="order_lookup", args={"order_id": "ORD-9999"}))
        return LLMResult(text="I couldn't find that order.\nSources: none\nHandoff: no")

    agent = make_agent(kb, responder)
    resp = agent.chat("Check ORD-9999")
    assert resp.tool_calls[0].result["found"] is False
    assert resp.handoff is True


def test_conversation_aware_retrieval_query_includes_prior_turn(kb):
    seen_queries = []

    def responder(turns, tools):
        return LLMResult(text="Answer.\nSources: none\nHandoff: no")

    agent = make_agent(kb, responder)
    agent.chat("Do you ship internationally?")
    resp = agent.chat("What about Canada?")
    # The retrieval query for the follow-up should include the earlier turn.
    assert "internationally" in resp.trace["retrieval_query"]
    assert "Canada" in resp.trace["retrieval_query"]


def test_memory_persists_across_turns(kb):
    def responder(turns, tools):
        return LLMResult(text="ok\nSources: none\nHandoff: no")

    agent = make_agent(kb, responder)
    agent.chat("first message")
    agent.chat("second message")
    # History holds two user + two model turns.
    assert len(agent._history) == 4
    assert "first message" in agent._history[0].text


def test_reset_clears_session(kb):
    def responder(turns, tools):
        return LLMResult(text="ok\nSources: none\nHandoff: no")

    agent = make_agent(kb, responder)
    agent.chat("hello")
    agent.reset()
    assert agent._history == []
    assert agent._user_messages == []


def test_retrieved_context_frames_data_as_untrusted(kb):
    """Every turn's context block must frame retrieved passages as untrusted."""
    captured = {}

    def responder(turns, tools):
        captured["text"] = _last_user_text(turns)
        return LLMResult(text="ok\nSources: none\nHandoff: no")

    agent = make_agent(kb, responder)
    agent.chat("What is the return window?")
    assert "UNTRUSTED" in captured["text"]


def test_context_block_labels_superseded_and_draft_sources():
    """Precedence labels are rendered whenever such sources are in the set."""
    from aster_agent.embeddings import TfidfEmbedder
    from aster_agent.prompts import format_context_block
    from aster_agent.retrieval import KnowledgeBase as KB

    kb2 = KB.build(TfidfEmbedder(), use_cache=False)
    # Retrieve a wide set so the superseded/draft returns docs are included.
    retrieved = kb2.search("return window 30 45 60 days legacy migration", top_k=14)
    block = format_context_block(retrieved)
    assert "SUPERSEDED" in block
    assert "NON-AUTHORITATIVE" in block


def test_missing_markers_defaults_safely(kb):
    def responder(turns, tools):
        return LLMResult(text="Just an answer with no markers.")

    agent = make_agent(kb, responder)
    resp = agent.chat("hi")
    assert resp.sources == []
    assert resp.handoff is False
    assert resp.answer == "Just an answer with no markers."
