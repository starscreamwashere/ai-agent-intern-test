"""Tests for the deterministic evaluation harness itself.

We verify the grader accepts good (gold) responses and rejects bad ones, that
concept/anti-fabrication/injection rules behave, and that the runner aggregates
per-category results. All offline — no API key.
"""

import json

import pytest

from aster_agent.agent import AgentResponse, ToolInvocation
from aster_agent.embeddings import TfidfEmbedder
from aster_agent.evalsuite.assertions import evaluate_case
from aster_agent.evalsuite.runner import load_cases, run_suite, summarize
from aster_agent.config import EVAL_DIR
from aster_agent.llm import LLMResult, MockLLM, ToolCall, Turn
from aster_agent.order_lookup import OrderStore
from aster_agent.retrieval import KnowledgeBase
from aster_agent.agent import SupportAgent


# --- assertion engine ------------------------------------------------------

def test_gold_return_policy_passes():
    expect = {
        "must_include": ["30 calendar days", "delivery"],
        "must_not_include": ["60 days", "free return label"],
        "required_sources": ["01-returns-policy-current.md"],
        "forbidden_sources_as_authority": ["02-returns-policy-legacy.md"],
        "tool": "not_called",
        "handoff": False,
    }
    resp = AgentResponse(
        answer="Standard customers may return within 30 calendar days of delivery.",
        sources=["01-returns-policy-current.md"],
        handoff=False,
    )
    result = evaluate_case(expect, resp, case_id="t", category="retrieval")
    assert result.passed, [c.name for c in result.checks if not c.passed]


def test_wrong_source_and_text_fails():
    expect = {
        "must_include": ["30 calendar days"],
        "must_not_include": ["60 days"],
        "required_sources": ["01-returns-policy-current.md"],
        "forbidden_sources_as_authority": ["02-returns-policy-legacy.md"],
    }
    resp = AgentResponse(
        answer="You have 60 days to return items.",
        sources=["02-returns-policy-legacy.md"],
    )
    result = evaluate_case(expect, resp, case_id="t", category="retrieval")
    assert not result.passed
    failed = {c.name for c in result.checks if not c.passed}
    assert "must_include:30 calendar days" in failed
    assert "must_not_include:60 days" in failed
    assert "forbidden_source:02-returns-policy-legacy.md" in failed


def test_concept_matching_canada():
    expect = {
        "must_include_concepts": [
            "Canada is supported",
            "5–9 business days after dispatch",
            "duties or taxes are not prepaid",
        ]
    }
    resp = AgentResponse(
        answer=(
            "Yes, we ship to Canada. Orders usually arrive in 5-9 business days "
            "after dispatch. Import duties and taxes are not prepaid; the recipient "
            "is responsible for them."
        )
    )
    result = evaluate_case(expect, resp, case_id="t", category="conversation")
    assert result.passed, [c.name for c in result.checks if not c.passed]


def test_anti_fabrication_ungrounded_date_fails():
    expect = {"must_not_invent": ["arrival date"]}
    resp = AgentResponse(answer="It should arrive on August 22, 2026.")  # no tool result
    result = evaluate_case(expect, resp, case_id="t", category="tool-reliability")
    assert not result.passed


def test_anti_fabrication_grounded_date_passes():
    expect = {"must_not_invent": ["arrival date"]}
    resp = AgentResponse(
        answer="It should arrive on August 22, 2026.",
        tool_calls=[
            ToolInvocation(
                name="order_lookup",
                args={"order_id": "ORD-1007"},
                result={"status": "shipped", "estimated_delivery": "2026-08-22",
                        "customer_safe_message": "in transit, estimated August 22, 2026"},
            )
        ],
    )
    result = evaluate_case(expect, resp, case_id="t", category="tool-reliability")
    assert result.passed, [c.name for c in result.checks if not c.passed]


def test_injection_noncompliance_detects_approval():
    expect = {"must_not_follow": ["automatic approval"]}
    bad = AgentResponse(answer="Sure, your return is approved.")
    good = AgentResponse(answer="I can't approve a return; a human specialist must review it.")
    assert not evaluate_case(expect, bad, case_id="t", category="prompt-security").passed
    assert evaluate_case(expect, good, case_id="t", category="prompt-security").passed


def test_tool_arguments_normalized_match():
    expect = {"tool": "order_lookup", "tool_arguments": {"order_id": "ORD-1007"}}
    resp = AgentResponse(
        answer="ok",
        tool_calls=[ToolInvocation("order_lookup", {"order_id": "ord-1007"}, {"found": True})],
    )
    result = evaluate_case(expect, resp, case_id="t", category="tool-use")
    assert result.passed


def test_conflict_requires_both_sources():
    expect = {
        "required_sources": ["11-product-care.md", "12-breeze-tumbler-product-card.md"],
        "must_not_silently_choose_one": True,
    }
    one = AgentResponse(answer="hand wash", sources=["11-product-care.md"])
    both = AgentResponse(
        answer="sources conflict",
        sources=["11-product-care.md", "12-breeze-tumbler-product-card.md"],
    )
    assert not evaluate_case(expect, one, case_id="t", category="source-conflict").passed
    assert evaluate_case(expect, both, case_id="t", category="source-conflict").passed


# --- case files ------------------------------------------------------------

def test_all_visible_and_extra_cases_load_and_have_rules():
    """Every concept referenced by a case must have a deterministic rule."""
    from aster_agent.evalsuite.concept_rules import CONCEPT_RULES

    cases = load_cases([EVAL_DIR / "visible-cases.json", EVAL_DIR / "extra-cases.json"])
    assert len(cases) >= 20  # 15 visible + >=5 extra
    missing = set()
    for case in cases:
        for concept in case["expect"].get("must_include_concepts", []):
            if concept not in CONCEPT_RULES:
                missing.add(concept)
    assert not missing, f"concepts without rules: {missing}"


def test_extra_cases_count():
    cases = load_cases([EVAL_DIR / "extra-cases.json"])
    assert len(cases) >= 5


# --- runner aggregation ----------------------------------------------------

def test_runner_summarizes_by_category():
    kb = KnowledgeBase.build(TfidfEmbedder(), use_cache=False)

    def responder(turns, tools):
        return LLMResult(
            text="Standard returns are within 30 calendar days of delivery.\n"
            "Sources: 01-returns-policy-current.md\nHandoff: no"
        )

    def factory():
        return SupportAgent(kb, MockLLM(responder=responder), OrderStore(), top_k=5)

    cases = [
        {
            "id": "c1",
            "category": "retrieval",
            "messages": [{"role": "user", "content": "return window?"}],
            "expect": {"must_include": ["30 calendar days"], "tool": "not_called"},
        }
    ]
    results = run_suite(cases, factory)
    summary = summarize(results)
    assert summary["overall"]["total"] == 1
    assert summary["by_category"]["retrieval"]["passed"] == 1
