"""Evaluation runner: load cases, run each in a fresh session, report results.

Runs the real agent (Gemini) by default; grading is fully deterministic (see
assertions.py). Reports per-case and per-category results and writes a
machine-readable JSON summary.

    python -m aster_agent.evalsuite.runner                 # visible + extra cases
    python -m aster_agent.evalsuite.runner --only visible  # visible cases only
    python -m aster_agent.evalsuite.runner --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from ..agent import AgentResponse, SupportAgent
from ..build import build_agent, build_knowledge_base
from ..config import EVAL_DIR, load_config
from .assertions import CaseResult, evaluate_case

VISIBLE_CASES = EVAL_DIR / "visible-cases.json"
EXTRA_CASES = EVAL_DIR / "extra-cases.json"


def load_cases(paths: list[Path]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for case in data.get("cases", []):
            case = dict(case)
            case["_source_file"] = path.name
            cases.append(case)
    return cases


def run_case(agent: SupportAgent, case: dict[str, Any]) -> tuple[CaseResult, AgentResponse]:
    """Run all messages in a case in one session; grade the final response."""
    last: AgentResponse | None = None
    for msg in case["messages"]:
        if msg.get("role") == "user":
            last = agent.chat(msg["content"])
    assert last is not None, f"case {case['id']} has no user messages"
    result = evaluate_case(
        case["expect"], last, case_id=case["id"], category=case.get("category", "uncategorized")
    )
    return result, last


def run_suite(
    cases: list[dict[str, Any]],
    agent_factory: Callable[[], SupportAgent],
) -> list[tuple[CaseResult, AgentResponse]]:
    results: list[tuple[CaseResult, AgentResponse]] = []
    for case in cases:
        agent = agent_factory()  # fresh, isolated session per case
        try:
            results.append(run_case(agent, case))
        except Exception as exc:  # a crash is a failed case, not a crashed run
            cr = CaseResult(
                case_id=case["id"],
                category=case.get("category", "uncategorized"),
                passed=False,
                error=repr(exc),
            )
            results.append((cr, AgentResponse(answer="")))
    return results


def summarize(results: list[tuple[CaseResult, AgentResponse]]) -> dict[str, Any]:
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    cases_out = []
    n_passed = 0
    for cr, resp in results:
        by_category[cr.category]["total"] += 1
        if cr.passed:
            by_category[cr.category]["passed"] += 1
            n_passed += 1
        cases_out.append(
            {
                "id": cr.case_id,
                "category": cr.category,
                "passed": cr.passed,
                "checks_passed": cr.n_passed,
                "checks_total": cr.n_total,
                "error": cr.error,
                "failed_checks": [
                    {"name": c.name, "detail": c.detail} for c in cr.checks if not c.passed
                ],
                "answer": resp.answer,
                "sources": resp.sources,
                "tool_calls": [{"name": t.name, "args": t.args} for t in resp.tool_calls],
                "handoff": resp.handoff,
            }
        )
    return {
        "overall": {"passed": n_passed, "total": len(results),
                    "pass_rate": round(n_passed / len(results), 4) if results else 0.0},
        "by_category": {k: v for k, v in sorted(by_category.items())},
        "cases": cases_out,
    }


def print_report(summary: dict[str, Any]) -> None:
    print("\n=== Per-case results ===")
    for c in summary["cases"]:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['id']:<34} ({c['category']}, {c['checks_passed']}/{c['checks_total']} checks)")
        if not c["passed"]:
            if c["error"]:
                print(f"         error: {c['error']}")
            for fc in c["failed_checks"]:
                detail = f" — {fc['detail']}" if fc["detail"] else ""
                print(f"         x {fc['name']}{detail}")

    print("\n=== Per-category ===")
    for cat, v in summary["by_category"].items():
        print(f"  {cat:<24} {v['passed']}/{v['total']}")

    o = summary["overall"]
    print(f"\n=== Overall: {o['passed']}/{o['total']} cases passed ({o['pass_rate']*100:.1f}%) ===\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the agent evaluation suite.")
    parser.add_argument("--only", choices=["visible", "extra", "all"], default="all")
    parser.add_argument("--output", type=Path, default=EVAL_DIR / "results.json")
    parser.add_argument("--label", default="run", help="label stored in the results file")
    args = parser.parse_args(argv)

    paths = {
        "visible": [VISIBLE_CASES],
        "extra": [EXTRA_CASES],
        "all": [VISIBLE_CASES, EXTRA_CASES],
    }[args.only]
    cases = load_cases(paths)

    config = load_config()
    if not config.has_api_key:
        print("GEMINI_API_KEY is not set. Set it in .env to run the live eval.", file=sys.stderr)
        return 1

    print(f"Building knowledge base ({config.effective_embed_backend} embeddings)...", file=sys.stderr)
    kb = build_knowledge_base(config)
    print(f"Running {len(cases)} cases against {config.gemini_model}...", file=sys.stderr)

    def factory() -> SupportAgent:
        return build_agent(config, kb=kb)  # fresh session, shared KB

    results = run_suite(cases, factory)
    summary = summarize(results)
    summary["label"] = args.label
    summary["model"] = config.gemini_model

    print_report(summary)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}", file=sys.stderr)

    return 0 if summary["overall"]["passed"] == summary["overall"]["total"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
