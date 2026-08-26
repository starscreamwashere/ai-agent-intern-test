"""Minimal interactive CLI for the support agent.

Usage:
    python -m aster_agent.cli            # interactive chat
    python -m aster_agent.cli --debug    # also print a trace after each answer

The response clearly separates the answer, its sources, and whether a human
handoff is recommended. `--debug` dumps the structured trace (retrieval scores,
tool calls, sanitized results) for observability. A fuller trace view arrives in
the observability stage.
"""

from __future__ import annotations

import argparse
import sys

from .build import build_agent
from .config import load_config
from .observability import JsonTracer


def _print_response(resp) -> None:
    print("\nAgent:")
    print(resp.answer)
    if resp.sources:
        print("\nSources: " + ", ".join(resp.sources))
    if resp.handoff:
        print("\n[Recommending human handoff]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aster & Row support agent (CLI)")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="stream a structured JSON trace per turn to stderr",
    )
    args = parser.parse_args(argv)

    config = load_config()
    if not config.has_api_key:
        print(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        return 1

    print("Building knowledge base and agent...", file=sys.stderr)
    tracer = JsonTracer(sys.stderr, enabled=args.debug)
    agent = build_agent(config, tracer=tracer)
    print(
        "Aster & Row support agent. Type your message, 'reset' to clear the "
        "session, or 'exit' to quit.\n"
    )

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        if message.lower() in ("exit", "quit"):
            break
        if message.lower() == "reset":
            agent.reset()
            print("[session reset]\n")
            continue
        try:
            resp = agent.chat(message)
        except Exception as exc:  # keep the REPL alive on transient API errors
            print(f"\n[error] {exc}\n", file=sys.stderr)
            continue
        _print_response(resp)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
