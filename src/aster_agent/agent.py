"""SupportAgent: retrieval + tool loop + multi-turn memory + structured output.

Per turn:
1. Build a conversation-aware retrieval query (recent user turns + current) so
   follow-ups like "What about Canada?" retrieve the right passages.
2. Retrieve top-k passages with precedence weighting.
3. Run the LLM with the order_lookup tool available. Execute any tool calls,
   feeding sanitized results back, until the model returns a text answer.
4. Parse the machine-readable Sources/Handoff markers, and force a handoff when a
   tool result requires human review. Return a structured AgentResponse plus a
   full trace for observability.

Sessions are isolated: each SupportAgent instance owns one conversation's memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .llm import LLMClient, ToolCall, ToolSpec, Turn
from .order_lookup import OrderStore
from .prompts import SYSTEM_PROMPT, format_context_block
from .retrieval import KnowledgeBase, RetrievedChunk

MAX_TOOL_ITERATIONS = 4
RETRIEVAL_HISTORY_TURNS = 3  # how many recent user turns to fold into the query

ORDER_LOOKUP_TOOL = ToolSpec(
    name="order_lookup",
    description=(
        "Look up the current status of a customer order by its order ID "
        "(for example ORD-1007). Returns only customer-safe fields. Call this "
        "whenever the user asks about a specific order and has provided an ID."
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The order ID to look up, e.g. 'ORD-1007'.",
            }
        },
        "required": ["order_id"],
    },
)

_SOURCES_RE = re.compile(r"^\s*sources?\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_HANDOFF_RE = re.compile(r"^\s*handoff\s*:\s*(yes|no|true|false)\s*$", re.IGNORECASE | re.MULTILINE)
_FILENAME_RE = re.compile(r"[0-9A-Za-z._\-]+\.md")


@dataclass
class ToolInvocation:
    name: str
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass
class AgentResponse:
    answer: str
    sources: list[str] = field(default_factory=list)
    handoff: bool = False
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_called(self) -> bool:
        return bool(self.tool_calls)


def _parse_markers(text: str) -> tuple[str, list[str], bool | None]:
    """Extract Sources/Handoff markers, returning (clean_answer, sources, handoff)."""
    sources: list[str] = []
    handoff: bool | None = None

    m = _SOURCES_RE.search(text)
    if m:
        raw = m.group(1).strip()
        if raw.lower() not in ("none", "n/a", ""):
            sources = _FILENAME_RE.findall(raw)

    h = _HANDOFF_RE.search(text)
    if h:
        handoff = h.group(1).lower() in ("yes", "true")

    # Strip the marker lines from the customer-facing answer.
    clean = _SOURCES_RE.sub("", text)
    clean = _HANDOFF_RE.sub("", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean, sources, handoff


class SupportAgent:
    def __init__(self, kb: KnowledgeBase, llm: LLMClient, order_store: OrderStore, *, top_k: int = 5) -> None:
        self.kb = kb
        self.llm = llm
        self.order_store = order_store
        self.top_k = top_k
        self._history: list[Turn] = []          # persisted user/model turns
        self._user_messages: list[str] = []      # for building retrieval queries

    def _retrieval_query(self, message: str) -> str:
        recent = self._user_messages[-(RETRIEVAL_HISTORY_TURNS - 1):] if RETRIEVAL_HISTORY_TURNS > 1 else []
        return " ".join([*recent, message]).strip()

    def _execute_tool(self, call: ToolCall) -> dict[str, Any]:
        if call.name == "order_lookup":
            order_id = str(call.args.get("order_id", ""))
            return self.order_store.lookup(order_id)
        return {"error": "unknown_tool", "message": f"No such tool: {call.name}"}

    def chat(self, message: str) -> AgentResponse:
        # 1. Retrieval (conversation-aware).
        query = self._retrieval_query(message)
        retrieved = self.kb.search(query, top_k=self.top_k)
        context_block = format_context_block(retrieved)

        # 2. Assemble turns: persisted history + this turn's context + message.
        turns: list[Turn] = list(self._history)
        turns.append(Turn(role="user", text=f"{context_block}\n\nCustomer message: {message}"))

        tool_invocations: list[ToolInvocation] = []
        forced_handoff = False
        answer_text = ""

        # 3. Tool loop.
        for _ in range(MAX_TOOL_ITERATIONS):
            result = self.llm.respond(
                system_instruction=SYSTEM_PROMPT, turns=turns, tools=[ORDER_LOOKUP_TOOL]
            )
            if result.tool_call is not None:
                tool_result = self._execute_tool(result.tool_call)
                tool_invocations.append(
                    ToolInvocation(
                        name=result.tool_call.name,
                        args=result.tool_call.args,
                        result=tool_result,
                    )
                )
                if tool_result.get("requires_human"):
                    forced_handoff = True
                turns.append(Turn(role="model", tool_call=result.tool_call))
                turns.append(
                    Turn(role="tool", tool_name=result.tool_call.name, tool_result=tool_result)
                )
                continue
            answer_text = result.text or ""
            break
        else:
            # Ran out of iterations without a text answer.
            answer_text = (
                "I'm having trouble completing that request right now. "
                "Let me hand you to a human specialist."
            )
            forced_handoff = True

        # 4. Parse markers and finalize.
        clean_answer, sources, handoff_marker = _parse_markers(answer_text)
        handoff = bool(forced_handoff or handoff_marker)

        # Persist memory: the user message (with context stripped for compactness)
        # and the model's final answer.
        self._user_messages.append(message)
        self._history.append(Turn(role="user", text=f"Customer message: {message}"))
        self._history.append(Turn(role="model", text=clean_answer))

        trace = {
            "user_message": message,
            "retrieval_query": query,
            "retrieved": [r.to_debug() for r in retrieved],
            "tool_calls": [
                {"name": t.name, "args": t.args, "result": t.result} for t in tool_invocations
            ],
            "raw_answer": answer_text,
            "sources": sources,
            "handoff": handoff,
        }

        return AgentResponse(
            answer=clean_answer,
            sources=sources,
            handoff=handoff,
            retrieved=retrieved,
            tool_calls=tool_invocations,
            trace=trace,
        )

    def reset(self) -> None:
        self._history.clear()
        self._user_messages.clear()
