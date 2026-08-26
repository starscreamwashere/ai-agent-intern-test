"""LLM abstraction with a Gemini implementation and a scriptable mock.

The agent talks to a provider-neutral interface so its control flow (retrieval,
the tool loop, memory, source/handoff parsing) is testable offline with a mock,
while the real Gemini call lives behind the same surface.

Turn model (provider-neutral):
* Turn(role="user"|"model"|"tool", ...)
  - a "user" or "model" turn carries `text`
  - a "model" turn that called a tool carries `tool_call`
  - a "tool" turn carries `tool_name` + `tool_result` (a JSON-able dict)

LLMResult is either a text answer or a single tool call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class Turn:
    role: str  # "user" | "model" | "tool"
    text: str = ""
    tool_call: ToolCall | None = None
    tool_name: str | None = None
    tool_result: dict[str, Any] | None = None


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the arguments object


@dataclass
class LLMResult:
    text: str | None = None
    tool_call: ToolCall | None = None


class LLMClient(Protocol):
    def respond(
        self, *, system_instruction: str, turns: list[Turn], tools: list[ToolSpec]
    ) -> LLMResult: ...


# --------------------------------------------------------------------------- #
# Gemini implementation                                                        #
# --------------------------------------------------------------------------- #

class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.6-flash",
        temperature: float = 0.0,
        *,
        min_interval: float = 0.0,
    ) -> None:
        from google import genai

        from .ratelimit import get_generate_limiter

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self._limiter = get_generate_limiter(min_interval) if min_interval > 0 else None

    def _to_contents(self, turns: list[Turn]):
        from google.genai import types

        contents = []
        for t in turns:
            if t.role == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=t.text)]))
            elif t.role == "model":
                if t.tool_call is not None:
                    contents.append(
                        types.Content(
                            role="model",
                            parts=[
                                types.Part(
                                    function_call=types.FunctionCall(
                                        name=t.tool_call.name, args=t.tool_call.args
                                    )
                                )
                            ],
                        )
                    )
                else:
                    contents.append(types.Content(role="model", parts=[types.Part(text=t.text)]))
            elif t.role == "tool":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=t.tool_name, response=t.tool_result or {}
                                )
                            )
                        ],
                    )
                )
        return contents

    def _to_tools(self, tools: list[ToolSpec]):
        from google.genai import types

        if not tools:
            return None
        declarations = [
            types.FunctionDeclaration(
                name=t.name, description=t.description, parameters=t.parameters
            )
            for t in tools
        ]
        return [types.Tool(function_declarations=declarations)]

    def respond(
        self, *, system_instruction: str, turns: list[Turn], tools: list[ToolSpec]
    ) -> LLMResult:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.temperature,
            tools=self._to_tools(tools),
            # Manual function calling: we execute + record tool calls ourselves.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        from .ratelimit import call_with_retry

        resp = call_with_retry(
            lambda: self._client.models.generate_content(
                model=self.model, contents=self._to_contents(turns), config=config
            ),
            limiter=self._limiter,
        )

        # Look for a function call in the first candidate's parts.
        candidate = resp.candidates[0] if resp.candidates else None
        if candidate and candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                fc = getattr(part, "function_call", None)
                if fc:
                    args = dict(fc.args) if fc.args else {}
                    return LLMResult(tool_call=ToolCall(name=fc.name, args=args))

        return LLMResult(text=(resp.text or "").strip())


# --------------------------------------------------------------------------- #
# Scriptable mock for offline tests                                            #
# --------------------------------------------------------------------------- #

@dataclass
class MockLLM:
    """Deterministic LLM stand-in driven by a scripted responder.

    `responder(turns, tools)` returns an LLMResult. It receives the full turn
    history each call, so a script can decide to emit a tool call first and a
    text answer once the tool result is present.
    """

    responder: Callable[[list[Turn], list[ToolSpec]], LLMResult]
    calls: list[list[Turn]] = field(default_factory=list)

    def respond(
        self, *, system_instruction: str, turns: list[Turn], tools: list[ToolSpec]
    ) -> LLMResult:
        self.calls.append(list(turns))
        return self.responder(turns, tools)
