"""Regression test for the Gemini tool-call replay (thought_signature bug).

Gemini 3.x thinking models return a `thought_signature` on the function-call part
that must be echoed back verbatim on the follow-up turn, or the API returns
400 INVALID_ARGUMENT. The agent must replay the original model content rather than
reconstructing the call from name+args. This test verifies `_to_contents` replays
the stored raw content. It builds the client without network via __new__.
"""

from google.genai import types

from aster_agent.llm import GeminiClient, ToolCall, Turn


def _client() -> GeminiClient:
    # Avoid constructing a real genai.Client (no network / key needed).
    return object.__new__(GeminiClient)


def test_tool_call_raw_content_is_replayed_verbatim():
    # Simulate the original model content carrying an opaque thought_signature.
    original = types.Content(
        role="model",
        parts=[types.Part(function_call=types.FunctionCall(name="order_lookup", args={"order_id": "ORD-1007"}))],
    )
    turns = [
        Turn(role="user", text="where is ORD-1007?"),
        Turn(role="model", tool_call=ToolCall(name="order_lookup", args={"order_id": "ORD-1007"}, raw=original)),
        Turn(role="tool", tool_name="order_lookup", tool_result={"status": "shipped"}),
    ]
    contents = _client()._to_contents(turns)
    # The model turn must be the exact original object (signature preserved),
    # not a freshly reconstructed Content.
    assert original in contents


def test_tool_call_without_raw_is_reconstructed():
    turns = [Turn(role="model", tool_call=ToolCall(name="order_lookup", args={"order_id": "X"}))]
    contents = _client()._to_contents(turns)
    assert len(contents) == 1
    assert contents[0].parts[0].function_call.name == "order_lookup"
