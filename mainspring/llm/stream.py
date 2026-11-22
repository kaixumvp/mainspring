"""Stream accumulation: fold ``ChatChunk``s from ``StreamingLLMGateway`` into one ``LLMResponse``.

The Decide node consumes a stream chunk by chunk (emitting token events) and folds the
chunks into the same ``LLMResponse`` the non-streaming ``chat()`` path produces, so every
downstream step is streaming-agnostic.
"""

from __future__ import annotations

import json

from mainspring.ports import ChatChunk, LLMResponse, ToolCall, Usage


def accumulate_response(chunks: list[ChatChunk]) -> LLMResponse:
    """Fold ordered chunks into a single response.

    Tool-call fragments are joined by their stream `index`; `id`/`name` come from the first
    fragment that carries them. Malformed argument JSON degrades to `{}` (same tolerance as
    the non-streaming gateway). Usage is taken from the last chunk that carries one.
    """
    content: list[str] = []
    calls: dict[int, dict] = {}
    usage: Usage | None = None
    for chunk in chunks:
        if chunk.content_delta:
            content.append(chunk.content_delta)
        for delta in chunk.tool_call_deltas:
            call = calls.setdefault(delta.index, {"id": "", "name": "", "arguments": ""})
            if delta.id:
                call["id"] = delta.id
            if delta.name:
                call["name"] = delta.name
            call["arguments"] += delta.arguments_delta
        if chunk.usage is not None:
            usage = chunk.usage

    tool_calls: list[ToolCall] = []
    for index in sorted(calls):
        call = calls[index]
        args: dict = {}
        if call["arguments"]:
            try:
                args = json.loads(call["arguments"])
            except json.JSONDecodeError:
                args = {}
        tool_calls.append(ToolCall(id=call["id"], name=call["name"], arguments=args))

    return LLMResponse(
        content="".join(content) or None,
        tool_calls=tool_calls,
        usage=usage or Usage(),
    )


__all__ = ["accumulate_response"]
