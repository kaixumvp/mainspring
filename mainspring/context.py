"""Default ContextManager (v2.1): sliding window + tool-result truncation.

Zero dependencies and side-effect free — it satisfies the ``ports.ContextManager``
protocol so the runtime can compact long histories before every Decide call. A host that
wants summarization instead of truncation implements the same protocol and captures its
own ``LLMGateway``; this default only ever drops or shortens messages.
"""

from __future__ import annotations


class SlidingWindowContext:
    """Keep the head system messages plus the newest `max_messages` turns.

    Dropping from the front can strand tool messages whose assistant tool-call parent was
    dropped (which OpenAI-compatible APIs reject), so the cut point always advances past
    leading tool messages. Oversized tool results are truncated in place even when the
    window is not exceeded — a single huge tool output is the most common context blowup.
    """

    def __init__(self, max_messages: int = 40, max_tool_chars: int = 4000) -> None:
        self._max_messages = max_messages
        self._max_tool_chars = max_tool_chars

    def fit(self, messages: list[dict]) -> list[dict]:
        out = [self._truncate(m) for m in messages]
        if len(out) <= self._max_messages:
            return out

        head = 0
        while head < len(out) and out[head].get("role") == "system":
            head += 1
        if head >= len(out):
            return out  # all system messages — nothing droppable

        # keep `head` system messages + newest tail; cut >= what makes the total fit
        cut = max(head, len(out) + head - self._max_messages)
        while cut < len(out) and out[cut].get("role") == "tool":
            cut += 1  # never strand a tool message at the tail's front
        if cut <= head:
            return out
        marker = {
            "role": "system",
            "content": f"[{cut - head} earlier messages omitted by context compaction]",
        }
        return [*out[:head], marker, *out[cut:]]

    def _truncate(self, message: dict) -> dict:
        if message.get("role") != "tool":
            return message
        content = message.get("content")
        if not isinstance(content, str) or len(content) <= self._max_tool_chars:
            return message
        trimmed = dict(message)
        trimmed["content"] = content[: self._max_tool_chars] + (
            f"…[truncated {len(content) - self._max_tool_chars} chars]"
        )
        return trimmed


__all__ = ["SlidingWindowContext"]
