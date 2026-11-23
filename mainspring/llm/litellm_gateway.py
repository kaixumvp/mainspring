import json
import logging
from collections.abc import AsyncIterator

from mainspring._optional import require_optional
from mainspring.llm.structured import parse_structured
from mainspring.ports import ChatChunk, LLMResponse, ToolCall, ToolCallDelta, Usage

logger = logging.getLogger(__name__)


class LiteLLMGateway:
    """LiteLLM implementation of the LLM Gateway (single vendor, OpenAI-compatible)."""

    def __init__(self, model: str = "", api_key: str = "", api_base: str | None = None) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base

    async def chat(self, messages, model=None, tools=None, schema=None) -> LLMResponse:
        litellm = require_optional("litellm", "litellm")
        model = model or self._model
        kwargs: dict = {"model": model, "messages": messages}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if tools:
            kwargs["tools"] = tools
        if schema is not None:
            # Provider-agnostic JSON mode; the schema itself is enforced framework-side.
            kwargs["response_format"] = {"type": "json_object"}

        logger.debug("llm.chat: model=%s messages=%d tools=%d", model, len(messages), len(tools or []))
        resp = await litellm.acompletion(**kwargs)
        msg = resp.choices[0].message

        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            args: dict = {}
            raw = getattr(tc.function, "arguments", None)
            if raw:
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append(
                ToolCall(id=getattr(tc, "id", "") or "", name=tc.function.name, arguments=args)
            )

        usage = resp.usage.model_dump() if resp.usage else {}
        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            parsed=parse_structured(msg.content, schema) if schema is not None else None,
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )

    def chat_stream(self, messages, model=None, tools=None) -> AsyncIterator[ChatChunk]:
        """Stream normalized chunks. Tries to include final usage; providers that reject
        ``stream_options`` are retried once without it."""
        return self._stream(messages, model or self._model, tools)

    async def _stream(self, messages, model, tools):
        litellm = require_optional("litellm", "litellm")
        kwargs: dict = {"model": model, "messages": messages, "stream": True}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if tools:
            kwargs["tools"] = tools

        logger.debug("llm.chat_stream: model=%s messages=%d tools=%d", model, len(messages), len(tools or []))
        try:
            kwargs["stream_options"] = {"include_usage": True}
            stream = await litellm.acompletion(**kwargs)
        except Exception:
            kwargs.pop("stream_options", None)
            stream = await litellm.acompletion(**kwargs)

        async for chunk in stream:
            yield _normalize_chunk(chunk)


def _normalize_chunk(chunk) -> ChatChunk:
    content = ""
    deltas: list[ToolCallDelta] = []
    for choice in getattr(chunk, "choices", None) or []:
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        content += getattr(delta, "content", None) or ""
        for tc in getattr(delta, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            deltas.append(
                ToolCallDelta(
                    index=getattr(tc, "index", 0) or 0,
                    id=getattr(tc, "id", None),
                    name=getattr(fn, "name", None) if fn is not None else None,
                    arguments_delta=(getattr(fn, "arguments", "") or "") if fn is not None else "",
                )
            )
    usage = getattr(chunk, "usage", None)
    return ChatChunk(
        content_delta=content,
        tool_call_deltas=deltas,
        usage=Usage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        ) if usage is not None else None,
    )
