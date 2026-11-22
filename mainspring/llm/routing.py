"""LLM routing and fault tolerance (V0.4).

``FallbackLLMGateway`` tries a list of ``LLMGateway``s in order and fails over to the next
on error. ``FaultTolerantGateway`` decorates a gateway so a malformed or failing response
degrades to a valid empty ``LLMResponse`` instead of raising — structured-output fallback.
Both are plain ``LLMGateway`` implementations, so they compose with anything.
"""

from __future__ import annotations

import logging

from ouroboros.ports import ChatChunk, LLMGateway, LLMResponse, Usage

logger = logging.getLogger(__name__)


class FallbackLLMGateway:
    """Route across vendors with failover. `last_used` records which gateway served the last call.

    Streaming failover only happens *before* the first chunk: once deltas have been emitted,
    a mid-stream error is re-raised instead of re-streaming from another vendor (which would
    duplicate output). Gateways without ``chat_stream`` are skipped in streaming mode.
    """

    def __init__(self, gateways: list[LLMGateway]) -> None:
        if not gateways:
            raise ValueError("FallbackLLMGateway requires at least one gateway")
        self._gateways = gateways
        self.last_used: str | None = None

    async def chat(self, messages, model=None, tools=None) -> LLMResponse:
        last_exc: Exception | None = None
        for gateway in self._gateways:
            try:
                resp = await gateway.chat(messages, model=model, tools=tools)
                self.last_used = getattr(gateway, "_model", None) or type(gateway).__name__
                return resp
            except Exception as exc:  # noqa: BLE001 — failover on any vendor error
                last_exc = exc
                logger.warning("gateway %s failed, falling back: %s", type(gateway).__name__, exc)
        raise RuntimeError("all LLM gateways failed") from last_exc

    def chat_stream(self, messages, model=None, tools=None):
        return self._stream(messages, model, tools)

    async def _stream(self, messages, model, tools):
        last_exc: Exception | None = None
        for gateway in self._gateways:
            stream_fn = getattr(gateway, "chat_stream", None)
            if stream_fn is None:
                continue
            first = True
            try:
                async for chunk in stream_fn(messages, model=model, tools=tools):
                    first = False
                    yield chunk
                self.last_used = getattr(gateway, "_model", None) or type(gateway).__name__
                return
            except Exception as exc:  # noqa: BLE001 — failover only pre-first-chunk
                if not first:
                    raise
                last_exc = exc
                logger.warning("gateway %s stream failed before first chunk, falling back: %s",
                               type(gateway).__name__, exc)
        raise RuntimeError("no streaming-capable gateway succeeded") from last_exc


class FaultTolerantGateway:
    """Catch gateway errors / malformed tool-calls and return a valid empty response.

    Streaming: an error *before* the first chunk degrades to a single chunk carrying
    ``on_error``; a mid-stream error is re-raised (the connection is already broken).
    """

    def __init__(self, gateway: LLMGateway, *, on_error: str = "") -> None:
        self._gateway = gateway
        self._on_error = on_error

    async def chat(self, messages, model=None, tools=None) -> LLMResponse:
        try:
            resp = await self._gateway.chat(messages, model=model, tools=tools)
        except Exception as exc:  # noqa: BLE001 — structured-output fallback
            logger.warning("gateway error, falling back to empty response: %s", exc)
            return LLMResponse(content=self._on_error, usage=Usage())
        # Tolerate a response whose tool_calls were not parsed (None) by normalizing to [].
        if getattr(resp, "tool_calls", None) is None:
            resp.tool_calls = []
        return resp

    def chat_stream(self, messages, model=None, tools=None):
        return self._stream(messages, model, tools)

    async def _stream(self, messages, model, tools):
        stream_fn = getattr(self._gateway, "chat_stream", None)
        if stream_fn is None:
            yield ChatChunk(content_delta=self._on_error, usage=Usage())
            return
        first = True
        try:
            async for chunk in stream_fn(messages, model=model, tools=tools):
                first = False
                yield chunk
        except Exception as exc:  # noqa: BLE001 — degrade pre-first-chunk, re-raise mid-stream
            if not first:
                raise
            logger.warning("gateway stream failed, degrading to on_error chunk: %s", exc)
            yield ChatChunk(content_delta=self._on_error, usage=Usage())


__all__ = ["FallbackLLMGateway", "FaultTolerantGateway"]
