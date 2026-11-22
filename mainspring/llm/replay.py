"""Deterministic record/replay of LLM responses (V1.1).

Record a run's responses with ``RecordingGateway``, then replay them offline with
``ReplayGateway`` so the *same* run reproduces the *same* result — the foundation for
offline evaluation and regression. ``dump_recording`` / ``load_recording`` persist a
recording to JSON.

Streaming: ``RecordingGateway`` passes chunks through while recording the folded response
once the stream ends; ``ReplayGateway`` synthesizes each recorded response as a single
chunk, so streaming consumers stay deterministic offline too.
"""

from __future__ import annotations

import json
from pathlib import Path

from mainspring.ports import ChatChunk, LLMResponse, Usage
from mainspring.llm.stream import accumulate_response
from mainspring.llm.structured import parse_structured, schema_kw


class RecordingGateway:
    """Wraps a gateway and records every response, in call order.

    Recordings store the raw response only: `parsed` is stripped (it is re-derivable from
    `content` + `schema`), so dumps stay plain JSON independent of Pydantic internals.
    """

    def __init__(self, gateway) -> None:
        self._gateway = gateway
        self.recorded: list[LLMResponse] = []

    async def chat(self, messages, model=None, tools=None, schema=None) -> LLMResponse:
        resp = await self._gateway.chat(messages, model=model, tools=tools, **schema_kw(schema))
        self.recorded.append(resp.model_copy(update={"parsed": None}))
        return resp

    def chat_stream(self, messages, model=None, tools=None):
        return self._stream(messages, model, tools)

    async def _stream(self, messages, model, tools):
        stream_fn = getattr(self._gateway, "chat_stream", None)
        if stream_fn is None:
            raise RuntimeError("inner gateway does not support streaming")
        chunks: list[ChatChunk] = []
        async for chunk in stream_fn(messages, model=model, tools=tools):
            chunks.append(chunk)
            yield chunk
        self.recorded.append(accumulate_response(chunks))


class ReplayGateway:
    """Plays back pre-recorded responses deterministically, ignoring any real LLM."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def chat(self, messages, model=None, tools=None, schema=None) -> LLMResponse:
        if not self._responses:
            raise RuntimeError("replay exhausted: no recorded response for this call")
        self.calls += 1
        resp = self._responses.pop(0)
        if schema is not None:
            # Recordings are raw; re-derive the parsed instance to reproduce the live run.
            resp = resp.model_copy(update={"parsed": parse_structured(resp.content, schema)})
        return resp

    def chat_stream(self, messages, model=None, tools=None):
        return self._stream(messages, model, tools)

    async def _stream(self, messages, model, tools):
        resp = await self.chat(messages, model=model, tools=tools)
        yield ChatChunk(content_delta=resp.content or "", usage=resp.usage)


def dump_recording(responses: list[LLMResponse], path) -> None:
    """Persist responses as raw JSON; `parsed` is excluded (re-derivable via `schema`)."""
    Path(path).write_text(
        json.dumps([r.model_dump(exclude={"parsed"}) for r in responses], ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def load_recording(path) -> list[LLMResponse]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [LLMResponse(**item) for item in data]


__all__ = ["RecordingGateway", "ReplayGateway", "dump_recording", "load_recording"]
