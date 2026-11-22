"""Structured output (V3.1): `chat(..., schema=PydanticModel)` across the gateway chain."""

import json

import litellm
import pytest
from pydantic import BaseModel

from mainspring.llm import (
    CachingGateway,
    FallbackLLMGateway,
    FaultTolerantGateway,
    LiteLLMGateway,
    RecordingGateway,
    ReplayGateway,
    TieredRouter,
)
from mainspring.ports import LLMResponse, StructuredOutputError, Usage


class Answer(BaseModel):
    score: int
    reason: str


class _FakeUsage:
    def model_dump(self):
        return {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeCompletionResponse:
    def __init__(self, content=None):
        self.choices = [_FakeChoice(_FakeMessage(content))]
        self.usage = _FakeUsage()


class _CapturingGateway:
    """Inner mock that records the kwargs it receives."""

    def __init__(self, response: LLMResponse) -> None:
        self._response = response
        self.seen: list[dict] = []

    async def chat(self, messages, model=None, tools=None, **kw) -> LLMResponse:
        self.seen.append({"messages": messages, "model": model, "tools": tools, **kw})
        return self._response


class _LegacyGateway:
    """Inner mock predating V3.1: rejects any keyword beyond model/tools."""

    async def chat(self, messages, model=None, tools=None) -> LLMResponse:
        return LLMResponse(content="legacy", usage=Usage())


MSG = [{"role": "user", "content": "rate it"}]


# --------------------------------------------------------------------------- LiteLLMGateway
@pytest.mark.asyncio
async def test_litellm_schema_sends_response_format_and_parses(monkeypatch):
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _FakeCompletionResponse('{"score": 9, "reason": "good"}')

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    resp = await LiteLLMGateway(model="m").chat(MSG, schema=Answer)

    assert captured["response_format"] == {"type": "json_object"}
    assert isinstance(resp.parsed, Answer)
    assert resp.parsed.score == 9
    assert resp.parsed.reason == "good"


@pytest.mark.asyncio
async def test_litellm_without_schema_no_response_format(monkeypatch):
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _FakeCompletionResponse("plain")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    resp = await LiteLLMGateway(model="m").chat(MSG)

    assert "response_format" not in captured
    assert resp.parsed is None


@pytest.mark.asyncio
async def test_litellm_schema_invalid_json_raises(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _FakeCompletionResponse("not-json")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(StructuredOutputError):
        await LiteLLMGateway(model="m").chat(MSG, schema=Answer)


@pytest.mark.asyncio
async def test_litellm_schema_wrong_shape_raises(monkeypatch):
    async def fake_acompletion(**kwargs):
        return _FakeCompletionResponse('{"score": "high"}')  # wrong type + missing field

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    with pytest.raises(StructuredOutputError):
        await LiteLLMGateway(model="m").chat(MSG, schema=Answer)


# --------------------------------------------------------------------------- decorators
@pytest.mark.asyncio
async def test_fallback_forwards_schema():
    inner = _CapturingGateway(LLMResponse(content="x", parsed=Answer(score=1, reason="r")))
    resp = await FallbackLLMGateway([inner]).chat(MSG, schema=Answer)

    assert inner.seen[0]["schema"] is Answer
    assert isinstance(resp.parsed, Answer)


@pytest.mark.asyncio
async def test_fallback_omits_schema_for_legacy_inner():
    """Schema=None is not forwarded, so V3.0-era gateways keep working."""
    resp = await FallbackLLMGateway([_LegacyGateway()]).chat(MSG)

    assert resp.content == "legacy"


@pytest.mark.asyncio
async def test_fault_tolerant_degrades_structured_output_error():
    class Raising:
        async def chat(self, messages, model=None, tools=None, **kw) -> LLMResponse:
            raise StructuredOutputError("response failed validation against Answer")

    resp = await FaultTolerantGateway(Raising(), on_error="fallback").chat(MSG, schema=Answer)

    assert resp.content == "fallback"
    assert resp.parsed is None
    assert resp.tool_calls == []


@pytest.mark.asyncio
async def test_recording_strips_parsed_caller_keeps_it():
    inner = _CapturingGateway(LLMResponse(content='{"score": 5, "reason": "ok"}', parsed=Answer(score=5, reason="ok")))
    rec = RecordingGateway(inner)

    resp = await rec.chat(MSG, schema=Answer)

    assert isinstance(resp.parsed, Answer)
    assert rec.recorded[0].parsed is None
    assert rec.recorded[0].content == '{"score": 5, "reason": "ok"}'


@pytest.mark.asyncio
async def test_replay_rederives_parsed_from_content_and_schema():
    replay = ReplayGateway([LLMResponse(content='{"score": 7, "reason": "fine"}', usage=Usage())])

    resp = await replay.chat(MSG, schema=Answer)
    assert isinstance(resp.parsed, Answer)
    assert resp.parsed.score == 7

    replay2 = ReplayGateway([LLMResponse(content='{"score": 7, "reason": "fine"}', usage=Usage())])
    assert (await replay2.chat(MSG)).parsed is None  # no schema → no re-derivation


@pytest.mark.asyncio
async def test_replay_rederive_invalid_content_raises():
    replay = ReplayGateway([LLMResponse(content="bad")])

    with pytest.raises(StructuredOutputError):
        await replay.chat(MSG, schema=Answer)


@pytest.mark.asyncio
async def test_caching_keys_on_schema():
    inner = _CapturingGateway(LLMResponse(content="x", parsed=Answer(score=1, reason="r")))
    cache = CachingGateway(inner)

    class Other(BaseModel):
        value: int

    await cache.chat(MSG, schema=Answer)
    await cache.chat(MSG, schema=Other)
    await cache.chat(MSG, schema=Answer)

    assert cache.misses == 2
    assert cache.hits == 1


@pytest.mark.asyncio
async def test_tiered_router_forwards_schema():
    inner = _CapturingGateway(LLMResponse(content="x", parsed=Answer(score=1, reason="r")))
    router = TieredRouter(inner, cheap_model="c", expensive_model="e", classifier=lambda m: False)

    await router.chat(MSG, schema=Answer)

    assert router.last_model == "c"
    assert inner.seen[0]["schema"] is Answer


# --------------------------------------------------------------------------- persistence
def test_dump_load_recording_stays_raw_json(tmp_path):
    rec = LLMResponse(content='{"score": 3, "reason": "meh"}', usage=Usage())
    path = tmp_path / "rec.json"

    from mainspring.llm import dump_recording, load_recording

    dump_recording([rec], path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    assert "parsed" not in raw[0]
    loaded = load_recording(path)
    assert loaded[0].parsed is None
    assert loaded[0].content == rec.content


def test_llm_response_parsed_defaults_to_none():
    assert LLMResponse().parsed is None
