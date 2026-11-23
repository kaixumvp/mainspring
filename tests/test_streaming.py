import pytest
from conftest import MockLLMGateway, MockStreamingGateway, build_default_plugins

from mainspring import AgentDefinition, AgentRuntime, RunContext, RuntimeDeps
from mainspring.core.resilience import RetryPolicy
from mainspring.llm.base import LLMResponse, Usage
from mainspring.llm.replay import RecordingGateway, ReplayGateway
from mainspring.llm.routing import FallbackLLMGateway, FaultTolerantGateway
from mainspring.llm.stream import accumulate_response
from mainspring.ports import ChatChunk, ToolCallDelta


def _definition(**overrides) -> AgentDefinition:
    kwargs = {"model": "m", "system_prompt": "s", "plugins": build_default_plugins()}
    kwargs.update(overrides)
    return AgentDefinition(**kwargs)


# --------------------------------------------------------------------------- accumulator
def test_accumulate_content_and_usage():
    chunks = [
        ChatChunk(content_delta="3*8"),
        ChatChunk(content_delta="+100 = "),
        ChatChunk(content_delta="124", usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)),
    ]
    resp = accumulate_response(chunks)
    assert resp.content == "3*8+100 = 124"
    assert resp.tool_calls == []
    assert resp.usage.total_tokens == 3


def test_accumulate_tool_calls_by_index():
    chunks = [
        ChatChunk(tool_call_deltas=[ToolCallDelta(index=0, id="c1", name="calculator")]),
        ChatChunk(tool_call_deltas=[ToolCallDelta(index=0, arguments_delta='{"expression":')],
                  ),
        ChatChunk(tool_call_deltas=[ToolCallDelta(index=0, arguments_delta='"3*8"}')]),
        ChatChunk(usage=Usage(total_tokens=7)),
    ]
    resp = accumulate_response(chunks)
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == "c1"
    assert resp.tool_calls[0].name == "calculator"
    assert resp.tool_calls[0].arguments == {"expression": "3*8"}
    assert resp.usage.total_tokens == 7


def test_accumulate_malformed_arguments_degrade_to_empty():
    chunks = [
        ChatChunk(tool_call_deltas=[ToolCallDelta(index=0, id="c1", name="t", arguments_delta="{oops")]),
    ]
    resp = accumulate_response(chunks)
    assert resp.tool_calls[0].arguments == {}


# --------------------------------------------------------------------------- runtime streaming
@pytest.mark.asyncio
async def test_runtime_streams_deltas_to_event_sink():
    events = []

    async def sink(event):
        events.append(event)

    llm = MockStreamingGateway([LLMResponse(content="3*8+100 = 124", usage=Usage(total_tokens=9))])
    deps = RuntimeDeps(llm=llm, event_sink=sink, retry=RetryPolicy(max_retries=0))

    res = await AgentRuntime().run(_definition(), "计算", RunContext(run_id="r1"), deps)

    assert res.status == "completed"
    assert res.result == "3*8+100 = 124"
    deltas = [e.data["delta"] for e in events if e.event == "message.delta"]
    assert deltas and "".join(deltas) == "3*8+100 = 124"
    assert all(e.run_id == "r1" for e in events if e.event == "message.delta")
    # token_usage merged from the final usage chunk
    assert res.token_usage["total_tokens"] == 9


@pytest.mark.asyncio
async def test_runtime_does_not_stream_without_event_sink():
    llm = MockStreamingGateway([LLMResponse(content="done", usage=Usage())])
    deps = RuntimeDeps(llm=llm, retry=RetryPolicy(max_retries=0))

    res = await AgentRuntime().run(_definition(), "hi", RunContext(run_id="r1"), deps)

    assert res.status == "completed"
    assert llm.calls[0]["streamed"] is False


@pytest.mark.asyncio
async def test_non_streaming_gateway_unaffected():
    llm = MockLLMGateway([LLMResponse(content="done", usage=Usage())])
    events = []

    async def sink(event):
        events.append(event)

    deps = RuntimeDeps(llm=llm, event_sink=sink, retry=RetryPolicy(max_retries=0))
    res = await AgentRuntime().run(_definition(), "hi", RunContext(run_id="r1"), deps)

    assert res.status == "completed"
    assert not [e for e in events if e.event == "message.delta"]


# --------------------------------------------------------------------------- decorators
@pytest.mark.asyncio
async def test_fallback_stream_failover_before_first_chunk():
    class BrokenStream:
        async def chat(self, messages, model=None, tools=None):
            raise RuntimeError("no chat")

        def chat_stream(self, messages, model=None, tools=None):
            return self._stream()

        async def _stream(self):
            raise RuntimeError("vendor down")
            yield ChatChunk()  # pragma: no cover — makes this an async generator

    class WorkingStream:
        def chat_stream(self, messages, model=None, tools=None):
            return self._stream()

        async def _stream(self):
            yield ChatChunk(content_delta="ok")

    gw = FallbackLLMGateway([BrokenStream(), WorkingStream()])
    chunks = [c async for c in gw.chat_stream([{"role": "user", "content": "x"}])]
    assert [c.content_delta for c in chunks] == ["ok"]


@pytest.mark.asyncio
async def test_fallback_stream_reraises_mid_stream():
    class MidStreamFailure:
        def chat_stream(self, messages, model=None, tools=None):
            return self._stream()

        async def _stream(self):
            yield ChatChunk(content_delta="part")
            raise RuntimeError("connection dropped")

    gw = FallbackLLMGateway([MidStreamFailure()])
    with pytest.raises(RuntimeError, match="connection dropped"):
        _ = [c async for c in gw.chat_stream([{"role": "user", "content": "x"}])]


@pytest.mark.asyncio
async def test_fault_tolerant_stream_degrades_before_first_chunk():
    class BrokenStream:
        def chat_stream(self, messages, model=None, tools=None):
            raise RuntimeError("vendor down")
            yield ChatChunk()  # pragma: no cover

    gw = FaultTolerantGateway(BrokenStream(), on_error="fallback text")
    chunks = [c async for c in gw.chat_stream([{"role": "user", "content": "x"}])]
    assert [c.content_delta for c in chunks] == ["fallback text"]


# --------------------------------------------------------------------------- record / replay
@pytest.mark.asyncio
async def test_recording_gateway_records_streamed_response():
    class Inner:
        def chat_stream(self, messages, model=None, tools=None):
            return self._stream()

        async def _stream(self):
            yield ChatChunk(content_delta="he")
            yield ChatChunk(content_delta="llo")
            yield ChatChunk(usage=Usage(total_tokens=5))

    rec = RecordingGateway(Inner())
    chunks = [c async for c in rec.chat_stream([{"role": "user", "content": "x"}])]
    assert "".join(c.content_delta for c in chunks) == "hello"
    assert len(rec.recorded) == 1
    assert rec.recorded[0].content == "hello"
    assert rec.recorded[0].usage.total_tokens == 5


@pytest.mark.asyncio
async def test_replay_gateway_streams_recorded_response_as_single_chunk():
    replay = ReplayGateway([LLMResponse(content="world", usage=Usage(total_tokens=3))])
    chunks = [c async for c in replay.chat_stream([{"role": "user", "content": "x"}])]
    assert len(chunks) == 1
    assert chunks[0].content_delta == "world"
    assert chunks[0].usage.total_tokens == 3
    assert replay.calls == 1
