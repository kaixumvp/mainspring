"""DevUI tests: offline demo agent + stdlib HTTP server (zero external dependencies).

The server tests use only ``urllib`` against a port-0 instance with instant streaming
(``stream_delay=0``). An SSE reader thread collects events while the test drives the
run/approve HTTP endpoints.
"""

from __future__ import annotations

import json
import queue
import threading
import urllib.request

import pytest

from mainspring.core.runtime import AgentRuntime, RunContext
from mainspring.devui.demo import (
    DEMO_TASK,
    ScriptedGateway,
    build_demo_definition,
    build_demo_deps,
    build_script,
)
from mainspring.devui.server import DevUIServer
from mainspring.llm.stream import accumulate_response
from mainspring.ports import ChatChunk


# --------------------------------------------------------------------------- demo agent
@pytest.mark.asyncio
async def test_demo_run_pauses_on_admin_tool_then_completes():
    """Full offline loop: calculator runs, transfer pauses, resume after approval finishes."""
    runtime = AgentRuntime()
    definition = build_demo_definition()
    deps = build_demo_deps(stream_delay=0)
    context = RunContext(tenant_id="devui", user_id="demo", session_id="devui")

    paused = await runtime.run(definition, DEMO_TASK, context, deps)
    assert paused.status == "awaiting_human"
    assert paused.pending_approvals == [
        {"resource_type": "tool", "resource_id": "tool_transfer", "action": "transfer"}
    ]
    assert [s.action for s in paused.steps if s.node == "decide"] == ["calculator", "transfer"]
    assert [s.action for s in paused.steps if s.node == "execute"] == [None, None]

    resumed = await runtime.resume(
        definition, paused.steps[-1].run_id, [paused.pending_approvals[0]["resource_id"]], deps
    )
    assert resumed.status == "completed"
    assert resumed.result == "转账完成！123×45=5535，已把 100 元转给 alice。"
    assert resumed.pending_approvals == []


@pytest.mark.asyncio
async def test_scripted_gateway_streams_content_word_by_word():
    resp = build_script()[2]  # the FINISH turn: content only
    gw = ScriptedGateway([resp], delay=0)
    chunks = [c async for c in gw.chat_stream([])]
    content_chunks = [c for c in chunks if c.content_delta]
    assert len(content_chunks) > 1  # streamed in pieces, not one blob
    folded = accumulate_response(chunks)
    assert folded.content == resp.content
    assert folded.tool_calls == []
    assert folded.usage.total_tokens == resp.usage.total_tokens


@pytest.mark.asyncio
async def test_scripted_gateway_streams_tool_calls():
    chunks = [c async for c in ScriptedGateway(build_script(), delay=0).chat_stream([])]
    assert isinstance(chunks[0], ChatChunk)
    assert chunks[0].tool_call_deltas  # first scripted turn is the calculator call
    folded = accumulate_response(chunks)
    assert folded.content in (None, "")
    assert folded.tool_calls[0].name == "calculator"
    assert folded.tool_calls[0].arguments == {"expression": "123*45"}
    assert folded.usage.total_tokens == 138


@pytest.mark.asyncio
async def test_script_exhaustion_raises():
    gw = ScriptedGateway(build_script())
    for _ in build_script():
        await gw.chat([])
    with pytest.raises(RuntimeError):
        await gw.chat([])


# --------------------------------------------------------------------------- HTTP server
def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _run_id(result) -> str:
    for step in result.get("steps", []):
        if step.get("run_id"):
            return step["run_id"]
    raise AssertionError(f"no run_id in result: {result}")


@pytest.fixture()
def server():
    srv = DevUIServer(port=0, stream_delay=0)
    srv.start()
    yield srv
    srv.stop()


def test_index_page_served(server):
    with urllib.request.urlopen(server.url("/"), timeout=10) as resp:
        assert resp.status == 200
        body = resp.read().decode("utf-8")
    assert "mainspring dev" in body
    assert "EventSource" in body


def test_run_and_approve_over_http(server):
    status, paused = _post(server.url("/run"), {"task": DEMO_TASK})
    assert status == 200
    assert paused["status"] == "awaiting_human"
    assert paused["pending_approvals"][0]["resource_id"] == "tool_transfer"

    status, resumed = _post(
        server.url("/approve"),
        {"run_id": _run_id(paused), "approvals": [a["resource_id"] for a in paused["pending_approvals"]]},
    )
    assert status == 200
    assert resumed["status"] == "completed"
    assert "5535" in resumed["result"]


def test_approve_unknown_run_404s(server):
    status, body = _post(server.url("/approve"), {"run_id": "run_nope", "approvals": []})
    assert status == 404
    assert "unknown run_id" in body["error"]


def test_sse_stream_carries_the_full_event_sequence(server):
    events: queue.Queue = queue.Queue()
    threading.Thread(target=_sse_reader, args=(server.url("/events"), events), daemon=True).start()
    assert events.get(timeout=10)["event"] == "_reader.ready"

    _, paused = _post(server.url("/run"), {"task": DEMO_TASK})
    _, resumed = _post(
        server.url("/approve"),
        {"run_id": _run_id(paused), "approvals": [a["resource_id"] for a in paused["pending_approvals"]]},
    )
    assert resumed["status"] == "completed"

    seen: list[dict] = []
    for _ in range(200):
        try:
            seen.append(events.get(timeout=0.2))
        except queue.Empty:
            break
        if seen[-1].get("event") == "run.completed":
            break
    names = [e.get("event") for e in seen]
    assert "run.started" in names
    assert "step.completed" in names
    assert "message.delta" in names
    assert "run.awaiting_human" in names
    assert "run.approved" in names
    assert "run.completed" in names
    assert [e for e in seen if e.get("event") == "message.delta"]  # final answer streamed


def _sse_reader(url, events):
    """Block on the SSE stream, pushing parsed events onto the queue."""
    with urllib.request.urlopen(url, timeout=60) as resp:
        assert resp.status == 200
        events.put({"event": "_reader.ready"})
        for line in resp:
            if line.startswith(b"data: "):
                events.put(json.loads(line[6:]))
