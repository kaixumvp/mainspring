import pytest

from conftest import MockLLMGateway, build_default_plugins
from mainspring import AgentDefinition, AgentRuntime, RunContext, RuntimeConfig, RuntimeDeps
from mainspring.context import SlidingWindowContext
from mainspring.core.resilience import RetryPolicy
from mainspring.llm.base import LLMResponse, Usage


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def test_under_limit_returns_truncated_only():
    cm = SlidingWindowContext(max_messages=10)
    msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
    assert cm.fit(msgs) == msgs


def test_over_limit_keeps_head_system_and_newest():
    cm = SlidingWindowContext(max_messages=4)
    msgs = [_msg("system", "be nice")] + [_msg("user", f"u{i}") for i in range(6)]
    out = cm.fit(msgs)
    assert out[0] == msgs[0]
    assert any("omitted by context compaction" in m["content"] for m in out)
    assert out[-1] == msgs[-1]
    assert len(out) <= 5  # marker adds one on top of the window


def test_cut_never_strands_leading_tool_messages():
    cm = SlidingWindowContext(max_messages=3)
    msgs = [_msg("user", "old")] + [_msg("tool", "t1"), _msg("tool", "t2")] + [_msg("assistant", "a"), _msg("user", "latest")]
    out = cm.fit(msgs)
    # cut advances past the tool messages, so the kept tail starts at the assistant turn
    assert out[0]["role"] == "system"  # the omission marker
    assert out[1] == msgs[3]  # assistant "a", not a tool message
    assert out[-1] == msgs[-1]


def test_tool_results_truncated_even_under_limit():
    cm = SlidingWindowContext(max_messages=50, max_tool_chars=10)
    long_tool = _msg("tool", "x" * 100)
    out = cm.fit([long_tool])
    assert out[0]["content"].startswith("x" * 10)
    assert "truncated 90 chars" in out[0]["content"]


def test_all_system_messages_returned_intact():
    cm = SlidingWindowContext(max_messages=2)
    msgs = [_msg("system", "a"), _msg("system", "b")]
    assert cm.fit(msgs) == msgs


# --------------------------------------------------------------------------- runtime integration
def _def(**overrides) -> AgentDefinition:
    kw = {"model": "m", "system_prompt": "s", "plugins": build_default_plugins()}
    kw.update(overrides)
    return AgentDefinition(**kw)


@pytest.mark.asyncio
async def test_runtime_compacts_history_with_max_history_messages():
    llm = MockLLMGateway([LLMResponse(content="done", usage=Usage())])
    deps = RuntimeDeps(llm=llm, retry=RetryPolicy(max_retries=0))

    history = [{"role": "user", "content": f"old question {i}"} for i in range(20)]
    history.append({"role": "user", "content": "current question"})

    res = await AgentRuntime().run(
        _def(runtime=RuntimeConfig(max_history_messages=4)), "current question",
        RunContext(run_id="r1"), deps, history=history,
    )

    assert res.status == "completed"
    sent = llm.calls[0]["messages"]
    user_msgs = [m for m in sent if m.get("role") == "user"]
    assert len(user_msgs) <= 5
    assert user_msgs[-1]["content"] == "current question"


@pytest.mark.asyncio
async def test_runtime_uses_injected_context_manager():
    class RecordingContext:
        def __init__(self) -> None:
            self.calls = 0

        def fit(self, messages):
            self.calls += 1
            return [m for m in messages if m.get("role") != "user" or m["content"] == "keep"]

    llm = MockLLMGateway([LLMResponse(content="done", usage=Usage())])
    ctx = RecordingContext()
    deps = RuntimeDeps(llm=llm, context_manager=ctx, retry=RetryPolicy(max_retries=0))

    res = await AgentRuntime().run(
        _def(), "keep", RunContext(run_id="r1"), deps,
        history=[{"role": "user", "content": "drop me"}, {"role": "user", "content": "keep"}],
    )

    assert res.status == "completed"
    assert ctx.calls >= 1
    sent = llm.calls[0]["messages"]
    assert not any(m.get("content") == "drop me" for m in sent)


@pytest.mark.asyncio
async def test_runtime_no_compaction_by_default():
    llm = MockLLMGateway([LLMResponse(content="done", usage=Usage())])
    deps = RuntimeDeps(llm=llm, retry=RetryPolicy(max_retries=0))

    history = [{"role": "user", "content": f"old {i}"} for i in range(60)]
    res = await AgentRuntime().run(_def(), "hi", RunContext(run_id="r1"), deps, history=history)

    assert res.status == "completed"
    sent = llm.calls[0]["messages"]
    assert sum(1 for m in sent if str(m.get("content", "")).startswith("old")) == 60
