"""Deterministic tests for v3.5 pre-request context budgeting (no LLM, no network)."""

import math

import pytest
from conftest import MockLLMGateway, build_default_plugins

from mainspring import AgentDefinition, AgentRuntime, ContextPolicy, RunContext, RuntimeConfig, RuntimeDeps
from mainspring.core.context import (
    CharTokenEstimator,
    ContextBudgetExceeded,
    build_context,
    limit_tool_result,
)
from mainspring.llm.base import LLMResponse, ToolCall, Usage
from mainspring.plugins.base import ToolAsPlugin
from mainspring.tools.decorator import tool

SYSTEM = "You are a test assistant."


class _FixedEstimator:
    """Estimator counting message/tool list lengths — makes budget math exact in tests."""

    def estimate(self, messages, tools=None):
        return len(messages) + len(tools or [])


def _policy(**overrides) -> ContextPolicy:
    base = dict(
        context_window_tokens=10000,
        max_output_tokens=0,
        safety_margin_tokens=0,
        chars_per_token=3.0,
    )
    base.update(overrides)
    return ContextPolicy(**base)


def _history(n_turns: int, filler: str = "turn content") -> list[dict]:
    msgs = []
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"{filler} {i}"})
        msgs.append({"role": "assistant", "content": f"reply {i}"})
    return msgs


# ------------------------------------------------------------------ estimator + limiter
def test_char_estimator_counts_messages_and_tools():
    est = CharTokenEstimator(chars_per_token=2.0)
    one = '{"role": "user", "content": "ab"}'
    assert est.estimate([{"role": "user", "content": "ab"}], None) == math.ceil(len(one) / 2.0)


def test_char_estimator_rejects_bad_ratio():
    with pytest.raises(ValueError):
        CharTokenEstimator(chars_per_token=0)


def test_limit_tool_result_marks_incomplete():
    assert limit_tool_result("short", 100) == "short"
    out = limit_tool_result("x" * 500, 100)
    assert out.startswith("x" * 100)
    assert "truncated to 100 chars of 500" in out
    assert "incomplete" in out


# ------------------------------------------------------------------ builder unit tests
def test_builder_under_budget_passes_through_unchanged():
    history = _history(3)
    built = build_context(
        system_prompt=SYSTEM,
        plan=[{"id": 1, "description": "step one"}],
        memory=[{"content": "past fact"}],
        messages=history,
        tools=[{"type": "function", "function": {"name": "calculator"}}],
        policy=_policy(),
        estimator=_FixedEstimator(),
    )
    assert built.messages[0] == {"role": "system", "content": SYSTEM}
    assert built.messages[1]["content"].startswith("Current plan:")
    assert built.messages[2]["content"].startswith("Relevant past memories")
    assert built.messages[3:] == history
    assert built.tools[0]["function"]["name"] == "calculator"
    assert built.actions == []
    assert built.estimated_tokens == sum(built.breakdown.values())


def test_builder_drops_earliest_history_with_marker_and_keeps_pairing():
    history = []
    for i in range(6):
        history.append({"role": "user", "content": f"task {i}"})
        history.append({
            "role": "assistant",
            "content": f"calling {i}",
            "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "calc", "arguments": "{}"}}],
        })
        history.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i}"})
    built = build_context(
        system_prompt=SYSTEM,
        messages=history,
        tools=[],
        policy=_policy(context_window_tokens=10),
        estimator=_FixedEstimator(),
    )
    assert any("omitted by token budget" in m.get("content", "") for m in built.messages)
    assert any(m.get("role") == "user" and m.get("content") == "task 5" for m in built.messages)
    # no stranded tool messages: each tool message follows its assistant tool-call parent
    for i, m in enumerate(built.messages):
        if m.get("role") == "tool":
            prev = built.messages[i - 1]
            assert prev.get("role") == "assistant" and prev.get("tool_calls")
            assert prev["tool_calls"][0]["id"] == m["tool_call_id"]
    assert any("history: dropped" in a for a in built.actions)


def test_builder_raises_when_protected_content_exceeds():
    with pytest.raises(ContextBudgetExceeded) as excinfo:
        build_context(
            system_prompt=SYSTEM,
            messages=[{"role": "user", "content": "task"}],
            tools=[],
            policy=_policy(context_window_tokens=10),
        )
    exc = excinfo.value
    assert "context budget exceeded" in str(exc)
    assert exc.breakdown["system"] > 0
    assert exc.input_limit == 10


def test_builder_misconfigured_policy_raises():
    with pytest.raises(ContextBudgetExceeded, match="misconfigured"):
        build_context(
            system_prompt=SYSTEM,
            messages=[{"role": "user", "content": "task"}],
            tools=[],
            policy=_policy(context_window_tokens=1000, max_output_tokens=900, safety_margin_tokens=200),
        )


def test_tool_schema_budget_exceeded_raises_instead_of_truncating():
    tools = [{"type": "function", "function": {"name": f"tool{i}", "description": "d" * 50}} for i in range(5)]
    with pytest.raises(ContextBudgetExceeded, match="shrink the agent tool allowlist"):
        build_context(
            system_prompt=SYSTEM,
            messages=[{"role": "user", "content": "task"}],
            tools=tools,
            policy=_policy(tool_schema_token_budget=10),
        )


def test_memory_budget_trims_least_relevant_tail():
    memory = [{"content": f"fact {i}"} for i in range(3)]
    built = build_context(
        system_prompt=SYSTEM,
        memory=memory,
        messages=[{"role": "user", "content": "task"}],
        tools=[],
        policy=_policy(memory_token_budget=95, chars_per_token=1.0),
        estimator=CharTokenEstimator(chars_per_token=1.0),
    )
    mem_msg = next(m for m in built.messages if m["content"].startswith("Relevant past memories"))
    assert "fact 0" in mem_msg["content"] and "fact 1" in mem_msg["content"]
    assert "fact 2" not in mem_msg["content"]
    assert any("kept 2 of 3" in a for a in built.actions)


def test_custom_estimator_drives_drop_decisions():
    history = _history(20)
    built = build_context(
        system_prompt=SYSTEM,
        messages=history,
        tools=[],
        policy=_policy(context_window_tokens=10),
        estimator=_FixedEstimator(),
    )
    # input_limit=10 counts messages: 1 system + 1 marker + 8 kept history
    assert built.estimated_tokens <= 10
    assert len(built.messages) == 10
    assert any("omitted by token budget" in m.get("content", "") for m in built.messages)


# ------------------------------------------------------------------ runtime end-to-end
def _definition(policy: ContextPolicy | None, **runtime) -> AgentDefinition:
    return AgentDefinition(
        model="test-model",
        system_prompt=SYSTEM,
        plugins=build_default_plugins(),
        runtime=RuntimeConfig(context_policy=policy, **runtime),
    )


async def test_run_over_budget_fails_without_calling_the_llm():
    llm = MockLLMGateway(responses=[LLMResponse(content="never", usage=Usage())])
    deps = RuntimeDeps(llm=llm)
    result = await AgentRuntime().run(
        _definition(_policy(context_window_tokens=10)), "task", RunContext(), deps
    )
    assert result.status == "failed"
    assert "context budget exceeded" in result.error
    assert llm.calls == []  # the oversized request was never sent


async def test_run_compacts_history_and_exposes_diagnostics():
    llm = MockLLMGateway(responses=[LLMResponse(content="done", usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15))])
    deps = RuntimeDeps(llm=llm, token_estimator=_FixedEstimator())
    history = _history(25, filler="a fairly long turn of conversation")

    result = await AgentRuntime().run(
        _definition(_policy(context_window_tokens=20)), "the task", RunContext(), deps, history=history
    )
    assert result.status == "completed"
    sent = llm.calls[0]["messages"]
    assert any("omitted by token budget" in m.get("content", "") for m in sent)
    assert len([m for m in sent if m.get("role") in ("user", "assistant")]) < len(history)
    decide_step = result.steps[0]
    assert decide_step.node == "decide"
    ctx = decide_step.output["context"]
    assert ctx["estimated_tokens"] <= ctx["input_limit"]
    assert any("history: dropped" in a for a in ctx["actions"])


async def test_run_limits_oversized_tool_result_with_marker():
    @tool(description="returns a huge text")
    def bigtext() -> str:
        return "x" * 5000

    llm = MockLLMGateway(responses=[
        LLMResponse(tool_calls=[ToolCall(id="c1", name="bigtext", arguments={})], usage=Usage(prompt_tokens=50, completion_tokens=10, total_tokens=60)),
        LLMResponse(content="finished", usage=Usage(prompt_tokens=50, completion_tokens=10, total_tokens=60)),
    ])
    deps = RuntimeDeps(llm=llm)
    policy = _policy(context_window_tokens=0, tool_result_chars=100)

    result = await AgentRuntime().run(
        AgentDefinition(
            model="test-model",
            system_prompt=SYSTEM,
            plugins=[*build_default_plugins(), ToolAsPlugin(bigtext)],
            runtime=RuntimeConfig(context_policy=policy),
        ),
        "get text",
        RunContext(),
        deps,
    )
    assert result.status == "completed"
    execute_step = next(s for s in result.steps if s.node == "execute")
    assert "truncated to 100 chars of 5000" in execute_step.output["result"]


async def test_plan_node_budget_failure_fails_the_run():
    llm = MockLLMGateway(responses=[LLMResponse(content="[]", usage=Usage())])
    deps = RuntimeDeps(llm=llm)
    # chars_per_token=1 makes the long planner system prompt exceed the tiny input limit
    policy = _policy(
        context_window_tokens=2600, max_output_tokens=2000, safety_margin_tokens=520, chars_per_token=1.0
    )

    result = await AgentRuntime().run(
        _definition(policy, enable_planning=True), "plan me", RunContext(), deps
    )
    assert result.status == "failed"
    assert "context budget exceeded" in result.error
    assert llm.calls == []


async def test_run_without_policy_sends_history_untouched():
    llm = MockLLMGateway(responses=[LLMResponse(content="ok", usage=Usage())])
    deps = RuntimeDeps(llm=llm)
    history = _history(5)

    result = await AgentRuntime().run(
        _definition(None), "the task", RunContext(), deps, history=history
    )
    assert result.status == "completed"
    sent = llm.calls[0]["messages"]
    assert not any("omitted by token budget" in m.get("content", "") for m in sent)
    assert sent[-len(history):] == history
