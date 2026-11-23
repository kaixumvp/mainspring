"""Offline demo agent for the DevUI: deterministic, no API key required.

A ``ScriptedGateway`` plays back three scripted Decide turns so the full
Plan → Decide → Execute → Observe loop (plus the human-in-the-loop pause) runs
without any LLM backend:

  1. decide → ``calculator("123*45")``      (read permission, executes directly)
  2. decide → ``transfer(100, "alice")``    (admin permission → awaiting_human)
  3. decide → FINISH with the final answer  (after approvals are granted)

``build_demo_definition()`` / ``build_demo_deps()`` are the drop-in
``AgentDefinition`` + ``RuntimeDeps`` the DevUI server hosts.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator

from mainspring.core.checkpointer import InMemoryCheckpointer
from mainspring.core.runtime import AgentDefinition, RuntimeDeps
from mainspring.plugins.base import ToolAsPlugin
from mainspring.ports import (
    ChatChunk,
    LLMResponse,
    PermissionContext,
    PermissionDecision,
    ToolCall,
    ToolCallDelta,
    Usage,
    check_permission,
)
from mainspring.tools.builtin.calculator import CalculatorTool
from mainspring.tools.decorator import tool

DEMO_TASK = "计算 123×45，然后把 100 元转给 alice。"
DEMO_MODEL = "scripted-demo"


class ScriptedGateway:
    """A scripted LLM gateway: chat() pops the next response, chat_stream() streams it.

    Content streams in word/character pieces (so the DevUI can show token-level
    ``message.delta`` events); tool calls arrive as one complete delta per call.
    ``delay`` throttles the stream (0 in tests, ~0.02s in the live DevUI).
    """

    def __init__(self, responses: list[LLMResponse], *, delay: float = 0.02) -> None:
        self._responses = list(responses)
        self.delay = delay
        self.calls = 0

    async def chat(self, messages, model=None, tools=None, schema=None) -> LLMResponse:
        if not self._responses:
            raise RuntimeError("script exhausted: the demo agent made more LLM calls than scripted")
        self.calls += 1
        return self._responses.pop(0)

    def chat_stream(self, messages, model=None, tools=None) -> AsyncIterator[ChatChunk]:
        return self._stream(messages, model, tools)

    async def _stream(self, messages, model, tools):
        resp = await self.chat(messages, model=model, tools=tools)
        if resp.tool_calls:
            await asyncio.sleep(self.delay)
            yield ChatChunk(
                tool_call_deltas=[
                    ToolCallDelta(
                        index=i,
                        id=tc.id,
                        name=tc.name,
                        arguments_delta=json.dumps(tc.arguments, ensure_ascii=False),
                    )
                    for i, tc in enumerate(resp.tool_calls)
                ]
            )
        for piece in re.findall(r"\S+\s*", resp.content or ""):
            await asyncio.sleep(self.delay)
            yield ChatChunk(content_delta=piece)
        yield ChatChunk(usage=resp.usage)


@tool(
    permission="admin",
    description="Transfer money from the demo account to another person. Requires human approval.",
)
def transfer(amount: float, to: str) -> str:
    """转移资金（演示用高危工具）。"""
    return f"已向 {to} 转账 {amount:g} 元"


class DemoPermissionChecker:
    """Wraps ``check_permission`` for the demo principal.

    The demo user is non-admin, holds ``run:execute``, and has an explicit allow-grant
    for the ``transfer`` tool — so that tool is *authorized* but, being
    ``permission="admin"``, still pauses for human approval.
    """

    def __init__(self) -> None:
        self._ctx = PermissionContext(
            permission_codes={"run:execute"},
            grant_keys={("tool", "transfer", "allow")},
        )

    def check(self, ctx, action, resource_type, resource_id,
              resource_permission="read", requires_approval=False, approved=None) -> PermissionDecision:
        return check_permission(
            self._ctx, action, resource_type, resource_id,
            resource_permission=resource_permission,
            requires_approval=requires_approval,
            approved=approved,
        )


def build_demo_definition() -> AgentDefinition:
    return AgentDefinition(
        agent_id="devui-demo",
        model=DEMO_MODEL,
        system_prompt="你是一个演示助手。需要算术时调用 calculator，转账时调用 transfer。",
        plugins=[ToolAsPlugin(CalculatorTool()), ToolAsPlugin(transfer)],
    )


def build_script() -> list[LLMResponse]:
    return [
        LLMResponse(
            tool_calls=[ToolCall(id="call_calc", name="calculator", arguments={"expression": "123*45"})],
            usage=Usage(prompt_tokens=120, completion_tokens=18, total_tokens=138),
        ),
        LLMResponse(
            tool_calls=[ToolCall(id="call_transfer", name="transfer", arguments={"amount": 100, "to": "alice"})],
            usage=Usage(prompt_tokens=180, completion_tokens=20, total_tokens=200),
        ),
        LLMResponse(
            content="转账完成！123×45=5535，已把 100 元转给 alice。",
            usage=Usage(prompt_tokens=240, completion_tokens=24, total_tokens=264),
        ),
    ]


def build_demo_deps(event_sink=None, *, stream_delay: float = 0.02, checkpointer=None) -> RuntimeDeps:
    """RuntimeDeps for one demo run. Inject a shared ``checkpointer`` so a host can
    ``resume`` a paused run across requests (the DevUI server does exactly that)."""
    return RuntimeDeps(
        llm=ScriptedGateway(build_script(), delay=stream_delay),
        permission_checker=DemoPermissionChecker(),
        permission_context=PermissionContext(permission_codes={"run:execute"}),
        event_sink=event_sink,
        checkpointer=checkpointer or InMemoryCheckpointer(),
    )


__all__ = [
    "DEMO_MODEL",
    "DEMO_TASK",
    "DemoPermissionChecker",
    "ScriptedGateway",
    "build_demo_definition",
    "build_demo_deps",
    "build_script",
]
