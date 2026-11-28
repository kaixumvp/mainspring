"""请求级上下文预算（v3.5）：``ContextPolicy`` + ``build_context`` + 运行时集成。

要点：
- ``build_context`` 是纯函数：在每次 Plan/Decide 前，把 system / plan / memory / history
  组装成预算内、可追溯的 Context View，并返回分项计数与压缩动作。
- 压缩顺序确定：记忆尾部裁剪 → 工具 schema 超限**响亮失败** → 历史前端丢弃（保留 system 头、
  最后一条用户消息与工具配对，裁剪处插省略标记）。
- 仍超预算则抛 ``ContextBudgetExceeded``（准入失败不重试）；运行时会把 run 置为 failed。
- ``RuntimeConfig.context_policy`` 默认 ``None``（完全关闭）；配置后 Decide 自动走预算。
运行：``python example/context_budget.py``
"""

import asyncio

from _shared import ScriptedLLM, finish, tool_call

from mainspring import (
    AgentDefinition,
    AgentRuntime,
    RunContext,
    RuntimeConfig,
    RuntimeDeps,
    SlidingWindowContext,
)
from mainspring.core.context import (
    ContextBudgetExceeded,
    ContextPolicy,
    build_context,
    limit_tool_result,
)
from mainspring.plugins.registry import PluginRegistry
from mainspring.tools.base import BaseTool, ToolResult


class LongOutputTool(BaseTool):
    """返回超长文本，用来观察工具结果截断。"""

    name = "long_output"
    description = "返回一段很长的日志"
    permission = "read"
    parameters = {"type": "object", "properties": {}, "required": []}

    async def run(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="ERROR 日志行；" * 200)


def build_budgeted_view() -> None:
    """纯逻辑：长历史 + 多条记忆，观察确定性压缩。"""
    policy = ContextPolicy(
        context_window_tokens=160,   # 总窗口
        max_output_tokens=20,        # 给模型回复预留
        safety_margin_tokens=20,     # 吸收估算误差
        memory_token_budget=33,      # 记忆超过预算时从尾部（最不相关）裁剪
        tool_schema_token_budget=1000,
    )
    history = [
        {"role": "user", "content": f"第{i}轮：这是一段用于撑大上下文的历史消息。"}
        for i in range(1, 9)
    ] + [{"role": "user", "content": "现在请总结以上内容"}]
    memory = [
        {"content": "用户偏好中文回答"},
        {"content": "用户是后端工程师"},
        {"content": "用户喜欢深色主题"},
    ]

    built = build_context(
        system_prompt="你是严谨的助手。",
        plan=None,
        memory=memory,
        messages=history,
        tools=[],
        policy=policy,
    )
    print("== build_context ==")
    print("输入预算:", built.input_limit, "| 估算 tokens:", built.estimated_tokens)
    print("分项（system/memory/history/tools）:", built.breakdown)
    print("压缩动作:", built.actions)
    print("组装后的首条消息:", built.messages[0]["content"])

    print("工具结果截断:", limit_tool_result("x" * 120, 40))


def tool_schema_over_budget() -> None:
    """工具白名单过大是配置错误：绝不静默截断，直接响亮失败。"""
    huge_schema = {
        "type": "function",
        "function": {
            "name": "huge",
            "description": "参数特别多的工具",
            "parameters": {
                "type": "object",
                "properties": {f"arg_{i}": {"type": "string", "description": "参数说明" * 5} for i in range(20)},
                "required": [],
            },
        },
    }
    policy = ContextPolicy(
        context_window_tokens=200,
        max_output_tokens=10,
        safety_margin_tokens=10,
        tool_schema_token_budget=5,
    )
    print("\n== 工具 schema 超预算 ==")
    try:
        build_context(
            system_prompt="system",
            messages=[{"role": "user", "content": "hi"}],
            tools=[huge_schema],
            policy=policy,
        )
    except ContextBudgetExceeded as exc:
        print("ContextBudgetExceeded:", exc)


def sliding_window_demo() -> None:
    """旧的 ContextManager 参考实现：滑窗 + 工具结果截断（宿主自定义扩展仍可注入）。"""
    messages = [{"role": "system", "content": "sys"}] + [
        {"role": "user", "content": f"m{i}"} for i in range(10)
    ]
    fitted = SlidingWindowContext(max_messages=5).fit(messages)
    print("\n== SlidingWindowContext ==")
    print("保留条数:", len(fitted), "| 内容:", [m["content"] for m in fitted])


async def runtime_integration() -> None:
    """把 ContextPolicy 配到 RuntimeConfig：Decide 自动预算，工具结果自动截断。"""
    registry = PluginRegistry()
    registry.register_tool(LongOutputTool())

    policy = ContextPolicy(
        context_window_tokens=4000,
        max_output_tokens=200,
        safety_margin_tokens=100,
        tool_result_chars=60,  # 工具结果进上下文前截断
    )
    definition = AgentDefinition(
        model="mock-model",
        system_prompt="你是运维助手",
        plugins=registry.all(),
        runtime=RuntimeConfig(context_policy=policy),
    )
    llm = ScriptedLLM([
        tool_call("long_output", {}),
        finish("日志已汇总。"),
    ])
    result = await AgentRuntime().run(
        definition, "分析今天的错误日志", RunContext(run_id="budget-1"), RuntimeDeps(llm=llm)
    )

    execute_step = next(s for s in result.steps if s.node == "execute")
    decide_step = next(s for s in result.steps if s.node == "decide")
    print("\n== RuntimeConfig(context_policy=...) ==")
    print("status:", result.status, "| result:", result.result)
    print("execute 结果前 80 字:", execute_step.output["result"][:80])
    print("是否被截断:", "truncated" in execute_step.output["result"])
    print("decide 的上下文诊断:", decide_step.output["context"])


async def main() -> None:
    build_budgeted_view()
    tool_schema_over_budget()
    sliding_window_demo()
    await runtime_integration()


if __name__ == "__main__":
    asyncio.run(main())
