"""可观测性：``Metrics`` + ``Tracer`` 注入（缺失依赖时自动退化）。

要点：
- ``RuntimeDeps.metrics`` 接收 ``Metrics`` 协议；框架自带 ``InMemoryMetrics`` 便于本地观察。
- 运行时自动记录：run 状态/耗时、每步耗时、迭代数、token 数、工具调用次数/耗时。
- ``Tracer`` 依赖可选 ``opentelemetry-api``；未安装时所有方法都是 no-op，安装后由宿主配置 exporter。
运行：``python example/observability.py``
"""

import asyncio

from _shared import ScriptedLLM, finish, tool_call

from mainspring import AgentDefinition, AgentRuntime, RunContext, RuntimeDeps
from mainspring.observability import InMemoryMetrics, get_tracer
from mainspring.plugins.registry import PluginRegistry
from mainspring.tools.registry import build_default_registry


async def main() -> None:
    registry = PluginRegistry()
    for tool in build_default_registry().all():
        registry.register_tool(tool)

    metrics = InMemoryMetrics()
    tracer = get_tracer("example-agent")
    print("Tracer 是否可用:", tracer.enabled, "（未配置 exporter 时 span 不会导出）")

    llm = ScriptedLLM([
        tool_call("calculator", {"expression": "6*7"}),
        finish("6*7 = 42"),
    ])
    definition = AgentDefinition(model="mock-model", system_prompt="你是助手", plugins=registry.all())
    result = await AgentRuntime().run(
        definition, "计算 6*7", RunContext(run_id="obs-1"),
        RuntimeDeps(llm=llm, metrics=metrics, tracer=tracer),
    )

    print("run:", result.status, "| result:", result.result)
    print("\n计数器（名称 + 标签 = 值）:")
    for (name, labels), value in metrics.counters.items():
        print(f"  {name}{dict(labels)} = {value}")
    print("\n耗时观测（名称, 毫秒, 标签）:")
    for name, value, labels in metrics.observations:
        print(f"  {name}: {value} ms {labels}")


if __name__ == "__main__":
    asyncio.run(main())
