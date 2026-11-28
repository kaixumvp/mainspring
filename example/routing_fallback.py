"""多厂商路由与容错：``FallbackLLMGateway`` / ``FaultTolerantGateway``。

要点：
- 两者都是普通 ``LLMGateway`` 实现，可与 ``CachingGateway`` / ``TieredRouter`` 任意组合。
- ``FallbackLLMGateway`` 按顺序尝试多个厂商，主厂商失败自动切到下一个。
- ``FaultTolerantGateway`` 把异常/畸形响应当场降级为合法空响应，适合“宁可退化不可崩溃”的链路。
- 流式 failover 只在**首个 chunk 之前**发生；已吐出 delta 后再失败会重新抛出，避免重复输出。
运行：``python example/routing_fallback.py``
"""

import asyncio

from mainspring import AgentDefinition, AgentRuntime, RunContext, RuntimeDeps
from mainspring.llm.routing import FallbackLLMGateway, FaultTolerantGateway
from mainspring.ports import LLMResponse, Usage


class FailingGateway:
    """模拟主厂商故障。"""

    async def chat(self, messages, model=None, tools=None, schema=None) -> LLMResponse:
        raise RuntimeError("主厂商 503 Service Unavailable")


class BackupGateway:
    """备用厂商；``_model`` 会被 ``last_used`` 记录。"""

    _model = "backup-model"

    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    async def chat(self, messages, model=None, tools=None, schema=None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=self._text, usage=Usage(total_tokens=3))


async def fallback_demo() -> None:
    backup = BackupGateway("备用厂商已接管。")
    router = FallbackLLMGateway([FailingGateway(), backup])
    resp = await router.chat([{"role": "user", "content": "你好"}])
    print("== FallbackLLMGateway ==")
    print("响应:", resp.content, "| last_used:", router.last_used, "| 备用调用次数:", backup.calls)


async def fault_tolerant_demo() -> None:
    safe = FaultTolerantGateway(FailingGateway(), on_error="服务降级：请稍后重试。")
    resp = await safe.chat([{"role": "user", "content": "你好"}])
    print("\n== FaultTolerantGateway ==")
    print("content:", resp.content, "| usage:", resp.usage)


async def in_agent() -> None:
    """把 fallback 链直接当作 RuntimeDeps.llm：主厂商失败时 Agent 仍能完成。"""
    backup = BackupGateway("已切换备用模型完成回答。")
    llm = FallbackLLMGateway([FailingGateway(), backup])
    definition = AgentDefinition(model="mock-model", system_prompt="你是助手", plugins=[])
    result = await AgentRuntime().run(
        definition, "继续回答", RunContext(run_id="fallback-agent"), RuntimeDeps(llm=llm)
    )
    print("\n== Agent 内 fallback ==")
    print("status:", result.status, "| result:", result.result, "| last_used:", llm.last_used)


async def main() -> None:
    await fallback_demo()
    await fault_tolerant_demo()
    await in_agent()


if __name__ == "__main__":
    asyncio.run(main())
