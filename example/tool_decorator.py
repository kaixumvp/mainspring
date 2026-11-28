"""``@tool`` 装饰器：类型化函数直接变成 ``BaseTool``。

要点：
- 函数签名（类型、默认值、``Annotated[..., Field(description=...)]``、``Literal``）自动生成
  OpenAI 参数 JSON Schema；docstring 即工具描述。
- 每次调用先经 Pydantic 校验/强转；同步函数跑在线程池，异步函数直接 await。
- 校验失败不抛异常，而是返回 ``ToolResult(success=False, error=...)``，交给 Agent 观察。
运行：``python example/tool_decorator.py``
"""

import asyncio
import json
from typing import Annotated, Literal

from _shared import ScriptedLLM, finish, tool_call
from pydantic import Field

from mainspring import AgentDefinition, AgentRuntime, RunContext, RuntimeDeps
from mainspring.plugins.registry import PluginRegistry
from mainspring.tools.decorator import tool


@tool
def convert_currency(
    amount: float,
    currency: Annotated[str, Field(description="目标币种，如 USD / EUR")] = "USD",
) -> str:
    """把人民币金额换算为目标币种（示例固定汇率）。"""
    rates = {"USD": 7.2, "EUR": 7.8}
    rate = rates.get(currency.upper(), 7.2)
    return f"{amount} CNY ≈ {amount / rate:.2f} {currency.upper()}"


@tool(name="transfer", permission="write")
def transfer_funds(
    to_account: str,
    amount: float,
    level: Literal["normal", "urgent"] = "normal",
) -> str:
    """发起转账（write 权限，演示高危声明）。"""
    return f"已向 {to_account} 转账 {amount} 元（{level}）"


@tool(description="异步 ping，description 可覆盖 docstring")
async def ping(host: str) -> str:
    await asyncio.sleep(0)
    return f"{host} pong"


async def main() -> None:
    # 1. 签名即契约：看自动生成的工具元数据
    print("== 自动生成的工具契约 ==")
    print("convert_currency.parameters:")
    print(json.dumps(convert_currency.parameters, ensure_ascii=False, indent=2))
    print("description:", convert_currency.description)
    print("transfer permission:", transfer_funds.permission, "| ping description:", ping.description)

    # 2. 直接调用：参数会被校验/强转（"100" -> 100）
    print("\n== 直接调用 ==")
    print((await convert_currency.run(amount="100", currency="usd")).output)
    print((await ping.run(host="localhost")).output)

    # 3. 校验失败 → ToolResult(success=False)，不中断程序
    failed = await convert_currency.run(amount="not-a-number")
    print("校验失败:", failed.success, "| error 首行:", (failed.error or "").splitlines()[0])

    # 4. 注册进 Registry，交给 Agent 选择调用
    registry = PluginRegistry()
    for fn_tool in (convert_currency, transfer_funds, ping):
        registry.register_tool(fn_tool)
    print("\n暴露给 LLM 的函数:", [s["function"]["name"] for s in registry.to_openai_schema()])

    llm = ScriptedLLM([
        tool_call("convert_currency", {"amount": 100, "currency": "USD"}),
        finish("100 元约合 13.89 美元。"),
    ])
    definition = AgentDefinition(model="mock-model", system_prompt="你是财务助手", plugins=registry.all())
    result = await AgentRuntime().run(
        definition, "100 元人民币等于多少美元？", RunContext(run_id="tool-decorator"), RuntimeDeps(llm=llm)
    )
    print("Agent result:", result.result)


if __name__ == "__main__":
    asyncio.run(main())
