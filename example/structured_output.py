"""结构化输出：``chat(schema=PydanticModel)`` → ``LLMResponse.parsed``。

要点：
- ``schema`` 是网关协议的一部分：网关可以尽力提示厂商用 ``response_format``，
  但**校验始终由框架侧 ``parse_structured`` 完成**，因此不同厂商行为一致。
- 校验成功：``resp.parsed`` 是 Pydantic 模型实例，不再手工 ``json.loads``。
- 校验失败：抛 ``StructuredOutputError``；可用 ``FaultTolerantGateway`` 降级为空响应。
运行：``python example/structured_output.py``
"""

import asyncio
from typing import Literal

from pydantic import BaseModel

from mainspring.llm.routing import FaultTolerantGateway
from mainspring.llm.structured import parse_structured
from mainspring.ports import LLMResponse, StructuredOutputError


class Ticket(BaseModel):
    """工单抽取的目标结构。"""

    title: str
    priority: Literal["low", "medium", "high"]
    tags: list[str] = []


class SchemaGateway:
    """用固定文本模拟模型输出，并执行与真实网关相同的 schema 校验。"""

    def __init__(self, content: str | None) -> None:
        self._content = content

    async def chat(self, messages, model=None, tools=None, schema=None) -> LLMResponse:
        parsed = parse_structured(self._content, schema) if schema is not None else None
        return LLMResponse(content=self._content, parsed=parsed)


async def valid_structured_call() -> None:
    gateway = SchemaGateway('{"title": "无法登录", "priority": "high", "tags": ["auth"]}')
    resp = await gateway.chat([{"role": "user", "content": "把工单转成 JSON"}], schema=Ticket)
    print("== 合法输出 ==")
    print("parsed 类型:", type(resp.parsed).__name__)
    print("parsed 内容:", resp.parsed)
    print("字段访问:", resp.parsed.priority, resp.parsed.tags)


async def invalid_structured_call() -> None:
    gateway = SchemaGateway('{"title": "无法登录", "priority": "urgent"}')
    print("\n== 非法输出：StructuredOutputError ==")
    try:
        await gateway.chat([{"role": "user", "content": "把工单转成 JSON"}], schema=Ticket)
    except StructuredOutputError as exc:
        print("捕获:", type(exc).__name__, "->", exc)


async def fault_tolerant_fallback() -> None:
    """校验失败不阻断流程：降级为可读文本，parsed=None。"""
    safe = FaultTolerantGateway(
        SchemaGateway('{"title": "无法登录", "priority": "urgent"}'),
        on_error="模型输出无法解析，已降级为兜底回复。",
    )
    resp = await safe.chat([{"role": "user", "content": "把工单转成 JSON"}], schema=Ticket)
    print("\n== FaultTolerantGateway 降级 ==")
    print("content:", resp.content)
    print("parsed:", resp.parsed)


async def main() -> None:
    await valid_structured_call()
    await invalid_structured_call()
    await fault_tolerant_fallback()


if __name__ == "__main__":
    asyncio.run(main())
