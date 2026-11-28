"""百炼 DashScope 的 LLMGateway 适配器。

把 dashscope 的 Generation API 包装成 mainspring 的 LLMGateway 端口。
仅 async 调用路径，同步 SDK 调用通过 asyncio.to_thread 转到线程池。
"""
from __future__ import annotations

import asyncio

import dashscope
from http import HTTPStatus as HttpOK

from mainspring.ports import LLMResponse


class DashScopeGateway:
    """百炼 chat 适配器——实现 mainspring 的 LLMGateway 协议（chat 方法）。"""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._key = api_key
        if base_url:
            dashscope.base_http_api_url = base_url
        dashscope.base_http_api_url = base_url or "https://cn-beijing.maas.qianwenaiapi.com/api/v1"

    async def chat(self, messages: list[dict], model: str | None = None,
                   tools: list[dict] | None = None, schema=None) -> LLMResponse:
        resp = await asyncio.to_thread(
            dashscope.Generation.call,
            model=model or "qwen-turbo-latest",
            messages=messages,
            api_key=self._key,
        )
        if resp.status_code != HttpOK.OK:
            return LLMResponse(content=f"ERROR:{resp.status_code} {resp.message}")
        return LLMResponse(content=resp.output.text)