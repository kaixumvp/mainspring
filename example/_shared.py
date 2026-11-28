"""示例公共工具：离线可运行的脚本化 Mock LLM + 安全读取密钥。

所有示例默认离线运行（无需真实 API key）；要接真实模型时，设置环境变量并改用
`real_llm()`（见下方）。
"""

from __future__ import annotations

import os

from mainspring.llm.base import LLMResponse, ToolCall
from mainspring.llm.litellm_gateway import LiteLLMGateway
from mainspring.llm.structured import parse_structured


class ScriptedLLM:
    """按顺序返回预设响应的 LLM，并记录每次调用，用于离线演示与断言。

    - ``chat`` 依次 pop 一个响应（含 ``schema`` 参数，与 LLMGateway 协议一致）。
    - ``calls`` 记录每条请求的 messages/model/tools/schema，方便观察 memory/知识上下文是否被注入。
    """

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, messages, model=None, tools=None, schema=None) -> LLMResponse:
        self.calls.append({"messages": messages, "model": model, "tools": tools, "schema": schema})
        if not self._responses:
            raise RuntimeError(
                "ScriptedLLM 已无预设响应：请为每次 plan/decide 调用各准备一条 LLMResponse"
            )
        resp = self._responses.pop(0)
        if schema is not None and resp.parsed is None:
            # 与真实网关同一契约：schema 请求由框架侧校验，结果放在 parsed。
            resp = resp.model_copy(update={"parsed": parse_structured(resp.content, schema)})
        return resp


def tool_call(name: str, arguments: dict, tool_call_id: str = "call_1") -> LLMResponse:
    """decide 节点的响应：让 LLM 选择调用某个插件。"""
    return LLMResponse(tool_calls=[ToolCall(id=tool_call_id, name=name, arguments=arguments)])


def finish(text: str) -> LLMResponse:
    """decide 节点的响应：收尾给最终答案。"""
    return LLMResponse(content=text)


def real_llm(default_model: str = "deepseek/deepseek-chat"):
    """从环境变量读取真实 LLM 配置；未设置 key 时抛错提示。

    环境变量：
        MAINSPRING_MODEL / DEEPSEEK_API_KEY（或 MAINSPRING_API_KEY）
    """
    model = os.getenv("MAINSPRING_MODEL", default_model)
    api_key = os.environ.get("MAINSPRING_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "请设置环境变量 MAINSPRING_API_KEY（或 DEEPSEEK_API_KEY）后重试。"
        )
    return LiteLLMGateway(model=model, api_key=api_key), model
