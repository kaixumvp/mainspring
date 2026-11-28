"""流式输出：token 级 ``message.delta`` + ``ChatChunk`` 折叠。

要点：
- 网关同时实现 ``chat()`` 与 ``chat_stream()`` 时，Decide 在注入 ``event_sink`` 后自动走流式。
- 每个 ``ChatChunk.content_delta`` 被转成 ``message.delta`` 事件；最终仍折叠成同一个 ``LLMResponse``。
- 未注入 ``event_sink``（或网关不支持 ``chat_stream``）时自动回退 ``chat()``，结果完全一致。
- ``accumulate_response`` 也可单独使用：把带 ``tool_call_deltas`` 的分片拼回完整工具调用。
运行：``python example/streaming.py``
"""

import asyncio

from mainspring import AgentDefinition, AgentRuntime, RunContext, RuntimeDeps
from mainspring.llm.stream import accumulate_response
from mainspring.plugins.registry import PluginRegistry
from mainspring.ports import ChatChunk, LLMResponse, ToolCallDelta, Usage
from mainspring.tools.registry import build_default_registry


class StreamingMock:
    """既支持 ``chat()`` 也支持 ``chat_stream()`` 的离线网关（用来观察分支）。"""

    def __init__(self, final_text: str, chunk_size: int = 4) -> None:
        self._final_text = final_text
        self._chunks = [final_text[i : i + chunk_size] for i in range(0, len(final_text), chunk_size)]
        self.chat_calls = 0
        self.stream_calls = 0

    async def chat(self, messages, model=None, tools=None, schema=None) -> LLMResponse:
        self.chat_calls += 1
        return LLMResponse(content=self._final_text, usage=Usage(total_tokens=5))

    def chat_stream(self, messages, model=None, tools=None):
        self.stream_calls += 1

        async def gen():
            for piece in self._chunks:
                yield ChatChunk(content_delta=piece)
            yield ChatChunk(usage=Usage(prompt_tokens=3, completion_tokens=2, total_tokens=5))

        return gen()


def build_definition() -> AgentDefinition:
    registry = PluginRegistry()
    for tool in build_default_registry().all():
        registry.register_tool(tool)
    return AgentDefinition(model="mock-model", system_prompt="你是助手", plugins=registry.all())


async def streamed_run() -> None:
    """注入了 event_sink → 走 chat_stream，逐个 delta 产出事件。"""
    llm = StreamingMock("3*8+100 = 124")
    deltas: list[str] = []

    async def sink(event) -> None:
        if event.event == "message.delta":
            deltas.append(event.data["delta"])

    result = await AgentRuntime().run(
        build_definition(), "计算 3*8+100", RunContext(run_id="stream-1"),
        RuntimeDeps(llm=llm, event_sink=sink),
    )
    print("== 注入 event_sink：流式 ==")
    print("delta 片段:", deltas)
    print("chat_stream 调用:", llm.stream_calls, "| chat 调用:", llm.chat_calls)
    print("折叠后的最终结果:", result.result)


async def non_streaming_run() -> None:
    """没有 event_sink → 能力检查失败，自动回退 chat()。"""
    llm = StreamingMock("3*8+100 = 124")
    result = await AgentRuntime().run(
        build_definition(), "计算 3*8+100", RunContext(run_id="stream-2"),
        RuntimeDeps(llm=llm),
    )
    print("\n== 不注入 event_sink：自动回退 ==")
    print("chat_stream 调用:", llm.stream_calls, "| chat 调用:", llm.chat_calls)
    print("最终结果:", result.result)


def fold_tool_call_fragments() -> None:
    """工具调用同样可以分片到达：按 index 拼接 id/name/arguments。"""
    chunks = [
        ChatChunk(tool_call_deltas=[
            ToolCallDelta(index=0, id="call_1", name="calculator", arguments_delta='{"expr'),
        ]),
        ChatChunk(tool_call_deltas=[
            ToolCallDelta(index=0, arguments_delta='ession": "3*8+100"}'),
        ]),
        ChatChunk(usage=Usage(total_tokens=7)),
    ]
    resp = accumulate_response(chunks)
    call = resp.tool_calls[0]
    print("\n== 手动折叠工具调用分片 ==")
    print("name:", call.name, "| arguments:", call.arguments, "| usage:", resp.usage.total_tokens)


async def main() -> None:
    await streamed_run()
    await non_streaming_run()
    fold_tool_call_fragments()


if __name__ == "__main__":
    asyncio.run(main())
