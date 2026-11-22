import json
import logging
from collections.abc import Awaitable, Callable

from ouroboros.core.state import FINISH, AgentState
from ouroboros.ports import ChatChunk, LLMGateway, LLMResponse
from ouroboros.llm.stream import accumulate_response

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM = "You are a general-purpose assistant. Call the calculator tool for arithmetic problems."

# Async callable receiving each content delta as it arrives (token-level event emission).
DeltaCallback = Callable[[str], Awaitable[None]]


async def run_decide(
    state: AgentState,
    llm: LLMGateway,
    model: str,
    tool_schemas: list[dict],
    system_prompt: str | None = None,
    on_delta: DeltaCallback | None = None,
) -> dict:
    system = system_prompt or _DEFAULT_SYSTEM
    messages: list[dict] = [{"role": "system", "content": system}]

    plan = state.get("plan") or []
    if plan:
        plan_text = "\n".join(f"{s['id']}. {s['description']}" for s in plan)
        messages.append({"role": "system", "content": f"Current plan:\n{plan_text}"})

    # Recalled long-term memories / knowledge fallback must reach the LLM even when the
    # (optional) Plan node is disabled — Decide is the only guaranteed LLM call.
    memory_ctx = state.get("memory_ctx") or []
    if memory_ctx:
        ctx_text = "\n".join(m.get("content", "") for m in memory_ctx)
        messages.append(
            {"role": "system", "content": f"Relevant past memories about this user/task:\n{ctx_text}"}
        )

    messages.extend(state.get("messages") or [])

    if on_delta is not None and hasattr(llm, "chat_stream"):
        resp = await _stream_response(llm, messages, model, tool_schemas or None, on_delta)
    else:
        resp = await llm.chat(messages=messages, model=model, tools=tool_schemas or None)

    if resp.tool_calls:
        tool_calls = [
            {"id": tc.id or f"call_{tc.name}", "name": tc.name, "arguments": tc.arguments}
            for tc in resp.tool_calls
        ]
        messages_history = list(state.get("messages") or [])
        messages_history.append({
            "role": "assistant",
            "content": resp.content,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                    },
                }
                for tc in tool_calls
            ],
        })
        first = tool_calls[0]
        logger.debug("decide: action=%s (%d tool calls)", first["name"], len(tool_calls))
        return {
            "action": first["name"],
            "action_input": first["arguments"],
            "tool_call_id": first["id"],
            "tool_calls": tool_calls,
            "messages": messages_history,
            "token_usage": resp.usage.model_dump(),
        }

    logger.debug("decide: finish")
    return {"action": FINISH, "result": resp.content or "", "token_usage": resp.usage.model_dump()}


async def _stream_response(
    llm: LLMGateway,
    messages: list[dict],
    model: str,
    tools: list[dict] | None,
    on_delta: DeltaCallback,
) -> LLMResponse:
    """Consume a streaming chat, forwarding content deltas live, then fold to one response.

    A retried Decide re-streams from scratch, so consumers must treat deltas as ephemeral
    and the folded response as the only authoritative result.
    """
    chunks: list[ChatChunk] = []
    async for chunk in llm.chat_stream(messages=messages, model=model, tools=tools):
        chunks.append(chunk)
        if chunk.content_delta:
            await on_delta(chunk.content_delta)
    return accumulate_response(chunks)
