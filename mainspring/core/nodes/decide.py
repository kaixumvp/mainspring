import json
import logging
from collections.abc import Awaitable, Callable

from mainspring.core.context import BUDGET_ERROR_KEY, ContextBudgetExceeded, build_context
from mainspring.core.state import FINISH, AgentState
from mainspring.ports import ChatChunk, LLMGateway, LLMResponse, TokenEstimator
from mainspring.llm.stream import accumulate_response

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
    *,
    context_policy=None,
    token_estimator: TokenEstimator | None = None,
) -> dict:
    system = system_prompt or _DEFAULT_SYSTEM

    plan = state.get("plan") or []
    memory_ctx = state.get("memory_ctx") or []
    history = state.get("messages") or []

    diagnostics: dict | None = None
    if context_policy is not None and context_policy.context_window_tokens > 0:
        # Budget admission before the LLM call; on rejection return a sentinel the runtime
        # turns into a run failure without retrying the same oversized request.
        try:
            built = build_context(
                system_prompt=system,
                plan=plan,
                memory=memory_ctx,
                messages=history,
                tools=list(tool_schemas),
                policy=context_policy,
                estimator=token_estimator,
            )
        except ContextBudgetExceeded as exc:
            logger.error(
                "decide: context budget exceeded run_id=%s estimated=%d limit=%d actions=%s",
                state.get("run_id"), exc.estimated_tokens, exc.input_limit, exc.actions,
            )
            return {BUDGET_ERROR_KEY: str(exc)}
        messages = built.messages
        tools = built.tools
        if built.actions:
            logger.warning("decide: context compacted run_id=%s actions=%s", state.get("run_id"), built.actions)
        diagnostics = {
            "estimated_tokens": built.estimated_tokens,
            "input_limit": built.input_limit,
            "breakdown": built.breakdown,
            "actions": built.actions,
        }
    else:
        messages = [{"role": "system", "content": system}]
        if plan:
            plan_text = "\n".join(f"{s['id']}. {s['description']}" for s in plan)
            messages.append({"role": "system", "content": f"Current plan:\n{plan_text}"})
        if memory_ctx:
            ctx_text = "\n".join(m.get("content", "") for m in memory_ctx)
            messages.append(
                {"role": "system", "content": f"Relevant past memories about this user/task:\n{ctx_text}"}
            )
        messages.extend(history)
        tools = tool_schemas or None

    if on_delta is not None and hasattr(llm, "chat_stream"):
        resp = await _stream_response(llm, messages, model, tools, on_delta)
    else:
        resp = await llm.chat(messages=messages, model=model, tools=tools)

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
        update: dict = {
            "action": first["name"],
            "action_input": first["arguments"],
            "tool_call_id": first["id"],
            "tool_calls": tool_calls,
            "messages": messages_history,
            "token_usage": resp.usage.model_dump(),
        }
        if diagnostics is not None:
            update["context"] = diagnostics
        return update

    logger.debug("decide: finish")
    update = {"action": FINISH, "result": resp.content or "", "token_usage": resp.usage.model_dump()}
    if diagnostics is not None:
        update["context"] = diagnostics
    return update


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
