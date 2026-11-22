"""Pre-request context budgeting (v3.5): build a budgeted, traceable context view.

Three-data separation (see documentation/mainspring/上下文管理设计方案.md): the full run
record lives with the host; the execution state drives the loop; the Context View built
here is a per-request, disposable input assembled under a token budget. This module is
pure logic — no network IO, no model calls, deterministic — so it is unit-testable
offline. Assembly order is canonical: system prompt, plan, memory, then history.
"""

from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel, Field

# Internal state key the decide/plan nodes return on budget failure; the runtime raises
# ContextBudgetExceeded outside the retry wrapper so a rejected request is never retried.
BUDGET_ERROR_KEY = "_budget_error"


class ContextBudgetExceeded(Exception):
    """Protected context still exceeds the input budget after deterministic compression.

    Carries the per-source breakdown and every compression action taken; the message
    contains only counts, never raw prompt or tool content.
    """

    def __init__(
        self,
        message: str,
        *,
        input_limit: int = 0,
        estimated_tokens: int = 0,
        breakdown: dict[str, int] | None = None,
        actions: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.input_limit = input_limit
        self.estimated_tokens = estimated_tokens
        self.breakdown = breakdown or {}
        self.actions = actions or []


class ContextPolicy(BaseModel):
    """Budget knobs for one agent/model. ``context_window_tokens=0`` disables admission control."""

    context_window_tokens: int = 0  # 0 = per-request budgeting disabled
    max_output_tokens: int = 2048  # reserved for the model reply, deducted up front
    safety_margin_tokens: int = 512  # absorbs estimator error and provider format overhead
    memory_token_budget: int | None = None  # None = unbounded
    tool_schema_token_budget: int | None = None  # None = unbounded
    tool_result_chars: int = 0  # 0 = tool results unlimited
    chars_per_token: float = 3.0  # heuristic estimator; CJK hosts tune lower or inject a tokenizer


class CharTokenEstimator:
    """Conservative chars-per-token heuristic; never provider-exact.

    CJK text costs ~1 char per token, so the 3.0 default UNDERESTIMATES it — tune
    ``chars_per_token`` per content or inject a tokenizer-backed ``TokenEstimator``.
    """

    def __init__(self, chars_per_token: float = 3.0) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.chars_per_token = chars_per_token

    def estimate(self, messages: list[dict], tools: list[dict] | None = None) -> int:
        chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
        for t in tools or []:
            chars += len(json.dumps(t, ensure_ascii=False))
        return math.ceil(chars / self.chars_per_token)


class ContextBuildResult(BaseModel):
    """The final per-request view plus budget diagnostics (counts only, no raw content)."""

    messages: list[dict]
    tools: list[dict]
    input_limit: int
    estimated_tokens: int
    breakdown: dict[str, int]  # system | memory | history | tools
    actions: list[str] = Field(default_factory=list)


def limit_tool_result(text: str, max_chars: int) -> str:
    """Bound a tool result before it enters the model context; the marker states incompleteness."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"…[truncated to {max_chars} chars of {len(text)} — content incomplete]"


def build_context(
    *,
    system_prompt: str,
    plan: list[dict] | None = None,
    memory: list[dict] | None = None,
    messages: list[dict],
    tools: list[dict],
    policy: ContextPolicy,
    estimator: Any = None,
) -> ContextBuildResult:
    """Assemble the budgeted request view; raises ``ContextBudgetExceeded`` instead of
    silently dropping protected content (system prompt, current user task, tool pairing).

    Deterministic compression order: trim least-relevant memory tail → drop earliest
    history turns (never stranding tool messages, never touching the last user turn).
    Tool schemas are never truncated: an oversized allowlist is an explicit error telling
    the host to shrink it.
    """
    est = estimator or CharTokenEstimator(policy.chars_per_token)
    input_limit = policy.context_window_tokens - policy.max_output_tokens - policy.safety_margin_tokens
    if input_limit <= 0:
        raise ContextBudgetExceeded(
            f"context policy misconfigured: window={policy.context_window_tokens} minus "
            f"output reserve={policy.max_output_tokens} and safety margin={policy.safety_margin_tokens} "
            "leaves no input budget",
            input_limit=input_limit,
            estimated_tokens=0,
            breakdown={},
            actions=[],
        )

    history = [dict(m) for m in messages]
    memory_kept = [dict(m) for m in (memory or [])]
    tools_out = [dict(t) for t in tools]
    actions: list[str] = []

    def memory_message(items: list[dict]) -> list[dict]:
        if not items:
            return []
        text = "\n".join(m.get("content", "") for m in items)
        return [{"role": "system", "content": f"Relevant past memories about this user/task:\n{text}"}]

    def assemble(history_msgs: list[dict]) -> tuple[list[dict], dict[str, int], int]:
        system_msgs = [{"role": "system", "content": system_prompt}]
        if plan:
            plan_text = "\n".join(f"{s['id']}. {s['description']}" for s in plan)
            system_msgs.append({"role": "system", "content": f"Current plan:\n{plan_text}"})
        mem_msgs = memory_message(memory_kept)
        final = system_msgs + mem_msgs + history_msgs
        breakdown = {
            "system": est.estimate(system_msgs, None),
            "memory": est.estimate(mem_msgs, None),
            "history": est.estimate(history_msgs, None),
            "tools": est.estimate([], tools_out),
        }
        total = est.estimate(final, tools_out)
        return final, breakdown, total

    def droppable_cut(history_msgs: list[dict]) -> int | None:
        """Index of the earliest droppable message, past the system head; None when only
        the protected last user turn remains. A tool message at the head is itself
        droppable — its assistant tool-call parent was dropped in an earlier iteration."""
        last_user = -1
        for i, m in enumerate(history_msgs):
            if m.get("role") == "user":
                last_user = i
        if last_user <= 0:
            return None
        cut = 0
        while cut < len(history_msgs) and history_msgs[cut].get("role") == "system":
            cut += 1
        return cut if cut < last_user else None

    # 1. memory: recall order is relevance order — drop the tail (least relevant) to fit.
    if policy.memory_token_budget is not None and memory_kept:
        while memory_kept and est.estimate(memory_message(memory_kept), None) > policy.memory_token_budget:
            memory_kept.pop()
        if len(memory_kept) != len(memory or []):
            actions.append(f"memory: kept {len(memory_kept)} of {len(memory or [])} items within budget")

    # 2. tools: never silently truncated — an oversized allowlist is a configuration error.
    if policy.tool_schema_token_budget is not None:
        tools_est = est.estimate([], tools_out)
        if tools_est > policy.tool_schema_token_budget:
            _, breakdown, total = assemble(history)
            raise ContextBudgetExceeded(
                f"tool schemas need {tools_est} tokens, over the {policy.tool_schema_token_budget} "
                "tool_schema_token_budget — shrink the agent tool allowlist",
                input_limit=input_limit,
                estimated_tokens=total,
                breakdown=breakdown,
                actions=actions,
            )

    # 3. history: drop earliest turns until the request fits, reserving a slot for the
    # omission marker so the marker's own cost stays inside the budget.
    marker_template = "[{} earlier messages omitted by token budget]"
    history = [{"role": "system", "content": marker_template.format(0)}] + history
    final, breakdown, total = assemble(history)
    dropped = 0
    while total > input_limit:
        cut = droppable_cut(history)
        if cut is None:
            break
        history = history[:cut] + history[cut + 1:]
        dropped += 1
        final, breakdown, total = assemble(history)
    if dropped:
        # Dropping an assistant(tool_calls) can leave its tool result stranded right
        # after the system head; pop it so the kept view never opens with an orphan.
        i = 0
        while i < len(history) and history[i].get("role") == "system":
            i += 1
        while i < len(history) and history[i].get("role") == "tool":
            history.pop(i)
            dropped += 1
        history[0] = {"role": "system", "content": marker_template.format(dropped)}
        final, breakdown, total = assemble(history)
        # The real count's digits can cost a few more tokens than the placeholder's.
        while total > input_limit:
            cut = droppable_cut(history)
            if cut is None:
                break
            history = history[:cut] + history[cut + 1:]
            dropped += 1
            history[0] = {"role": "system", "content": marker_template.format(dropped)}
            final, breakdown, total = assemble(history)
        actions.append(f"history: dropped {dropped} earliest messages within budget")
    else:
        history.pop(0)  # nothing was omitted; remove the placeholder marker
        final, breakdown, total = assemble(history)

    if total > input_limit:
        raise ContextBudgetExceeded(
            f"context budget exceeded: estimated {total} > input limit {input_limit} "
            f"(window {policy.context_window_tokens} - output {policy.max_output_tokens} - margin {policy.safety_margin_tokens})",
            input_limit=input_limit,
            estimated_tokens=total,
            breakdown=breakdown,
            actions=actions,
        )

    return ContextBuildResult(
        messages=final,
        tools=tools_out,
        input_limit=input_limit,
        estimated_tokens=total,
        breakdown=breakdown,
        actions=actions,
    )


__all__ = [
    "BUDGET_ERROR_KEY",
    "CharTokenEstimator",
    "ContextBudgetExceeded",
    "ContextBuildResult",
    "ContextPolicy",
    "build_context",
    "limit_tool_result",
]
