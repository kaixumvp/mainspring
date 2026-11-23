import json
import logging

from mainspring.core.context import BUDGET_ERROR_KEY, ContextBudgetExceeded, build_context
from mainspring.core.state import AgentState, PlanStep
from mainspring.ports import LLMGateway

logger = logging.getLogger(__name__)

_PLAN_SYSTEM = (
    "You are a task planner. Break the user's task into several executable sub-steps, "
    'and output only a JSON array where each item has the format {"description": "..."}.'
)


def _parse_plan(content: str | None) -> list[PlanStep]:
    if not content:
        return []
    text = content.strip()
    # Tolerantly strip markdown code fences
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    steps: list[PlanStep] = []
    for i, item in enumerate(data, start=1):
        desc = str(item.get("description") or "") if isinstance(item, dict) else str(item)
        steps.append({"id": i, "description": desc, "status": "pending"})
    return steps


async def run_plan(
    state: AgentState,
    llm: LLMGateway,
    model: str,
    *,
    context_policy=None,
    token_estimator=None,
) -> dict:
    task = state["task"]
    memory_ctx = state.get("memory_ctx") or []
    if context_policy is not None and context_policy.context_window_tokens > 0:
        try:
            built = build_context(
                system_prompt=_PLAN_SYSTEM,
                plan=None,
                memory=memory_ctx,
                messages=[{"role": "user", "content": task}],
                tools=[],
                policy=context_policy,
                estimator=token_estimator,
            )
        except ContextBudgetExceeded as exc:
            logger.error(
                "plan: context budget exceeded run_id=%s estimated=%d limit=%d",
                state.get("run_id"), exc.estimated_tokens, exc.input_limit,
            )
            return {BUDGET_ERROR_KEY: str(exc)}
        messages = built.messages
    else:
        messages = [
            {"role": "system", "content": _PLAN_SYSTEM},
            {"role": "user", "content": task},
        ]
        # Inject recalled long-term memories as planning context (V0.2 baseline recall).
        if memory_ctx:
            ctx_text = "\n".join(m.get("content", "") for m in memory_ctx)
            messages.append(
                {"role": "system", "content": f"Relevant past memories about this user/task:\n{ctx_text}"}
            )
    plan: list[PlanStep] = []
    usage: dict = {}
    try:
        resp = await llm.chat(messages=messages, model=model)
        plan = _parse_plan(resp.content)
        usage = resp.usage.model_dump()
    except Exception:
        logger.warning("plan failed, falling back to a single-step plan", exc_info=True)
        plan = []
    if not plan:
        plan = [{"id": 1, "description": task, "status": "pending"}]
    logger.debug("plan: run_id=%s steps=%d", state.get("run_id"), len(plan))
    return {"plan": plan, "token_usage": usage}
