"""多 Agent 协作：agent-skill 委派子 Agent + 子 Agent 深度保护。

要点：
- ``AgentSkill`` 只负责声明 ``agent_ref``；真正的“启动子 Agent”由宿主注入的
  ``run_subagent`` 回调完成——可以在回调里创建新的 ``AgentDefinition``/``RuntimeDeps``
  再跑一次 ``AgentRuntime``。
- 父 Agent 的 ``InvokeContext.subagent_depth`` 传给子 Agent 的 ``RunContext`` 时 +1。
- 运行时在 Execute 前检查 ``subagent_depth >= max_subagent_depth`` 并拒绝继续委派。
运行：``python example/multi_agent.py``
"""

import asyncio

from _shared import ScriptedLLM, finish, tool_call

from mainspring import AgentDefinition, AgentRuntime, RunContext, RuntimeConfig, RuntimeDeps
from mainspring.plugins.registry import PluginRegistry
from mainspring.skills.manager import SkillManager

# 回调里记录父 Agent 发起的委派，以及子 Agent 自己的步骤，便于最后打印。
DELEGATIONS: list[tuple[str, str, int]] = []
CHILD_STEPS: list[list] = []


async def run_subagent(agent_ref: str, task: str, ctx) -> str:
    """宿主实现：真正启动一个独立的子 Agent，并返回它的最终答案。"""
    DELEGATIONS.append((agent_ref, task, ctx.subagent_depth))
    if agent_ref != "log-specialist":
        return f"[unknown agent] {agent_ref}"

    # 子 Agent：只配一个更深的 agent-skill，用来验证深度保护。
    child_llm = ScriptedLLM([
        tool_call("skill_escalate", {"task": "再向上委派"}),
        finish("日志已分析：发现 3 个 ERROR。"),
    ])
    child_manager = SkillManager(run_subagent=run_subagent)
    child_registry = PluginRegistry()
    child_registry.register(child_manager.build({
        "name": "escalate",
        "type": "agent",
        "description": "继续向上委派（应被深度保护拦下）",
        "manifest": {
            "parameters": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]},
        },
        "body": {"agent_ref": "super-specialist"},
    }))
    child_definition = AgentDefinition(
        model="mock-child",
        system_prompt="你是日志专家",
        plugins=child_registry.all(),
        runtime=RuntimeConfig(max_subagent_depth=1),
    )
    child_result = await AgentRuntime().run(
        child_definition,
        task,
        RunContext(run_id=f"{ctx.run_id}-child", subagent_depth=ctx.subagent_depth + 1),
        RuntimeDeps(llm=child_llm),
    )
    CHILD_STEPS.append(child_result.steps)
    return child_result.result or ""


async def main() -> None:
    # 父 Agent：一个 delegate agent-skill
    parent_manager = SkillManager(run_subagent=run_subagent)
    parent_registry = PluginRegistry()
    parent_registry.register(parent_manager.build({
        "name": "delegate",
        "type": "agent",
        "description": "把日志分析委派给日志专家子 Agent",
        "manifest": {
            "parameters": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]},
        },
        "body": {"agent_ref": "log-specialist"},
    }))
    parent_llm = ScriptedLLM([
        tool_call("skill_delegate", {"task": "分析今天的错误日志"}),
        finish("父 Agent：日志专家已完成分析。"),
    ])
    parent_definition = AgentDefinition(
        model="mock-parent",
        system_prompt="你是调度助手",
        plugins=parent_registry.all(),
        runtime=RuntimeConfig(max_subagent_depth=1),
    )

    parent_result = await AgentRuntime().run(
        parent_definition, "帮我分析今天的日志", RunContext(run_id="parent-1"), RuntimeDeps(llm=parent_llm)
    )
    print("父 Agent:", parent_result.result)
    delegate_step = next(s for s in parent_result.steps if s.node == "execute")
    print("父 Agent 的委派结果:", delegate_step.output["result"])

    print("\n委派记录 (agent_ref, task, 发起时深度):", DELEGATIONS)
    child_steps = CHILD_STEPS[0]
    print("子 Agent 步骤:", [s.node for s in child_steps])
    child_execute = next(s for s in child_steps if s.node == "execute")
    print("子 Agent 再次委派是否被拦下:", child_execute.output["result"])


if __name__ == "__main__":
    asyncio.run(main())
