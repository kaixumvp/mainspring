"""离线评测（V1.1）：Golden Set + Judge + run_evaluation。

要点：
- `GoldenSet` 是一组 `(task, reference)` 用例。
- `Judge` 端口给每次输出打分（此处用 `HeuristicJudge`；生产接 LLM-as-Judge）。
- `run_evaluation` 遍历用例跑 Agent 并汇总评测结果。
运行：``python example/evaluation.py``
"""

import asyncio

from _shared import ScriptedLLM, finish

from mainspring import AgentDefinition, AgentRuntime, RunContext, RuntimeDeps
from mainspring.core.resilience import RetryPolicy
from mainspring.evaluation import Case, GoldenSet, HeuristicJudge, run_evaluation
from mainspring.plugins.registry import PluginRegistry
from mainspring.tools.registry import build_default_registry


async def main() -> None:
    registry = PluginRegistry()
    for tool in build_default_registry().all():
        registry.register_tool(tool)
    definition = AgentDefinition(model="mock-model", system_prompt="你是助手", plugins=registry.all())

    # 每条用例会跑 plan → decide(finish)，脚本化 LLM 按序给出各用例的回答
    llm = ScriptedLLM([
        finish("3*8+100 = 124"),
        finish("1+1 = 2"),
        finish("抱歉，我不知道。"),  # 故意答错，观察 verdict.passed=False
    ])
    deps = RuntimeDeps(llm=llm, retry=RetryPolicy(max_retries=0))

    golden = GoldenSet(cases=[
        Case(id="add", task="3*8+100 等于多少？", reference="124"),
        Case(id="sum", task="1+1 等于多少？", reference="2"),
        Case(id="unknown", task="2+2 等于多少？", reference="4"),
    ])

    results = await run_evaluation(
        AgentRuntime(), definition, deps, golden, HeuristicJudge(), RunContext(run_id="eval")
    )
    for r in results:
        print(f"[{r.case_id}] passed={r.verdict.passed} score={r.verdict.score} "
              f"reason={r.verdict.reason} output={r.output!r}")
    print("通过率:", f"{sum(r.verdict.passed for r in results)}/{len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
