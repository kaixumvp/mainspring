"""持久化检查点：``SQLiteCheckpointer`` 让 HITL 暂停可跨进程恢复。

要点：
- ``InMemoryCheckpointer`` 只适合进程内测试；生产用 ``mainspring.checkpoint`` 的
  SQLite / Redis / PostgreSQL 实现，协议完全相同。
- 运行时在每次 Decide 后、Execute 前保存快照；暂停后由 ``resume`` 从快照续跑。
- 本示例用新 ``AgentRuntime`` + 新 checkpointer 实例读取同一个 SQLite 文件，模拟进程重启。
- ``:memory:`` 会被显式拒绝（每次调用新连接会得到一个空库，正是要避免的“重启丢状态”）。
运行：``python example/checkpoint_sqlite.py``
"""

import asyncio
import tempfile
from pathlib import Path

from _shared import ScriptedLLM, finish, tool_call

from mainspring import AgentDefinition, AgentRuntime, RunContext, RuntimeDeps
from mainspring.checkpoint import SQLiteCheckpointer
from mainspring.plugins.registry import PluginRegistry
from mainspring.ports import PermissionContext, PermissionDecision, check_permission
from mainspring.tools.base import BaseTool, ToolResult


class DeleteAccountTool(BaseTool):
    name = "delete_account"
    description = "删除用户账号（高危，需审批）"
    permission = "admin"
    parameters = {
        "type": "object",
        "properties": {"user_id": {"type": "string"}},
        "required": ["user_id"],
    }

    async def run(self, user_id: str) -> ToolResult:
        return ToolResult(success=True, output=f"已删除账号 {user_id}")


class PermissionManager:
    def check(self, ctx, action, resource_type, resource_id,
              resource_permission="read", requires_approval=False, approved=None) -> PermissionDecision:
        return check_permission(ctx, action, resource_type, resource_id,
                                resource_permission, requires_approval, approved)


def build_deps(llm, checkpointer) -> RuntimeDeps:
    return RuntimeDeps(
        llm=llm,
        permission_checker=PermissionManager(),
        permission_context=PermissionContext(
            permission_codes={"run:execute"},
            grant_keys={("tool", "delete_account", "allow")},
        ),
        checkpointer=checkpointer,
    )


async def main() -> None:
    registry = PluginRegistry()
    registry.register_tool(DeleteAccountTool())
    definition = AgentDefinition(
        model="mock-model", system_prompt="你是管理员助手", plugins=registry.all()
    )
    llm = ScriptedLLM([
        tool_call("delete_account", {"user_id": "u42"}),
        finish("账号 u42 已删除。"),
    ])

    try:
        SQLiteCheckpointer(":memory:")
    except ValueError as exc:
        print("拒绝 :memory::", exc)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "checkpoints.db"

        # ---- 进程 A：跑到高危操作，暂停 ----
        cp_a = SQLiteCheckpointer(db_path)
        paused = await AgentRuntime().run(
            definition, "删除账号 u42", RunContext(run_id="run-sqlite"), build_deps(llm, cp_a)
        )
        print("\n进程 A:", paused.status, "| 待审批:", paused.pending_approvals)

        # ---- 进程 B：只拿到同一个文件路径 ----
        cp_b = SQLiteCheckpointer(db_path)
        print("磁盘上的 run:", await cp_b.list_runs())

        resumed = await AgentRuntime().resume(
            definition, "run-sqlite", ["delete_account"], build_deps(llm, cp_b)
        )
        print("进程 B 恢复:", resumed.status, "| 结果:", resumed.result)
        print("完成后磁盘上的 run:", await cp_b.list_runs(), "（成功后自动清理）")


if __name__ == "__main__":
    asyncio.run(main())
