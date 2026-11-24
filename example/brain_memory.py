"""示例：把 BrainMemory 接进 mainspring 的 MemoryManager（零外部依赖，可离线跑）。

接线方式与框架设计一致：BrainMemory 实现 ports.LongTermMemory 端口，
MemoryManager 原样使用；写入选材（抽取 + Reflector 打分 + 查重）仍由
Consolidator 负责 —— BrainMemory 只替换「长期记忆怎么存、怎么取、怎么淘汰」。

运行：cd 仓库根 && .venv/Scripts/python example/brain_memory.py
"""
from __future__ import annotations

import asyncio

from mainspring.memory.brain import BrainConfig, BrainMemory
from mainspring.memory.consolidator import Consolidator
from mainspring.memory.manager import MemoryManager


class InMemoryWorking:
    """进程内 WorkingMemory 桩（省去 redis 依赖）：窗口 + TTL 语义不变。"""

    def __init__(self, window_size: int = 20) -> None:
        self._window = window_size
        self._store: dict[str, list[dict]] = {}

    async def append(self, session_id: str, message: dict) -> None:
        hist = self._store.setdefault(session_id, [])
        hist.append(message)
        del hist[: max(0, len(hist) - self._window)]

    async def get_history(self, session_id: str) -> list[dict]:
        return list(self._store.get(session_id, []))


async def main() -> None:
    brain = BrainMemory(BrainConfig(), embedder=None)   # None = 离线哈希 embedding
    manager = MemoryManager(
        working=InMemoryWorking(),
        vector_store=brain,
        consolidator=Consolidator(brain, importance_threshold=0.30),
    )
    db = None  # v1 进程内实现，db 参数忽略（真实部署接 sqlite/ANN 适配器）

    session = "sess-demo"
    for role, text in (
        ("user", "下午好"),
        ("user", "我不喜欢咖啡，每次会议都要准备茶"),
        ("user", "明天帮我把那个报表再看一眼"),
        ("assistant", "好的，已记录你的偏好，明天提醒你检查报表。"),
    ):
        await manager.write(session, {"role": role, "content": text})

    kept = await manager.consolidate(db, session, "tenant-1", "u-1")
    print(f"巩固写入 {kept} 条（低于 importance 阈值的闲聊被过滤）")
    print(f"当前热层 {brain.diagnostics()['hot']} 条 / 冷层 {brain.diagnostics()['archive']} 条")

    for q in ("用户喝茶还是喝咖啡？", "报表什么时候要检查？"):
        hits = await manager.recall(db, "tenant-1", "u-1", q, top_k=2)
        print(f"\n召回「{q}」→")
        for h in hits:
            print(f"  [{h.similarity}] {h.content[:36]}  (importance={h.importance})")

    print("\ntier 分位:", brain.diagnostics()["tier_p10_p50_p90"],
          "（命中晋升已生效，跨 0 即为非 FIFO 壳）")
    print("性能:", brain.perf_summary())


if __name__ == "__main__":
    asyncio.run(main())
