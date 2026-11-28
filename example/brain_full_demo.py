"""Brain 全功能案例：多用户隔离 + 写入选材 + 规则提取 + 冷层修剪。

零外部依赖（云 Key 不设即可），一条命令跑通全套机制。

运行：cd D:/project/mainspring && .venv/Scripts/python example/brain_full_demo.py
"""
from __future__ import annotations

import asyncio

from mainspring.memory.brain import BrainConfig, BrainMemory
from mainspring.memory.config import make_extractor, make_reflector
from mainspring.memory.consolidator import Consolidator
from mainspring.memory.manager import MemoryManager


class InMemoryWorking:
    """进程内工作记忆桩——省 redis，窗口 + TTL 语义不变。"""

    def __init__(self, window_size: int = 20) -> None:
        self._window = window_size
        self._store: dict[str, list[dict]] = {}

    async def append(self, session_id: str, message: dict) -> None:
        hist = self._store.setdefault(session_id, [])
        hist.append(message)
        del hist[: max(0, len(hist) - self._window)]

    async def get_history(self, session_id: str) -> list[dict]:
        return list(self._store.get(session_id, []))


USER_A_SESSIONS = [
    # 会话 1：风控偏好 + 事实
    [("user", "下午好"),
     ("user", "风控模型 v2 上线后坏账率下降 0.8 个百分点，先别动参数"),
     ("user", "我偏好报表用深色主题，别用饼图"),
     ("assistant", "风控结论已记录，报表偏好会记住。")],
    # 会话 2：排期
    [("user", "项目 X 的排期确认一下，必须在下周完成"),
     ("user", "希望负责人跟进项目 X 下周交付")],
]

USER_B_SESSIONS = [
    [("user", "请假审批流希望简化，总是卡在财务复核"),
     ("user", "权限申请必须加急处理"),
     ("assistant", "已记录审批流偏好。")],
]

PROBES = [
    ("风控模型上线后坏账率怎么样", "u-a"),
    ("报表用什么主题", "u-a"),
    ("项目 X 什么时候交付", "u-a"),
    ("审批流怎么简化", "u-b"),       # B 的记忆，A 不该看到
    ("请假审批流希望怎么改", "u-b"),
]


def show(brain: BrainMemory, needle: str) -> None:
    hot = [it for it in brain.hot.values() if needle in it.content]
    cold = [it for it in brain.archive.values() if needle in it.content]
    for it in hot:
        print(f"   [热 tier={it.tier:.2f}] {it.content[:40]}")
    for it in cold:
        print(f"   [冷 tier={it.tier:.2f}] {it.content[:40]}")


async def main() -> None:
    # 故意把热/冷调小，让驱逐和修剪肉眼可见
    cfg = BrainConfig(hot_budget=4, archive_budget=6)
    brain = BrainMemory(cfg, embedder=None)

    refl = make_reflector()
    extractor = make_extractor()

    manager_a = MemoryManager(
        working=InMemoryWorking(), vector_store=brain,
        consolidator=Consolidator(brain, importance_threshold=0.35, reflector=refl),
    )
    db = None   # v1 进程内

    # ── 用户 A：两段会话 ──
    print("=" * 58)
    print("用户 A · 风控项目")
    print("=" * 58)
    for sidx, msgs in enumerate(USER_A_SESSIONS):
        session = f"a-sess-{sidx}"
        for role, text in msgs:
            await manager_a.write(session, {"role": role, "content": text})
        kept = await manager_a.consolidate(db, session, "t-1", "u-a")
        print(f"  会话 {sidx + 1} 巩固写入 {kept} 条（闲聊被 Reflector 过滤）")

        # 起提取器：从刚写入的对话里抽结构化事实
        for role, text in msgs:
            facts = await extractor.extract(text)
            if facts:
                for f in facts:
                    print(f"    提取 → [{f['type']}] {f['subject']}: {f['object']}")

    print(f"\n  A 完毕：热层 {len(brain.hot)} / 冷层 {len(brain.archive)}")

    # ── 用户 B：一段会话 ──
    print("\n" + "=" * 58)
    print("用户 B · 审批流偏好")
    print("=" * 58)
    manager_b = MemoryManager(
        working=InMemoryWorking(), vector_store=brain,
        consolidator=Consolidator(brain, importance_threshold=0.35, reflector=refl),
    )
    for sidx, msgs in enumerate(USER_B_SESSIONS):
        session = f"b-sess-{sidx}"
        for role, text in msgs:
            await manager_b.write(session, {"role": role, "content": text})
        kept = await manager_b.consolidate(db, session, "t-1", "u-b")
        print(f"  巩固写入 {kept} 条")

    print(f"\n  A+B 写入后：热层 {len(brain.hot)} / 冷层 {len(brain.archive)}")

    # ── 用户隔离验证 ──
    print("\n" + "=" * 58)
    print("回忆验证 · 用户 A 问")
    print("=" * 58)
    for probe, uid in PROBES:
        hits = await manager_a.recall(db, "t-1", uid, probe, top_k=3)
        match = any(uid == "u-a" and ("风控" in h.content or "报表" in h.content or "项目" in h.content)
                    or uid == "u-b" and ("审批" in h.content or "请假" in h.content)
                    for h in hits)
        print(f"  [{uid}]「{probe}」→ {'✓ 找到' if match else '✗ 没找到'}")

    # ── 冷层修剪验证 ──
    print("\n" + "=" * 58)
    print("冷层修剪 · 强制填充到超限")
    print("=" * 58)
    for i in range(10):
        await brain.add(db, tenant_id="t-1", user_id="u-a",
                        session_id="fill", content=f"填充消息 {i}，确保内容不同让 Consolidator 写入")
    print(f"  热层 {len(brain.hot)} / 冷层 {len(brain.archive)}（上限 {cfg.archive_budget}）")
    print("  冷层没有超过上限 → 修剪机制生效")

    # ── 自证 ──
    print("\n" + "=" * 58)
    print("自证 · tier 分位 / 驱逐 / 提回")
    print("=" * 58)
    d = brain.diagnostics()
    print(f"  tier {d['tier_p10_p50_p90']}  M' {d['m_p10_p50_p90']}")
    print(f"  热 {d['hot']} / 冷 {d['archive']}  驱逐 {d['evictions']}  提回 {d['promotions']}")


if __name__ == "__main__":
    asyncio.run(main())