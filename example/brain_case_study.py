"""案例：项目周期助手 —— BrainMemory 怎么「记住、忘记、想起来」（离线可跑）。

三幕故事，对应记忆机制的三步：
  第 1 幕 活跃期：用户在工作，记忆被写入（写入选材 + 打分过滤）
  第 2 幕 遗忘期：用户去忙别的项目，旧记忆被挤出热层进冷层（价值驱逐，
            **归档不删除**——人类遗忘是可访问性下降，不是信息丢失）
  第 3 幕 回归期：用户回到旧项目，冷层命中 → 提回热层 + 层级晋升

配套文档：documentation/BrainMemory使用案例.md
运行：cd 仓库根 && .venv/Scripts/python example/brain_case_study.py
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


def show(brain: BrainMemory, needle: str, tag: str) -> None:
    """打印热层/冷层里包含 needle 的条目——让驱逐和提回肉眼可见。"""
    hot = [it for it in brain.hot.values() if needle in it.content]
    cold = [it for it in brain.archive.values() if needle in it.content]
    print(f"  {tag}: 热层 {len(hot)} 条 / 冷层 {len(cold)} 条")
    for it in hot[:3]:
        print(f"      [热] tier={it.tier:.2f} uses={it.uses} 「{it.content[:30]}」")
    for it in cold[:3]:
        print(f"      [冷] tier={it.tier:.2f} uses={it.uses} 「{it.content[:30]}」")


def stats(brain: BrainMemory) -> str:
    s = brain.stats
    return (f"命中统计: 热层命中={s['hot_hit']} 冷层命中={s['archive_hit']} "
            f"未中={s['miss']} 驱逐={s['evictions']} 提回={s['promotions']}")


async def main() -> None:
    # 热层只留 8 个格子：故意调小，让「驱逐」在十几条消息内就肉眼可见
    brain = BrainMemory(BrainConfig(hot_budget=8), embedder=None)
    manager = MemoryManager(
        working=InMemoryWorking(),
        vector_store=brain,
        consolidator=Consolidator(brain, importance_threshold=0.35),
    )
    db = None  # v1 进程内实现，db 参数忽略

    print("=" * 64)
    print("第 1 幕 · 活跃期：用户在工作，记忆被写入")
    print("=" * 64)
    for role, text in (
        ("user", "下午好"),
        ("user", "风控模型 v2 上线后坏账率下降 0.8 个百分点，先别动参数"),
        ("user", "我偏好报表用深色主题，别用饼图"),
        ("assistant", "好的，风控模型结论已记录，报表偏好也会记住。"),
    ):
        await manager.write("sess-A", {"role": role, "content": text})
    kept = await manager.consolidate(db, "sess-A", "tenant-1", "u-1")
    print(f"  巩固写入 {kept} 条（「下午好」这类闲聊被 importance 阈值过滤）")
    show(brain, "风控", "风控项目")
    show(brain, "报表", "报表偏好")
    print(f"  {stats(brain)}\n")

    print("=" * 64)
    print("第 2 幕 · 遗忘期：六周里用户忙别的项目")
    print("=" * 64)
    filler = (
        "项目 X 的排期需要确认一下，必须在下周完成",
        "项目 X 下周要交付，希望负责人跟进进度",
        "机房巡检安排在下周三，不要安排其它变更",
        "预算审批流程希望简化，总是卡在财务复核",
        "权限申请总是要等两天，希望加急处理",
        "合规审查清单必须更新到最新版本",
        "新员工培训材料需要重新整理一遍",
        "合同模板的审批流希望下周上线",
        "应急演练脚本必须每季度更新一次",
        "数据备份策略希望改为每日增量",
        "监控告警阈值总是误报，需要重新标定",
        "外包人员的权限回收流程需要复核",
        "年底审计的材料清单必须提前准备",
        "测试环境的机器希望统一升级版本",
        "报销流程总是太慢，希望简化审批链",
        "会议室预订系统需要接入日历同步",
    )
    for i, text in enumerate(filler):
        await manager.write(f"sess-other-{i}", {"role": "user", "content": text})
        await manager.consolidate(db, f"sess-other-{i}", "tenant-1", "u-1")
    print("  16 个其它项目会话写入后：")
    show(brain, "风控", "风控项目")
    show(brain, "报表", "报表偏好")
    print(f"  热层 {len(brain.hot)} 格 / 冷层 {len(brain.archive)} 格"
          f"（容量 {brain.cfg.hot_budget}）：旧项目条目被挤出热层 → 进冷层，归档不删除")
    print(f"  {stats(brain)}\n")

    print("=" * 64)
    print("第 3 幕 · 回归期：用户回到风控项目")
    print("=" * 64)
    q = "风控模型上线后的结论是什么"
    print(f"  用户问：「{q}」")
    hits = await manager.recall(db, "tenant-1", "u-1", q, top_k=3)
    for h in hits:
        print(f"    相似度 {h.similarity:.3f} 「{h.content[:34]}」")
    print(f"  冷层命中 {brain.stats['archive_hit']} 次 → 提回热层（这就是『想起来』："
          f"被挤出去的记忆没有丢，取回时自动回到热层并晋升）")
    print(f"  {stats(brain)}")
    show(brain, "风控", "风控项目")

    print("\n  再问一遍同一个问题（热层命中，不再付冷层取回代价）：")
    hits = await manager.recall(db, "tenant-1", "u-1", q, top_k=3)
    for h in hits:
        print(f"    相似度 {h.similarity:.3f} 「{h.content[:34]}」")
    print(f"  {stats(brain)}")
    show(brain, "风控", "风控项目")

    print("\n" + "=" * 64)
    print("收尾 · 自证与性能（anti-FIFO-shell + 计数器）")
    print("=" * 64)
    d = brain.diagnostics()
    print(f"  tier 分位 {d['tier_p10_p50_p90']}（跨度 {d['tier_spread']}，"
          f"恒 0 就是套壳 FIFO）")
    print(f"  M' 分位 {d['m_p10_p50_p90']}（铺开 = 存储强度有差异，没饱和）")
    print(f"  热层 {d['hot']} / 冷层 {d['archive']}，τ 范围 {d['tau_range']}")
    print(f"  性能: {brain.perf_summary()['query']}")
    print("\n  想看全量对照（vs LRU/FIFO/TTL 的三档基准 + 仪表盘）：")
    print("    .venv/Scripts/python benchmarks/memory_bench.py")
    print("    然后浏览器打开 benchmarks/bench_dashboard.html")


if __name__ == "__main__":
    asyncio.run(main())
