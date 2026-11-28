"""真实语料回放：脱敏后的考勤项目群聊，验 BrainMemory 的写入 / 召回 / 回归。

诚实标注：
  - 内容全真（脱敏：人名→角色、手机号/验证码抹除）
  - 时钟合成：话题切换 +5000 事件、整场对话重复 3 个 epoch（epoch 间 +50000），
    模拟「隔段时间回到同一话题」——真实日志只有一天，不合成时钟就测不到时间侧。

时序按因果交错：话题回归的 burst 开头**先提问、后写入**该 burst 的消息，
与真实使用一致（用户回来先问，助手此时只有旧记忆可用）。

写入门槛单独评估（mainspring 自带 Reflector 对中文技术群聊误杀严重，见报告），
回放本身把门槛关掉（threshold=0），只测记忆机制。

对照 LRU（同热层预算、无冷层——缓存踢了就是踢了；探针命中时按缓存语义重排）。
产出 benchmarks/real_replay/real_replay_report.json + 控制台报告。
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path

from mainspring.memory.brain import BrainConfig, BrainMemory
from mainspring.memory.reflect import Reflector

LOG = Path(__file__).with_name("chat_log_anonymized.txt")

TOPIC_KEYWORDS = {
    "规则核算": ["打卡规则", "加班规则", "核算", "修改规则"],
    "考勤周期": ["考勤周期", "统计周期", "统计方案", "当月周期", "周期"],
    "考勤组": ["考勤组", "考勤档案"],
    "权限": ["权限", "按钮", "菜单"],
    "验证码": ["验证码"],
    "节假日": ["节假日"],
    "排班": ["排班"],
    "报表": ["报表", "表报"],
    "回归测试": ["用例", "回归"],
    "晚报": ["晚报", "存量", "SIT", "UAT", "待解决", "激活"],
}

PROBES = {
    "规则核算": "修改打卡规则后怎么核算日考勤",
    "考勤周期": "如何确定考勤周期",
    "考勤组": "员工没有考勤组是否展示日考勤",
    "权限": "模型权限怎么配置",
    "验证码": "验证码是多少",
    "节假日": "节假日同步失败怎么办",
    "排班": "按需排班要改成什么",
    "报表": "报表问题怎么处理",
    "回归测试": "失败用例怎么分配",
    "晚报": "今天的bug晚报",
}

GAP_SWITCH = 5000.0
GAP_EPOCH = 50000.0
EPOCHS = 3
HOT_BUDGET = 10
TOP_K = 5


def make_embedder():
    """云 embedding（阿里云百炼 dashscope）：有 key 用云，没有退回哈希占位。

    Key 只从进程环境变量读（DASHSCOPE_API_KEY），不落盘。
    """
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        return None, "hash:512d"
    import dashscope
    from http import HTTPStatus
    dashscope.base_http_api_url = "https://cn-beijing.maas.qianwenaiapi.com/api/v1"

    class DashScopeEmbedder:
        model = "qwen3.7-text-embedding-flash"

        def __init__(self) -> None:
            self.calls = 0

        def _embed_sync(self, texts: list[str]) -> list[list[float]]:
            resp = dashscope.TextEmbedding.call(model=self.model, input=texts, api_key=key)
            if resp.status_code != HTTPStatus.OK:
                raise RuntimeError(
                    f"dashscope embedding failed: {resp.status_code} {resp.code} {resp.message}")
            self.calls += 1
            return [e["embedding"] for e in resp.output["embeddings"]]

        async def embed(self, texts: list[str]) -> list[list[float]]:
            return await asyncio.to_thread(self._embed_sync, texts)

    return DashScopeEmbedder(), "dashscope:qwen3.7-text-embedding-flash"


def parse_log() -> list[tuple[str, str]]:
    msgs: list[tuple[str, str]] = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([^:：]+)[:：]\s?(.*)$", line)
        if m and len(m.group(1)) <= 10:
            msgs.append((m.group(1), m.group(2).strip()))
        elif msgs:
            msgs[-1] = (msgs[-1][0], msgs[-1][1] + " " + line)
    return msgs


def topic_of(text: str) -> str:
    best, best_n = "其它", 0
    for name, kws in TOPIC_KEYWORDS.items():
        n = sum(text.count(k) for k in kws)
        if n > best_n:
            best, best_n = name, n
    return best


def build_timeline(msgs: list[tuple[str, str]]) -> list[dict]:
    events, t, prev = [], 0.0, None
    for epoch in range(EPOCHS):
        if epoch > 0:
            t += GAP_EPOCH
        for spk, txt in msgs:
            top = topic_of(txt)
            if prev is not None and top != prev:
                t += GAP_SWITCH
            prev = top
            events.append({"t": t, "epoch": epoch, "speaker": spk,
                           "topic": top, "text": txt})
            t += 1.0
    return events


class LRUNoArchive:
    name = "LRU"

    def __init__(self, budget: int) -> None:
        self.H = budget
        self.hot: list[str] = []

    def insert(self, key: str) -> None:
        if key in self.hot:
            self.hot.remove(key)
        self.hot.append(key)
        del self.hot[: max(0, len(self.hot) - self.H)]


async def main() -> None:
    msgs = parse_log()
    events = build_timeline(msgs)
    refl = Reflector()
    embedder, embedder_tag = make_embedder()

    # ---- 写入门槛单独评估（不参与回放） ----
    filter_scores = [{"text": ev["text"],
                       "score": await refl.score(ev["text"])} for ev in events]
    killed_at_035 = [s for s in filter_scores if s["score"] < 0.35]
    key_killed = [s for s in killed_at_035 if any(
        k in s["text"] for k in ("考勤周期", "打卡规则", "加班规则", "统计方案", "核算"))]

    # ---- 回放：门槛关掉，只测记忆机制 ----
    brain = BrainMemory(BrainConfig(hot_budget=HOT_BUDGET), embedder=embedder)
    lru = LRUNoArchive(HOT_BUDGET)
    topic_id: dict[str, int] = {}
    key_topic: dict[str, str] = {}
    probe_results = []
    burst_seen: dict[str, int] = {}
    epoch_probed: set[tuple[str, int]] = set()
    prev_topic: str | None = None

    for ev in events:
        topic = ev["topic"]
        if topic != prev_topic:
            burst_seen[topic] = burst_seen.get(topic, 0) + 1
            is_return = burst_seen[topic] > 1
            if is_return and topic in PROBES and (topic, ev["epoch"]) not in epoch_probed:
                epoch_probed.add((topic, ev["epoch"]))
                pre_archive = set(brain.archive.keys())
                hits = await brain.recall(None, "tenant-1", "u-1", PROBES[topic], top_k=TOP_K)
                relevant = [h for h in hits if key_topic.get(h.id) == topic]
                cold_rescue = any(h.id in pre_archive for h in relevant)
                topic_keys = {k for k, tp in key_topic.items() if tp == topic}
                lru_hit = bool(topic_keys & set(lru.hot))
                if lru_hit:
                    for k in topic_keys & set(lru.hot):
                        lru.insert(k)
                sim = max((h.similarity or 0.0 for h in relevant), default=0.0)
                probe_results.append({
                    "topic": topic, "epoch": ev["epoch"], "t": ev["t"],
                    "relevant_top5": bool(relevant), "cold_rescue": cold_rescue,
                    "lru_hit": lru_hit, "best_sim": round(sim, 3),
                })
        prev_topic = topic
        tid = topic_id.setdefault(topic, len(topic_id))
        key = BrainMemory._content_key(ev["text"])
        key_topic[key] = topic
        brain.insert_key(key, ev["t"], tid, content=ev["text"])
        lru.insert(key)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus": {
            "messages": len(msgs), "epochs": EPOCHS, "events": len(events),
            "gap_switch": GAP_SWITCH, "gap_epoch": GAP_EPOCH,
            "hot_budget": HOT_BUDGET, "clock_synthetic": True,
            "content_real_anonymized": True, "embedder": embedder_tag,
            "embed_calls": getattr(embedder, "calls", 0),
        },
        "write_filter_finding": {
            "threshold_tested": 0.35,
            "killed": len(killed_at_035), "total": len(filter_scores),
            "key_conclusions_killed": [s["text"][:48] for s in key_killed[:6]],
        },
        "probes": probe_results,
        "summary": {
            "n_probes": len(probe_results),
            "brain_relevant": sum(1 for p in probe_results if p["relevant_top5"]),
            "brain_cold_rescue": sum(1 for p in probe_results if p["cold_rescue"]),
            "lru_hit": sum(1 for p in probe_results if p["lru_hit"]),
            "best_sim_mean": round(sum(p["best_sim"] for p in probe_results)
                                   / max(1, len(probe_results)), 3),
        },
        "diagnostics": brain.diagnostics(),
        "perf": brain.perf_summary(),
    }
    here = Path(__file__).resolve().parent
    (here / "real_replay_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    s = report["summary"]
    w = report["write_filter_finding"]
    print(f"语料：{len(msgs)} 条真实消息 × {EPOCHS} epoch = {len(events)} 个事件（时钟合成）"
          f" · embedding: {embedder_tag}")
    print(f"\n【写入选材实测】mainspring 自带打分在阈值 0.35 下误杀 "
          f"{w['killed']}/{w['total']}（{100.0 * w['killed'] / w['total']:.0f}%），"
          f"其中含业务关键结论 {len(w['key_conclusions_killed'])} 条，例如：")
    for t in w["key_conclusions_killed"]:
        print(f"  ✗ 被滤掉: {t}")
    print("  → 回放实验因此把写入门槛关掉，只测记忆机制；门槛修复是独立结论（见报告）")

    print(f"\n【话题回归探针】共 {s['n_probes']} 次（话题第 2 次出现起、每 epoch 一次）：")
    print(f"  {'话题':<8}{'探针':>5}{'Brain相关':>9}{'冷层救援':>9}{'LRU命中':>8}")
    by_topic: dict[str, list[dict]] = {}
    for p in probe_results:
        by_topic.setdefault(p["topic"], []).append(p)
    for topic, ps in by_topic.items():
        print(f"  {topic:<8}{len(ps):>5}{sum(1 for p in ps if p['relevant_top5']):>9}"
              f"{sum(1 for p in ps if p['cold_rescue']):>9}"
              f"{sum(1 for p in ps if p['lru_hit']):>8}")
    print(f"\n汇总：Brain 相关命中 {s['brain_relevant']}/{s['n_probes']}"
          f"（其中冷层救援 {s['brain_cold_rescue']} 次，相关命中平均相似度 "
          f"{s['best_sim_mean']}）；LRU 热层命中 {s['lru_hit']}/{s['n_probes']}")
    d = report["diagnostics"]
    print(f"自证：tier 分位 {d['tier_p10_p50_p90']}、M' 分位 {d['m_p10_p50_p90']}、"
          f"热层 {d['hot']} / 冷层 {d['archive']}、驱逐 {d['evictions']}、提回 {d['promotions']}")
    print(f"性能：query p99 {report['perf']['query']['p99_ms']}ms，"
          f"QPS {report['perf']['query']['qps']}")
    print("\n[已写出 real_replay_report.json]")


if __name__ == "__main__":
    asyncio.run(main())
