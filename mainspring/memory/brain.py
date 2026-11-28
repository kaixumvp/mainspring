"""BrainMemory：人脑机制实测筛选后的长期记忆后端（memory-math 子集）。

设计出处是 memory-math 项目（memmath/agent_memory.py）的四层推导 + 基准实测，
这里只搬**被实测验证值钱的子集**，被否决的机制保留开关但默认关闭，理由写在注释里：

  1. 层级稳定度（L2 离散级联 + 桥）
     tau(tier) = ladder_base · rho^tier，对数刻度。新条目 tier=0，
     命中按 Montanari-Seung 规则晋升：tier += ms_gain · p_reuse。
     等比阶梯自动产生幂律长尾（d = ln(1/q)/ln(rho)），调两个数定全部时间尺度。

  2. 价值驱逐（实测唯一赢家：phased +11.6% 相对 LRU）
     热层有界，驱逐按 value = p_reuse · (0.3 + 0.7·O) · loss 取最小者。
     O 是提取强度（随 tau 衰减），loss 只在冷层有损（fidelity<1）时起作用。
     收益全部来自「topic 内份额 × O」的驱逐排序，不是时间风险预测——
     时间项在消融里贡献 0.00pp，但保留，因为它同时是「冷层该提回谁」的判据。

  3. 有损冷层（L4 信息瓶颈）
     被驱逐条目进冷层而不是删除：M' 乘 fidelity 打折，取回算部分价值
     （archive_value）。删除是策略错误——人类遗忘是可访问性下降，不是信息丢失。

  4. 自证（anti-FIFO-shell）
     diagnostics() 报 tier/m 分位。真塌成 FIFO 时 tier 恒 0、m 早早饱和，
     基准数字就不用看了。

已知限制（诚实标注）：v1 为进程内实现（db 参数忽略，重启即失）；语义召回是
冷层线性扫描，生产应换 ANN 索引——两个端口（Embedder / LongTermMemory）就是
为这个替换留的缝。
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from mainspring.ports import Embedder, MemoryEntry


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# ---------------------------------------------------------------- 复用概率预测

class ReusePredictor:
    """在线估 p(下次复用)。两条独立证据，缺任何一条都会退化回近因排序。

    rate   : topic 的访问强度（按时间半衰的计数）—— 认「现在正忙的那块」
    hazard : topic 自己的回访间隔分布（log 分桶直方图）—— 认「该回来了」

    hazard 是唯一能给出「很久没用、但马上要回来」判断的项：
    残余危险率 P(lookahead 内回访 | 已 age 没回访)。它不是 age 的单调函数——
    真实回访间隔双峰（爆发内几百、爆发间上万），LRU/TTL 把整条分布压成
    单调递减函数，结构上看不见第二个峰。
    """

    def __init__(self, n_topics: int, halflife: float = 3000.0, bins: int = 16,
                 tmin: float = 1.0, tmax: float = 1e5):
        self.halflife = halflife
        self.bins = bins
        self.tmin = tmin
        self.span = math.log(tmax / tmin)
        self.rate = [0.0] * n_topics
        self.last = [-1.0] * n_topics
        self.seen = [0] * n_topics
        self.hist = [[0.0] * bins for _ in range(n_topics)]

    def _ensure(self, topic: int) -> None:
        """自动归簇会动态开出新 topic，预测器数组随需扩容。"""
        while len(self.rate) <= topic:
            self.rate.append(0.0)
            self.last.append(-1.0)
            self.seen.append(0)
            self.hist.append([0.0] * self.bins)

    def _bin(self, dt: float) -> int:
        if dt <= self.tmin:
            return 0
        return min(self.bins - 1, int(math.log(dt / self.tmin) / self.span * self.bins))

    def observe(self, topic: int, t: float) -> None:
        self._ensure(topic)
        self.seen[topic] += 1
        if self.last[topic] < 0.0:
            self.rate[topic] = 1.0
        else:
            dt = t - self.last[topic]
            self.hist[topic][self._bin(dt)] += 1.0
            self.rate[topic] = self.rate[topic] * 0.5 ** (dt / self.halflife) + 1.0
        self.last[topic] = t

    def tail(self, topic: int, dt: float, lookahead: float) -> tuple[float, float]:
        """(还活着的 episode 数, 其中在 dt+lookahead 之内回访的数目)。"""
        h = self.hist[topic]
        b0 = self._bin(max(self.tmin, dt))
        alive = sum(h[b0:])
        if alive <= 0.0:
            return 0.0, 0.0
        b1 = self._bin(dt + lookahead)
        soon = sum(h[b0:b1 + 1]) if b1 >= b0 else h[b0]
        return alive, soon

    def return_prob(self, topic: int, t: float, lookahead: float) -> float:
        """P(lookahead 内回访 | 到现在已经 age 没回访) = 尾概率之比。"""
        self._ensure(topic)
        if self.last[topic] < 0.0:
            return 0.0
        alive, soon = self.tail(topic, t - self.last[topic], lookahead)
        return soon / alive if alive > 0.0 else 0.0


def _tail_ratio(h: list[float] | None, b0: int, b1: int) -> float:
    if not h:
        return 0.0
    alive = sum(h[b0:])
    if alive <= 0.0:
        return 0.0
    return (sum(h[b0:b1 + 1]) if b1 >= b0 else h[b0]) / alive


# ---------------------------------------------------------------- 配置 / 条目

@dataclass
class BrainConfig:
    """memory-math Config 的忠实移植。默认值就是实测用的默认值。

    被否决机制的开关保留但默认关闭（item_weight=0、min_cluster=1），
    用途是负对照——见 benchmarks/memory_bench.py 的消融。
    """

    hot_budget: int = 150
    ladder_base: float = 8.0
    rho: float = 3.0
    n_tiers: float = 7.0
    m_init: float = 0.05
    eta: float = 0.55
    e0: float = 0.30
    effort_weight: float = 1.0
    kappa: float = 0.0
    ms_gain: float = 0.20
    min_cluster: float = 1.0
    target_r: float = 0.42
    horizon: float = 3e5
    archive_value: float = 0.25
    slot_cost: float = 0.40
    topic_halflife: float = 3000.0
    lookahead: float = 900.0
    item_weight: float = 0.0
    item_shrink: float = 10.0
    topic_weight: float = 0.45
    use_share: bool = True
    fidelity: float = 0.70
    promote_p: float = 0.30
    archive_budget: int = 100000  # 冷层上限，超了按最后想起时间淘汰（0=不限）

    def top_tau(self) -> float:
        return self.ladder_base * self.rho ** self.n_tiers


@dataclass
class BrainItem:
    key: str
    content: str = ""
    topic: int = 0
    tier: float = 0.0
    m: float = 0.05
    t_touch: float = 0.0
    uses: int = 0
    p_time: float = -1.0
    p_val: float = 0.0
    archived: bool = False
    importance: float = 0.5
    confidence: float = 0.5
    tenant_id: str = ""
    user_id: str | None = None
    session_id: str | None = None
    created_at: datetime | None = None
    vec: list[float] | None = None

    def tau(self, cfg: BrainConfig) -> float:
        return cfg.ladder_base * cfg.rho ** self.tier

    def O(self, cfg: BrainConfig, t: float) -> float:
        return math.exp(-max(0.0, t - self.t_touch) / self.tau(cfg))

    def R(self, cfg: BrainConfig, t: float) -> float:
        return self.m * self.O(cfg, t)


# ---------------------------------------------------------------- 离线降级 embedding

def hash_embed(text: str, dim: int = 512) -> list[float]:
    """零依赖确定性 embedding：中文按字二元组、ASCII 按词与字符三元组，
    特征哈希 + L2 归一化。两条提取通道分开，混合中英字符串也能正常重叠。

    仅用于离线/无宿主 embedding 的演示与基准冒烟；生产请注入 ports.Embedder。
    md5 派生符号保证跨进程稳定（不用内置 hash()——它受 PYTHONHASHSEED 影响）。
    """
    v = [0.0] * dim
    s = (text or "").strip().lower()
    if not s:
        return v
    grams: list[str] = []
    cjk = [c for c in s if "\u4e00" <= c <= "\u9fff"]
    grams.extend("".join(t) for t in zip(cjk, cjk[1:]))
    grams.extend(re.findall(r"[a-z0-9]+", s))
    grams.extend("".join(t) for t in zip(s, s[1:], s[2:])
                 if all(c.isascii() and c.isalnum() for c in t))
    for g in grams:
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest()[:8], 16)
        v[h % dim] += 1.0 if (h >> 31) & 1 else -1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------- 记忆引擎

class BrainMemory:
    """同步策略引擎（可离线基准）+ ports.LongTermMemory 异步门面。

    引擎时间 t 是**事件时钟**：benchmark 传显式步数；门面自增。
    topic_halflife / lookahead 因此以「事件数」为单位——真实部署若接墙钟，
    需要按事件率重标定，这是 memory-math 已知限制的如实继承。
    """

    name = "memory-model"

    def __init__(self, cfg: BrainConfig | None = None, n_topics: int = 9,
                 embedder: Embedder | None = None, seed: int = 0) -> None:
        self.cfg = cfg or BrainConfig()
        self.pred = ReusePredictor(n_topics, self.cfg.topic_halflife)
        self.n_keys = 0
        self.hot: dict[str, BrainItem] = {}
        self.archive: dict[str, BrainItem] = {}
        self.ibins: dict[str, list[float]] = {}
        self.ghist = [0.0] * self.pred.bins
        self.topic_in_hot = [0] * n_topics
        self.embedder = embedder
        self._t = 0.0
        self._topic_centers: list[list[float]] = []
        self._topic_by_key: dict[str, int] = {}
        self.stats = {"hot_hit": 0, "archive_hit": 0, "miss": 0,
                      "slots": 0, "evictions": 0, "promotions": 0}
        self._lat: dict[str, list[int]] = {"insert": [], "query": [], "consolidate": []}
        self._lat_cap = 10000

    def _lat_append(self, op: str, ns: int) -> None:
        lst = self._lat[op]
        lst.append(ns)
        if len(lst) > self._lat_cap:
            del lst[: len(lst) - self._lat_cap]

    # ---- 门面时钟与元数据 --------------------------------------------------

    def _now(self) -> float:
        return self._t

    def _advance(self) -> float:
        self._t += 1.0
        return self._t

    # ---- 引擎核心（benchmark 直接调用） ------------------------------------

    def insert_key(self, key: str, t: float, topic: int, content: str = "") -> None:
        t0 = time.perf_counter_ns()
        it = self.archive.pop(key, None)
        if it is None and key not in self.hot:
            it = BrainItem(key=key, content=content, topic=topic,
                           m=self.cfg.m_init, t_touch=t)
            self.n_keys += 1
        elif it is None:
            it = self.hot[key]
        if key not in self.hot:
            self._hot_add(it)
        self.hot[key] = it
        self._ensure_space(t)
        self._lat_append("insert",time.perf_counter_ns() - t0)

    def query_key(self, key: str, t: float, topic: int) -> str:
        t0 = time.perf_counter_ns()
        it = self.hot.get(key)
        if it is not None:
            self._touch(it, t)
            self.stats["hot_hit"] += 1
            self._lat_append("query",time.perf_counter_ns() - t0)
            return "hot"
        it = self.archive.get(key)
        if it is not None:
            del self.archive[key]
            it.m = _clamp(it.m * self.cfg.fidelity, 0.0, 1.0)
            self._hot_add(it)
            self._touch(it, t)
            self.hot[key] = it
            self.stats["archive_hit"] += 1
            self.stats["promotions"] += 1
            self._ensure_space(t)
            self._lat_append("query",time.perf_counter_ns() - t0)
            return "archive"
        self.pred.observe(topic, t)
        self.stats["miss"] += 1
        self._lat_append("query",time.perf_counter_ns() - t0)
        return "miss"

    def consolidate_now(self, t: float, budget: int) -> int:
        t0 = time.perf_counter_ns()
        c = self.cfg
        pool = [(it, True) for it in self.hot.values() if it.R(c, t) < c.target_r]
        pool += [(it, False) for it in self.archive.values()
                 if it.R(c, t) < c.target_r and self.p_reuse(it, t) >= c.promote_p]
        pool.sort(key=lambda x: -self._value(x[0], t))
        used = 0
        for it, in_hot in pool[:budget]:
            if not in_hot:
                self.archive.pop(it.key, None)
                self._hot_add(it)
                self.hot[it.key] = it
                self.stats["promotions"] += 1
            self._touch(it, t, record=False)
            used += 1
        if used:
            self.stats["slots"] += used
            self._ensure_space(t)
        self._lat_append("consolidate",time.perf_counter_ns() - t0)
        return used

    # ---- 内部机制（与 memory-math 同式） -----------------------------------

    def _hot_add(self, it: BrainItem) -> None:
        while len(self.topic_in_hot) <= it.topic:
            self.topic_in_hot.append(0)
        self.topic_in_hot[it.topic] += 1

    def _hot_del(self, it: BrainItem) -> None:
        self.topic_in_hot[it.topic] -= 1

    def _item_hist(self, key: str) -> list[float]:
        h = self.ibins.get(key)
        if h is None:
            h = [0.0] * self.pred.bins
            self.ibins[key] = h
        return h

    def _item_return(self, it: BrainItem, t: float) -> float:
        """条目级 hazard，向全局直方图收缩。**默认权重 0 —— 实测判死**：

        item_weight   phased   stationary
          0.0          11.66       2.33
          0.2          12.05       1.94
          0.5          10.65       2.04
          1.0          10.09       2.13

        条目级区分度已由 fraction() 提供，单条目 1~2 段复用历史估出的
        危险率非 0 即 1，纯噪声。保留作负对照。
        """
        c = self.cfg
        age = max(self.pred.tmin, t - it.t_touch)
        b0 = self.pred._bin(age)
        b1 = self.pred._bin(age + c.lookahead)
        h = self.ibins.get(it.key)
        n_i = sum(h) if h else 0.0
        pi = _tail_ratio(h, b0, b1) if h else 0.0
        pp = _tail_ratio(self.ghist, b0, b1)
        k = c.item_shrink
        return (n_i * pi + k * pp) / (n_i + k) if n_i + k > 0.0 else 0.0

    def _touch(self, it: BrainItem, t: float, record: bool = True) -> None:
        c = self.cfg
        r_now = it.R(c, t)
        effort = _clamp(1.0 - r_now, 0.0, 1.0)
        gain = _clamp(c.e0 + (1.0 - c.e0) * effort * (1.0 + c.effort_weight * effort)
                      + c.kappa * effort, 0.0, 1.0)
        it.m = _clamp(it.m + c.eta * (1.0 - it.m) * gain, 0.0, 1.0 - 1e-9)
        if record and it.uses >= 1:
            b = self.pred._bin(max(1.0, t - it.t_touch))
            self._item_hist(it.key)[b] += 1.0
            self.ghist[b] += 1.0
        it.t_touch = t
        it.uses += 1
        self.pred.observe(it.topic, t)
        self._promote(it, t)

    def _promote(self, it: BrainItem, t: float) -> None:
        """MS 最优规则：dS/S ∝ 预测复用概率。min_cluster 是 L2 BCM 判据
        （gain·m ≥ 4·decay）的落点，默认 1.0 = 关（实测 ≤0.25pp，负对照用）。"""
        c = self.cfg
        step = c.ms_gain * self.p_reuse(it, t)
        if c.min_cluster > 1.0:
            m = self.topic_in_hot[it.topic]
            step *= min(1.0, m / c.min_cluster)
        it.tier = _clamp(it.tier + step, 0.0, self.cfg.n_tiers)

    def p_reuse(self, it: BrainItem, t: float) -> float:
        """P(lookahead 内再用到)。写成概率的「或」合并而非加权和——
        加权和会被 clamp 顶到 1，所有条目同值 ⇒ 预测器变常数 ⇒ 变 FIFO 壳。"""
        c = self.cfg
        if it.p_time == t:
            return it.p_val
        p_item = _clamp(c.item_weight * self._item_return(it, t), 0.0, 1.0) \
            if c.item_weight > 0.0 else 0.0
        p_topic = self.pred.return_prob(it.topic, t, c.lookahead) * self.fraction(it)
        it.p_time, it.p_val = t, _clamp(
            1.0 - (1.0 - p_item) * (1.0 - c.topic_weight * p_topic), 0.0, 1.0)
        return it.p_val

    def fraction(self, it: BrainItem) -> float:
        if not self.cfg.use_share:
            return 1.0
        tot = self.pred.seen[it.topic]
        return it.uses / tot if tot > 0.0 else 1.0 / max(1, self.n_keys)

    def _value(self, it: BrainItem, t: float) -> float:
        """一个热层格子的价值 = 会再用 × 现在抓得住 × 丢了掉多少。"""
        c = self.cfg
        loss = 1.0 if c.fidelity >= 1.0 else (0.4 + 0.6 * it.m)
        return self.p_reuse(it, t) * (0.3 + 0.7 * it.O(c, t)) * loss

    def _ensure_space(self, t: float) -> None:
        while len(self.hot) > self.cfg.hot_budget:
            worst = min(self.hot.values(), key=lambda i: self._value(i, t))
            del self.hot[worst.key]
            self._hot_del(worst)
            worst.archived = True
            worst.t_touch = t  # 入冷时间戳：修剪时按此排序（最久没被想起的先淘汰）
            self.archive[worst.key] = worst
            self.stats["evictions"] += 1
        if self.cfg.archive_budget > 0:
            self._trim_cold(t)

    def _trim_cold(self, t: float) -> None:
        """冷层修剪：超上限时按 t_touch 淘汰最久没被想起的条目（模仿长期遗忘）。"""
        while len(self.archive) > self.cfg.archive_budget:
            oldest = min(self.archive.values(), key=lambda it: it.t_touch)
            del self.archive[oldest.key]

    # ---- 诊断 ---------------------------------------------------------------

    def diagnostics(self) -> dict:
        pool = list(self.hot.values()) + list(self.archive.values())
        if not pool:
            return {"n_items": 0}
        tiers = sorted(it.tier for it in pool)
        ms = sorted(it.m for it in pool)

        def q(a: list[float], p: float) -> float:
            return a[min(len(a) - 1, int(p * len(a)))]

        return {
            "n_items": len(pool), "hot": len(self.hot), "archive": len(self.archive),
            "tier_p10_p50_p90": [round(q(tiers, p), 2) for p in (0.1, 0.5, 0.9)],
            "tier_spread": round(tiers[-1] - tiers[0], 2),
            "m_p10_p50_p90": [round(q(ms, p), 3) for p in (0.1, 0.5, 0.9)],
            "m_mean": round(sum(ms) / len(ms), 3),
            "tau_range": [round(self.cfg.ladder_base, 1), round(self.cfg.top_tau(), 1)],
            "m_below_target": sum(1 for it in pool if it.m < self.cfg.target_r),
            "hot_per_topic": dict(enumerate(self.topic_in_hot)),
            **self.stats,
        }

    def perf_summary(self) -> dict:
        """每操作延迟（ns→ms）与吞吐。p50/p99 出自本进程实测样本。"""
        out: dict[str, dict] = {}
        for op, samples in self._lat.items():
            if not samples:
                out[op] = {"ops": 0, "p50_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0,
                           "max_ms": 0.0, "qps": 0.0}
                continue
            s = sorted(samples)
            ms = [x / 1e6 for x in s]
            total_s = sum(s) / 1e9
            out[op] = {
                "ops": len(s),
                "p50_ms": round(ms[len(ms) // 2], 4),
                "p99_ms": round(ms[min(len(ms) - 1, int(0.99 * len(ms)))], 4),
                "mean_ms": round(sum(ms) / len(ms), 4),
                "max_ms": round(ms[-1], 4),
                "qps": round(len(s) / total_s, 1) if total_s > 0 else 0.0,
            }
        return out

    # ---- ports.LongTermMemory 门面 ------------------------------------------

    @staticmethod
    def _content_key(content: str) -> str:
        return hashlib.sha1((content or "").strip().encode("utf-8")).hexdigest()[:16]

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if self.embedder is not None:
            return await self.embedder.embed(texts)
        return [hash_embed(x) for x in texts]

    def _assign_topic(self, text: str, vec: list[float]) -> int:
        """未显式给 topic 时按 embedding 最近质心归簇；太远则开新簇。"""
        best_t, best_s = -1, 0.30
        for i, c in enumerate(self._topic_centers):
            sim = _cosine(vec, c)
            if sim > best_s:
                best_t, best_s = i, sim
        if best_t < 0:
            self._topic_centers.append(vec[:])
            self.topic_in_hot.append(0)
            return len(self._topic_centers) - 1
        c = self._topic_centers[best_t]
        for j in range(len(c)):
            c[j] = 0.9 * c[j] + 0.1 * vec[j]
        return best_t

    async def add(self, db, *, tenant_id: str, user_id: str | None,
                  session_id: str | None, content: str,
                  importance: float = 0.5, confidence: float = 0.5,
                  topic: int | None = None) -> str:
        """写入：同用户同内容幂等（返回既有 id），不同用户同内容各自存储；
        显式 topic 优先，否则按内容自动归簇（归簇映射是内容级全局缓存）。"""
        t0 = time.perf_counter_ns()
        uid = user_id or ""
        chash = self._content_key(content)
        key = f"user:{uid}:{chash}"
        t = self._advance()
        if topic is None:
            if chash in self._topic_by_key:
                topic = self._topic_by_key[chash]
            else:
                (vec,) = await self._embed([content])
                topic = self._assign_topic(content, vec)
                self._topic_by_key[chash] = topic
        if key in self.hot or key in self.archive:
            return key
        it = BrainItem(key=key, content=content, topic=topic, m=self.cfg.m_init,
                       t_touch=t, importance=importance, confidence=confidence,
                       tenant_id=tenant_id, user_id=user_id, session_id=session_id,
                       created_at=datetime.now(UTC))
        self.n_keys += 1
        self._hot_add(it)
        self.hot[key] = it
        self._ensure_space(t)
        self._lat_append("insert",time.perf_counter_ns() - t0)
        return key

    async def recall(self, db, tenant_id: str, user_id: str | None,
                     query: str, top_k: int = 5) -> list[MemoryEntry]:
        """语义召回：按 user_id 隔离扫描热层+冷层，命中者走晋升与统计。

        隔离语义：门面写入的条目一律带 user_id，严格只对该用户可见；
        引擎路径插入的条目（无 user_id）视为未标属主、对所有召回可见——
        基准/测试走引擎路径，业务走门面路径。
        v1 是线性扫描（热层 ≤hot_budget 条 + 冷层全量）——生产规模请换 ANN。
        """
        (qvec,) = await self._embed([query])
        scored: list[tuple[float, BrainItem]] = []
        for it in self.hot.values():
            if it.user_id and it.user_id != user_id:
                continue
            scored.append((await self._sim(it, qvec), it))
        for it in self.archive.values():
            if it.user_id and it.user_id != user_id:
                continue
            scored.append((await self._sim(it, qvec), it))
        scored.sort(key=lambda x: -x[0])
        t = self._advance()
        out: list[MemoryEntry] = []
        for sim, it in scored[:max(0, top_k)]:
            self.query_key(it.key, t, it.topic)
            out.append(MemoryEntry(
                id=it.key, tenant_id=it.tenant_id or tenant_id, user_id=it.user_id or user_id,
                session_id=it.session_id, content=it.content or it.key,
                importance=it.importance, confidence=it.confidence,
                source="brain", similarity=round(sim, 4), created_at=it.created_at,
            ))
        return out

    async def _sim(self, it: BrainItem, qvec: list[float]) -> float:
        return _cosine(qvec, await self._item_vec(it))

    async def _item_vec(self, it: BrainItem) -> list[float]:
        """条目向量缓存：云 API 按调用计费，内容不可变，每条只算一次。"""
        if it.vec is not None:
            return it.vec
        if self.embedder is None:
            it.vec = hash_embed(it.content)
        else:
            vecs = await self.embedder.embed([it.content])
            it.vec = vecs[0]
        return it.vec

    async def consolidate(self, db, budget: int = 20) -> int:
        """热层维护：跌破 target_r 的条目按价值重放。宿主可在会话结束后调度。"""
        return self.consolidate_now(self._advance(), budget)


__all__ = ["BrainConfig", "BrainItem", "BrainMemory", "ReusePredictor", "hash_embed"]
