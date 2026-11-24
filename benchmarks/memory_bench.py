"""BrainMemory 基准：与 LRU / FIFO / 固定 TTL / 近因衰减在同一查询流上正面对照。

移植自 memory-math 的 experiments（memmath/agent_memory.benchmark），需求生成器
逐式一致，目的是在 mainspring 里**复现**那组实测数字（平稳 ≈ LRU、组级轮换更差、
条目级周期回归 +11% 量级），并产出给非算法读者看的验收报告：

  - 三个大白话数字：热层命中率 / 总召回率 / 每千次查询成本（并排对照 LRU）
  - 性能计数器：insert / query / consolidate 的 p50、p99、QPS（本进程实测）
  - 自证非套壳：tier 分位、m 分位、has_real_state
  - 验收线（阶段 0 定档）：
      达到   = 同一 regime 下 utility ≥ LRU（成本已内嵌在 utility 里）
      突出   = phased：热命中 ≥ 1.1×LRU 且 utility 跨 3 个 seed 全胜
      性能   = query p99 < 2ms、insert p99 < 10ms（纯 Python 进程内实现）

运行：cd 仓库根 && .venv/Scripts/python benchmarks/memory_bench.py
产出：benchmarks/bench_report.json（数据）+ benchmarks/bench_dashboard.html（仪表盘）
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from mainspring.memory.brain import BrainConfig, BrainMemory

# ---------------------------------------------------------------- 朴素基线（移植）


class Baseline:
    """所有基线共享同一假设：被驱逐条目落入无损冷层，取回算部分价值。
    所以各策略的差别只在「热层这一格里放谁」，不在「丢不丢」。"""

    name = "base"

    def __init__(self, hot_budget: int, n_topics: int = 1) -> None:
        self.H = hot_budget
        self.hot: list[str] = []
        self.seen: set[str] = set()
        self.stats = {"hot_hit": 0, "archive_hit": 0, "miss": 0,
                      "slots": 0, "evictions": 0, "promotions": 0}

    def _admit(self, key: str, t: float = 0.0) -> None:
        raise NotImplementedError

    def on_hot_hit(self, key: str, t: float) -> None:
        pass

    def insert(self, key: str, t: float = 0.0, topic: int = 0) -> None:
        self.seen.add(key)
        n0 = len(self.hot)
        self._admit(key, t)
        self.stats["evictions"] += max(0, n0 - len(self.hot))

    def query(self, key: str, t: float = 0.0, topic: int = 0) -> str:
        self.seen.add(key)
        if key in self.hot:
            self.stats["hot_hit"] += 1
            self.on_hot_hit(key, t)
            return "hot"
        if key in self.seen:
            self.stats["archive_hit"] += 1
            self.stats["promotions"] += 1
            n0 = len(self.hot)
            self._admit(key, t)
            self.stats["evictions"] += max(0, n0 - len(self.hot))
            return "archive"
        self.stats["miss"] += 1
        return "miss"

    def consolidate(self, t: float, budget: int) -> int:
        return 0


class LRU(Baseline):
    name = "LRU"

    def _admit(self, key: str, t: float = 0.0) -> None:
        if key in self.hot:
            self.hot.remove(key)
        self.hot.append(key)
        del self.hot[: max(0, len(self.hot) - self.H)]

    def on_hot_hit(self, key: str, t: float) -> None:
        self._admit(key, t)


class FIFO(Baseline):
    name = "FIFO"

    def _admit(self, key: str, t: float = 0.0) -> None:
        if key in self.hot:
            return
        self.hot.append(key)
        del self.hot[: max(0, len(self.hot) - self.H)]


class FixedTTL(Baseline):
    """固定 TTL —— L1 里「单一时间常数」假设的策略化。"""

    name = "fixed-TTL"

    def __init__(self, hot_budget: int, n_topics: int = 1, ttl: float = 400.0) -> None:
        super().__init__(hot_budget, n_topics)
        self.ttl = ttl
        self.stamp: dict[str, float] = {}

    def _expire(self, t: float) -> None:
        for k in [k for k, s in self.stamp.items() if t - s > self.ttl]:
            if k in self.hot:
                self.hot.remove(k)
            self.stamp.pop(k, None)

    def _admit(self, key: str, t: float = 0.0) -> None:
        self.stamp[key] = t
        if key not in self.hot:
            self.hot.append(key)
        self._expire(t)
        del self.hot[: max(0, len(self.hot) - self.H)]

    def on_hot_hit(self, key: str, t: float) -> None:
        self.stamp[key] = t

    def query(self, key: str, t: float = 0.0, topic: int = 0) -> str:
        self._expire(t)
        return super().query(key, t, topic)


class RecencyDecay(Baseline):
    """per-item 近因衰减驱逐：有逐项时间常数，但没有 M'、没有复用预测。"""

    name = "recency-decay"

    def __init__(self, hot_budget: int, n_topics: int = 1, tau: float = 600.0) -> None:
        super().__init__(hot_budget, n_topics)
        self.tau = tau
        self.stamp: dict[str, float] = {}

    def _admit(self, key: str, t: float = 0.0) -> None:
        self.stamp[key] = t
        if key not in self.hot:
            self.hot.append(key)
        while len(self.hot) > self.H:
            worst = min(self.hot, key=lambda k: self.stamp.get(k, 0.0))
            self.hot.remove(worst)
            self.stamp.pop(worst, None)

    def on_hot_hit(self, key: str, t: float) -> None:
        self.stamp[key] = t


# ---------------------------------------------------------------- 需求生成（移植）


def _sample(rng: random.Random, w: list[float]) -> int:
    tot, u, acc = sum(w), rng.random(), 0.0
    if tot <= 0.0:
        return 0
    for i, wi in enumerate(w):
        acc += wi
        if u * tot <= acc:
            return i
    return len(w) - 1


def _demand(n_admitted: int, zipf: list[float], topic_of: list[int], t: int,
            regime: str, switch_period: int, n_blocks: int,
            period: list[float] | None = None, offset: list[float] | None = None,
            pulse: float = 120.0) -> list[float]:
    """三档需求：stationary 平稳 Zipf（LRU 主场）、cyclic 组级轮换（证伪栏）、
    phased 条目级周期回归（LRU 结构性打不赢的形态）。"""
    if n_admitted == 0:
        return []
    if regime == "phased":
        out = []
        for i in range(n_admitted):
            P = period[i] if period else 1.0
            off = offset[i] if offset else 0.0
            d = (P - ((t - off) % P)) % P
            out.append(math.exp(-d / pulse) * zipf[i] ** 0.25)
        return out
    if regime != "cyclic":
        return zipf[:n_admitted]
    hot_block = (t // switch_period) % n_blocks
    return [zipf[i] * (6.0 if (topic_of[i] % n_blocks) == hot_block else 0.05)
            for i in range(n_admitted)]


# ---------------------------------------------------------------- 基准（移植）


def _admit_all(policies: list, key: str, t: float, topic: int) -> None:
    for p in policies:
        if isinstance(p, BrainMemory):
            p.insert_key(key, t, topic)
        else:
            p.insert(key, t, topic)


def _query_all(policies: list, key: str, t: float, topic: int) -> None:
    for p in policies:
        if isinstance(p, BrainMemory):
            p.query_key(key, t, topic)
        else:
            p.query(key, t, topic)


def benchmark(n_items: int = 900, n_topics: int = 9, T: int = 30000,
              hot_budget: int = 120, regime: str = "stationary",
              switch_period: int = 5000, seed: int = 7, arrival_rate: float = 0.06,
              consolidation_every: int = 300, consolidation_budget: int = 20,
              n_blocks: int = 3, effort_weight: float = 1.0, pulse: float = 120.0,
              lookahead: float | None = None, archive_value: float | None = None,
              slot_cost: float | None = None, target_r: float | None = None,
              cfg_over: dict | None = None) -> dict:
    rng = random.Random(seed)
    zipf = [1.0 / (i + 1) ** 1.15 for i in range(n_items)]
    topic_of = [i % n_topics for i in range(n_items)]
    keys = [f"i{i}" for i in range(n_items)]
    period = [200.0 * (100.0 ** (i / max(1, n_items - 1))) for i in range(n_items)]
    offset = [i * 0.6180339887 * period[i] for i in range(n_items)]

    cfg = BrainConfig(hot_budget=hot_budget, horizon=float(T * 8),
                      effort_weight=effort_weight, **(cfg_over or {}))
    if target_r is not None:
        cfg.target_r = target_r
    if lookahead is not None:
        cfg.lookahead = lookahead
    if archive_value is not None:
        cfg.archive_value = archive_value
    if slot_cost is not None:
        cfg.slot_cost = slot_cost
    mm = BrainMemory(cfg, n_topics, seed=seed)
    policies: list[Baseline] = [LRU(hot_budget), FIFO(hot_budget),
                                FixedTTL(hot_budget), RecencyDecay(hot_budget), mm]

    n_admitted = 0
    for t in range(T):
        if n_admitted < n_items and rng.random() < arrival_rate:
            _admit_all(policies, keys[n_admitted], float(t), topic_of[n_admitted])
            n_admitted += 1
        w = _demand(n_admitted, zipf, topic_of, t, regime, switch_period, n_blocks,
                    period, offset, pulse)
        if not w:
            continue
        pick = _sample(rng, w)
        _query_all(policies, keys[pick], float(t), topic_of[pick])
        if t % consolidation_every == 0:
            mm.consolidate_now(float(t), consolidation_budget)

    out = {}
    for p in policies:
        s = p.stats
        n = max(1, s["hot_hit"] + s["archive_hit"] + s["miss"])
        out[p.name] = {
            "queries": n,
            "hot_rate": round(s["hot_hit"] / n, 4),
            "archive_rate": round(s["archive_hit"] / n, 4),
            "total_recall": round((s["hot_hit"] + s["archive_hit"]) / n, 4),
            "utility": round((s["hot_hit"] + cfg.archive_value * s["archive_hit"]
                              - cfg.slot_cost * s["slots"]) / n, 4),
            "slots_per_1k_q": round(1000.0 * s["slots"] / n, 2),
            "evictions": s["evictions"],
            "promotions": s["promotions"],
        }
    lru_u = out["LRU"]["utility"]
    return {
        "regime": regime, "T": T, "n_items": n_items, "n_admitted": n_admitted,
        "hot_budget": hot_budget, "target_r": round(cfg.target_r, 3),
        "lookahead": cfg.lookahead, "archive_value": cfg.archive_value,
        "effort_weight": effort_weight, "seed": seed, "policies": out,
        "gain_vs_LRU_pct": round(100.0 * (out["memory-model"]["utility"] - lru_u) / lru_u, 2)
        if lru_u else None,
        "diagnostics": mm.diagnostics(),
        "perf": mm.perf_summary(),
    }


# ---------------------------------------------------------------- 验收线（阶段 0 定档）

ACCEPT = {
    "reach": "同一 regime 下 memory-model 的 utility ≥ LRU（成本已内嵌在 utility 里）",
    "outstanding": "phased 下热命中 ≥ 1.1×LRU，且 utility 跨 3 个 seed 全胜",
    "perf": "query p99 < 2ms、insert p99 < 10ms（纯 Python 进程内，事件时钟）",
}

PREDICTOR_VARIANTS = {
    "full": {},
    "no_share": {"use_share": False},
    "no_topic_timing": {"lookahead": 1e12},
    "no_predictor": {"topic_weight": 0.0},
    "item_hazard_added": {"item_weight": 0.2},
}


def _mean(rows: list[float]) -> float:
    return sum(rows) / len(rows)


def compute_verdicts(results: dict) -> dict:
    v = {}
    for regime, rows in results.items():
        util_m = [_r["policies"]["memory-model"]["utility"] for _r in rows]
        util_l = [_r["policies"]["LRU"]["utility"] for _r in rows]
        hot_m = [_r["policies"]["memory-model"]["hot_rate"] for _r in rows]
        hot_l = [_r["policies"]["LRU"]["hot_rate"] for _r in rows]
        v[regime] = {
            "gain_pct_mean": round(_mean([_r["gain_vs_LRU_pct"] or 0.0 for _r in rows]), 2),
            "hot_model": round(_mean(hot_m), 4),
            "hot_lru": round(_mean(hot_l), 4),
            "hot_ratio": round(_mean(hot_m) / max(1e-9, _mean(hot_l)), 3),
            "utility_model": round(_mean(util_m), 4),
            "utility_lru": round(_mean(util_l), 4),
            "reach": _mean(util_m) >= _mean(util_l) - 1e-4,
            "win_every_seed": all(m >= l - 1e-4 for m, l in zip(util_m, util_l)),
        }
    v["phased"]["outstanding"] = (
        v["phased"]["hot_ratio"] >= 1.1 and v["phased"]["win_every_seed"]
        and v["phased"]["gain_pct_mean"] >= 5.0
    )
    perf = results["phased"][-1]["perf"]
    v["perf"] = {
        "query_p99_ms": perf["query"]["p99_ms"],
        "insert_p99_ms": perf["insert"]["p99_ms"],
        "query_qps": perf["query"]["qps"],
        "insert_qps": perf["insert"]["qps"],
        "pass": perf["query"]["p99_ms"] < 2.0 and perf["insert"]["p99_ms"] < 10.0,
    }
    return v


# ---------------------------------------------------------------- 仪表盘


def _bars(rows: list[tuple[str, float, str, str]]) -> str:
    mx = max(abs(x[1]) for x in rows) or 1.0
    parts = []
    for label, val, note, color in rows:
        w = abs(val) / mx * 100
        parts.append(
            f'<div class="bar-row"><span class="lab">{label}</span>'
            f'<div class="track"><div class="fill" style="width:{w:.1f}%;background:{color}"></div></div>'
            f'<span class="num">{note}</span></div>'
        )
    return "".join(parts)


def write_dashboard(report: dict, path: Path) -> None:
    verdicts = report["verdicts"]
    reg_rows = []
    for regime, label in (("stationary", "平稳 Zipf"), ("cyclic", "组级周期轮换"),
                          ("phased", "条目级周期回归")):
        v = verdicts[regime]
        color = "#4c9aff" if regime == "stationary" else "#e06c5f" if regime == "cyclic" else "#5ac48a"
        reg_rows.append((label, v["gain_pct_mean"],
                         f"{v['gain_pct_mean']:+.2f}% · 热命中 {v['hot_model']:.3f} vs LRU {v['hot_lru']:.3f}",
                         color))
    abl = report["ablation"]
    base_u = abl["full"]["utility"]
    abl_rows = []
    for tag, label in (("no_topic_timing", "关时间风险预测"), ("no_share", "关主题内份额"),
                       ("no_predictor", "全关（→FIFO）"), ("item_hazard_added", "加条目级风险")):
        d = abl[tag]["delta_vs_full_pct"]
        abl_rows.append((label, d, f"{d:+.2f}pp · 效用 {abl[tag]['utility']:.4f}",
                         "#8b93a3" if tag == "no_topic_timing" else "#b08ae0"
                         if tag == "no_share" else "#e06c5f" if tag == "no_predictor" else "#4c9aff"))
    d = report["diagnostics"]
    perf = verdicts["perf"]
    html = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>BrainMemory 验收仪表盘</title>
<style>
body{{font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif;max-width:1000px;margin:24px auto;padding:0 18px;color:#20242c;background:#f6f7f9}}
h1{{font-size:20px}} h2{{font-size:16px;border-left:3px solid #d9a441;padding-left:8px;margin-top:28px}}
.card{{background:#fff;border:1px solid #e3e6ec;border-radius:8px;padding:14px 18px;margin:10px 0}}
.bar-row{{display:flex;align-items:center;gap:10px;margin:6px 0;font-size:13px}}
.bar-row .lab{{width:150px;color:#5c6470;text-align:right}}
.bar-row .track{{flex:1;height:16px;background:#eef0f4;border-radius:5px;position:relative;overflow:hidden}}
.bar-row .fill{{position:absolute;top:0;bottom:0;left:0;border-radius:4px}}
.bar-row .num{{width:300px;font-family:ui-monospace,monospace;font-size:12.5px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border:1px solid #e3e6ec;padding:5px 9px;text-align:left}}
th{{background:#f0f2f6;color:#5c6470}} td.num{{text-align:right;font-family:ui-monospace,monospace}}
.badge{{display:inline-block;padding:2px 10px;border-radius:10px;font-size:12.5px;font-weight:500}}
.ok{{background:#e3f5ea;color:#1e7a44}} .no{{background:#fdeaea;color:#b3392f}}
.note{{color:#5c6470;font-size:12.5px;margin-top:6px}}
</style></head><body>
<h1>BrainMemory 验收仪表盘（mainspring 分支 feature/memory-brain）</h1>
<div class="note">{report['generated_at']} · 基准参数 T=30000、条目 900、热层 120、冷层价值 0.25 · 每档 3 个 seed</div>

<h2>一、三个大白话数字：三档需求 × 相对 LRU 的效用增益</h2>
<div class="card">{_bars(reg_rows)}
<div class="note">效用 = (热命中 + 0.25×冷命中 − 0.40×整理槽位) / 查询数。冷层免费（=1）时 LRU 就是准最优，
所以这套对照用 0.25 的冷层代价——真 agent 里冷层取回 = 重新读文件/塞上下文，值远小于 1。</div></div>

<h2>二、收益分解：预测器逐项消融（phased）</h2>
<div class="card">{_bars(abl_rows)}
<div class="note">memory-math 实测结论：收益全部来自「主题内份额 × O」的驱逐排序；时间风险预测 0.00pp；
全关退化 FIFO 是 −17pp 量级。本栏验证移植没把机制搬丢。</div></div>

<h2>三、性能计数器（本进程实测，事件时钟）</h2>
<div class="card"><table>
<tr><th>操作</th><th>次数</th><th>p50</th><th>p99</th><th>均值</th><th>最大</th><th>QPS</th></tr>
<tr><td>query（热/冷/未中）</td><td class="num">{report['perf']['query']['ops']}</td><td class="num">{report['perf']['query']['p50_ms']}ms</td><td class="num">{report['perf']['query']['p99_ms']}ms</td><td class="num">{report['perf']['query']['mean_ms']}ms</td><td class="num">{report['perf']['query']['max_ms']}ms</td><td class="num">{perf['query_qps']}</td></tr>
<tr><td>insert（写入+驱逐）</td><td class="num">{report['perf']['insert']['ops']}</td><td class="num">{report['perf']['insert']['p50_ms']}ms</td><td class="num">{report['perf']['insert']['p99_ms']}ms</td><td class="num">{report['perf']['insert']['mean_ms']}ms</td><td class="num">{report['perf']['insert']['max_ms']}ms</td><td class="num">{perf['insert_qps']}</td></tr>
<tr><td>consolidate（整理批次）</td><td class="num">{report['perf']['consolidate']['ops']}</td><td class="num">{report['perf']['consolidate']['p50_ms']}ms</td><td class="num">{report['perf']['consolidate']['p99_ms']}ms</td><td class="num">{report['perf']['consolidate']['mean_ms']}ms</td><td class="num">{report['perf']['consolidate']['max_ms']}ms</td><td class="num">{report['perf']['consolidate']['qps']}</td></tr>
</table>
<div class="note">性能验收线：query p99 &lt; 2ms、insert p99 &lt; 10ms →
<span class="badge {'ok' if perf['pass'] else 'no'}">{'通过' if perf['pass'] else '未通过'}</span>
（纯 Python 进程内；真实部署接 embedding/ANN 时，延迟大头在宿主注入的 embedder 上，不在本策略。）</div></div>

<h2>四、自证非套壳（anti-FIFO-shell）</h2>
<div class="card"><table>
<tr><th>指标</th><th>值</th><th>判读</th></tr>
<tr><td>tier 分位 p10/p50/p90</td><td class="num">{d['tier_p10_p50_p90']}</td><td>真塌成 FIFO 时 tier 恒 0</td></tr>
<tr><td>tier 跨度</td><td class="num">{d['tier_spread']}</td><td>&gt;1 才算有层级状态</td></tr>
<tr><td>M' 分位 p10/p50/p90</td><td class="num">{d['m_p10_p50_p90']}</td><td>FIFO 壳会早早饱和到同一值</td></tr>
<tr><td>热层/冷层条目</td><td class="num">{d['hot']} / {d['archive']}</td><td>驱逐在发生、归档在积累</td></tr>
</table></div>

<h2>五、验收裁决</h2>
<div class="card"><table>
<tr><th>regime</th><th>相对 LRU（3 seed 均值）</th><th>达到（效用 ≥ LRU）</th></tr>
<tr><td>stationary</td><td class="num">{verdicts['stationary']['gain_pct_mean']:+.2f}%</td><td><span class="badge {'ok' if verdicts['stationary']['reach'] else 'no'}">{'是' if verdicts['stationary']['reach'] else '否'}</span></td></tr>
<tr><td>cyclic</td><td class="num">{verdicts['cyclic']['gain_pct_mean']:+.2f}%</td><td><span class="badge {'ok' if verdicts['cyclic']['reach'] else 'no'}">{'是' if verdicts['cyclic']['reach'] else '否'}</span></td></tr>
<tr><td>phased</td><td class="num">{verdicts['phased']['gain_pct_mean']:+.2f}%（热命中比 {verdicts['phased']['hot_ratio']:.2f}×）</td><td><span class="badge {'ok' if verdicts['phased']['reach'] else 'no'}">{'是' if verdicts['phased']['reach'] else '否'}</span></td></tr>
</table>
<div class="note" style="margin-top:8px">「效果突出」= phased 热命中 ≥ 1.1×LRU 且跨 seed 全胜且增益 ≥ 5% →
<span class="badge {'ok' if verdicts['phased']['outstanding'] else 'no'}">{'达成' if verdicts['phased']['outstanding'] else '未达成'}</span>
· 诚实预期：平稳场景 ≈ LRU 是设计内的（近因本就是复用概率的极好代理），模型买的是非平稳/周期回归那一栏。</div></div>
</body></html>"""
    path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------- 主流程


def run_suite() -> dict:
    t0 = time.time()
    results: dict = {}
    for regime in ("stationary", "cyclic", "phased"):
        rows = []
        for seed in (7, 8, 9):
            rows.append(benchmark(regime=regime, seed=seed))
            print(f"  [{regime} seed={seed}] "
                  f"LRU={rows[-1]['policies']['LRU']['utility']:.4f} "
                  f"model={rows[-1]['policies']['memory-model']['utility']:.4f} "
                  f"gain={rows[-1]['gain_vs_LRU_pct']}%")
        results[regime] = rows

    ablation = {}
    for tag, ov in PREDICTOR_VARIANTS.items():
        r = benchmark(regime="phased", seed=7, cfg_over=ov)
        mm = r["policies"]["memory-model"]
        ablation[tag] = {"utility": mm["utility"], "hot_rate": mm["hot_rate"],
                         "gain_vs_LRU_pct": r["gain_vs_LRU_pct"]}
    base_u = ablation["full"]["utility"]
    for tag in ablation:
        ablation[tag]["delta_vs_full_pct"] = round(
            100.0 * (ablation[tag]["utility"] - base_u) / base_u, 2)

    verdicts = compute_verdicts(results)
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "branch": "feature/memory-brain",
        "acceptance_lines": ACCEPT,
        "verdicts": verdicts,
        "results": {regime: [{k: row[k] for k in ("seed", "gain_vs_LRU_pct", "policies")}
                             for row in rows] for regime, rows in results.items()},
        "ablation": ablation,
        "diagnostics": results["phased"][-1]["diagnostics"],
        "perf": results["phased"][-1]["perf"],
    }
    here = Path(__file__).resolve().parent
    json_path = here / "bench_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    html_path = here / "bench_dashboard.html"
    write_dashboard(report, html_path)

    print("\n== 验收裁决 ==")
    for regime in ("stationary", "cyclic", "phased"):
        v = verdicts[regime]
        print(f"  {regime:<12} 增益 {v['gain_pct_mean']:+.2f}%  热命中 {v['hot_model']:.4f} "
              f"vs LRU {v['hot_lru']:.4f}  达到={'是' if v['reach'] else '否'}")
    print(f"  phased 突出={'是' if verdicts['phased']['outstanding'] else '否'}"
          f"（热命中比 {verdicts['phased']['hot_ratio']:.2f}×）")
    p = verdicts["perf"]
    print(f"  性能: query p99={p['query_p99_ms']}ms insert p99={p['insert_p99_ms']}ms "
          f"query QPS={p['query_qps']}  通过={'是' if p['pass'] else '否'}")
    print(f"\n[已写出 {json_path.name} / {html_path.name}]  总耗时 {time.time() - t0:.1f}s")
    return report


if __name__ == "__main__":
    run_suite()
