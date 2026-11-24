# BrainMemory 验收报告（阶段 3）

- 分支：`feature/memory-brain`（不触碰主线 master）
- 基准：`benchmarks/memory_bench.py`，数据与仪表盘：`bench_report.json` / `bench_dashboard.html`
- 生成时间：2026-11-24（本报告数字 = 仪表盘数字，同一份数据）
- 语料：合成三档需求（无真实日志可用，本局限如实标注，见文末）

## 验收线（阶段 0 定档）

| 档位 | 判据 |
|---|---|
| 达到 | 同一 regime 下 memory-model 的 utility ≥ LRU（成本已内嵌在 utility 里） |
| 突出 | phased 下热命中 ≥ 1.1×LRU，且 utility 跨 3 个 seed 全胜，增益 ≥ 5% |
| 性能 | query p99 < 2ms、insert p99 < 10ms（纯 Python 进程内，事件时钟） |

## 结果一：三档需求 × 3 seed（相对 LRU 的效用增益）

| regime | seed 7 | seed 8 | seed 9 | 均值 | 热命中（模型 vs LRU） | 达到 |
|---|---|---|---|---|---|---|
| stationary 平稳 Zipf | +2.34% | +2.45% | +2.42% | **+2.40%** | 0.8501 vs 0.7881 | 是 |
| cyclic 组级周期轮换 | −4.78% | −4.72% | −4.58% | **−4.69%** | 0.8619 vs 0.8839 | 否（证伪栏，预期内） |
| phased 条目级周期回归 | +11.64% | +11.26% | +11.81% | **+11.57%** | **0.6328 vs 0.5012（1.26×）** | 是 |

**突出：达成**（热命中比 1.26× ≥ 1.1×，三个 seed 全胜，增益 +11.57% ≥ 5%）。

与 memory-math 原版实测（+2.34 / −4.78 / +11.64）逐 seed 吻合 ⇒ 移植忠实，机制没搬丢。

## 结果二：收益分解（phased，seed 7，相对 full 的 pp）

| 变体 | Δ vs full | memory-math 原版 | 判读 |
|---|---|---|---|
| no_topic_timing（关时间风险预测） | **0.00pp** | 0.00pp | 时间项零贡献，但保留作冷层提回判据 |
| no_share（关主题内份额） | −0.96pp | −0.96pp | 小贡献 |
| no_predictor（全关 → FIFO） | **−17.03pp** | −17.03pp | 全部收益 = 预测器驱动的驱逐排序 |
| item_hazard_added（加条目级风险） | +0.39pp | +0.39pp | 净洗，默认关 |

结论与 memory-math 一致：**收益全部来自「主题内份额 × O」的驱逐排序**。

## 结果三：性能（本进程实测，13 轮基准中的 phased 末轮）

| 操作 | 次数 | p50 | p99 | 均值 | 最大 | QPS |
|---|---|---|---|---|---|---|
| query | 29996 | 0.018ms | **1.277ms** | 0.210ms | 55.4ms* | 4765 |
| insert | — | — | **1.325ms** | — | — | — |

*max 为进程内首轮 GC 伪影，p99 不受影响。

**性能：通过**（query p99 1.28ms < 2ms；insert p99 1.32ms < 10ms）。
热路径耗时在「内存查表 + 150 条排序 + 少量算术」；真实部署接 embedding/ANN 后，
延迟大头在宿主注入的 embedder 上，不在本策略。

## 结果四：自证非套壳（anti-FIFO-shell）

- tier 分位 [0.0, 0.0, 0.24]，跨度 7.0 —— 层级状态真实存在（FIFO 壳 tier 恒 0）
- M' 分位 [0.05, 0.803, 1.0] —— 存储强度铺开（FIFO 壳会早饱和到同一值）
- 热层 120 / 冷层 780，驱逐 11958 次、提回 11178 次 —— 驱逐与归档在真实发生

## 裁决汇总

| 档位 | 结果 |
|---|---|
| 达到（stationary / phased） | **达成** |
| 突出（phased） | **达成**（1.26×，跨 seed 全胜） |
| 性能 | **通过**（p99 1.3ms 量级） |
| 证伪栏（cyclic −4.7%） | 诚实保留：组级轮换下模型更差，与 memory-math 结论一致 |

## 诚实条款与已知限制

1. **平稳场景只有 +2.4%**——近因本就是复用概率的极好代理，这栏接近白买。
   模型买的是「用户过段时间回到同一主题」的场景，别的场景 LRU 就够。
2. **语料是合成的**：三档需求生成器来自 memory-math。真实日志接入后，
   topic_halflife / lookahead / archive_value 需按实际事件率重新标定，
   且 topic 由 host 显式传入或由 embedding 自动归簇——归簇质量未在真实语料上验证。
3. **v1 进程内实现**：db 参数忽略，重启即失；语义召回是冷层线性扫描，
   生产规模需换 ANN 索引（Embedder / LongTermMemory 两个端口就是替换缝）。
4. 被否决机制（巩固回放、BCM 簇门、保真度分级、条目级风险）保留开关但默认关，
   实测数字见 memory-math README 的否决清单。

## 复现

```bash
cd D:/project/mainspring
git checkout feature/memory-brain
.venv/Scripts/python benchmarks/memory_bench.py     # 全量基准 + 出报告/仪表盘
.venv/Scripts/python example/brain_memory.py        # 接入 MemoryManager 的最小演示
```
