# DevUI：`mainspring dev` 可视化控制台

> 零依赖开发者控制台：一条命令起一个本地页面，看一次 run 的**步骤时间线、逐 token 输出、原始事件流和人工审批**。内置离线 demo agent——全程不调真实 LLM、不需要 API key。

## 启动

```bash
mainspring dev                          # http://127.0.0.1:8765
mainspring dev --port 0                 # 随机端口（控制台打印实际地址）
mainspring dev --delay 0                # token 流无间隔（默认 0.02s/块）
```

页面三个面板：

| 面板 | 数据来源 |
|------|----------|
| 步骤时间线 | `step.completed` 事件：`decide`（含动作与参数）→ `execute`（含工具结果）→ `observe` |
| 实时输出 | `message.delta` 事件逐块拼接，带 `[decide #N]` 分段标记 |
| 事件追踪 | 原始 `RuntimeEvent` JSON 流水 |
| 审批卡 | `run.awaiting_human` 事件弹出，列出待批资源，一键批准 |

## 内置 demo 剧情

默认任务「计算 123×45，然后把 100 元转给 alice」走完框架的三个关键机制：

1. `decide` → 调 `calculator`（`read` 权限，直接执行）
2. `decide` → 调 `transfer`（`permission="admin"`，**暂停** `awaiting_human`）
3. 点「批准全部」→ `resume` 同 run 续跑 → `decide` 收尾输出答案

demo 完全由 [mainspring/devui/demo.py](../../application/mainspring/mainspring/devui/demo.py) 定义：`ScriptedGateway` 脚本化三轮 LLM 响应（流式按字/词分块），`build_demo_definition()` / `build_demo_deps()` 是标准的 `AgentDefinition` + `RuntimeDeps`——和你在业务代码里写的没有任何区别。

## 架构（可嵌入宿主）

[mainspring/devui/server.py](../../application/mainspring/mainspring/devui/server.py) 是纯 stdlib 实现，整个数据通路如下：

```
浏览器 ── EventSource ── GET /events ──┐
浏览器 ── fetch ──────── POST /run ────┤
浏览器 ── fetch ──────── POST /approve ┤
                                       ▼
                     ThreadingHTTPServer（端口可配，port=0 随机）
                                       │
                ┌──────────────────────┼──────────────────────┐
                │ asyncio 后台线程      │ 事件总线（queue）      │
                │  AgentRuntime        │  EventSink → 队列     │
                │  run / resume        │  → 广播给每个 SSE 订阅 │
                └──────────────────────┴──────────────────────┘
```

- **事件总线桥**：框架的 `EventSink`（async 回调）→ `queue.Queue` → 每个 SSE 订阅者一个独立队列（慢消费者丢事件不拖垮他人），15s 心跳保活。
- **run/approve 同 run 复用 deps**：`ScriptedGateway` 的脚本位置是 run 的进度，`resume` 必须复用同一份 deps（服务器按 `run_id` 持有），否则重放会从头开始。
- **零依赖**：`http.server` + `threading` + 原生 JS（单文件 `static/index.html`），不引入 FastAPI/ws 库，`mainspring dev` 开箱即用。

把这个模式搬进自己的宿主只需三步：用你的 `LiteLLMGateway` 换掉 `ScriptedGateway`、用你的 `DbCheckpointer` 换掉 `InMemoryCheckpointer`、把 `/events` 的 SSE 换成你已有的传输——`EventSink` 拿到的 `RuntimeEvent` 数据与传输无关。

## 测试

`tests/test_devui.py`：demo 全流程（暂停/批准/完成）、脚本网关流式折叠、HTTP 端点、SSE 事件序列——全部零外部依赖（`urllib` + 端口 0 临时起服）。
