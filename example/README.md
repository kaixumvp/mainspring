# Mainspring 示例

每个文件对应一个模块/能力，展示「如何单独使用」以及「如何经 `RuntimeDeps` 配合 Agent 运行」。所有示例默认**离线运行**（用 `_shared.ScriptedLLM` 脚本化 Mock，无需真实 API key）。

## 运行方式

```bash
# 先安装（poetry 环境下 mainspring 已可导入）
poetry install

# 跑单个示例
poetry run python example/first_example.py

# 一次跑完全部示例（独立子进程，逐项汇总通过/失败）
poetry run python example/run_all.py
```

未安装时从仓库根目录指定源码路径（`example/` 内部的 `_shared` 由脚本目录自动导入）：

```bash
PYTHONPATH=. python example/first_example.py
PYTHONPATH=. python example/run_all.py
```

要接真实模型，设置环境变量后改用 `_shared.real_llm()`：

```bash
export MAINSPRING_API_KEY=sk-...   # 或 DEEPSEEK_API_KEY
export MAINSPRING_MODEL=deepseek/deepseek-chat
```

## 清单

### 入门与装配

| 文件 | 能力 | 说明 |
|------|------|------|
| `first_example.py` | 最小闭环 | `AgentRuntime.run` 跑一个带工具调用的 Agent |
| `plugins_tools.py` | 插件 / 工具 | 自定义 `BaseTool`、完整 `Plugin`、`PluginRegistry`、`ToolAsPlugin` |
| `tool_decorator.py` | `@tool` 装饰器 | 类型化函数直出工具：自动 JSON Schema、参数校验、sync/async |
| `skills_example.py` | 技能 | 四形态 Skill（prompt/function/flow/agent）+ `SkillManager` |
| `multi_agent.py` | 多 Agent | `agent-skill` 委派子 Agent + `max_subagent_depth` 深度保护 |
| `ports_example.py` | 端口全景 | 实现并注入全部契约，一次接线跑完整 Agent |

### 记忆与知识

| 文件 | 能力 | 说明 |
|------|------|------|
| `memory.py` | 记忆 | Working / LongTerm / Consolidator / Reflector / MemoryManager |
| `embedding.py` | 向量化 | `HashEmbedder` 离线向量 + 余弦相似度去重；生产换 `LiteLLMEmbedder` |
| `knowledge.py` | 知识 | `Knowledge`/`Retriever` 端口 + `KnowledgePlugin` + 知识回退 |

### 安全与运行控制

| 文件 | 能力 | 说明 |
|------|------|------|
| `permissions_hitl.py` | 权限 / HITL | `check_permission` + 高危暂停 → `resume` 续跑（内存 checkpoint） |
| `checkpoint_sqlite.py` | 持久化检查点 | `SQLiteCheckpointer`：暂停后跨进程/重启恢复同一 run |
| `events_streaming.py` | 事件 | `EventSink` 端口 + `RuntimeEvent` + 宿主侧 SSE 序列化 |
| `streaming.py` | token 级流式 | `StreamingLLMGateway` + `message.delta` + `accumulate_response` 折叠 |
| `structured_output.py` | 结构化输出 | `chat(schema=PydanticModel)` → `LLMResponse.parsed` + 降级 |
| `context_budget.py` | 上下文预算 | `ContextPolicy` / `build_context` / 工具结果截断 / 超预算失败 |
| `reliability.py` | 可靠性 | Retry / 熔断 / 超时 / 子 Agent 深度 |

### 生产化

| 文件 | 能力 | 说明 |
|------|------|------|
| `routing_fallback.py` | 路由 / 容错 | `FallbackLLMGateway` 多厂商切换 + `FaultTolerantGateway` 降级 |
| `observability.py` | 可观测性 | `InMemoryMetrics` 指标 + `Tracer` 注入（无 OTel 时自动 no-op） |
| `security.py` | 安全 | 工具结果边界标注 `wrap_tool_result` + `Redactor` 递归脱敏 |
| `http_tools.py` | HTTP / OpenAPI / MCP | 本地起服务演示内置 `http`、`HttpOpenApiTool`、`McpTool` 装配 |
| `replay.py` | 确定性重放 | `RecordingGateway` / `ReplayGateway`，同一 run 可离线复现 |
| `evaluation.py` | 离线评测 | `GoldenSet` + `Judge` + `run_evaluation` |
| `cost.py` | 成本感知 | `CachingGateway` / `TieredRouter` / `TokenBudget` + 预算中止 |

### 公共工具

| 文件 | 说明 |
|------|------|
| `_shared.py` | `ScriptedLLM` + `real_llm()`（示例间复用，非 API） |
| `run_all.py` | 一键运行全部示例并汇总（独立子进程） |

## 按需求找示例

| 我想要…… | 看这里 |
|-----------|--------|
| 让输出逐 token 流式显示 | `streaming.py`、`events_streaming.py` |
| 让模型输出直接变成 Pydantic 实例 | `structured_output.py` |
| 用最少代码定义工具 | `tool_decorator.py` |
| 控制长对话的上下文窗口与工具结果体积 | `context_budget.py` |
| 高危操作先暂停、审批后再继续 | `permissions_hitl.py` |
| 审批暂停后进程重启也不丢状态 | `checkpoint_sqlite.py` |
| 主厂商故障时自动切换备用模型 | `routing_fallback.py` |
| 接监控、看每步耗时与 token | `observability.py`、`cost.py` |
| 防止工具输出泄露密钥 / 提示注入 | `security.py` |
| 把内部 HTTP API / OpenAPI / MCP 变成工具 | `http_tools.py` |
| 编排父子 Agent、限制递归深度 | `multi_agent.py`、`reliability.py` |
| 上线前跑回归与离线评测 | `replay.py`、`evaluation.py` |

## 与 Agent 配合的关键入口

- **注入端口**：`RuntimeDeps(llm=..., permission_checker=..., permission_context=..., long_term_memory=..., knowledge=..., event_sink=..., checkpointer=..., context_manager=..., token_estimator=..., retry=..., circuit_breaker=..., tracer=..., metrics=..., redactor=...)`
- **装配插件**：`AgentDefinition(model=..., system_prompt=..., plugins=registry.all(), runtime=RuntimeConfig(...))`
- **驱动运行**：`AgentRuntime().run(definition, task, RunContext(...), deps)` → `RunResult`
- **暂停恢复**：`AgentRuntime().resume(definition, run_id, approvals, deps)`（需 `checkpointer`）
- **运行时开关**：`RuntimeConfig(enable_planning=..., max_iterations=..., timeout_seconds=..., max_total_tokens=..., max_subagent_depth=..., context_policy=...)`
