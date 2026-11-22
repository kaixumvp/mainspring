# Changelog

本文件按 [SemVer](https://semver.org/) 记录 Mainspring 的变更。v1.0 起：破坏性变更只出现在 major，弃用至少跨一个 minor。

## [3.5.0] - 2026-11-22

### 新增
- **请求级上下文预算治理**（`RuntimeConfig.context_policy`，默认 `None` 关闭，行为零变化）：「三数据分离」（完整运行记录归宿主 / 执行状态驱动循环 / 每次请求组装可丢弃的 Context View）在框架内的落地，长跑（几百步）不再靠宿主编外裁剪或撞上下文窗口。
  - `ContextPolicy`：`context_window_tokens`（0=禁用准入）、`max_output_tokens`、`safety_margin_tokens`、`memory_token_budget`、`tool_schema_token_budget`、`tool_result_chars`、`chars_per_token`。
  - `TokenEstimator` 端口（`RuntimeDeps.token_estimator`）：宿主可注入 tokenizer 级精确估算；默认 `CharTokenEstimator`（字符数/`chars_per_token` 启发式，CJK 内容约 1 字符/token，默认 3.0 会**低估**——中文宿主请调低或注入真实估算器）。
  - `build_context()`（`mainspring/core/context.py`，纯函数、无 IO、确定性）：组装顺序 = system → plan → memory → history；压缩顺序 = 记忆尾部裁剪（相关性顺序）→ 工具 schema 预算检查 → 历史从前端丢弃；受保护内容（system 头、最后一条用户消息、工具调用/结果配对）绝不静默丢弃；裁剪处插入 `[N earlier messages omitted by token budget]` 标记（标记自身成本也计入预算）。
  - **超预算 = 响亮失败，绝不静默截断**：工具 schema 超 `tool_schema_token_budget` 抛 `ContextBudgetExceeded`（错误信息直接告诉宿主缩小工具白名单）；压缩后仍超限同样抛出。Decide/Plan 节点以哨兵键返回，运行时在 `run_with_retry` **之外**抛出——准入失败不是瞬时错误，绝不对同一个超预算请求重试。
  - `tool_result_chars`：工具结果进模型上下文前截断，追加 `…[truncated to N chars of M — content incomplete]` 标记（默认 0=不限）。
  - Decide 节点诊断：每次预算化请求在 step output 暴露 `context` 诊断（`estimated_tokens`/`input_limit`/`breakdown`/`actions`，只含计数不含原文）。
- **工具 schema 改为每次 Decide 现算**（`registry.to_openai_schema()` 从 run 级移到请求级），schema 成本参与该请求的预算核算。

### 变更
- `run_plan` / `run_decide` / `run_execute` 签名新增关键字参数（`context_policy` / `token_estimator` / `result_limit`），全部带默认值；未启用 `context_policy` 时消息组装路径与 3.4.0 完全一致。
- `mainspring/__init__.py` 导出 `ContextPolicy`。

## [3.4.0] - 2026-11-22

### 新增
- **DevUI 开发者控制台**（`mainspring dev`，v2.3「开发者体验」交付项）：零依赖（纯 stdlib）的本地单页控制台，可视化一次 run 的步骤时间线、逐 token `message.delta` 输出、原始 `RuntimeEvent` 事件流与人工审批卡。
  - `mainspring/devui/server.py`：`ThreadingHTTPServer` + SSE（`GET /`、`GET /events`、`POST /run`、`POST /approve`）；框架的 `EventSink` 经事件总线桥接到 SSE 订阅者（15s 心跳、慢消费者丢事件不拖垮他人）；asyncio 循环跑在后台线程，请求经 `run_coroutine_threadsafe` 调度；`port=0` 随机端口。
  - `mainspring/devui/demo.py`：**离线 demo agent**（`ScriptedGateway` 脚本化三轮 LLM 响应，流式按字/词分块；无需 API key）。剧情覆盖三个关键机制：`calculator` 工具调用（read 直接执行）→ `transfer` 工具（`permission="admin"`）触发 `awaiting_human` 暂停 → 批准后 `resume` 同 run 续跑收尾。`build_demo_definition()` / `build_demo_deps()` 产出标准 `AgentDefinition` + `RuntimeDeps`。
  - `mainspring/cli.py`：`mainspring dev [--host] [--port] [--delay]` 入口，`[tool.poetry.scripts]` 注册。
  - 前端单页 `devui/static/index.html`（原生 JS 无构建）：步骤时间线 / 实时 delta 面板 / 原始事件流水 / 审批卡。
- **文档站（分层 Markdown）**：`documentation/mainspring/` 新增上手路线：快速上手（3 分钟、离线 10 行跑通）→ `@tool` 开发工具 → 插件与技能 → 进阶（Ports / HITL / 重放）→ DevUI，导航索引已更新。
- `tests/test_devui.py` 8 项：demo 全流程（暂停/批准/完成）、脚本网关流式折叠与工具调用、HTTP 端点、SSE 完整事件序列——全部零外部依赖（`urllib` + 端口 0 临时起服）。

### 变更
- `python = ">=3.11"`（移除 `<3.15` 上界）在本版随发行物正式生效——3.3.0 的 PyPI 元数据仍为 `<3.15`，`pip install mainspring==3.4.0` 起与新约束一致。
- wheel 打包声明 DevUI 静态资源（`mainspring/devui/static/index.html`），PyPI 安装后 `mainspring dev` 可用。

## [3.3.0] - 2026-11-22

### 新增
- **Checkpointer 官方实现**（`mainspring/checkpoint/`）：`Checkpointer` 端口（save/load/delete + list_runs）的三个零配置持久化后端，行为与 `InMemoryCheckpointer` 完全互换，宿主换后端不改 resume 语义：
  - `SQLiteCheckpointer`：stdlib `sqlite3`，单文件单表，WAL + `busy_timeout=5000`；每次操作独立连接（checkpoint 低频路径，比连接池更简单安全），跨进程安全。`:memory:` 显式拒绝——每次操作开新连接会让它每次都是空库。
  - `RedisCheckpointer`：注入 `redis.asyncio.Redis`（与 `WorkingBrain` 同一模式，框架不负责建连接）。键前缀 `cp:` 可配；TTL 默认关闭（过期即丢失快照与 resume 语义冲突，要开显式传 `ttl_seconds`）。
  - `PostgresCheckpointer`：接受宿主提供的任意 DBAPI 2.0 连接（psycopg2/psycopg/…），`import mainspring` 永远不需要 PG 驱动。表在首次使用时懒建（`CREATE TABLE IF NOT EXISTS`，可共存于宿主已有的迁移表）。
- **共享 JSON 编解码信封**（`checkpoint/codec.py`）：`{"schema_version": 1, "state": ...}`。版本不符或载荷损坏在 `load` 时抛 `ValueError`（响亮失败，绝不静默用错状态 resume）；`default=str` 兜底非常规 JSON 值（Datetime 等）。
- 快照语义与 `InMemoryCheckpointer` 对齐：`load` 返回深拷贝（宿主改返回值不影响已存快照），save 覆盖旧快照，`list_runs` 全部排序稳定。

## [3.2.0] - 2026-11-22

### 新增
- **`@tool` 装饰器**（`mainspring/tools/decorator.py`）：类型化函数直接成为 `BaseTool`。函数签名经 Pydantic 生成 OpenAI 风格参数 JSON Schema（无默认值的参数进 `required`、`Annotated[Field(description)]` 成为参数描述、`Literal` → `enum`），docstring 成为工具描述；每次调用先 `validate_call` 校验并强制类型（JSON 参数里的 `"5"` → `5`），校验失败或函数异常降级为 `ToolResult(success=False, error=...)`，不会打断运行循环。
- 同步函数在 worker 线程执行（不阻塞事件循环），异步函数内联执行；返回 `dict`/`str`/`ToolResult` 均正确折叠进 `ToolResult`。
- 用法：裸用 `@tool` 或 `@tool(name=..., description=..., permission=...)`；原函数经 `tool.fn` 可达。导出：`from mainspring.tools import tool, FunctionTool`；与 `ToolRegistry` / `ToolAsPlugin` 直接兼容。

## [3.1.0] - 2026-11-22

### 新增
- **结构化输出**：`chat(messages, model, tools, schema=PydanticModel)`——请求 Pydantic 模型直出。`LLMResponse.parsed` 承载校验后的模型实例；`LiteLLMGateway` 向供应商下发 `response_format={"type": "json_object"}`（尽力而为的提示），框架侧以 `model_validate_json` 强制契约，失败抛 `StructuredOutputError`（`ports.py`）。`FaultTolerantGateway` 将其与其它网关错误同等降级为空响应（`parsed=None`）。
- `schema` 在装饰器链全量透传：`Fallback` / `FaultTolerant` / `Recording` / `Replay` / `Caching` / `TieredRouter`。透传仅在 `schema` 非 None 时携带关键字，V3.0 时代自研网关不受影响；`CachingGateway` 缓存键纳入 `schema`。
- 录制/重放保持确定性：录制只存原始响应（`parsed` 剥离，dump 为纯 JSON），`ReplayGateway` 按 `content` + `schema` 重推导 `parsed`。
- `parse_structured`（`llm/structured.py`）：自研网关可复用同一校验契约。结构化输出暂只覆盖非流式 `chat()`（`chat_stream` 不含 `schema`）。

## [3.0.0] - 2026-11-22

### 变更（破坏性）
- **全量改名 `ouroboros` → `mainspring`**：发行名与导入包名同步改（PyPI 上 `ouroboros` 已被生信包占用、`ouroboros-ai` 定位撞车；外部用户为零，选择一次到位而非分两步）。`pip install mainspring`，`import mainspring`。
  受影响导入写法（逐一替换即可）：

  | 旧 | 新 |
  |---|---|
  | `from ouroboros import AgentRuntime` | `from mainspring import AgentRuntime` |
  | `from ouroboros.ports import LLMGateway` | `from mainspring.ports import LLMGateway` |
  | `from ouroboros.core.runtime import AgentDefinition` | `from mainspring.core.runtime import AgentDefinition` |
  | `from ouroboros.tools.registry import build_default_registry` | `from mainspring.tools.registry import build_default_registry` |

  日志命名空间同步从 `ouroboros` 改为 `mainspring`（`logging.getLogger("mainspring")`）。

## [2.1.0] - 2026-11-21

### 新增
- **流式输出**：`StreamingLLMGateway` 可选协议 + `ChatChunk` / `ToolCallDelta`（`ports.py`），`LiteLLMGateway.chat_stream` 归一化分块实现；Decide 节点边流式边发 `message.delta` 事件（token 级）。仅当注入 `event_sink` 且网关支持 `chat_stream` 时启用流式，其余情况自动回退 `chat()`，下游步骤与流式无关（`llm/stream.py` 折叠为同一 `LLMResponse`）。
- 装饰器流式透传：`FallbackLLMGateway`（首块前才 failover，流中错误不再换厂商以免重复输出）、`FaultTolerantGateway`（首块前降级为 `on_error` 单块）、`RecordingGateway`（边转发边记录折叠响应）、`ReplayGateway`（录制响应合成为单块流，离线重放可跑流式消费方）。
- **上下文治理**：`ContextManager` 端口（`ports.py`）+ 零依赖默认实现 `SlidingWindowContext`（`mainspring/context.py`：保留头部 system 消息 + 最新窗口；切点跳过孤立 tool 消息；超长工具结果就地截断）。`RuntimeDeps.context_manager` 注入自定义策略，或 `RuntimeConfig.max_history_messages` 零配置启用；历史在被压缩后**永久丢弃**（这正是长任务不爆窗口的机制）。
- `RuntimeConfig.enable_planning`（见下方变更）与 `test_streaming.py` / `test_context_window.py` 测试组。

### 变更（行为，非 API）
- **Plan 节点默认关闭**：`enable_planning=False`（此前每次 run 固定先花一次 LLM 调用生成计划，且 `PlanStep.status` 从未被跟踪）。默认路径为 `Decide → Execute → Observe` 循环，省一次 LLM 调用；需要旧行为的宿主显式传 `RuntimeConfig(enable_planning=True)`。记忆召回上下文（`memory_ctx`）改由 Decide 注入，plan 开关不影响召回链路。

## [2.0.0] - 2026-10-10

### 变更（破坏性）
- **导入包名 `src` → `ouroboros`**：发行包 `ouroboros` 此前导出的顶层包字面就叫 `src`，
  与宿主自己的 src-layout 直接冲突，也与本文件 v1.0 段声明的公共 API 面（「顶层 `ouroboros`」）
  从未一致过。所有 `from src.x import y` 改为 `from ouroboros.x import y`。
  公共 API 面的**符号**未变，`__all__` 无改动，仅包名前缀变化。

  受影响的导入写法（逐一替换即可）：

  | 旧 | 新 |
  |---|---|
  | `from src import AgentRuntime` | `from ouroboros import AgentRuntime` |
  | `from src.ports import LLMGateway` | `from ouroboros.ports import LLMGateway` |
  | `from src.core.runtime import AgentDefinition` | `from ouroboros.core.runtime import AgentDefinition` |
  | `from src.tools.registry import build_default_registry` | `from ouroboros.tools.registry import build_default_registry` |

  日志命名空间同步从 `src` 改为 `ouroboros`（`logging.getLogger("ouroboros")`）。

### 移除
- `langgraph` 依赖。核心循环自 v0.3 起就是 `core/runtime.py` 里的手写状态机，
  `build_graph`/`run_agent` 在 v1.0.0 已删除，该依赖自那时起就无人 import。
  连带移除了 `langgraph` / `langgraph-checkpoint` / `langgraph-prebuilt` / `langgraph-sdk` /
  `langchain-core` / `langchain-protocol` / `xxhash` / `ormsgpack` 共 8 个传递依赖。

## [1.1.0] - 2026-08-26

### 新增
- **确定性重放**：`RecordingGateway` / `ReplayGateway` / `dump_recording` / `load_recording`（`llm/replay.py`），同一 run 可离线复现。
- **评测端口**：`Judge` / `JudgeVerdict` 端口（`ports.py`）+ `Case` / `GoldenSet` / `run_evaluation` / `HeuristicJudge`（`evaluation/`）。
- **成本感知**：`CachingGateway`（prompt 缓存复用）、`TieredRouter` + `message_length_classifier`（模型分级路由）、`TokenBudget`（token 预算）。
- `RuntimeConfig.max_total_tokens`：运行期 token 预算，超限中止（`status="failed"`）。

## [1.0.0] - 2026-08-26

### 移除（破坏性）
- 删除 `build_graph` / `run_agent` deprecated shim（`core/graph.py`），请改用 `AgentRuntime`。

### 新增
- `py.typed` 标记，类型完备（PEP 561）。
- 公共 API 面冻结：顶层 `mainspring` 与各子包 `__all__` 显式导出。
- 兼容矩阵：Python 3.11 – 3.14。

## [0.4.0] - 2026-08-26

### 新增
- 可观测：`observability/`（`Tracer` 可选依赖 `opentelemetry-api` + `Metrics` 端口），`trace_id` 经 `InvokeContext` 透传。
- 安全：`FunctionSkill` 默认子进程沙箱；`security/`（注入边界标注 + `Redactor` 脱敏）。
- 韧性：`FallbackLLMGateway` + `FaultTolerantGateway`（`llm/routing.py`）。
- 并发：单轮多 `tool_calls` 并行执行。

## [0.3.0] - 2026-08-26

### 新增
- 端口收口：全部框架↔宿主契约集中到 `ports.py`。
- `AgentRuntime` 门面 + 强类型（`AgentDefinition` / `RunContext` / `RuntimeConfig` / `RuntimeDeps` / `Step` / `RunResult`）。
- 可靠执行：`RetryPolicy` + `CircuitBreaker`、超时、子 Agent 深度、checkpoint 暂停/恢复。
- 事件解耦：`RuntimeEvent` + `EventSink`。
- 知识端口：`Knowledge` / `Retriever` + `KnowledgePlugin`；记忆 `reflect` / 遗忘 / 充分性。

## [0.2.0] - 2026-08-24

### 新增
- 插件协议、四形态 Skill、MCP/HTTP 工具、记忆（Working/Consolidator）、授权、事件；从平台单体独立成包。

## [0.1.0]

### 新增
- 运行循环 + Builtin 工具 + LLM 网关。
