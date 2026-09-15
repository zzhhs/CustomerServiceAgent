# CustomerServiceAgent

基于 FastAPI、LangGraph、Pydantic v2 和 DeepSeek 的中心化多智能体客服系统。

当前版本提供一条可运行的纵向链路：

```text
HTTP API → Load Context → Orchestrator → Plan Validator → Task Scheduler
         → QA / Order / After-sales → Result Aggregator → Update Memory
```

`TaskScheduler` 按依赖关系动态释放任务：同一就绪层中互不依赖的任务并发执行，下游任务
会等待所有前置依赖完成；任一依赖失败时，对应下游任务不会执行。

系统会在进程内保存最近的会话消息，并记录尚未补齐的业务字段。例如用户先说“我要申请
退货”，系统会返回 `requires_input=true` 和 `requested_fields=["order_id"]`；用户在同一
会话中补充订单号后，系统会恢复原始退货目标并继续执行。该会话存储是数据库接入前的
临时实现，服务重启后会清空。回复生成器会读取同一用户、同一会话的最近历史，根据已有
的称呼、语气和详略偏好调整回复；订单状态等业务事实仍只采用本轮工具执行结果。

执行失败时，编排器现在支持有预算的闭环：

```text
Execute
  ├─ 临时模型错误 → Retry + Exponential Backoff
  ├─ 执行失败且未确认 → Replan → Validate → Execute
  ├─ 等待确认/确认后失败 → 不自动改计划
  └─ 成功或预算耗尽 → Synthesize
```

默认每个任务最多执行 3 次、整个请求最多重规划 1 次。`TaskResult` 会返回
`attempts`、`error_code` 和 `retryable`，响应 `metadata` 会给出总任务尝试次数和
重规划次数。

运行时由 DeepSeek 完成顶层任务规划，并驱动 `QAAgent`、`OrderAgent` 和
`AfterSalesAgent` 选择各自允许的工具、观察工具结果并结束领域任务，最后再合成回复。
业务规则、权限校验和写操作仍由 Python 代码控制。订单、工单数据库和向量库当前采用
进程内模拟实现，已经预置可测试的假数据。

```text
Orchestrator Agent
  → Specialist Agent
    → Tool / Service
      → Repository / 模拟外部系统
```

## 本地运行

要求 Python 3.12+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
cp .env.example .env
# 编辑 .env，填写 DEEPSEEK_API_KEY
uv sync
uv run uvicorn app.main:app --reload
```

接口文档：<http://127.0.0.1:8000/docs>

```bash
curl http://127.0.0.1:8000/health
```

复合任务示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user-1","conversation_id":"conversation-1","message":"查询订单 A123，如果已经签收并且符合条件就申请退货"}'
```

首次调用会返回 `confirmation_token`。确认写操作时，把该值原样放入下一次请求的
`confirmation_token` 字段。令牌与用户、会话和服务端保存的原始计划绑定，默认十分钟
过期，并且只能成功使用一次；确认阶段不会再次调用 Planner。

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"user-1","conversation_id":"conversation-1","message":"确认申请退货","confirmation_token":"第一次响应返回的令牌"}'
```

## 配置

复制 `.env.example` 为 `.env`，填写你的 DeepSeek Token：

```dotenv
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your-token
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
CONFIRMATION_TTL_SECONDS=600
TASK_MAX_ATTEMPTS=3
RETRY_BASE_DELAY_SECONDS=0.25
MAX_REPLANS=1
```

模型参与顶层 `RoutePlan` 生成、专业 Agent 工具决策/观察、RAG 回答和最终回复合成。
模型只能选择 Agent 被授予的工具，不直接访问订单或工单数据层。

仅运行离线测试时可以设置 `LLM_PROVIDER=deterministic`；生产运行不建议使用它。

本地开发默认使用 `AUTH_MODE=disabled`，此时请求体必须提供 `user_id`。部署环境必须
设置 `APP_ENV=production`、`AUTH_MODE=jwt` 和足够长的 `AUTH_SECRET`，用户身份将从
Bearer JWT 的 `sub` 中取得，请求体里的 `user_id` 会被忽略。JWT 会验证 HS256 签名、
`exp`、`iss` 和 `aud`。实际接入 API Gateway 后，可以把这个验证器替换为网关提供的
JWT 身份上下文。

## Observability

项目使用 OpenTelemetry SDK，并包含以下埋点：

- FastAPI HTTP 服务端 Span。
- `orchestration.invoke/plan/validate/execute/replan/synthesize` Span。
- `agent.task` Span，包含 Agent、Action 和尝试次数。
- `gen_ai.chat` Span，包含模型名和输入/输出 token 数。
- 请求数、任务状态、重试次数、重规划次数和任务耗时指标。
- 带 trace/span ID 的结构化 JSON 日志。
- 开启遥测时，HTTP 响应通过 `X-Trace-ID` 返回可用于排查的链路 ID。

本地直接查看遥测数据：

```dotenv
LOG_FORMAT=json
TELEMETRY_ENABLED=true
OTEL_EXPORTER=console
```

发送到支持 OTLP/HTTP 的 Collector：

```dotenv
LOG_FORMAT=json
TELEMETRY_ENABLED=true
OTEL_EXPORTER=otlp
OTEL_ENDPOINT=http://localhost:4318
```

生产环境通常由 OpenTelemetry Collector 再转发到 Jaeger、Tempo、Prometheus 或
其他可观测性平台。日志和 Span 不记录用户消息、模型提示词、JWT、确认令牌等敏感内容。

## 模拟数据

订单：

- `user-1 / A123`：已签收，可退货。
- `user-1 / B456`：运输中，不可退货。
- `user-2 / C789`：已签收，可退货，用于验证订单归属隔离。

模拟向量库包含退货政策、退款到账时间、配送时效和人工客服时间。检索结果来源会写入
`QAResult.sources`。

## 质量检查

```bash
uv run ruff check .
uv run mypy app
uv run pytest
```
