# molo

**Mo**del **Lo**op —— 一个轻量级 Rust Agent 框架，用于构建 LLM 智能体。

[English](../README.md) · 简体中文

molo 提供了推理循环、LLM 通信、工具调用、上下文管理等全套组件，让你通过
积木式组装来构建 Agent，而不用自己手写循环。它面向 tokio 生态，兼容任何
OpenAI 兼容接口，并遵循开放协议（工具的 MCP 协议、技能的 Agent Skills 协议）。

- **内置推理循环** —— `ReActAgent` 实现了经典的"思考 → 调用工具 → 回传结果 →
  作答"循环，自带工具轮次上限、协作式取消、Usage 聚合、事件通道与 tracing 跨度。
- **一切皆可插拔** —— `Provider`（LLM）、`Memory`（上下文）、`Tool`（能力）都是
  trait，按场景替换实现，无需改动循环本身。
- **真实世界的集成** —— MCP 客户端（`McpClient`）接入外部工具，Agent Skills
  （`SkillRegistry`）渐进式披露，JSON Schema 校验的结构化输出。
- **可观测、可控制** —— 通过 `run_stream` 逐 token 流式输出，通过 `EventChannel`
  订阅运行事件，通过 `MessageChannel` 与外部对话（人工确认、Agent 间对话），
  通过 `CancellationToken` 协作式取消。
- **零绑定** —— 不捆绑运行时、不开隐藏线程；用你自己的 tokio 运行时、
  你自己的 tracing subscriber、你自己的 HTTP 栈。

## 快速开始

一个最小 Agent 需要三样东西：与 LLM 通信的 `Provider`、管理上下文的 `Memory`、
（可选）提供外部能力的 `Tool`。推理循环内置于 `ReActAgent`，`react_agent!`
宏一步完成组装。

```toml
[dependencies]
molo = "0.2"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

不依赖真实 API 自测循环，用 `FakeProvider` 注入脚本化回复：

```rust
use molo::{react_agent, Agent, FakeProvider, FakeReply};

let mut agent = react_agent!(
    FakeProvider::new([FakeReply::Text("Hello".into())]),
    "You are a helpful assistant",
);
let answer = agent.run("Are you there?").await?;
assert_eq!(answer, "Hello");
```

接入真实 LLM 时，把 `FakeProvider` 换成 `OpenAiProvider`（任意 OpenAI 兼容
接口），用 `RetryProvider` 包裹以获得重试/超时保护，并挂上工具：

```rust
use molo::{react_agent, Agent, OpenAiProvider};

let provider = OpenAiProvider::new(
    std::env::var("MOLO_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
    std::env::var("MOLO_API_KEY").unwrap_or_default(),
    std::env::var("MOLO_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string()),
);

let mut agent = react_agent!(
    provider,
    [Calculator],                          // 你的工具
    "You are a helpful assistant. Use the calculator tool for calculations.",
);
let answer = agent.run("What is (1 + 2) * 3?").await?;
println!("{answer}");
```

这就是一个完整、支持流式、支持工具调用的 Agent。`run` 返回最终回答；
`run_stream` 逐 token 产出 `MessageChunk` 事件（文本增量、工具调用、
工具结果、`Done`）。

## 核心概念

molo 按领域模块组织 —— 每个模块只负责一个概念。crate 根统一 re-export
各模块的核心类型，`use molo::...` 即可覆盖大多数场景：

| 概念 | 职责 | 核心类型 |
| --- | --- | --- |
| Agent | 推理循环 | `Agent`、`ReActAgent`、`react_agent!`、`TypedAgent`、`CancellableAgent` |
| Provider | LLM 通信 | `Provider`、`OpenAiProvider`、`RetryProvider`、`FakeProvider` |
| Memory | 上下文管理 | `Memory`、`InMemoryMemory`、`WindowMemory`、`SummarizeStrategy` |
| Tool | 外部能力 | `Tool`、`ToolRegistry`、`SharedState`、`#[molo::tool]` |
| Skill | 能力包（Agent Skills 协议） | `Skill`、`SkillRegistry`、`LoadSkillTool` |
| MCP | 外部工具服务器 | `McpClient`、`McpTool` |
| Message | 对话消息模型 | `Message`、`ContentBlock`、`ToolCall` |
| MessageChannel | 外部对话（人/Agent） | `CliMessageChannel`、`MpscChannel`、`BroadcastChannel`、`WatchChannel` |
| EventChannel | 运行过程观测 | `BroadcastEventChannel`、`MpscEventChannel` |

**实现选型**（详见 crate 文档）：

- **上下文** —— 短会话或不限预算用 `InMemoryMemory`；长会话按 token 预算裁剪
  最旧轮次用 `WindowMemory`；叠加 `SummarizeStrategy` 把超预算消息压缩为一条摘要。
- **与 LLM 通信** —— 开发期用 `FakeProvider`（脚本化回复，无需 API）；
  生产用 `OpenAiProvider` + `RetryProvider`。
- **外部对话** —— 人与终端交互用 `CliMessageChannel`；进程内一对一 Agent 对话
  用 `MpscChannel`；一对多广播 / 最新值通知用 `BroadcastChannel` / `WatchChannel`。
- **观测过程** —— `BroadcastEventChannel`（多订阅者，慢者丢弃最旧事件）或
  `MpscEventChannel`（单订阅者，容量内不丢弃）。

## 特性亮点

### 工具：`#[molo::tool]` 一行定义

手写一个工具大约需要 25 行样板代码；宏从 async 函数生成结构体、schema、
参数解析与错误转换：

```rust
use molo::tool::{SharedState, ToolError};

#[molo::tool(description = "Evaluates a math expression, e.g. \"(1 + 2) * 3\"")]
async fn calculator(args: CalcArgs) -> Result<String, ToolError> {
    let value = evalexpr::eval(&args.expression)
        .map_err(|e| ToolError::Execution(e.to_string()))?;
    Ok(value.to_string())
}
```

工具调用失败**不会**中止循环 —— 错误文本回传给模型，由模型决定下一步。
`SharedState` 让工具间共享类型化状态，在调用时注入。

### 结构化（类型化）输出

`run_typed` 根据返回类型推导出的 JSON Schema 校验模型回复；校验失败时
把错误回传给模型重试（默认 3 次）：

```rust
let weather: Weather = agent.run_typed("How is the weather in Beijing today?").await?;
println!("{}°C, {}", weather.temperature, weather.condition);
```

### MCP 客户端

把外部 MCP 服务器暴露的工具接入 Agent。`McpClient` 支持 stdio 子进程与
Streamable HTTP 两种传输（基于 `rmcp`）：

```rust
let mut client = McpClient::from_command("filesystem", ["/path/to/server"]);
let mut registry = ToolRegistry::new();
for tool in client.tools().await? {
    registry.register(tool);
}
```

### 技能（Agent Skills 开放协议）

技能 = 一个包含 `SKILL.md`（YAML frontmatter + Markdown）的目录。核心机制是
**渐进式披露**：模型最初只看到一行"名称 + 描述"的菜单；当任务匹配时，
通过 `load_skill` 工具读取正文。技能通过 `allowed-tools` 声明依赖的工具。

### 子 Agent

`SubAgentTool` / `SubAgentPool` 让主 Agent 委托子 Agent：持久专家
（`from_agent`）、一次性工厂（`from_factory`）、或在调用参数中当场定义的
标准 ReAct 子 Agent（`from_react` / `spawn_react`）。命名池让子 Agent 在
多轮对话间存活，便于后续追问。

### 协作式取消

支持取消的 Agent 实现 `CancellableAgent`。每次运行携带独立的
`CancellationToken`；取消只作用于当前一次运行，中途停止不留残留，
下一轮对话从新 token 开始。

### 流式与事件

- `Agent::run_stream` —— `MessageChunk` 事件：`Delta` / `ToolCall` /
  `ToolResult` / `Done` / `Cancelled`。
- `Provider::stream_chat` —— 原始 `StreamEvent` 流（文本增量、推理、工具调用）。
- `EventChannel` —— 订阅一次运行的 `AgentEvent` 流，实现解耦观测。

### 可观测性

循环在固定位置发出 `tracing` 跨度（`agent.run`、`llm_request`、`tool`，
按轮次分组）。框架不安装 subscriber —— 由你自行接入（如
`tracing-subscriber`），需要的话自行接入 OpenTelemetry。

## 示例

所有示例位于 `examples/`；大多数从 `.env` 读取 `MOLO_API_KEY` /
`MOLO_BASE_URL` / `MOLO_MODEL`（复制 `.example.env` 为 `.env` 并填入真实值）。
不依赖真实 API 的自包含示例以 ✦ 标注。

### 核心循环

| 示例 | 运行 | 演示内容 |
| --- | --- | --- |
| ✦ `react_agent` | `cargo run --example react_agent` | 内置 `ReActAgent` + `react_agent!` 宏（流式 / 普通两种模式） |
| `agent` | `cargo run --example agent` | 手写 `Agent` trait 循环 |
| `tool_agent` | `cargo run --example tool_agent` | 基于 `Provider::chat` / `stream_chat` 的工具循环 |
| `sub_agent` | `cargo run --example sub_agent` | 子 Agent 委托，`SubAgentTool` / `SubAgentPool` |

### 工具

| 示例 | 运行 | 演示内容 |
| --- | --- | --- |
| ✦ `tool_registry` | `cargo run --example tool_registry` | 注册表完整 API：register / names / schemas / call / subset |
| ✦ `tool_macro` | `cargo run --example tool_macro` | `#[molo::tool]` 一键定义工具 |
| ✦ `shared_state` | `cargo run --example shared_state` | `SharedState` 的三种用法 |
| ✦ `mcp` | `cargo run --example mcp` | MCP 客户端适配，自包含假服务器 |

### Provider

| 示例 | 运行 | 演示内容 |
| --- | --- | --- |
| `chat` | `cargo run --example chat` | 真实模型普通对话 |
| `chat_stream` | `cargo run --example chat_stream` | 流式对话，逐 token 打印 |
| ✦ `fake_provider` | `cargo run --example fake_provider` | 脚本化回复，测试自己的循环 |
| `retry` | `cargo run --example retry` | `RetryProvider` 包装器 |
| `usage` | `cargo run --example usage` | 每次运行的执行摘要（token、轮次） |
| `trace` | `cargo run --example trace` | 把 tracing 跨度渲染到控制台 |

### 记忆

| 示例 | 运行 | 演示内容 |
| --- | --- | --- |
| ✦ `window_memory` | `cargo run --example window_memory` | `WindowMemory` 裁剪与自定义裁剪策略 |
| `window_memory_agent` | `cargo run --example window_memory_agent` | 带 token 预算窗口的真实模型 Agent |
| ✦ `summarize` | `cargo run --example summarize` | `SummarizeStrategy`：把旧消息压缩为摘要 |
| `summarize_agent` | `cargo run --example summarize_agent` | 带摘要压缩的流式真实模型 Agent |

### 通道

| 示例 | 运行 | 演示内容 |
| --- | --- | --- |
| ✦ `message_channel` | `cargo run --example message_channel` | 全部四种通道实现（Cli / Mpsc / Broadcast / Watch） |
| `confirm_agent` | `cargo run --example confirm_agent` | 通过 `MessageChannel` 实现人工确认 |
| `mpsc_agent` | `cargo run --example mpsc_agent` | 通过 `MpscChannel` 的双 Agent 对话 |
| `broadcast_agent` | `cargo run --example broadcast_agent` | 一对多广播通知 |
| `watch_agent` | `cargo run --example watch_agent` | 通过 `WatchChannel` 发布最新状态 |
| ✦ `event_channel` | `cargo run --example event_channel` | 订阅事件流（自包含） |
| `event_channel_agent` | `cargo run --example event_channel_agent` | 通过 `EventChannel` 观测真实模型 Agent |

### 输出、技能、取消

| 示例 | 运行 | 演示内容 |
| --- | --- | --- |
| ✦ `structured` | `cargo run --example structured` | `run_typed` 类型化输出与 JSON Schema 校验 |
| ✦ `skill` | `cargo run --example skill` | 技能：发现、渐进式披露、激活 |
| ✦ `cancellation` | `cargo run --example cancellation` | 中途协作式取消，之后继续对话 |

## 配置

示例从 `.env` 读取配置（复制 `.example.env` 为 `.env` 并填入真实值）；
环境变量可直接覆盖：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `MOLO_API_KEY` | *(空)* | API Key；无鉴权的本地端点（如 Ollama）可留空 |
| `MOLO_BASE_URL` | `https://api.openai.com/v1` | OpenAI 兼容接口地址 |
| `MOLO_MODEL` | `gpt-4o-mini` | 模型名称 |
| `MOLO_MAX_TOKENS` | — | 窗口/上下文预算（window、summarize 示例） |
| `MOLO_SUMMARY_MAX_TOKENS` | `150` | 摘要输出上限（summarize 示例） |

## 说明

- **tokio 生态** —— 所有 async API 都要求 tokio 运行时；molo 不自带运行时。
- **MSRV** —— Rust 1.97（edition 2024）。
- **许可证** —— MIT OR Apache-2.0。

## 文档

- Rustdoc：`cargo doc --no-deps`
- [English README](../README.md)
