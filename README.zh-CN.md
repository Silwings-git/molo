<div align="center">

# molo

**Mo**del **Lo**op · 可嵌入的 Rust agent runtime 与 harness framework

用一组可组合的构件搭建安全、可扩展的 LLM agent：模型交互、推理循环、
上下文管理、工具调用、结构化输出、可观测性，以及受治理的副作用执行。

[![crates.io](https://img.shields.io/crates/v/molo.svg)](https://crates.io/crates/molo)
[![docs.rs](https://docs.rs/molo/badge.svg)](https://docs.rs/molo)
[![CI](https://github.com/Silwings-git/molo/actions/workflows/ci.yml/badge.svg)](https://github.com/Silwings-git/molo/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/molo.svg)](LICENSE-APACHE)
[![MSRV](https://img.shields.io/badge/rustc-1.88%2B-orange.svg)](https://github.com/Silwings-git/molo)

[English](README.md) · 简体中文

</div>

---

molo 是 framework 和 SDK，不是面向最终用户的 agent 产品。你可以用它的基础组件
组装自己的 agent：provider adapter、agent kernel、memory、tools、effect request、
harness policy、approval、audit、transcript、workspace primitives、MCP 和 skills。

根 `molo` crate 是 `molo-core`、`molo-agent`、`molo-harness`、`molo-coding`、
`molo-mcp`、`molo-skills`、`molo-openai` 和 `molo-macros` 的 facade。你可以用
facade 获得简单的 `molo::...` 导入路径，也可以直接依赖 focused crates 来减小依赖面。

## ✨ 特性

- **内置推理循环** — `ReActAgent` 实现经典的
  "think → call tools → feed results back → answer" 循环，内置 tool round 限制、
  协作式取消、usage 聚合、事件通道和可选 tracing spans。
- **组件都可替换** — `Provider`（LLM）、`Memory`（上下文）、`Tool`（模型可见能力）、
  `Harness`、policy、approval 和 executor 都是 trait 或可替换组件。
- **受治理的副作用** — 有副作用的 tool 返回 `EffectRequest`；harness 代码在
  observation 回到 agent 前执行 policy、approval、sandbox/network 规则、audit、
  transcript、redaction 和输出限制。
- **Coding workload primitives** — 可选的 workspace、patch、command、git、仓库搜索、
  instruction 和 coding-context API，用于构建 coding agent 产品。
- **现实集成能力** — 可选 OpenAI-compatible provider、MCP client/effect adapter、
  支持 progressive disclosure 的 Agent Skills，以及基于 JSON Schema 校验的 typed output。
- **无运行时锁定** — 不捆绑 runtime，不隐藏线程。使用你的 tokio runtime、tracing
  subscriber、HTTP stack 和应用策略。

## 🚀 快速开始

一个最小 agent 需要一个和 LLM 通信的 `Provider`、一个管理上下文的 `Memory`，
以及可选的外部工具。推理循环内置在 `ReActAgent` 中，`react_agent!` 宏可以一次性
完成常见装配。

```toml
[dependencies]
molo = "0.3"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

默认 feature set 保持精简。按需开启可选能力：

```toml
molo = { version = "0.3", features = ["openai"] }      # OpenAiProvider
molo = { version = "0.3", features = ["macros"] }      # #[molo::tool]
molo = { version = "0.3", features = ["structured"] }  # TypedAgent / validation
molo = { version = "0.3", features = ["harness"] }     # HarnessRuntime
molo = { version = "0.3", features = ["coding"] }      # workspace/command/git SDK
molo = { version = "0.3", features = ["mcp"] }         # MCP adapter
molo = { version = "0.3", features = ["skills"] }      # Agent Skills
molo = { version = "0.3", features = ["full"] }        # 全部可选能力
```

### 不需要 API 的自测

使用 `FakeProvider` 注入脚本化回复，不需要真实 LLM：

```rust
use molo::{react_agent, Agent, FakeProvider, FakeReply};

let mut agent = react_agent!(
    FakeProvider::new([FakeReply::Text("Hello".into())]),
    "You are a helpful assistant",
);
let answer = agent.run("Are you there?").await?;
assert_eq!(answer, "Hello");
```

### 连接真实 LLM

启用 `features = ["openai"]` 后，可以换成 `OpenAiProvider`（任意
OpenAI-compatible endpoint），再用 `RetryProvider` 增加 retry/timeout 保护，并挂载工具：

```rust
use molo::{react_agent, Agent, OpenAiProvider};

let provider = OpenAiProvider::new(
    std::env::var("MOLO_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
    std::env::var("MOLO_API_KEY").unwrap_or_default(),
    std::env::var("MOLO_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string()),
);

let mut agent = react_agent!(
    provider,
    [Calculator],                              // your tools
    "You are a helpful assistant. Use the calculator tool for calculations.",
);
let answer = agent.run("What is (1 + 2) * 3?").await?;
println!("{answer}");
```

这就是一个完整的、支持 streaming 的 tool-calling agent。`run` 返回最终回答；
`run_stream` 产出 token-by-token 的 `MessageChunk` 事件（text delta、tool call、
tool result、`Done`）。

### 结构化 run

只需要答案文本时使用 `run`。需要 run metadata、调用方指定的 run id、request-scoped
model options、deadline 或 multi-block input 时，使用 `run_request` 或
`run_request_with_context`：

```rust
use molo::{Agent, RunContext, RunRequest};
use std::time::Duration;

let output = agent
    .run_request_with_context(
        RunRequest::text("Summarize this session"),
        RunContext::new("request-42").with_timeout(Duration::from_secs(30)),
    )
    .await?;

println!("{}", output.answer);
println!("{} tokens", output.summary.usage.total_tokens);
```

Typed output 走同样的结构化路径：`run_typed_request` 同时返回反序列化后的值和原始
`RunOutput`。该能力需要 `features = ["structured"]`。

## 🧩 核心概念

molo 按领域模块组织，每个模块对应一个概念。crate root 会 re-export 核心 item，
大多数场景直接 `use molo::...` 即可：

| 概念 | 作用 | 关键类型 |
| --- | --- | --- |
| Agent | 推理循环和 step-wise kernel | `Agent`、`AgentKernel`、`AgentAction`、`ReActAgent`、`react_agent!`、`RunContext`；`TypedAgent` 需要 `structured` |
| Provider | LLM 通信 | `Provider`、`ProviderCapabilities`、`ProviderRequestContext`、`RetryProvider`、`FakeProvider`；`OpenAiProvider` 需要 `openai` |
| Memory | 上下文管理 | `Memory`、`InMemoryMemory`、`WindowMemory`、`SummarizeStrategy` |
| Tool | 模型可见能力 | `Tool`、`ToolSchema`、`ToolPolicy`、`ToolOutput`、`ToolResult`、`ToolRegistry`、`SharedState`；`#[molo::tool]` 需要 `macros` |
| Effect | 副作用边界 | `EffectRequest`、`EffectObservation`、`EffectKind`、`RiskLevel` |
| Harness | 受治理的 effect 执行 | `HarnessRuntime`、`Harness`、`BasicHarness`、`EffectExecutor`、`PolicyEngine`、`ApprovalBroker`、`AuditSink`、`TranscriptStore`，需要 `harness` |
| Coding | coding workload primitives | `LocalWorkspace`、`WorkspacePath`、`CodingEffectExecutor`、`CommandExecutor`、`GitInspector`、`RepoSearcher`、`InstructionResolver`、`CodingContextProvider`，需要 `coding` |
| Skill | 能力包 | `Skill`、`SkillRegistry`、`SkillLayer`、`LoadSkillTool`，需要 `skills` |
| MCP | 外部 tool server | `McpClient`、`McpDirectTool`，需要 `mcp`；`McpEffectTool`、`McpEffectExecutor` 需要 `mcp,harness` |
| Message | 对话模型 | `Message`、`ContentBlock`、`ToolCall` |
| MessageChannel | 外部对话 | `MpscChannel`、`BroadcastChannel`、`WatchChannel`；`CliMessageChannel` 需要 `cli-channel` |
| EventChannel | run 过程观察 | `BroadcastEventChannel`、`MpscEventChannel` |

### 如何选择实现

- **上下文** — 短会话或无界会话用 `InMemoryMemory`；需要 token 预算时用
  `WindowMemory` 裁剪旧轮次；需要压缩旧消息时加 `SummarizeStrategy`。
- **连接 LLM** — 开发和测试用 `FakeProvider`（脚本化回复、不需要 API），生产集成用
  `OpenAiProvider` + `RetryProvider`，并启用 `openai`。
- **外部对话** — 一对一进程内 agent 对话用 `MpscChannel`；一对多广播或 latest-value
  通知用 `BroadcastChannel` / `WatchChannel`；终端人机交互用 `CliMessageChannel`。
- **过程观察** — 多订阅者场景用 `BroadcastEventChannel`，单订阅者场景用
  `MpscEventChannel`。

## 🛠 重点能力

### 使用 `#[molo::tool]` 一次性定义工具

需要 `features = ["macros"]`。

手写一个工具大约需要 25 行样板代码；宏会从 async function 生成 struct、schema、
参数解析、output 包装和错误转换：

```rust
use molo::tool::{SharedState, ToolError};

#[molo::tool(description = "Evaluates a math expression, e.g. \"(1 + 2) * 3\"")]
async fn calculator(args: CalcArgs) -> Result<String, ToolError> {
    let value = evalexpr::eval(&args.expression)
        .map_err(|e| ToolError::Execution(e.to_string()))?;
    Ok(value.to_string())
}
```

工具调用失败不会中断 loop。错误文本会反馈给模型，由模型决定下一步。`SharedState`
可让工具共享 typed state，并在调用时注入。

手写工具返回 `ToolResult`：纯工具返回 `ToolOutput::text(...).into()`；有副作用的工具
返回 `EffectRequest`，交给外层 harness 治理和执行。

### Harness runtime

需要 `features = ["harness"]`。

`HarnessRuntime` 是 effect-producing agent 的外层循环。它持有 `Provider`，驱动
`AgentKernel`，把 effect request 交给 `Harness`，再把 observation 回灌给 kernel。
`BasicHarness` 组合 policy engine、approval broker、effect executor、audit sink、
transcript store、output limiter 和 redactor。

```text
tool call -> EffectRequest -> Harness policy/approval/audit -> executor -> Observation
```

### Coding workload SDK

需要 `features = ["coding"]`。

coding SDK 提供构建 coding-agent 产品所需的 primitives，而不是把 molo 本身做成 CLI：
`WorkspacePath` 拒绝绝对路径和路径穿越；`LocalWorkspace` 将 read/list/write/patch
限制在 canonical root 内；`CommandExecutor` 用明确 argv、timeout 和输出限制执行命令；
`GitInspector` / `RepoSearcher` / `InstructionResolver` / `CodingContextProvider`
将仓库上下文和 chat memory 分开。

模型可见的 side-effecting tool 会构造 typed payload，例如 `ReadFilePayload`、
`WriteFilePayload`、`ApplyPatchPayload`、`CommandPayload` 或 `GitPayload`；
`CodingEffectExecutor` 在 harness 完成 policy、approval、audit 和 transcript 后解码并执行。

详见 [Coding Agents](docs/coding-agent.md)。

### Reference CLI

仓库内包含一个 `publish = false` 的参考 CLI package：

```bash
cargo run -p molo-cli -- --help
cargo run -p molo-cli -- chat --no-stream "hello"
cargo run -p molo-cli -- --workspace . code "inspect this repo"
cargo run -p molo-cli -- review --json
```

二进制名是 `molo`。默认使用 fake-provider mode，因此 smoke test 不需要 API key。
使用 `--provider openai --base-url ...` 和 `--api-key-env NAME` 可以连接
OpenAI-compatible provider。

### Typed（结构化）输出

需要 `features = ["structured"]`。

`run_typed` 会用从目标类型派生出的 JSON Schema 校验模型回复；校验失败时会把错误反馈给
模型重试：

```rust
let weather: Weather = agent.run_typed("How is the weather in Beijing today?").await?;
println!("{}°C, {}", weather.temperature, weather.condition);
```

### MCP client

需要 `features = ["mcp"]`。

可以将外部 MCP server 暴露的工具带入 agent。`McpClient` 支持 stdio child-process 和
Streamable HTTP transports（基于 `rmcp`）。direct path 适合原型和低风险 server：

```rust
let mut client = McpClient::from_command("filesystem", "mcp-filesystem", ["/workspace"]);
let mut registry = ToolRegistry::new();
for tool in client.tools().await? {
    registry.register_with_source(tool.clone(), tool.source())?;
}
```

需要治理外部副作用时，启用 `features = ["mcp", "harness"]`，并使用
`McpEffectTool` + `McpEffectExecutor`。

### Skills（Agent Skills 协议）

需要 `features = ["skills"]`。

一个 skill 是包含 `SKILL.md`（YAML frontmatter + Markdown）的目录。核心机制是
**progressive disclosure**：模型先只看到 name + description 的一行菜单；任务匹配时，
再通过 `load_skill` tool 读取正文。`SkillLayer::assemble` 返回 prompt fragment 和可选
`load_skill` tool；host 显式拼接 prompt 并注册 tool。

### Sub-agents

`SubAgentTool` / `SubAgentPool` 让主 agent 可以委托给 sub-agent：持久专家
（`from_agent`）、临时工厂（`from_factory`），或由调用参数现场定义的 ReAct sub-agent
（`from_react` / `spawn_react`）。命名 pool 会让 sub-agent 在多轮中保持可继续对话。

### 协作式取消

取消信息由 `RunContext` 携带。把带 `CancellationToken` 的 context 传给
`run_request_with_context` 或 `run_stream_request_with_context`；取消按 run 生效，
中途停止回复不会影响下一轮。

### Streaming、events 与 observability

- `Agent::run_stream` — `MessageChunk` 事件：`Delta` / `ToolCall` /
  `ToolResult` / `Done` / `Cancelled`。
- `Provider::stream_chat` — 原始 `StreamEvent` stream（delta、reasoning、tool call）。
- `EventChannel` — 订阅 run 的 `AgentEvent` stream；`EventChannelStats` 报告 drop 和 lag。
- `AgentEventRecord` — 可选的序列化、redacted event 摘要，适合日志和 devtools。
- 启用 `features = ["tracing"]` 后，loop 会在固定点发出 `tracing` spans。molo 不安装
  subscriber；你可以接入自己的 `tracing-subscriber`。

## 📚 示例

所有示例都在 `examples/`。大多数真实 provider 示例会从 `.env` 读取 `MOLO_API_KEY` /
`MOLO_BASE_URL` / `MOLO_MODEL`（可复制 `.example.env` 为 `.env`）。不需要真实 API 的
自包含示例用 ✦ 标记。

### Core loop

| 示例 | 命令 | 展示内容 |
| --- | --- | --- |
| `react_agent` | `cargo run --example react_agent --features openai,structured` | 内置 `ReActAgent` + `react_agent!` 宏（stream / chat 模式） |
| `agent` | `cargo run --example agent --features openai,structured` | 手写 `Agent` trait loop |
| `tool_agent` | `cargo run --example tool_agent --features openai,structured` | 基于 `Provider::chat` / `stream_chat` 的 tool-call loop |
| `sub_agent` | `cargo run --example sub_agent --features openai,structured` | Sub-agent delegation，`SubAgentTool` / `SubAgentPool` |

### Tools、harness、coding

| 示例 | 命令 | 展示内容 |
| --- | --- | --- |
| ✦ `tool_registry` | `cargo run --example tool_registry --features structured` | Registry API：register / names / schemas / call / subset |
| ✦ `tool_macro` | `cargo run --example tool_macro --features macros` | 使用 `#[molo::tool]` 一次性定义工具 |
| ✦ `harness_runtime` | `cargo run --example harness_runtime --features harness` | `HarnessRuntime` 和 `BasicHarness` 的受治理 effect 执行 |
| ✦ `coding_workspace` | `cargo run --example coding_workspace --features coding` | 本地 workspace read/list/write primitives |
| ✦ `coding_harness` | `cargo run --example coding_harness --features coding` | typed coding effects 经 `BasicHarness` 执行 |
| ✦ `shared_state` | `cargo run --example shared_state` | `SharedState` 的三种使用方式 |
| ✦ `mcp` | `cargo run --example mcp --features mcp` | MCP client adapter，自包含 fake server |
| ✦ `mcp_governed` | `cargo run --example mcp_governed --features mcp,harness` | MCP tool 作为受治理 harness effect |

### Provider

| 示例 | 命令 | 展示内容 |
| --- | --- | --- |
| `chat` | `cargo run --example chat --features openai` | 使用真实模型的 plain chat |
| `chat_stream` | `cargo run --example chat_stream --features openai` | Streaming chat，逐 token 输出 |
| `multimodal` | `cargo run --example multimodal --features openai -- <image path>` | 多模态模型的图片输入（`ContentBlock::Image`） |
| ✦ `fake_provider` | `cargo run --example fake_provider` | 用脚本化回复测试自己的 loop |
| `retry` | `cargo run --example retry --features openai` | `RetryProvider` wrapper |
| `usage` | `cargo run --example usage --features openai,structured` | 每次 run 的执行摘要（tokens、rounds） |
| `trace` | `cargo run --example trace --features macros,tracing` | 在终端渲染 tracing spans |

### Memory

| 示例 | 命令 | 展示内容 |
| --- | --- | --- |
| ✦ `window_memory` | `cargo run --example window_memory` | `WindowMemory` 裁剪和自定义 trim strategies |
| `window_memory_agent` | `cargo run --example window_memory_agent --features openai` | 使用 token-budget window 的真实模型 agent |
| ✦ `summarize` | `cargo run --example summarize` | `SummarizeStrategy`：将旧消息压缩成 summary |
| `summarize_agent` | `cargo run --example summarize_agent --features openai,tracing` | 带 summary compression 的真实模型 streaming agent |

### Channels

| 示例 | 命令 | 展示内容 |
| --- | --- | --- |
| ✦ `message_channel` | `cargo run --example message_channel --features structured,cli-channel` | Cli / Mpsc / Broadcast / Watch channel |
| `confirm_agent` | `cargo run --example confirm_agent --features openai,structured,cli-channel` | 通过 `MessageChannel` 做人工确认 |
| `mpsc_agent` | `cargo run --example mpsc_agent --features openai,structured` | 两个 agent 通过 `MpscChannel` 对话 |
| `broadcast_agent` | `cargo run --example broadcast_agent --features openai,structured` | 一对多 broadcast notifications |
| `watch_agent` | `cargo run --example watch_agent --features openai,structured` | latest-value 状态发布 |
| ✦ `event_channel` | `cargo run --example event_channel` | 订阅 event stream |
| `event_channel_agent` | `cargo run --example event_channel_agent --features openai,structured` | 通过 `EventChannel` 观察真实模型 agent |

### Output、skills、cancellation

| 示例 | 命令 | 展示内容 |
| --- | --- | --- |
| ✦ `structured` | `cargo run --example structured --features structured` | 使用 `run_typed` 和 JSON Schema validation 的 typed output |
| ✦ `skill` | `cargo run --example skill --features skills` | Skills：discovery、progressive disclosure、activation |
| ✦ `skill_layer` | `cargo run --example skill_layer --features skills` | 不构造 `ReActAgent` 的 SkillLayer prompt/tool assembly |
| ✦ `cancellation` | `cargo run --example cancellation` | 中途取消回复后继续下一轮 |

## ⚙️ 配置

示例会读取 `.env`（复制 `.example.env` 到 `.env` 并填入真实值）；环境变量会直接覆盖：

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `MOLO_API_KEY` | *(empty)* | API key；本地无鉴权 endpoint 可留空 |
| `MOLO_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `MOLO_MODEL` | `gpt-4o-mini` | 模型名称 |
| `MOLO_MAX_TOKENS` | - | memory 示例中的 window/context budget |
| `MOLO_SUMMARY_MAX_TOKENS` | `150` | summarize 示例中的 summary 输出上限 |

## 📖 文档

- Rustdoc：`cargo doc --workspace --no-deps` 或 [docs.rs](https://docs.rs/molo)
- [架构](docs/architecture.md) — layer boundaries、依赖方向和副作用执行模型
- [Coding Agents](docs/coding-agent.md) — coding SDK 边界和示例
- [Changelog](CHANGELOG.md)

## 📦 状态

molo 是 pre-1.0 软件。当前版本适合实验、嵌入和集成工作。下游应用建议固定依赖版本。

## 📄 许可证

本项目使用 [MIT](LICENSE-MIT) 或 [Apache-2.0](LICENSE-APACHE) 双许可证，
你可以任选其一。
