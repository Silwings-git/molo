<div align="center">

# molo

**Mo**del **Lo**op · An embeddable Rust agent runtime and harness framework

molo is an embeddable Rust runtime and harness framework for building safe,
extensible tool-calling agents, with first-class support for coding-agent
workloads.

[![crates.io](https://img.shields.io/crates/v/molo.svg)](https://crates.io/crates/molo)
[![docs.rs](https://docs.rs/molo/badge.svg)](https://docs.rs/molo)
[![CI](https://github.com/Silwings-git/molo/actions/workflows/ci.yml/badge.svg)](https://github.com/Silwings-git/molo/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/molo.svg)](LICENSE-APACHE)
[![MSRV](https://img.shields.io/badge/rustc-1.97%2B-orange.svg)](https://github.com/Silwings-git/molo)

</div>

---

molo is a framework and SDK, not an end-user agent product. You assemble
agents — including CLI coding agents — from its building blocks: model
interaction, reasoning loop, context management, tool calling, structured
output, and observability. The target architecture adds two optional layers
on top: a harness that governs and executes side effects (approval, sandbox,
audit, transcript), and a coding-workload SDK (workspace, shell, git, patch,
repo context). See [Architecture](docs/architecture.md) for the layer
boundaries and dependency rules.

## Status

molo is in an early 0.x phase with a planned architecture evolution. The
`0.3.x` crate ships the agent-runtime side: the `ReActAgent` reasoning loop,
provider traits, memory, tools, effects, optional OpenAI-compatible provider,
MCP, skills, structured output, macros, observability, the optional `harness`
runtime for governed effect execution, and an optional `coding` SDK baseline
for workspace, patch, command, git, search, project instructions, and coding
context primitives. Public API breaking changes are expected throughout 0.x
when they serve the target architecture; each one ships with a migration path.

## ✨ Features

- **Built-in reasoning loop** — `ReActAgent` implements the classic
  "think → call tools → feed results back → answer" loop, with a tool-round
  limit, cooperative cancellation, usage aggregation, event channel, and
  optional tracing spans.
- **Everything is pluggable** — `Provider` (LLM), `Memory` (context) and
  `Tool` (capabilities) are traits; swap implementations per scenario without
  touching the loop.
- **Real-world integrations** — optional MCP client for external tools, Agent
  Skills with progressive disclosure, and structured output validated against
  JSON Schema.
- **Observable & controllable** — stream every token, subscribe to agent
  events, talk to the outside world (human confirmation, agent-to-agent
  conversation), and cancel runs cooperatively.
- **Zero lock-in** — no bundled runtime, no hidden threads. Your tokio
  runtime, your tracing subscriber, your HTTP stack.

## 🚀 Quick Start

A minimal agent needs three things: a `Provider` to talk to the LLM, a
`Memory` to manage context, and (optionally) tools for external capabilities.
The reasoning loop is built into `ReActAgent`, and the `react_agent!` macro
assembles everything in one call.

```toml
[dependencies]
molo = "0.3"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

The default feature set is intentionally small. Enable optional capabilities
when you use them:

```toml
molo = { version = "0.3", features = ["openai"] }      # OpenAiProvider
molo = { version = "0.3", features = ["macros"] }      # #[molo::tool]
molo = { version = "0.3", features = ["structured"] }  # TypedAgent / validation
molo = { version = "0.3", features = ["harness"] }     # HarnessRuntime
molo = { version = "0.3", features = ["coding"] }      # workspace/command/git SDK
molo = { version = "0.3", features = ["full"] }        # all optional features
```

### Self-test without an API

Use `FakeProvider` to inject scripted replies — no real LLM required:

```rust
use molo::{react_agent, Agent, FakeProvider, FakeReply};

let mut agent = react_agent!(
    FakeProvider::new([FakeReply::Text("Hello".into())]),
    "You are a helpful assistant",
);
let answer = agent.run("Are you there?").await?;
assert_eq!(answer, "Hello");
```

### Talk to a real LLM

Enable `features = ["openai"]`, swap in `OpenAiProvider` (any
OpenAI-compatible endpoint), wrap it in `RetryProvider` for retry/timeout
protection, and add tools:

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

That is a complete, streaming-capable, tool-calling agent. `run` returns the
final answer; `run_stream` yields token-by-token `MessageChunk` events
(text deltas, tool calls, tool results, `Done`).

### Structured runs

Use `run` when you only need the answer text. Use `run_request` or
`run_request_with_context` when you need run metadata, a caller-provided run
id, request-scoped model options, deadlines, or multi-block input:

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

Typed output has the same structured path via `run_typed_request`, returning
both the deserialized value and the raw `RunOutput`. It requires the
`structured` feature.

## 🧩 Core Concepts

molo is organized into domain modules — one concept per module. The crate
root re-exports each module's core items, so `use molo::...` covers most cases:

| Concept | What it does | Key types |
| --- | --- | --- |
| Agent | the reasoning loop | `Agent`, `AgentKernel`, `AgentAction`, `ReActAgent`, `react_agent!`, `CancellableAgent`; `TypedAgent` with `structured` |
| Provider | LLM communication | `Provider`, `ProviderCapabilities`, `ProviderRequestContext`, `RetryProvider`, `FakeProvider`; `OpenAiProvider` with `openai` |
| Memory | context management | `Memory`, `InMemoryMemory`, `WindowMemory`, `SummarizeStrategy` |
| Tool | model-visible capabilities | `Tool`, `ToolSchema`, `ToolPolicy`, `ToolOutput`, `ToolResult`, `ToolRegistry`, `SharedState`; `#[molo::tool]` with `macros` |
| Effect | side-effect boundary | `EffectRequest`, `EffectObservation`, `EffectKind`, `RiskLevel` |
| Harness | governed effect execution | `HarnessRuntime`, `Harness`, `BasicHarness`, `EffectExecutor`, `PolicyEngine`, `ApprovalBroker`, `AuditSink`, `TranscriptStore` with `harness` |
| Coding | coding-workload primitives | `LocalWorkspace`, `WorkspacePath`, `CodingEffectExecutor`, `CommandExecutor`, `GitInspector`, `RepoSearcher`, `InstructionResolver`, `CodingContextProvider` with `coding` |
| Skill | capability packs (Agent Skills protocol) | `Skill`, `SkillRegistry`, `SkillLayer`, `LoadSkillTool` with `skills` |
| MCP | external tool servers | `McpClient`, `McpDirectTool` with `mcp`; `McpEffectTool`, `McpEffectExecutor` with `mcp,harness` |
| Message | conversation model | `Message`, `ContentBlock`, `ToolCall` |
| MessageChannel | external conversation (human/agents) | `MpscChannel`, `BroadcastChannel`, `WatchChannel`; `CliMessageChannel` with `cli-channel` |
| EventChannel | observation of a run | `BroadcastEventChannel`, `MpscEventChannel` |

### Choosing between implementations

- **Context** — `InMemoryMemory` for short or unbounded sessions; `WindowMemory`
  trims the oldest turns to a token budget; add `SummarizeStrategy` to compress
  over-budget messages into a single summary.
- **Talking to the LLM** — `FakeProvider` for development (scripted replies,
  no API), `OpenAiProvider` + `RetryProvider` for production with `openai`.
- **External conversation** — `MpscChannel` for one-to-one in-process agent
  conversation, `BroadcastChannel` / `WatchChannel` for one-to-many broadcast
  and latest-value notifications, and `CliMessageChannel` for human-terminal
  interaction with `cli-channel`.
- **Observing the process** — `BroadcastEventChannel` (multiple subscribers,
  slow ones drop the oldest events) or `MpscEventChannel` (single subscriber,
  nothing dropped within capacity).

## 🛠 Highlights

### Tools: one-shot definition with `#[molo::tool]`

Requires `features = ["macros"]`.

Writing a tool by hand takes ~25 lines of boilerplate; the macro generates the
struct, schema, argument parsing, output wrapping, and error conversion from an
async function:

```rust
use molo::tool::{SharedState, ToolError};

#[molo::tool(description = "Evaluates a math expression, e.g. \"(1 + 2) * 3\"")]
async fn calculator(args: CalcArgs) -> Result<String, ToolError> {
    let value = evalexpr::eval(&args.expression)
        .map_err(|e| ToolError::Execution(e.to_string()))?;
    Ok(value.to_string())
}
```

A failed tool call does **not** abort the loop — the error text is fed back to
the model, which decides what to do next. `SharedState` lets tools share typed
state; it is injected at call time.

Hand-written tools return `ToolResult`: pure tools produce
`ToolOutput::text(...).into()`, while side-effecting tools return an
`EffectRequest` for an outer harness to govern and execute. The high-level
`Agent` helpers do not execute effects; enable `features = ["harness"]` and
use `HarnessRuntime`, or drive the step-wise `AgentKernel` boundary yourself.
Multiple effect requests from one step are surfaced as
`AgentAction::RequestEffects` and completed with `Observation::Effects`, so a
harness can choose sequential or parallel execution.

### Harness runtime

Requires `features = ["harness"]`.

`HarnessRuntime` is the outer loop for effect-producing agents: it owns the
`Provider`, drives an `AgentKernel`, sends effect requests through `Harness`,
and feeds observations back to the kernel. `BasicHarness` wires together a
policy engine, approval broker, effect executor, audit sink, transcript store,
output limiter, and redactor. Phase 4 includes test/application executors
(`NoopEffectExecutor`, `StaticEffectExecutor`, `RouterEffectExecutor`); concrete
filesystem/shell/git/patch executors are intentionally left to the coding layer.

### Coding workload SDK

Requires `features = ["coding"]`.

The coding SDK provides primitives for building coding-agent products without
making molo itself a CLI: `WorkspacePath` rejects absolute paths and traversal,
`LocalWorkspace` bounds reads/lists/writes/patches to a canonical root,
`CommandExecutor` runs explicit argv commands with timeouts and output limits,
and `GitInspector` / `RepoSearcher` / `InstructionResolver` /
`CodingContextProvider` keep repo context separate from chat memory.
Side-effecting model-visible tools should construct typed payloads such as
`ReadFilePayload`, `WriteFilePayload`, `ApplyPatchPayload`, `CommandPayload`,
or `GitPayload`; `CodingEffectExecutor` decodes those effects after harness
policy, approval, audit, and transcript handling.

See [Coding Agents](docs/coding-agent.md) for the SDK boundary and examples.

### Typed (structured) output

Requires `features = ["structured"]`.

`run_typed` validates the model's reply against a JSON Schema derived from
your return type; on validation failure the error is fed back to the model for
retry (3 attempts by default):

```rust
let weather: Weather = agent.run_typed("How is the weather in Beijing today?").await?;
println!("{}°C, {}", weather.temperature, weather.condition);
```

### MCP client

Requires `features = ["mcp"]`.

Bring tools exposed by external MCP servers into the agent. `McpClient`
supports the stdio child-process and Streamable HTTP transports (via `rmcp`).
The direct path is convenient for prototypes and low-risk servers:

```rust
let mut client = McpClient::from_command("filesystem", "mcp-filesystem", ["/workspace"]);
let mut registry = ToolRegistry::new();
for tool in client.tools().await? {
    registry.register_with_source(tool.clone(), tool.source())?;
}
```

For production external side effects, enable `features = ["mcp", "harness"]`
and use `McpEffectTool` + `McpEffectExecutor`; the model-visible tool returns
`EffectKind::Mcp`, and the harness owns policy, approval, network/sandbox,
audit, and transcript.

### Skills (Agent Skills open protocol)

Requires `features = ["skills"]`.

A skill is a directory containing `SKILL.md` (YAML frontmatter + Markdown).
The core mechanism is **progressive disclosure**: the model first sees only a
one-line menu of name + description; when a task matches, it reads the body via
the `load_skill` tool. Skills declare their tool dependencies with
`allowed-tools`, which is a policy hint or upper bound, not an authorization
grant. `SkillLayer` assembles the prompt fragment and loader tool explicitly;
`ReActAgent::with_skills` remains a convenience wrapper over that layer.
Skill scripts are not executed by the skill loader; script execution must be
converted into a harness-governed command/coding effect by the host.

### Sub-agents

`SubAgentTool` / `SubAgentPool` let the main agent delegate to sub-agents:
persistent experts (`from_agent`), transient factories (`from_factory`), or
on-the-spot ReAct sub-agents defined in the call arguments (`from_react` /
`spawn_react`). A named pool keeps sub-agents alive across turns for follow-up
conversations.

### Cooperative cancellation

Agents that support cancellation implement `CancellableAgent`. Each run
carries its own `CancellationToken`; cancellation applies per run, so stopping
mid-reply leaves no residue and the next turn starts fresh.

### Streaming, events & observability

- `Agent::run_stream` — `MessageChunk` events: `Delta` / `ToolCall` /
  `ToolResult` / `Done` / `Cancelled`.
- `Provider::stream_chat` — raw `StreamEvent` stream (deltas, reasoning,
  tool calls).
- `EventChannel` — subscribe to the `AgentEvent` stream of a run for
  decoupled best-effort observation; `EventChannelStats` reports drops and
  lag.
- `AgentEventRecord` — optional serializable, redacted event summaries for
  logs/devtools; raw deltas, reasoning and tool arguments are omitted by
  default.
- With `features = ["tracing"]`, loops emit `tracing` spans at fixed points
  (`agent.run`, `llm_request`, `tool`). No subscriber is installed — bring
  your own (e.g. `tracing-subscriber`), and wire OpenTelemetry yourself if
  needed.

## 📚 Examples

All examples live in `examples/`; most read `MOLO_API_KEY` /
`MOLO_BASE_URL` / `MOLO_MODEL` from `.env` (copy `.example.env` to `.env`).
Self-contained examples (no real API needed) are marked ✦.

### Core loop

| Example | Run | What it shows |
| --- | --- | --- |
| ✦ `react_agent` | `cargo run --example react_agent --features openai,structured` | The built-in `ReActAgent` + `react_agent!` macro (stream / chat modes) |
| `agent` | `cargo run --example agent --features openai,structured` | Hand-writing the `Agent` trait loop yourself |
| `tool_agent` | `cargo run --example tool_agent --features openai,structured` | Tool-call loop with `Provider::chat` / `stream_chat` |
| `sub_agent` | `cargo run --example sub_agent --features openai,structured` | Sub-agent delegation, `SubAgentTool` / `SubAgentPool` |

### Tools

| Example | Run | What it shows |
| --- | --- | --- |
| ✦ `tool_registry` | `cargo run --example tool_registry --features structured` | Registry full API: register / names / schemas / call / subset |
| ✦ `tool_macro` | `cargo run --example tool_macro --features macros` | One-shot tool definitions with `#[molo::tool]` |
| ✦ `harness_runtime` | `cargo run --example harness_runtime --features harness` | Governed effect execution with `HarnessRuntime` and `BasicHarness` |
| ✦ `coding_workspace` | `cargo run --example coding_workspace --features coding` | Local workspace read/list/write primitives |
| ✦ `coding_harness` | `cargo run --example coding_harness --features coding` | Typed coding effects executed through `BasicHarness` |
| ✦ `shared_state` | `cargo run --example shared_state` | Three ways to use `SharedState` |
| ✦ `mcp` | `cargo run --example mcp --features mcp` | MCP client adapter, self-contained fake server |
| ✦ `mcp_governed` | `cargo run --example mcp_governed --features mcp,harness` | MCP tool as a governed harness effect |

### Provider

| Example | Run | What it shows |
| --- | --- | --- |
| `chat` | `cargo run --example chat --features openai` | Plain chat with a real model |
| `chat_stream` | `cargo run --example chat_stream --features openai` | Streaming chat, tokens as they arrive |
| `multimodal` | `cargo run --example multimodal --features openai -- <image path>` | Image input (`ContentBlock::Image`) to a multimodal model |
| ✦ `fake_provider` | `cargo run --example fake_provider` | Scripted replies for testing your own loop |
| `retry` | `cargo run --example retry --features openai` | `RetryProvider` wrapper |
| `usage` | `cargo run --example usage --features openai,structured` | Per-run execution summary (tokens, rounds) |
| `trace` | `cargo run --example trace --features macros,tracing` | Rendering the tracing spans to the console |

### Memory

| Example | Run | What it shows |
| --- | --- | --- |
| ✦ `window_memory` | `cargo run --example window_memory` | `WindowMemory` trimming and custom trim strategies |
| `window_memory_agent` | `cargo run --example window_memory_agent --features openai` | Real-model agent with a token-budget window |
| ✦ `summarize` | `cargo run --example summarize` | `SummarizeStrategy`: compress old messages into a summary |
| `summarize_agent` | `cargo run --example summarize_agent --features openai,tracing` | Real-model streaming agent with summary compression |

### Channels

| Example | Run | What it shows |
| --- | --- | --- |
| ✦ `message_channel` | `cargo run --example message_channel --features structured,cli-channel` | All four channel implementations (Cli / Mpsc / Broadcast / Watch) |
| `confirm_agent` | `cargo run --example confirm_agent --features openai,structured,cli-channel` | Human confirmation via `MessageChannel` |
| `mpsc_agent` | `cargo run --example mpsc_agent --features openai,structured` | Two-agent conversation over `MpscChannel` |
| `broadcast_agent` | `cargo run --example broadcast_agent --features openai,structured` | One-to-many broadcast notifications |
| `watch_agent` | `cargo run --example watch_agent --features openai,structured` | Latest-value status publishing via `WatchChannel` |
| ✦ `event_channel` | `cargo run --example event_channel` | Subscribing to the event stream (self-contained) |
| `event_channel_agent` | `cargo run --example event_channel_agent --features openai,structured` | Real-model agent observed through an `EventChannel` |

### Output, skills, cancellation

| Example | Run | What it shows |
| --- | --- | --- |
| ✦ `structured` | `cargo run --example structured --features structured` | Typed output with `run_typed` and JSON Schema validation |
| ✦ `skill` | `cargo run --example skill --features skills` | Skills: discovery, progressive disclosure, activation |
| ✦ `skill_layer` | `cargo run --example skill_layer --features skills` | SkillLayer prompt/tool assembly without ReActAgent internals |
| ✦ `cancellation` | `cargo run --example cancellation` | Cooperative cancellation mid-reply, then continue |

## ⚙️ Configuration

Examples read configuration from `.env` (copy `.example.env` to `.env` and
fill in real values); environment variables override directly:

| Variable | Default | Used for |
| --- | --- | --- |
| `MOLO_API_KEY` | *(empty)* | API key; may be left empty for local endpoints without auth (e.g. Ollama) |
| `MOLO_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint |
| `MOLO_MODEL` | `gpt-4o-mini` | Model name |
| `MOLO_MAX_TOKENS` | — | Window/context budget (window/summarize examples) |
| `MOLO_SUMMARY_MAX_TOKENS` | `150` | Summary output cap (summarize examples) |

## 📖 Documentation

- Rustdoc: `cargo doc --no-deps` or [docs.rs](https://docs.rs/molo)
- [Architecture](docs/architecture.md) — layer boundaries, dependency
  direction, and the side-effect execution model

## 📄 License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE),
at your option.
