//! molo — an embeddable Rust agent runtime and harness framework.
#![warn(missing_docs)]
//!
//! molo = **Mo**del **Lo**op: an embeddable Rust runtime and harness
//! framework for building safe, extensible tool-calling agents, with
//! first-class support for coding-agent workloads.
//!
//! molo is a library, not an end-user agent product. You assemble agents
//! from its building blocks — model interaction, reasoning loop, memory,
//! tools, structured output, and observability. The target architecture also
//! includes an optional harness layer (approval, sandbox, audit, transcript)
//! that governs and executes side effects, and an optional coding-workload
//! SDK (workspace, shell, git, patch, repo context). The `0.2.x` crate
//! currently ships the agent-runtime side of that picture; the harness and
//! coding layers are being introduced as optional components in later
//! releases. Public API breaking changes are expected throughout 0.x when
//! they serve the target architecture, and each one ships with a migration
//! path.
//!
//! # Quick Start
//!
//! A minimal agent needs only three things: a [`Provider`] to talk to the
//! LLM, a [`Memory`] to manage context, and a [`Tool`] for external
//! capabilities; the reasoning loop is built into [`ReActAgent`], and the
//! [`react_agent!`] macro assembles everything in one go. To self-test the
//! loop, use [`FakeProvider`] to inject scripted replies without depending
//! on a real API:
//!
//! ```rust
//! # #[tokio::main]
//! # async fn main() -> Result<(), molo::AgentError> {
//! use molo::{react_agent, Agent, FakeProvider, FakeReply};
//!
//! let mut agent = react_agent!(
//!     FakeProvider::new([FakeReply::Text("Hello".into())]),
//!     "You are a helpful assistant",
//! );
//! let answer = agent.run("Are you there?").await?;
//! assert_eq!(answer, "Hello");
//! # Ok(())
//! # }
//! ```
//!
//! For a real LLM, swap [`FakeProvider`] for [`OpenAiProvider`]; for retry
//! and timeout protection, wrap it in [`RetryProvider`]; for interaction
//! with the outside world (human approval, agent-to-agent conversation),
//! attach a [`MessageChannel`]; to observe the reasoning process, attach an
//! [`EventChannel`].
//!
//! # Component Overview
//!
//! Code is organized into domain modules, one concept per module; the top
//! level re-exports each module's core items, so `use molo::...` covers
//! most cases without digging into module paths:
//!
//! - [`agent`] — the agent interface and reasoning loop — [`Agent`] /
//!   [`CancellableAgent`] / [`AgentError`], typed output via [`TypedAgent`]
//!   with the validator [`StructuredValidator`], the [`ReActAgent`]
//!   assembly with the [`react_agent!`] macro, sub-agent parts
//!   ([`SubAgentTool`](crate::agent::SubAgentTool) /
//!   [`SubAgentPool`](crate::agent::SubAgentPool)), streaming output chunks
//!   and run summaries ([`MessageChunk`] / [`RunSummary`]);
//! - [`provider`] — LLM communication — the [`Provider`] interface,
//!   request / response / usage models ([`ChatRequest`] / [`ChatResponse`] /
//!   [`Usage`]), and implementations [`OpenAiProvider`] / [`RetryProvider`] /
//!   [`FakeProvider`];
//! - [`tool`](mod@crate::tool): external capabilities — the [`Tool`]
//!   interface and tool definitions ([`ToolSchema`]), registration and
//!   execution ([`ToolRegistry`]), cross-tool shared state
//!   ([`SharedState`]), and the procedural macro for one-shot tool
//!   definitions [`tool`](macro@molo::tool);
//! - [`skill`] — skills — capability packages following the Agent Skills
//!   open protocol ([`Skill`] parsing / validation, [`SkillRegistry`]
//!   discovery and hot-swapping, progressive disclosure loading via
//!   [`LoadSkillTool`]);
//! - [`mcp`] — MCP client adapter — wiring tools exposed by external MCP
//!   servers into molo ([`McpClient`] connects and pulls, [`McpTool`]
//!   adapts tools, [`McpError`]);
//! - [`memory`] — context management — the [`Memory`] interface and
//!   implementations [`InMemoryMemory`] / [`WindowMemory`] (trims the
//!   oldest turns by token budget), summary compression
//!   [`SummarizeStrategy`] — old messages over budget are compressed into a
//!   single summary;
//! - [`message`] — the conversation message model — [`Message`] /
//!   [`ContentBlock`] / [`ToolCall`];
//! - [`message_channel`] — channels for external conversation —
//!   [`CliMessageChannel`] and three more implementations (request-reply /
//!   one-way notification);
//! - [`event_channel`] — observation channels — subscribe to the event
//!   stream of an agent run ([`AgentEvent`] payloads).
//!
//! # Choosing Between Implementations
//!
//! When several implementations share a responsibility, pick per scenario:
//!
//! - **Context**: short sessions or unbounded needs use
//!   [`InMemoryMemory`]; long sessions that must stay within budget use
//!   [`WindowMemory`];
//! - **Talking to the LLM**: for development, inject scripted replies with
//!   [`FakeProvider`], no real API needed; for production use
//!   [`OpenAiProvider`], wrapped in [`RetryProvider`] for retry / timeout
//!   protection;
//! - **External conversation**: use [`CliMessageChannel`] for
//!   human-terminal interaction, [`MpscChannel`] for one-to-one in-process
//!   agent conversation, [`BroadcastChannel`] / [`WatchChannel`] for
//!   one-to-many broadcast / latest-value change notifications;
//! - **Observing the reasoning process**: subscribe to agent events via
//!   [`BroadcastEventChannel`] (multiple subscribers; slow ones drop the
//!   oldest events) or [`MpscEventChannel`] (single subscriber; nothing is
//!   dropped within capacity).
//!
//! # Notes
//!
//! - molo targets the tokio ecosystem: all async APIs require a tokio
//!   runtime; the library ships no runtime of its own, so callers bring
//!   their own (examples uniformly use `#[tokio::main]`);
//! - cancellation is cooperative and opt-in: agents that support it
//!   implement [`CancellableAgent`]; each run carries a
//!   [`CancellationToken`] that any holder may request, and the loop
//!   responds at safe points;
//! - tool execution failure does not abort the reasoning loop: the error
//!   text is passed back to the model, which decides what to do next.

pub mod agent;
pub mod effect;
pub mod event_channel;
pub mod mcp;
pub mod memory;
pub mod message;
pub mod message_channel;
pub mod provider;
pub mod run;
pub mod skill;
pub mod tool;

// `extern crate self`: lets macro-expanded code (which hardcodes `::molo::`
// paths) and in-crate tests refer to this crate as `molo::`.
extern crate self as molo;

// Re-export the procedural macro and helper trait: code expanded by
// `#[molo::tool]` references `::molo::tool` and `::molo::async_trait`,
// which transitive dependencies cannot resolve from user crates, so they
// are routed through this crate's root.
pub use async_trait::async_trait;
pub use molo_macros::tool;

pub use agent::{
    Agent, AgentAction, AgentConfig, AgentError, AgentEvent, AgentKernel, CancellableAgent,
    MessageChunk, ModelObservation, ModelRequest, Observation, ReActAgent, ReActEvent,
    StructuredOutcome, StructuredValidator, TypedAgent,
};
pub use effect::{
    DisplayFormat, DisplayOutput, EffectKind, EffectObservation, EffectOutput, EffectRequest,
    EffectSource, EffectStatus, RiskLevel,
};
pub use event_channel::{BroadcastEventChannel, EventChannel, EventReceiver, MpscEventChannel};
pub use mcp::{McpClient, McpError, McpTool};
pub use memory::{
    Budget, CharTokenCounter, InMemoryMemory, Memory, MemoryError, SummarizeStrategy, TokenCounter,
    TrimResult, TrimStrategy, WindowDrop, WindowMemory,
};
pub use message::{ContentBlock, ImageContent, Message, ToolCall};
pub use message_channel::{
    BroadcastChannel, BroadcastReceiver, ChannelError, CliMessageChannel, IncomingMessage,
    MessageChannel, MpscChannel, WatchChannel, WatchReceiver,
};
pub use provider::{
    Backoff, ChatRequest, ChatResponse, FakeProvider, FakeReply, FinishReason, ModelOptions,
    OpenAiProvider, Provider, ProviderError, RetryPolicy, RetryProvider, Retryable, StreamEvent,
    StructuredOutputMode, Usage,
};
pub use run::{
    Artifact, RunContext, RunMetadata, RunOutput, RunRequest, RunSummary, TypedRunOutput, UserInput,
};
pub use skill::{AllowedTool, LoadSkillTool, Skill, SkillError, SkillRegistry};
pub use tool::{
    MissingTools, RegistryError, SharedState, SideEffectLevel, Tool, ToolContext, ToolError,
    ToolMemoryPolicy, ToolOutput, ToolPolicy, ToolRegistry, ToolResult, ToolSchema,
};

// Cooperative cancellation primitive (a standard tokio-util component):
// `CancellableAgent::run_cancellable` / `run_stream_cancellable` use it as
// the cancellation source for each run; also re-exported for transitive
// dependencies (tokio-util, licensed MIT OR Apache-2.0).
pub use tokio_util::sync::CancellationToken;
