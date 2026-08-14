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
//! SDK (workspace, shell, git, patch, repo context). Public API breaking
//! changes are expected throughout 0.x when they serve the target
//! architecture, and each one ships with a migration path.
//!
//! # Feature Flags
//!
//! The default dependency surface is the lightweight core path. Optional
//! capabilities are enabled explicitly:
//!
//! - `openai`: [`OpenAiProvider`] and OpenAI-compatible HTTP support.
//! - `structured`: typed output, [`StructuredValidator`], and JSON Schema
//!   validation.
//! - `macros`: the `#[molo::tool]` attribute macro. This also
//!   enables `structured` because macro-generated schemas use `schemars`.
//! - `skills`: Agent Skills protocol support.
//! - `mcp`: MCP client adapter support.
//! - `harness`: `HarnessRuntime` and governed effect execution.
//! - `coding`: coding-workload primitives on top of `harness`: workspace
//!   paths, file effects, commands, git inspection, repo search, project
//!   instructions, and context gathering.
//! - `cli-channel`: [`CliMessageChannel`] for stdin/stdout interaction.
//! - `tracing`: internal tracing spans and logs.
//! - `full`: all optional capabilities above.
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
//! For a real LLM, enable the `openai` feature and swap [`FakeProvider`] for
//! [`OpenAiProvider`]; for retry
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
//!   with the validator [`StructuredValidator`] when the `structured`
//!   feature is enabled, the [`ReActAgent`] assembly with the
//!   [`react_agent!`] macro, sub-agent parts
//!   ([`SubAgentTool`](crate::agent::SubAgentTool) /
//!   [`SubAgentPool`](crate::agent::SubAgentPool)), streaming output chunks
//!   and run summaries ([`MessageChunk`] / [`RunSummary`]);
//! - [`provider`] — LLM communication — the [`Provider`] interface,
//!   request / response / usage models ([`ChatRequest`] / [`ChatResponse`] /
//!   [`Usage`]), and implementations [`RetryProvider`] / [`FakeProvider`],
//!   plus [`OpenAiProvider`] with the `openai` feature;
//! - [`tool`](mod@crate::tool): external capabilities — the [`Tool`]
//!   interface and tool definitions ([`ToolSchema`]), registration and
//!   execution ([`ToolRegistry`]), cross-tool shared state
//!   ([`SharedState`]), and the `#[molo::tool]` procedural macro for
//!   one-shot tool definitions with the `macros` feature;
//! - `skill` — skills — capability packages following the Agent Skills
//!   open protocol, available with the `skills` feature;
//! - `mcp` — MCP client adapter, available with the `mcp` feature;
//! - [`memory`] — context management — the [`Memory`] interface and
//!   implementations [`InMemoryMemory`] / [`WindowMemory`] (trims the
//!   oldest turns by token budget), summary compression
//!   [`SummarizeStrategy`] — old messages over budget are compressed into a
//!   single summary;
//! - [`message`] — the conversation message model — [`Message`] /
//!   [`ContentBlock`] / [`ToolCall`];
//! - [`message_channel`] — channels for external conversation — in-process
//!   request-reply / notification channels, plus [`CliMessageChannel`] with
//!   the `cli-channel` feature;
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
//!   [`FakeProvider`], no real API needed; for production enable `openai`
//!   and use [`OpenAiProvider`], wrapped in [`RetryProvider`] for retry /
//!   timeout protection;
//! - **External conversation**: use [`MpscChannel`] for one-to-one
//!   in-process agent conversation, [`BroadcastChannel`] / [`WatchChannel`]
//!   for one-to-many broadcast / latest-value change notifications, and
//!   [`CliMessageChannel`] with `cli-channel` for human-terminal
//!   interaction;
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
#[cfg(feature = "coding")]
pub mod coding;
pub mod effect;
pub mod event_channel;
#[cfg(feature = "harness")]
pub mod harness;
#[cfg(feature = "mcp")]
pub mod mcp;
pub mod memory;
pub mod message;
pub mod message_channel;
pub mod observability;
pub mod provider;
pub mod run;
#[cfg(feature = "skills")]
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
#[cfg(feature = "macros")]
pub use molo_macros::tool;

pub use agent::{
    Agent, AgentAction, AgentConfig, AgentError, AgentEvent, AgentKernel, CancellableAgent,
    MessageChunk, ModelObservation, ModelRequest, Observation, ReActAgent, ReActEvent,
};
#[cfg(feature = "structured")]
pub use agent::{StructuredOutcome, StructuredValidator, TypedAgent};
#[cfg(feature = "coding")]
pub use coding::{
    AgentChangeTracker, ApplyPatchPayload, ApplyPatchTool, CliGitInspector, CodingContextBundle,
    CodingContextError, CodingContextInclude, CodingContextProvider, CodingContextRequest,
    CodingEffectExecutor, CodingError, CodingExecutorConfig, CommandError, CommandExecutor,
    CommandExecutorCapabilities, CommandOutput, CommandOutputLimit, CommandPayload, CommandRequest,
    CommandStatus, CommandTestRunner, ContentDigest, ContextBudget, DefaultCodingContextProvider,
    DefaultInstructionResolver, DependencyMetadata, DiffRequest, EnvPolicy, FileBody, FileContent,
    FilePatch, FileReadOptions, FileVersion, FileWriteContent, FileWriteResult, GitChangedFile,
    GitChangedFilesRequest, GitDiffRequest, GitError, GitHead, GitInspector, GitOperation,
    GitPayload, GitStatus, GitStatusRequest, GitStatusTool, InstructionBundle, InstructionError,
    InstructionFile, InstructionFileSpec, InstructionRequest, InstructionResolver,
    ListFilesPayload, ListFilesQuery, ListFilesTool, LocalCommandExecutor, LocalWorkspace,
    LocalWorkspaceConfig, OutputText, Patch, PatchConflict, PatchHunk, PatchOperation,
    PatchRequest, PatchResult, PolicyEnforcementReport, PtyMode, ReadFilePayload, ReadFileTool,
    RepoSearchRequest, RepoSearchResults, RepoSearcher, ResolvedPath, ResolvedPathKind,
    RipgrepSearcher, RunCommandTool, SearchError, SearchMatch, SearchMode, SearchPayload,
    SearchRepoTool, SnapshotRequest, SymlinkPolicy, TestRunError, TestRunRequest, TestRunner,
    TextEncoding, VerificationResult, Workspace, WorkspaceAccess, WorkspaceDiff, WorkspaceEntry,
    WorkspaceError, WorkspacePath, WorkspaceRoot, WorkspaceSearcher, WriteFilePayload,
    WriteFileRequest,
};
pub use effect::{
    DisplayFormat, DisplayOutput, EffectKind, EffectObservation, EffectOutput, EffectRequest,
    EffectSource, EffectStatus, RiskLevel,
};
pub use event_channel::{
    BroadcastEventChannel, EventChannel, EventChannelStats, EventReceiver, MpscEventChannel,
};
#[cfg(feature = "harness")]
pub use harness::{
    AgentActionSummary, AlwaysAllowApprovalBroker, AlwaysDenyApprovalBroker, ApprovalBroker,
    ApprovalDecision, ApprovalError, ApprovalRequest, AuditError, AuditEvent, AuditSink,
    BasicHarness, ClassifiedEffect, DefaultPolicyEngine, DefaultRiskClassifier, EffectExecutor,
    ExecutionError, ExecutionPolicy, ExecutionPolicySummary, Harness, HarnessConfig, HarnessError,
    HarnessRuntime, HarnessRuntimeConfig, HarnessRuntimeError, LimitedOutput, ModelSummary,
    NetworkPolicy, NoopAuditSink, NoopEffectExecutor, NoopRedactor, NoopTranscriptStore,
    OutputLimit, PatternRedactor, PolicyDecision, PolicyEngine, RawEffectOutput, RedactedText,
    Redactor, RouterEffectExecutor, SandboxPolicy, StaticApprovalBroker, StaticEffectExecutor,
    TranscriptError, TranscriptRecord, TranscriptStore, VecAuditSink, VecTranscriptStore,
};
#[cfg(feature = "mcp")]
pub use mcp::{
    McpCacheHint, McpClient, McpDirectTool, McpError, McpServerId, McpTool, McpToolCatalog,
    McpToolDescriptor, McpToolId, McpToolMode,
};
#[cfg(all(feature = "mcp", feature = "harness"))]
pub use mcp::{
    McpCallPayload, McpClientProvider, McpEffectExecutor, McpEffectTool, McpPermissionBridge,
    McpServerPolicy, McpToolCallOutput,
};
pub use memory::{
    Budget, CharTokenCounter, InMemoryMemory, Memory, MemoryError, SummarizeStrategy, TokenCounter,
    TrimResult, TrimStrategy, WindowDrop, WindowMemory,
};
pub use message::{ContentBlock, ImageContent, Message, ToolCall};
#[cfg(feature = "cli-channel")]
pub use message_channel::CliMessageChannel;
pub use message_channel::{
    BroadcastChannel, BroadcastReceiver, ChannelError, IncomingMessage, MessageChannel,
    MpscChannel, WatchChannel, WatchReceiver,
};
pub use observability::{AgentEventRecord, EventSeverity, RedactionRecord};
pub use provider::{
    Backoff, ChatRequest, ChatResponse, FakeProvider, FakeReply, FinishReason, ModelOptions,
    Provider, ProviderCapabilities, ProviderError, ProviderRequestContext, RetryPolicy,
    RetryProvider, Retryable, StreamEvent, Usage,
};
#[cfg(feature = "openai")]
pub use provider::{OpenAiProvider, StructuredOutputMode};
#[cfg(feature = "structured")]
pub use run::TypedRunOutput;
pub use run::{Artifact, RunContext, RunMetadata, RunOutput, RunRequest, RunSummary, UserInput};
#[cfg(feature = "skills")]
pub use skill::{
    AllowedTool, LoadSkillReferenceTool, LoadSkillTool, Skill, SkillActivationState, SkillError,
    SkillLayer, SkillLayerAssembly, SkillLayerConfig, SkillLayerManifest, SkillMode, SkillRegistry,
    SkillResourceStore, SkillSourceTrust,
};
pub use tool::{
    MissingTools, RegistryError, SharedState, SideEffectLevel, Tool, ToolContext, ToolError,
    ToolMemoryPolicy, ToolNamespace, ToolNamespaceKind, ToolOutput, ToolPolicy, ToolRegistry,
    ToolResult, ToolSchema, ToolSource, ToolTrustLevel,
};

// Cooperative cancellation primitive (a standard tokio-util component):
// `CancellableAgent::run_cancellable` / `run_stream_cancellable` use it as
// the cancellation source for each run; also re-exported for transitive
// dependencies (tokio-util, licensed MIT OR Apache-2.0).
pub use tokio_util::sync::CancellationToken;
