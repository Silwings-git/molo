//! molo — an embeddable Rust agent runtime and harness framework.
#![warn(missing_docs)]
//!
//! The `molo` crate is the facade over the molo workspace crates. It keeps
//! the ergonomic `molo::...` import path while the implementation is split
//! into focused crates:
//!
//! - `molo-core`: message, run, provider, tool, and effect protocols.
//! - `molo-agent`: agent runtime, memory, channels, and tool registry.
//! - `molo-harness`: governed effect execution.
//! - `molo-coding`: coding-workload primitives.
//! - `molo-mcp`: MCP adapter.
//! - `molo-skills`: Agent Skills protocol.
//! - `molo-openai`: OpenAI-compatible provider.
//!
//! # Feature Flags
//!
//! The default surface stays lightweight. Enable optional layers explicitly:
//!
//! - `openai`: `OpenAiProvider` and OpenAI-compatible HTTP/SSE support.
//! - `structured`: typed output and JSON Schema validation.
//! - `macros`: the `#[molo::tool]` attribute macro; also enables
//!   `structured`.
//! - `skills`: Agent Skills protocol support.
//! - `mcp`: MCP client/tool adapter support.
//! - `harness`: governed effect execution.
//! - `coding`: coding-workload primitives on top of `harness`.
//! - `cli-channel`: stdin/stdout message channel.
//! - `tracing`: internal tracing spans and logs.
//! - `full`: all optional capabilities above.
//!
//! # Quick Start
//!
//! A minimal agent needs a [`Provider`], memory, and an optional
//! [`ToolRegistry`]. The [`react_agent!`] macro assembles the default ReAct
//! runtime while keeping the familiar `molo::...` path:
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
//!
//! let answer = agent.run("Are you there?").await?;
//! assert_eq!(answer, "Hello");
//! # Ok(())
//! # }
//! ```
//!
//! Applications can keep using the facade crate, or depend on focused crates
//! such as `molo-core`, `molo-agent`, and `molo-harness` when they need a
//! smaller dependency surface.

pub mod agent {
    //! Agent runtime facade.
    pub use molo_agent::agent::*;
}

#[cfg(feature = "coding")]
pub mod coding {
    //! Coding-workload primitives facade.
    pub use molo_coding::*;
}

pub mod effect {
    //! Effect protocol facade.
    pub use molo_core::effect::*;
}

pub mod event_channel {
    //! Event channel facade.
    pub use molo_agent::event_channel::*;
}

#[cfg(feature = "harness")]
pub mod harness {
    //! Harness runtime facade.
    pub use molo_harness::*;
}

pub mod memory {
    //! Memory facade.
    pub use molo_agent::memory::*;
}

pub mod message {
    //! Message model facade.
    pub use molo_core::message::*;
}

pub mod message_channel {
    //! Message channel facade.
    pub use molo_agent::message_channel::*;
}

#[cfg(feature = "mcp")]
pub mod mcp {
    //! MCP adapter facade.
    pub use molo_mcp::*;
}

pub mod observability {
    //! Observability facade.
    pub use molo_core::observability::*;
}

pub mod provider {
    //! Provider facade.
    pub use molo_core::provider::*;
    #[cfg(feature = "openai")]
    pub use molo_openai::{OpenAiProvider, StructuredOutputMode};
}

pub mod run {
    //! Run protocol facade.
    pub use molo_core::run::*;
}

#[cfg(feature = "skills")]
pub mod skill {
    //! Agent Skills facade.
    pub use molo_skills::*;
}

pub mod tool {
    //! Tool protocol and registry facade.
    pub use molo_agent::tool::{MissingTools, RegistryError, ToolRegistry};
    pub use molo_core::tool::*;
}

// `extern crate self`: macro-expanded code hardcodes `::molo::...`.
extern crate self as molo;

pub use molo_agent::react_agent;
pub use molo_core::async_trait;
#[cfg(feature = "macros")]
pub use molo_macros::tool;

pub use agent::{
    Agent, AgentAction, AgentConfig, AgentError, AgentEvent, AgentKernel, MessageChunk,
    ModelObservation, ModelRequest, Observation, ReActAgent, ReActAgentBuilder, ReActEvent,
};
#[cfg(feature = "structured")]
pub use agent::{StructuredOutcome, StructuredValidator, TypedAgent};
#[cfg(feature = "coding")]
pub use coding::{
    AgentChangeTracker, ApplyPatchPayload, ApplyPatchTool, CliGitInspector, CodingContextBundle,
    CodingContextError, CodingContextInclude, CodingContextProvider, CodingContextRequest,
    CodingEffectExecutor, CodingError, CodingExecutorConfig, CodingPolicyClass, CodingPolicyEngine,
    CodingPolicyInput, CommandError, CommandExecutor, CommandExecutorBackend,
    CommandExecutorCapabilities, CommandExecutorIdentity, CommandOutput, CommandOutputLimit,
    CommandPattern, CommandPayload, CommandRequest, CommandStatus, CommandTaxonomy,
    CommandTestRunner, ContentDigest, ContextBudget, DefaultCodingContextProvider,
    DefaultInstructionResolver, DependencyMetadata, DiffRequest, EnvPolicy, FileBody, FileContent,
    FilePatch, FileReadOptions, FileVersion, FileWriteContent, FileWriteResult, GitChangedFile,
    GitChangedFilesRequest, GitDiffRequest, GitError, GitHead, GitInspector, GitOperation,
    GitPayload, GitStatus, GitStatusRequest, GitStatusTool, InstructionBundle, InstructionError,
    InstructionFile, InstructionFileSpec, InstructionRequest, InstructionResolver,
    ListFilesPayload, ListFilesQuery, ListFilesTool, LocalCommandExecutor, LocalWorkspace,
    LocalWorkspaceConfig, OutputText, Patch, PatchConflict, PatchHunk, PatchOperation,
    PatchRequest, PatchResult, PolicyCapabilityMode, PolicyEnforcementReport,
    PolicyEnforcementStatus, PtyMode, ReadFilePayload, ReadFileTool, RepoSearchRequest,
    RepoSearchResults, RepoSearcher, ResolvedPath, ResolvedPathKind, RipgrepSearcher,
    RunCommandTool, SearchError, SearchMatch, SearchMode, SearchPayload, SearchRepoTool,
    SnapshotRequest, SymlinkPolicy, TestRunError, TestRunRequest, TestRunner, TextEncoding,
    VerificationResult, Workspace, WorkspaceAccess, WorkspaceDiff, WorkspaceEntry, WorkspaceError,
    WorkspacePath, WorkspaceRoot, WorkspaceSearcher, WriteFilePayload, WriteFileRequest,
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
    McpCacheHint, McpClient, McpDirectTool, McpError, McpServerId, McpToolCatalog,
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
pub use molo_core::CancellationToken;
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
