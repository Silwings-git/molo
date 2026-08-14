//! Agent runtime, memory, channels, and tool registry for molo.
#![warn(missing_docs)]

pub mod agent;
pub mod event_channel;
pub mod memory;
pub mod message_channel;
pub mod tool;

pub use molo_core::{effect, message, observability, provider, run};

extern crate self as molo;

pub use agent::{
    Agent, AgentAction, AgentConfig, AgentError, AgentEvent, AgentKernel, MessageChunk,
    ModelObservation, ModelRequest, Observation, ReActAgent, ReActAgentBuilder, ReActEvent,
};
#[cfg(feature = "structured")]
pub use agent::{StructuredOutcome, StructuredValidator, TypedAgent};
pub use event_channel::{
    BroadcastEventChannel, EventChannel, EventChannelStats, EventReceiver, MpscEventChannel,
};
pub use memory::{
    Budget, CharTokenCounter, InMemoryMemory, Memory, MemoryError, SummarizeStrategy, TokenCounter,
    TrimResult, TrimStrategy, WindowDrop, WindowMemory,
};
#[cfg(feature = "cli-channel")]
pub use message_channel::CliMessageChannel;
pub use message_channel::{
    BroadcastChannel, BroadcastReceiver, ChannelError, IncomingMessage, MessageChannel,
    MpscChannel, WatchChannel, WatchReceiver,
};
pub use molo_core::CancellationToken;
#[cfg(feature = "structured")]
pub use molo_core::run::TypedRunOutput;
pub use molo_core::{
    Artifact, Backoff, ChatRequest, ChatResponse, ContentBlock, DisplayFormat, DisplayOutput,
    EffectKind, EffectObservation, EffectOutput, EffectRequest, EffectSource, EffectStatus,
    FakeProvider, FakeReply, FinishReason, ImageContent, Message, ModelOptions, Provider,
    ProviderCapabilities, ProviderError, ProviderRequestContext, RetryPolicy, RetryProvider,
    Retryable, RiskLevel, RunContext, RunMetadata, RunOutput, RunRequest, RunSummary, SharedState,
    SideEffectLevel, StreamEvent, Tool, ToolCall, ToolContext, ToolError, ToolMemoryPolicy,
    ToolNamespace, ToolNamespaceKind, ToolOutput, ToolPolicy, ToolResult, ToolSchema, ToolSource,
    ToolTrustLevel, Usage, UserInput, async_trait,
};
pub use tool::{MissingTools, RegistryError, ToolRegistry};
