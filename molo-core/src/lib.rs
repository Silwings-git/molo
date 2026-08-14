//! Core protocol types and traits for molo.
#![warn(missing_docs)]

pub mod agent;
pub mod effect;
pub mod message;
pub mod observability;
pub mod provider;
pub mod run;
pub mod tool;

// Allows examples generated in dependent facade crates to keep using
// `::molo::...` paths through the facade.
extern crate self as molo;

pub use agent::{AgentAction, ModelObservation, ModelRequest, Observation};
pub use async_trait::async_trait;
pub use effect::{
    DisplayFormat, DisplayOutput, EffectKind, EffectObservation, EffectOutput, EffectRequest,
    EffectSource, EffectStatus, RiskLevel,
};
pub use message::{ContentBlock, ImageContent, Message, ToolCall};
pub use observability::{AgentEventRecord, EventSeverity, RedactionRecord};
pub use provider::{
    Backoff, ChatRequest, ChatResponse, FakeProvider, FakeReply, FinishReason, ModelOptions,
    Provider, ProviderCapabilities, ProviderError, ProviderRequestContext, RetryPolicy,
    RetryProvider, Retryable, StreamEvent, Usage,
};
pub use run::{Artifact, RunContext, RunMetadata, RunOutput, RunRequest, RunSummary, UserInput};
pub use tokio_util::sync::CancellationToken;
pub use tool::{
    SharedState, SideEffectLevel, Tool, ToolContext, ToolError, ToolMemoryPolicy, ToolNamespace,
    ToolNamespaceKind, ToolOutput, ToolPolicy, ToolResult, ToolSchema, ToolSource, ToolTrustLevel,
};
