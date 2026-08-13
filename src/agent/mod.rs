//! Agents: reasoning loops.
//!
//! An Agent is expressed as a trait (consistent with the rest of the
//! library: Provider / Tool / Memory); the concrete reasoning loops (ReAct /
//! Plan & Execute / ...) are provided by each implementation.
//!
//! This module provides:
//! - Interfaces: [`Agent`] reasoning-loop trait, [`CancellableAgent`]
//!   optional cooperative cancellation, [`AgentEvent`] application-level
//!   event interface, [`AgentError`] run-failure reasons;
//! - The classic assembly: [`ReActAgent`] generic ReAct loop and its
//!   convenience macro [`react_agent!`](crate::react_agent);
//! - Sub-agent parts: [`SubAgentTool`] sub-agent as a tool, [`SubAgentPool`]
//!   named sub-agent pool (the main loop delegates sub-loops via tools);
//! - Structured output: [`TypedAgent`] typed-output interface,
//!   [`StructuredValidator`] validation component (validation / feedback
//!   messages / retry budget in one);
//! - Message chunks and summaries: [`MessageChunk`] / [`RunSummary`];
//! - Optional behavior configuration: [`AgentConfig`].
//!
//! Execution state such as goal / plan / step does not belong to the
//! [`Agent`] trait; each concrete loop manages it itself.
//!
//! # Examples
//!
//! Assemble an agent in one shot with [`react_agent!`](crate::react_agent)
//! and run a round of conversation:
//!
//! ```
//! # #[tokio::main]
//! # async fn main() -> Result<(), molo::AgentError> {
//! use molo::{react_agent, Agent, FakeProvider, FakeReply};
//!
//! let mut agent = react_agent!(
//!     FakeProvider::new([
//!         FakeReply::Text("Hello".into()),
//!         FakeReply::Text("Hello again".into()),
//!     ]),
//!     "You are a helpful assistant",
//! );
//! let answer = agent.run("hi").await?;
//! assert_eq!(answer, "Hello");
//!
//! let output = agent.run_request(molo::RunRequest::text("hi again")).await?;
//! assert_eq!(output.answer, "Hello again");
//! # Ok(())
//! # }
//! ```

mod config;
mod events;
mod react;
mod structured;
mod sub_agent;

pub use config::AgentConfig;
pub use events::ReActEvent;
pub use react::{
    ReActAgent, SerialToolRoundExecutor, ToolCallOutcome, ToolRoundCtx, ToolRoundExecutor,
};
pub use structured::{
    StructuredOutcome, StructuredValidator, structured_retry_message, validate_structured,
};
pub use sub_agent::{PoolError, SubAgentPool, SubAgentTool};

use crate::memory::MemoryError;
use crate::provider::ProviderError;
use crate::run::{RunContext, RunOutput, RunRequest, TypedRunOutput};
use futures::stream::BoxStream;
use std::fmt;
use tokio_util::sync::CancellationToken;

pub use crate::run::RunSummary;

/// Reasoning-loop interface: one `run` takes the user input, drives the
/// reasoning loop, and returns the final answer.
///
/// Every reasoning loop (the built-in [`ReActAgent`] and custom
/// implementations) implements this trait; implementations that want
/// cooperative cancellation additionally implement [`CancellableAgent`].
///
/// The streaming and non-streaming entry points share the same semantics:
/// the reply is either given whole ([`run`](Agent::run)) or returned as a
/// [`MessageChunk`] stream ([`run_stream`](Agent::run_stream), ending with
/// [`MessageChunk::Done`]).
#[async_trait::async_trait]
pub trait Agent {
    /// One structured run with caller-provided execution context.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError::Memory`] when context access fails;
    /// [`AgentError::Provider`] when communicating with the LLM fails;
    /// [`AgentError::TooManyToolRounds`] when the model keeps requesting
    /// tools beyond [`AgentConfig::max_tool_rounds`] without a final answer;
    /// [`AgentError::Cancelled`] when the run is cooperatively cancelled
    /// (the passed [`CancellationToken`] is requested);
    /// [`AgentError::DeadlineExceeded`] when the run-level deadline elapses.
    async fn run_request_with_context(
        &mut self,
        request: RunRequest,
        context: RunContext,
    ) -> Result<RunOutput, AgentError>;

    /// One structured run with a generated [`RunContext`].
    async fn run_request(&mut self, request: RunRequest) -> Result<RunOutput, AgentError> {
        self.run_request_with_context(request, RunContext::generated())
            .await
    }

    /// One run: record the user input, drive the reasoning loop, and return
    /// the model's final answer as text.
    async fn run(&mut self, input: &str) -> Result<String, AgentError> {
        Ok(self.run_request(RunRequest::text(input)).await?.answer)
    }

    /// Streaming structured run with caller-provided execution context.
    async fn run_stream_request_with_context<'a>(
        &'a mut self,
        request: RunRequest,
        context: RunContext,
    ) -> Result<BoxStream<'a, Result<MessageChunk, AgentError>>, AgentError> {
        let output = self.run_request_with_context(request, context).await?;
        Ok(Box::pin(futures::stream::iter([
            Ok(MessageChunk::Delta(output.answer)),
            Ok(MessageChunk::Done(output.summary)),
        ])))
    }

    /// Streaming structured run with a generated [`RunContext`].
    async fn run_stream_request<'a>(
        &'a mut self,
        request: RunRequest,
    ) -> Result<BoxStream<'a, Result<MessageChunk, AgentError>>, AgentError> {
        self.run_stream_request_with_context(request, RunContext::generated())
            .await
    }

    /// Streaming run: same semantics as [`run`](Agent::run), with the reply
    /// returned as a stream of message chunks (see [`MessageChunk`]), ending
    /// with [`MessageChunk::Done`]; errors are produced as `Err` items and
    /// terminate the stream (no Done afterwards).
    ///
    /// The default implementation is not truly streaming — it completes the
    /// structured run first, then emits a single [`MessageChunk::Delta`] with
    /// the final answer and a [`MessageChunk::Done`] carrying the run
    /// summary. Implementations that need per-token streaming or tool
    /// progress should override this method.
    async fn run_stream<'a>(
        &'a mut self,
        input: &'a str,
    ) -> Result<BoxStream<'a, Result<MessageChunk, AgentError>>, AgentError> {
        self.run_stream_request(RunRequest::text(input)).await
    }
}

/// Optional capability: cooperative cancellation.
///
/// opt-in — implementations that don't need cancellation don't implement
/// this trait (the methods don't even exist at compile time, so there's no
/// fake cancellation where "the default implementation ignores the token");
/// callers that need cancellation (such as interactive apps) call
/// [`run_cancellable`](CancellableAgent::run_cancellable)
/// / [`run_stream_cancellable`](CancellableAgent::run_stream_cancellable)
/// directly on the concrete type.
///
/// Each run carries a [`CancellationToken`], the cooperative cancellation
/// source for this run — any holder can cancel the same token (UI button /
/// timeout / external signal); implementations should check at safe points
/// and terminate promptly: `run_cancellable` returns
/// `Err([`AgentError::Cancelled`])`, while `run_stream_cancellable`
/// terminates with a [`MessageChunk::Cancelled`] terminal chunk (no `Done`).
/// Messages already recorded are kept, not rolled back.
///
/// # Examples
///
/// An already-cancelled token makes the run fail immediately; a fresh token
/// lets it proceed:
///
/// ```
/// # #[tokio::main]
/// # async fn main() -> Result<(), molo::AgentError> {
/// use molo::agent::{CancellableAgent, ReActAgent};
/// use molo::provider::{FakeProvider, FakeReply};
/// use molo::tool::ToolRegistry;
/// use molo::CancellationToken;
///
/// let mut agent = ReActAgent::new(
///     FakeProvider::new([FakeReply::Text("Hello".into())]),
///     ToolRegistry::new(),
///     "",
/// );
///
/// let cancelled = CancellationToken::new();
/// cancelled.cancel();
/// // Cancelled token: run returns Err(AgentError::Cancelled) immediately
/// assert!(agent.run_cancellable("hi", &cancelled).await.is_err());
///
/// // Fresh token: completes normally
/// let fresh = CancellationToken::new();
/// assert_eq!(agent.run_cancellable("hi", &fresh).await?, "Hello");
/// # Ok(())
/// # }
/// ```
#[async_trait::async_trait]
pub trait CancellableAgent: Agent {
    /// Run with a cancellation source. Returns `Err(AgentError::Cancelled)`
    /// on cancellation.
    async fn run_cancellable(
        &mut self,
        input: &str,
        token: &CancellationToken,
    ) -> Result<String, AgentError> {
        let context = RunContext::generated().with_cancellation(token.clone());
        Ok(self
            .run_request_with_context(RunRequest::text(input), context)
            .await?
            .answer)
    }

    /// Streaming run with a cancellation source: same semantics as
    /// [`run_cancellable`](CancellableAgent::run_cancellable); when
    /// cancelled, terminates with a [`MessageChunk::Cancelled`] terminal
    /// chunk (no `Done`).
    ///
    /// The default implementation is not truly streaming — the whole answer
    /// is given as a single [`MessageChunk::Delta`] chunk; implementations
    /// that need per-character streaming should override this method.
    async fn run_stream_cancellable<'a>(
        &'a mut self,
        input: &'a str,
        token: &CancellationToken,
    ) -> Result<BoxStream<'a, Result<MessageChunk, AgentError>>, AgentError> {
        let context = RunContext::generated().with_cancellation(token.clone());
        match self
            .run_stream_request_with_context(RunRequest::text(input), context)
            .await
        {
            Ok(stream) => Ok(stream),
            Err(AgentError::Cancelled) => Ok(Box::pin(futures::stream::iter([Ok(
                MessageChunk::Cancelled,
            )]))),
            Err(e) => Err(e),
        }
    }
}

/// Optional capability: typed output (opt-in — implementations that don't
/// need it don't implement it; the method doesn't even exist at compile
/// time, the same pattern as [`CancellableAgent`]).
///
/// [`run_typed`](TypedAgent::run_typed) has the same semantics as
/// [`Agent::run`] (records input, drives the reasoning loop), but
/// deserializes the final answer into the type parameter `U` once it passes
/// validation — this run auto-generates a JSON Schema from `U`
/// (`schemars`-derived), feeds validation failures back to the model for
/// retry, and reports [`AgentError::StructuredRetriesExhausted`] when the
/// budget is exhausted.
///
/// **Why separate from [`Agent`]**: trait generic methods are not
/// object-safe — putting it in `Agent` would immediately break
/// `Box<dyn Agent>` (sub-agent delegation, etc.); a separate trait leaves
/// `Box<dyn Agent>` unaffected, and code with the generic bound
/// `A: TypedAgent` can call it on any implementation.
///
/// **No default implementation**: validation retries happen inside the
/// reasoning loop (a failure is fed back to the model and the conversation
/// continues), while `Agent::run` is a one-shot call — a default
/// implementation couldn't retry within the loop; implementors assemble the
/// public parts [`StructuredValidator`] (validation / feedback messages /
/// retry budget in one) or the pure functions [`validate_structured`] /
/// [`structured_retry_message`] inside their own loops (the built-in
/// [`ReActAgent`] assembly is exactly this shape).
#[async_trait::async_trait]
pub trait TypedAgent: Agent {
    /// Typed structured run with caller-provided execution context.
    async fn run_typed_request_with_context<U>(
        &mut self,
        request: RunRequest,
        context: RunContext,
    ) -> Result<TypedRunOutput<U>, AgentError>
    where
        U: serde::de::DeserializeOwned + schemars::JsonSchema + Send + Sync;

    /// Typed structured run with a generated [`RunContext`].
    async fn run_typed_request<U>(
        &mut self,
        request: RunRequest,
    ) -> Result<TypedRunOutput<U>, AgentError>
    where
        U: serde::de::DeserializeOwned + schemars::JsonSchema + Send + Sync,
    {
        self.run_typed_request_with_context(request, RunContext::generated())
            .await
    }

    /// Typed run: the final answer is deserialized into `U` once validation
    /// passes.
    ///
    /// # Errors
    ///
    /// - [`AgentError::StructuredRetriesExhausted`][]: validation failures
    ///   are fed back to the model for retry, with the budget defined by the
    ///   implementation (see
    ///   [`AgentConfig::max_structured_retries`](crate::agent::AgentConfig)
    ///   for the built-in assembly); returned when the budget is exhausted
    ///   without success;
    /// - [`AgentError::StructuredParse`][]: validation passed but
    ///   deserialization failed;
    /// - otherwise the same as [`Agent::run`](Agent::run).
    async fn run_typed<U>(&mut self, input: &str) -> Result<U, AgentError>
    where
        U: serde::de::DeserializeOwned + schemars::JsonSchema + Send + Sync,
    {
        Ok(self
            .run_typed_request::<U>(RunRequest::text(input))
            .await?
            .value)
    }
}

/// Message chunks for a streaming run — the streaming output of one run,
/// sliced into pieces.
///
/// `Delta` / `ToolCall` / `ToolResult` are the streaming projection of the
/// message record (text the model is generating, recorded Assistant tool
/// requests, and returned ToolResult messages); `Done` / `Cancelled` are
/// terminal markers. These are not "events" — real events are the
/// application-level event abstraction
/// [`AgentEvent`](trait, where each Agent implementation defines its own
/// event variants (the framework doesn't anticipate them), flowing through
/// an event pipeline.
///
/// # Reasoning
///
/// Reasoning produces no chunks: reasoning deltas from thinking models do
/// not appear in this enum — matching on `MessageChunk::Reasoning` won't
/// compile, and that's intentional. To surface reasoning, attach an
/// [`EventChannel`](crate::event_channel::EventChannel) and subscribe to
/// [`ReActEvent::Reasoning`], or consume
/// [`StreamEvent::Reasoning`](crate::provider::StreamEvent::Reasoning) at
/// the Provider layer.
///
/// The enum carries `#[non_exhaustive]` (reserved for extension): matches
/// must include a wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MessageChunk {
    /// An increment of the reply text; increments within the same round are
    /// concatenated in order.
    Delta(String),
    /// The model requested a tool call (the tool_calls of a recorded
    /// Assistant message).
    ToolCall {
        /// The id of this call, matching [`ToolCall::id`](crate::ToolCall::id)
        /// in the recorded memory; used to pair multiple calls of the same
        /// tool within a round.
        id: String,
        /// The tool name.
        name: String,
        /// The arguments generated by the model (JSON text).
        arguments: String,
    },
    /// A tool execution completed (records a ToolResult message; on failure
    /// the content is the error text).
    ToolResult {
        /// The id of this execution, matching
        /// [`Message::ToolResult`](crate::Message::ToolResult) in the
        /// recorded memory; paired with the
        /// [`ToolCall`](MessageChunk::ToolCall) id in the same round.
        id: String,
        /// The tool name.
        name: String,
        /// The execution result text (the error text on failure).
        content: String,
    },
    /// The run ended normally; carries the execution summary for this run
    /// ([`RunSummary`]); the stream produces no further chunks afterwards.
    Done(RunSummary),
    /// The run was cooperatively cancelled (via the CancellationToken passed
    /// to run/run_stream); terminal chunk, the stream produces no further
    /// chunks afterwards (no Done).
    Cancelled,
}

/// Application-level event abstraction.
///
/// Each Agent implementation defines its own event types (tool lifecycle /
/// plan steps / retrieval / sub-agents, etc.); the framework doesn't
/// anticipate variants. Events are pushed through
/// [`EventChannel`](crate::event_channel::EventChannel) for external
/// subscription. Consumers downcast known types precisely via `as_any` (see
/// [`impl dyn AgentEvent`](AgentEvent) below) and fall back to
/// [`name`](AgentEvent::name) for unknown types.
///
/// Event payloads are uniformly `Arc<dyn AgentEvent>`: `Arc` covers the
/// clone requirement of broadcast channels, the trait itself needs no
/// `Clone`, and event types are zero-boilerplate.
pub trait AgentEvent: std::any::Any + Send + Sync + fmt::Debug {
    /// Event name: lets subscribers at least display a name for unknown
    /// types. Default = the type's full path; override for a short name
    /// (e.g. `"tool.started"`).
    fn name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

impl dyn AgentEvent {
    /// Typed access: `event.as_any().downcast_ref::<ToolStarted>()`.
    ///
    /// Declared as an inherent method rather than a trait default method:
    /// `&dyn AgentEvent` → `&dyn Any` is a trait-object upcast (Rust 1.86+,
    /// with `Any` as a supertrait), which can't be expressed directly in a
    /// trait default method.
    pub fn as_any(&self) -> &dyn std::any::Any {
        self as &dyn std::any::Any
    }
}

/// Reasons an Agent run can fail.
///
/// A tool execution failure is not an `AgentError` — it is fed back to the
/// model as text, and the model decides what to do next. `#[non_exhaustive]`
/// ensures future error categories won't be a breaking change.
///
/// # Examples
///
/// ```
/// use molo::AgentError;
///
/// // The round-limit error carries the limit value, useful for prompting
/// // the user to adjust the config
/// let err = AgentError::TooManyToolRounds(10);
/// assert!(err.to_string().contains("10"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AgentError {
    /// Context access failed.
    #[error("memory error: {0}")]
    Memory(#[from] MemoryError),
    /// Communication with the LLM failed.
    #[error("provider error: {0}")]
    Provider(#[from] ProviderError),
    /// The model kept requesting tools past the implementation's maximum
    /// number of rounds without giving a final answer. Increase via
    /// [`AgentConfig::max_tool_rounds`](crate::agent::AgentConfig) (chained
    /// `with_config(AgentConfig { max_tool_rounds: N, ..Default::default() })`).
    #[error(
        "model requested tools for more than {0} rounds; increase AgentConfig::max_tool_rounds (via with_config) if intended"
    )]
    TooManyToolRounds(usize),
    /// The run was cooperatively cancelled (via the CancellationToken passed
    /// to run/run_stream); already-recorded messages are kept, not rolled
    /// back.
    #[error("run cancelled")]
    Cancelled,
    /// Structured output: validation passed but deserializing into the
    /// target type failed — triggered when the JSON the schema allows is
    /// inconsistent with the serde representation of `run_typed`'s type
    /// parameter `U` (auto-generated schemas agree with `U` by default;
    /// conflicts come from `#[schemars(...)]` custom derives).
    #[error("structured output failed to deserialize: {0}")]
    StructuredParse(String),
    /// Structured output: validation failed more times than the configured
    /// limit. Increase via
    /// [`AgentConfig::max_structured_retries`](crate::agent::AgentConfig)
    /// (chained
    /// `with_config(AgentConfig { max_structured_retries: N, ..Default::default() })`).
    #[error(
        "structured output failed validation for more than {0} attempts; increase AgentConfig::max_structured_retries (via with_config) if intended"
    )]
    StructuredRetriesExhausted(usize),
    /// The run-level deadline elapsed before the run completed.
    #[error("run deadline exceeded")]
    DeadlineExceeded,
}
