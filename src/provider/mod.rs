//! Provider: the interface for communicating with an LLM.
//!
//! This module defines the `Provider` trait and its companion data types:
//! requests ([`ChatRequest`]), responses ([`ChatResponse`] / [`StreamEvent`]),
//! errors ([`ProviderError`]), and usage ([`Usage`]). The trait itself is
//! vendor-agnostic; the implementations ([`FakeProvider`] /
//! [`RetryProvider`], and [`OpenAiProvider`] with the `openai` feature) all
//! implement the same trait, and any implementation can be wrapped by
//! [`RetryProvider`] to gain retry capability.

mod fake;
#[cfg(feature = "openai")]
mod openai;
mod retry;

pub use fake::{FakeProvider, FakeReply};
#[cfg(feature = "openai")]
pub use openai::{OpenAiProvider, StructuredOutputMode};
pub use retry::{Backoff, RetryPolicy, RetryProvider, Retryable};

use crate::message::Message;
use crate::tool::ToolSchema;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::time::Duration;

/// The interface for chatting with an LLM.
///
/// Implementations are responsible for communicating with a specific LLM
/// service and mapping vendor responses back to this framework's [`Message`];
/// [`chat`](Provider::chat) returns the full reply at once, while
/// [`stream_chat`](Provider::stream_chat) returns the same reply incrementally
/// as a stream of events. Both share the same semantics and differ only in
/// delivery.
///
/// `Send + Sync` guarantees that `Box<dyn Provider>` can be held across
/// awaits in Agent implementations (for the same reason as
/// [`Tool`](crate::tool::Tool)).
///
/// # Examples
///
/// The calling convention is identical for every implementation; the example
/// below uses [`FakeProvider`]:
///
/// ```rust
/// # #[tokio::main]
/// # async fn main() -> Result<(), molo::ProviderError> {
/// use molo::provider::{ChatRequest, FakeProvider, FakeReply, Provider};
///
/// let fake = FakeProvider::new([FakeReply::Text("hi".into())]);
/// let response = fake.chat(ChatRequest::default()).await?;
/// assert_eq!(response.message, molo::message::Message::assistant("hi"));
/// # Ok(())
/// # }
/// ```
#[async_trait::async_trait]
pub trait Provider: Send + Sync {
    /// Model identifier exposed by this provider, when known.
    ///
    /// Agents copy this value into run summaries for observability. Providers
    /// that are not bound to one model can keep the default `None`.
    fn model(&self) -> Option<&str> {
        None
    }

    /// Sends one turn of conversation and returns the model's reply (text, or
    /// a request to call tools).
    ///
    /// # Errors
    ///
    /// Network failures / timeouts / rate limits / vendor business errors are
    /// all returned as [`ProviderError`]; see that type's docs for error
    /// classification and retry guidance.
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ProviderError>;

    /// Streams one turn of conversation, returning the Assistant's reply
    /// incrementally.
    ///
    /// Semantically identical to [`chat`](Provider::chat), except that the
    /// reply is returned as a stream of events: several [`StreamEvent::Delta`]
    /// items concatenated in order form the full reply, and the stream ends
    /// with [`StreamEvent::Done`].
    ///
    /// # Errors
    ///
    /// Failures during request setup (connection / timeout / vendor rejection)
    /// are returned as `Err`; event errors after the stream is established are
    /// produced as `Err` items in the stream, and no success events are
    /// produced after an error item (see [`StreamEvent`] for termination
    /// semantics).
    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError>;
}

/// `Box<dyn Provider>` is itself a Provider: re-exposes the trait object as a
/// value, for assembly patterns that need to "hold an instance and create a
/// new loop per call" (e.g., a sub-agent factory that captures a provider and
/// constructs a fresh loop for each invocation).
#[async_trait::async_trait]
impl Provider for Box<dyn Provider> {
    fn model(&self) -> Option<&str> {
        self.as_ref().model()
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ProviderError> {
        self.as_ref().chat(request).await
    }

    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError> {
        self.as_ref().stream_chat(request).await
    }
}

/// A single conversation request.
///
/// # Examples
///
/// ```rust
/// use molo::message::Message;
/// use molo::provider::ChatRequest;
///
/// let request = ChatRequest {
///     messages: vec![Message::user("hi")],
///     ..Default::default()
/// };
/// # let _ = request;
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    /// Conversation history (in order of occurrence, assembled by the caller).
    pub messages: Vec<Message>,
    /// Tool definitions offered to the model; when empty, the model sees no
    /// tools.
    pub tools: Vec<ToolSchema>,
    /// Model options; `Default` means all vendor defaults.
    pub options: ModelOptions,
}

/// Model options for one conversation.
///
/// Common parameters are provided as typed fields (temperature / max tokens,
/// where `None` means vendor default); **vendor-specific or framework-unknown
/// parameters go into [`extra`](ModelOptions::extra)** and are passed through
/// to the vendor verbatim under their wire field names — so users can use new
/// parameters without waiting for a framework update:
///
/// ```rust
/// use molo::ModelOptions;
///
/// let mut options = ModelOptions::default();
/// options.extra.insert("top_p".into(), serde_json::json!(0.9));
/// ```
///
/// Extra keys that collide with framework-managed fields are ignored in favor
/// of the typed fields.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelOptions {
    /// Sampling temperature; `None` means vendor default.
    pub temperature: Option<f32>,
    /// Maximum number of tokens for the reply; `None` means vendor default.
    pub max_tokens: Option<u32>,
    /// Vendor extension parameters: keys are wire request field names,
    /// serialized into the request body verbatim.
    pub extra: BTreeMap<String, serde_json::Value>,
    /// Structured output: the final answer must be JSON conforming to this
    /// **JSON Schema document** (the serialized `RootSchema` produced by
    /// schemars, or a hand-written schema).
    ///
    /// Two layers of semantics:
    /// - Provider side: compatible endpoints receive it via `response_format`
    ///   to best-effort constrain the model (the OpenAI-compatible
    ///   `json_schema` shape; unsupported endpoints ignore it or error);
    /// - Agent side: the final answer is **validated framework-side**, and on
    ///   mismatch the validation error is fed back to the model for a retry
    ///   (counted against the turn budget; see
    ///   [`ReActAgent::with_structured_output`](crate::agent::ReActAgent::with_structured_output),
    ///   available with the `structured` feature).
    ///
    /// `None` = free-form text reply.
    pub structured: Option<serde_json::Value>,
}

/// Token usage for one conversation.
///
/// Field names match the OpenAI wire format; `total_tokens` follows the
/// vendor's convention (not necessarily the sum of the other two). `Default`
/// = all zeros (endpoints that omit usage count as zero).
///
/// Usage serves two consumers: the Agent layer accumulates it per turn into
/// the end-of-loop summary, while external observability (logs / metrics)
/// reads it directly — consumers do not need to distinguish the source.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Input tokens for this turn.
    pub prompt_tokens: u32,
    /// Output tokens for this turn.
    pub completion_tokens: u32,
    /// Total for this turn (vendor convention).
    pub total_tokens: u32,
}

impl Usage {
    /// Constructs from input / output counts; the total is summed
    /// automatically.
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

/// Usage accumulates per turn (the Agent layer sums tokens across turns).
impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        self.prompt_tokens += rhs.prompt_tokens;
        self.completion_tokens += rhs.completion_tokens;
        self.total_tokens += rhs.total_tokens;
    }
}

/// The reply to one conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatResponse {
    /// This turn's reply: a single [`Message::Assistant`] message
    /// (text + reasoning + any tool requests from the same turn stay together,
    /// one-to-one with the vendor wire structure);
    /// when the model produces nothing it is still an empty Assistant.
    ///
    /// The Agent loop executes tool requests when it sees them and appends the
    /// results as [`Message::ToolResult`] before continuing the conversation.
    pub message: Message,
    /// Why the model ended its reply; vendor-specific reasons are surfaced via
    /// [`FinishReason::Other`].
    pub finish_reason: FinishReason,
    /// Token usage for this turn; always present (endpoints that omit it
    /// count as zero, see [`Usage`]).
    pub usage: Usage,
}

/// Why the model ended its reply.
///
/// Common reasons are typed (Stop / Length); vendor-specific or
/// framework-unknown reasons are surfaced via [`Other`](FinishReason::Other)
/// carrying the vendor's raw string — users can recognize new reasons without
/// waiting for a framework update. `#[non_exhaustive]` guarantees that adding
/// new common categories in the future is not a breaking change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FinishReason {
    /// The model ended naturally (including ending the turn by requesting
    /// tool calls).
    Stop,
    /// Truncated by hitting the max_tokens limit.
    Length,
    /// A vendor-specific or framework-unknown reason, carrying the vendor's
    /// raw string.
    Other(String),
}

/// An event in a streamed conversation reply.
///
/// One streamed reply = several [`StreamEvent::Delta`] /
/// [`StreamEvent::ToolCall`] increments + one closing [`StreamEvent::Done`];
/// the caller concatenates the Deltas in order to get the full reply.
///
/// Stream termination semantics: on normal termination `Done` is always the
/// last success event on the stream; errors terminate the stream with an
/// `Err` item, and no events are produced after the error item.
///
/// The enum is `#[non_exhaustive]` (reserved for extension): matches must
/// include a wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StreamEvent {
    /// An incremental fragment of the reply content.
    Delta(String),
    /// The model requests a tool call; argument fragments have already been
    /// aggregated by the Provider into full JSON text, with fields matching
    /// [`crate::ToolCall`] so the Agent can use them directly.
    ToolCall {
        /// Unique id of this call; the execution result is paired back to it
        /// via this id.
        id: String,
        /// Tool name, corresponding to the name in
        /// [`Tool::schema`](crate::tool::Tool::schema).
        name: String,
        /// Arguments generated by the model (JSON text).
        arguments: String,
    },
    /// An incremental fragment of the model's reasoning (thinking); the
    /// vendor delivers it in fragments during the stream, and the Provider
    /// forwards each fragment as it arrives (like [`StreamEvent::Delta`]) —
    /// consumers concatenate them in order to get the full text.
    ///
    /// Corresponds to the reasoning field of [`Message`]; the Agent must store
    /// it in this turn's message and carry it back verbatim in the history
    /// (otherwise thinking models like DeepSeek / Qwen3 reject the request).
    Reasoning(String),
    /// The model finished its reply; the stream produces no more events after
    /// this.
    Done {
        /// Why the model ended its reply.
        reason: FinishReason,
        /// Token usage for this turn; `None` when the endpoint did not return
        /// it (`include_usage` off or unsupported by the endpoint).
        usage: Option<Usage>,
    },
}

/// Why a Provider call failed.
///
/// The enum categories cover the cases that need distinguishing, with details
/// carried by fields; vendor-specific errors are mapped into this type at the
/// implementation boundary. `#[non_exhaustive]` guarantees that adding new
/// categories in the future is not a breaking change.
///
/// Error classification is the basis for retry decisions (see the `Default`
/// judgment of [`Retryable`]): Network / Timeout / RateLimited are worth
/// retrying, while `Api` is judged by status (5xx retried, 4xx not — retrying
/// would not change the outcome).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ProviderError {
    /// A business error returned by the vendor (auth failure / invalid
    /// arguments, etc.), carrying the HTTP status.
    ///
    /// `status: 0` means a non-HTTP source (e.g. response parse failure /
    /// empty choices); it is not a valid status code and naturally
    /// distinguishes "returned by the vendor" from "framework-side parse
    /// failure".
    // No "provider " prefix: the type name already expresses the domain,
    // avoiding a double prefix when wrapped by AgentError::Provider
    // ("provider error: provider api error …").
    #[error("api error (status {status}): {message}")]
    Api {
        /// HTTP status code; 0 = non-HTTP source (e.g. response parse
        /// failure).
        status: u16,
        /// Error description text.
        message: String,
    },
    /// Rate limited (HTTP 429): retrying is meaningful, and the default retry
    /// policy waits before retrying.
    ///
    /// `retry_after`: the wait duration parsed from the vendor's
    /// `Retry-After` response header (numeric seconds); `None` when absent
    /// (HTTP date formats are not parsed).
    #[error("rate limited")]
    RateLimited {
        /// The wait duration indicated by the vendor (numeric seconds);
        /// `None` when missing / not numeric.
        retry_after: Option<Duration>,
    },
    /// A network-layer failure (connection failure and other transport
    /// errors).
    ///
    /// Implementations map concrete transport errors (e.g. reqwest's) into
    /// carried text, so the error type does not depend on a concrete
    /// implementation library's types.
    #[error("network error: {0}")]
    Network(String),
    /// Request timeout, carrying the stage at which it occurred (see
    /// [`TimeoutStage`]): distinguishes "cannot connect", "total duration
    /// elapsed" and "event interval stalled" for easier diagnosis.
    #[error("request timed out during {0:?}")]
    Timeout(TimeoutStage),
}

/// The stage at which a timeout occurred: one-to-one with
/// [`OpenAiProvider`]'s four timeouts (connect / non-streaming total /
/// streaming event interval / streaming total), plus the error-response-body
/// read timeout and a generic transport timeout.
///
/// The enum is `#[non_exhaustive]` (reserved for extension): matches must
/// include a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TimeoutStage {
    /// Connect timeout (Client-level connect timeout).
    Connect,
    /// Request timeout: for non-streaming, covers until the response body is
    /// fully read; for streaming, covers the connect and response-header wait.
    Request,
    /// Streaming event interval exceeded: no data between two events (idle
    /// timeout).
    Idle,
    /// Streaming total duration exceeded: the wall-clock deadline has elapsed
    /// and an active but never-ending stream is terminated (stream timeout).
    StreamTotal,
    /// Total timeout for reading an error response body.
    ResponseBody,
    /// Generic transport timeout (reqwest cannot distinguish the stage, e.g.
    /// during connect or read).
    Transport,
}
