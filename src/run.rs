//! Structured run protocol shared by agent implementations and callers.
//!
//! The structured entry point is [`RunRequest`] + [`RunContext`] →
//! [`RunOutput`]. The older text helpers on [`Agent`](crate::Agent) remain
//! convenient wrappers for callers that only need the final answer string.
//!
//! # Examples
//!
//! ```rust
//! # #[tokio::main]
//! # async fn main() -> Result<(), molo::AgentError> {
//! use molo::{react_agent, Agent, FakeProvider, FakeReply, RunContext, RunRequest};
//! use std::time::Duration;
//!
//! let mut agent = react_agent!(FakeProvider::new([FakeReply::Text("Hello".into())]));
//! let output = agent
//!     .run_request_with_context(
//!         RunRequest::text("hi"),
//!         RunContext::new("request-42").with_timeout(Duration::from_secs(30)),
//!     )
//!     .await?;
//!
//! assert_eq!(output.run_id, "request-42");
//! assert_eq!(output.answer, "Hello");
//! assert_eq!(output.summary.rounds, 1);
//! # Ok(())
//! # }
//! ```

use crate::message::{ContentBlock, Message};
use crate::provider::{FinishReason, ModelOptions, Usage};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio_util::sync::CancellationToken;

/// Request, context, or output metadata for one run.
///
/// Metadata is caller- or implementation-owned key/value data. It is not
/// automatically inserted into model context; an agent, harness, provider
/// adapter, or application layer must opt in explicitly.
pub type RunMetadata = BTreeMap<String, serde_json::Value>;

/// User input accepted by a run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum UserInput {
    /// Plain text input, equivalent to [`Message::user`].
    Text(String),
    /// Multi-block user input, equivalent to [`Message::user_blocks`].
    Blocks(Vec<ContentBlock>),
}

impl UserInput {
    /// Constructs plain text input.
    pub fn text(input: impl Into<String>) -> Self {
        Self::Text(input.into())
    }

    /// Constructs multi-block input.
    pub fn blocks(blocks: Vec<ContentBlock>) -> Self {
        Self::Blocks(blocks)
    }

    /// Converts this input into the user [`Message`] recorded for the run.
    pub fn into_message(self) -> Message {
        match self {
            Self::Text(input) => Message::user(input),
            Self::Blocks(blocks) => Message::user_blocks(blocks),
        }
    }

    /// Returns the inner text when this input is plain text.
    ///
    /// Multi-block inputs return `None`; callers that need to inspect those
    /// should match on [`UserInput`] and handle blocks explicitly.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(input) => Some(input),
            Self::Blocks(_) => None,
        }
    }
}

impl From<String> for UserInput {
    fn from(input: String) -> Self {
        Self::Text(input)
    }
}

impl From<&str> for UserInput {
    fn from(input: &str) -> Self {
        Self::Text(input.to_string())
    }
}

impl From<Vec<ContentBlock>> for UserInput {
    fn from(blocks: Vec<ContentBlock>) -> Self {
        Self::Blocks(blocks)
    }
}

/// Input and model parameters for one run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunRequest {
    /// User input to record and send into the agent loop.
    pub input: UserInput,
    /// Request-scoped model options.
    ///
    /// `None` uses the agent's configured defaults. `Some(options)` replaces
    /// the configured defaults for this run. Typed output schemas override
    /// `options.structured`.
    pub options: Option<ModelOptions>,
    /// Caller-owned metadata for this run request.
    pub metadata: RunMetadata,
}

impl RunRequest {
    /// Constructs a text request.
    pub fn text(input: impl Into<String>) -> Self {
        Self {
            input: UserInput::text(input),
            options: None,
            metadata: RunMetadata::new(),
        }
    }

    /// Constructs a multi-block request.
    pub fn blocks(blocks: Vec<ContentBlock>) -> Self {
        Self {
            input: UserInput::blocks(blocks),
            options: None,
            metadata: RunMetadata::new(),
        }
    }

    /// Sets the model options for this request.
    pub fn with_options(mut self, options: ModelOptions) -> Self {
        self.options = Some(options);
        self
    }

    /// Sets caller-owned metadata for this request.
    pub fn with_metadata(mut self, metadata: RunMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

impl From<String> for RunRequest {
    fn from(input: String) -> Self {
        Self::text(input)
    }
}

impl From<&str> for RunRequest {
    fn from(input: &str) -> Self {
        Self::text(input)
    }
}

impl From<UserInput> for RunRequest {
    fn from(input: UserInput) -> Self {
        Self {
            input,
            options: None,
            metadata: RunMetadata::new(),
        }
    }
}

/// Execution controls and host-owned metadata for one run.
#[derive(Clone)]
pub struct RunContext {
    /// Stable correlation id for this run.
    pub run_id: String,
    /// Cooperative cancellation source for this run.
    pub cancellation: CancellationToken,
    /// Optional wall-clock deadline for the run.
    pub deadline: Option<Instant>,
    /// Host-owned execution metadata.
    pub metadata: RunMetadata,
}

impl fmt::Debug for RunContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RunContext")
            .field("run_id", &self.run_id)
            .field("cancellation", &"CancellationToken")
            .field("deadline", &self.deadline)
            .field("metadata", &self.metadata)
            .finish()
    }
}

impl RunContext {
    /// Constructs a context with a generated process-local run id.
    pub fn generated() -> Self {
        Self::new(generated_run_id())
    }

    /// Constructs a context with a caller-provided run id.
    pub fn new(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            cancellation: CancellationToken::new(),
            deadline: None,
            metadata: RunMetadata::new(),
        }
    }

    /// Sets the cooperative cancellation token.
    pub fn with_cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = token;
        self
    }

    /// Sets an absolute wall-clock deadline.
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Sets a deadline relative to now.
    pub fn with_timeout(self, timeout: Duration) -> Self {
        self.with_deadline(Instant::now() + timeout)
    }

    /// Sets host-owned execution metadata.
    pub fn with_metadata(mut self, metadata: RunMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.is_cancelled()
    }

    /// Whether the deadline has elapsed.
    pub fn is_expired(&self) -> bool {
        self.deadline
            .is_some_and(|deadline| Instant::now() >= deadline)
    }

    /// Remaining wall-clock time before the deadline.
    ///
    /// Returns `None` when no deadline is set, and `Some(Duration::ZERO)`
    /// after the deadline has elapsed.
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
    }
}

/// Creates a process-local unique run id.
pub(crate) fn generated_run_id() -> String {
    static START_NANOS: OnceLock<u128> = OnceLock::new();
    static PROCESS_RUN_COUNTER: AtomicU64 = AtomicU64::new(0);

    let start_nanos = *START_NANOS.get_or_init(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    });
    let n = PROCESS_RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("run-{start_nanos}-{n}")
}

/// A handle to an artifact produced by a run.
///
/// Phase 1 treats artifacts as references, not storage: this type carries no
/// bytes and does not define persistence, cleanup, or permissions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Artifact {
    /// Artifact id, unique within the producing application or store.
    pub id: String,
    /// Optional human-facing label.
    pub label: Option<String>,
    /// Optional MIME type.
    pub mime_type: Option<String>,
    /// Optional application- or store-owned URI.
    pub uri: Option<String>,
    /// Artifact metadata.
    pub metadata: RunMetadata,
}

/// Execution summary for one run.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunSummary {
    /// Number of conversation rounds.
    pub rounds: usize,
    /// Total number of tool executions.
    pub tool_calls: usize,
    /// Sum of token usage across provider turns.
    pub usage: Usage,
    /// Final direct-answer provider finish reason, when available.
    pub finish_reason: Option<FinishReason>,
    /// Wall-clock run latency.
    pub latency: Duration,
    /// Provider model identifier, when the provider exposes one.
    pub provider_model: Option<String>,
}

/// Structured result of one non-streaming run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunOutput {
    /// Run id copied from the [`RunContext`].
    pub run_id: String,
    /// Final assistant answer text.
    pub answer: String,
    /// Execution summary.
    pub summary: RunSummary,
    /// Final assistant message.
    pub final_message: Message,
    /// Artifact handles produced by this run.
    pub artifacts: Vec<Artifact>,
    /// Implementation-owned output metadata.
    pub metadata: RunMetadata,
}

/// Typed-output result paired with the raw structured run output.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedRunOutput<T> {
    /// Deserialized typed value.
    pub value: T,
    /// Raw run output, including the JSON text in [`RunOutput::answer`].
    pub output: RunOutput,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn user_input_converts_to_message() {
        assert_eq!(UserInput::text("hi").into_message(), Message::user("hi"));
        let blocks = vec![ContentBlock::Text("hi".into())];
        assert_eq!(
            UserInput::blocks(blocks.clone()).into_message(),
            Message::user_blocks(blocks)
        );
    }

    #[test]
    fn run_request_builders_set_fields() {
        let mut metadata = RunMetadata::new();
        metadata.insert("trace".into(), json!("abc"));
        let request = RunRequest::text("hi")
            .with_options(ModelOptions {
                temperature: Some(0.1),
                ..Default::default()
            })
            .with_metadata(metadata.clone());

        assert_eq!(request.input, UserInput::text("hi"));
        assert_eq!(
            request.options.as_ref().and_then(|o| o.temperature),
            Some(0.1)
        );
        assert_eq!(request.metadata, metadata);
    }

    #[test]
    fn run_context_helpers() {
        let a = RunContext::generated();
        let b = RunContext::generated();
        assert_ne!(a.run_id, b.run_id);
        assert!(a.run_id.starts_with("run-"));

        let named = RunContext::new("request-42");
        assert_eq!(named.run_id, "request-42");

        let token = CancellationToken::new();
        let cancelled = RunContext::new("cancel").with_cancellation(token.clone());
        token.cancel();
        assert!(cancelled.is_cancelled());

        let expired = RunContext::new("expired").with_deadline(Instant::now());
        assert!(expired.is_expired());
        assert_eq!(expired.remaining(), Some(Duration::ZERO));
    }
}
