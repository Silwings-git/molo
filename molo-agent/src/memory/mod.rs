//! Memory: manages the agent's context.
//!
//! The inference loop uses [`Memory`] to store conversation history and to
//! retrieve the full context for each request. This module provides two
//! ready-to-use implementations:
//!
//! - [`InMemoryMemory`] — stores all messages verbatim, never trims;
//! - [`WindowMemory`] — trims to a recent window against a budget, with
//!   pluggable token counting and trim strategies.
//!
//! Compaction, retrieval, persistence and similar strategies plug in by
//! implementing [`Memory`] (or injecting a [`TrimStrategy`] into
//! [`WindowMemory`]) without touching the inference loop. A built-in
//! summarization strategy is provided as [`SummarizeStrategy`].

mod in_memory;
mod summarize;
mod window;

pub use in_memory::InMemoryMemory;
pub use summarize::SummarizeStrategy;
pub use window::{
    Budget, CharTokenCounter, TokenCounter, TrimResult, TrimStrategy, WindowDrop, WindowMemory,
};

use crate::message::Message;

/// Manages the agent's context: decides which messages the model sees on each
/// turn.
///
/// The inference loop interacts with Memory in only two ways:
/// - hands each turn's new messages (user input, assistant replies, tool
///   results) to [`record`](Memory::record) /
///   [`record_protected`](Memory::record_protected);
/// - fetches the full context via [`context`](Memory::context) before each
///   request and passes it to the Provider.
///
/// How messages are stored, trimmed, and compacted is up to the implementation;
/// the inference loop is unaware of it. For example, [`WindowMemory`] trims to
/// a budget on retrieval and guarantees that a tool message and its Assistant
/// message stay paired; no implementation should split that pair (message
/// interface constraint). `Send + Sync` ensures `Box<dyn Memory>` can be held
/// across await points in agent implementations.
#[async_trait::async_trait]
pub trait Memory: Send + Sync {
    /// Records a message, appending it to the end of the session context.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] when storing or counting fails (e.g., a remote
    /// counting API is unavailable).
    async fn record(&mut self, message: Message) -> Result<(), MemoryError>;

    /// Records a **protected** message: implementations must not remove it when
    /// trimming.
    ///
    /// Used for persistent behavior guidance such as skill bodies — trimming
    /// one silently degrades behavior (the model keeps running but loses the
    /// dedicated instruction, with no visible error). The default
    /// implementation behaves like [`record`](Memory::record); implementations
    /// that trim should override it to exempt protected messages.
    ///
    /// # Errors
    ///
    /// Same as [`record`](Memory::record).
    async fn record_protected(&mut self, message: Message) -> Result<(), MemoryError> {
        self.record(message).await
    }

    /// Returns the full message sequence of the current context (in order of
    /// occurrence), which the agent hands to the Provider as-is.
    ///
    /// The returned value is a clone; mutating it does not affect Memory's
    /// internal state.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] when reading from storage or trimming fails.
    async fn context(&self) -> Result<Vec<Message>, MemoryError>;
}

/// Context access failed.
///
/// `#[non_exhaustive]`: new error variants can be added in the future without
/// breaking changes; matches should include a wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum MemoryError {
    /// Storage-layer error; in-memory implementations never produce it, but
    /// storage-backed implementations (e.g., persistence) do.
    ///
    /// The message carries no "memory error" prefix: the type name already
    /// conveys the domain, and it avoids a doubled prefix when wrapped by
    /// [`AgentError::Memory`](crate::agent::AgentError).
    #[error("{0}")]
    Storage(String),
    /// Token counting failed; remote counters (e.g., calling a counting API)
    /// can produce it, local counters never do.
    #[error("token counting failed: {0}")]
    TokenCount(String),
}
