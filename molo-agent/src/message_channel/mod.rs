//! MessageChannel: a message transport channel between an Agent and the outside world (a human or another Agent).
//!
//! The channel supports only two operations, defined by the [`MessageChannel`] trait:
//!
//! - [`MessageChannel::ask`] — send a question and wait for a reply (request-response), for scenarios that need the
//!   counterparty's confirmation (e.g. asking the user for consent before a tool runs);
//! - [`MessageChannel::notify`] — a one-way notification, sent without waiting for a reply.
//!
//! Interrupt semantics: an interrupt happens when a channel consumer (such as a tool) calls `ask`, and the call
//! returning is the resume. The channel itself is unaware of interrupts and takes no part in the resume flow.
//!
//! # Choosing a channel: which one when
//!
//! Pick a channel along two dimensions: "does it need a reply" and "how many receivers are there".
//!
//! | Need | Choice |
//! |------|------|
//! | Interacting with a human: terminal questions, confirmation, approval | [`CliMessageChannel`] (`cli-channel` feature) |
//! | One-on-one dialogue between two Agents, with request-response in both directions | [`MpscChannel`] |
//! | One-to-many broadcast notifications, no reply expected | [`BroadcastChannel`] |
//! | Observing changes of the latest state (status, heartbeat, progress) | [`WatchChannel`] |
//!
//! Selection notes:
//!
//! - **Need request-response** → [`MpscChannel`] (Agent) or
//!   [`CliMessageChannel`] (human, `cli-channel` feature); the other two don't support `ask`
//!   and return [`ChannelError::NotSupported`].
//! - **One-way notifications only** → [`BroadcastChannel`] or
//!   [`WatchChannel`]; neither waits for the counterparty's confirmation.
//!   The difference is semantic: broadcast is a message queue (bounded, drops old
//!   messages when consumers are slow); watch holds the latest value (unbounded, keeps only the newest).
//! - **Concurrent two-way questioning** → replies in `MpscChannel` are bound to their requests one-to-one, so both
//!   sides can ask each other concurrently without replies going astray; `CliMessageChannel` suits slow-paced
//!   "human confirms one at a time" interactions — while one request is unanswered, later requests queue up.
//! - **Concurrent access** → all implementations are `Send + Sync`, so they can be placed in an [`Arc`](std::sync::Arc)
//!   and shared across tasks; `ask` presents only one request at a time.
//!
//! Event observation (UI / environment subscription) does not go through this module, but through the
//! publish-subscribe of [`EventChannel`](crate::event_channel::EventChannel): conversation channels handle
//! "request-response / notifications", while event channels handle "observing the reasoning process".

/// Sends a question to the outside world and waits for a reply, or sends a one-way notification.
///
/// The receiver is not limited to humans: the same trait supports both
/// Agent-to-Agent dialogue (one-to-one `ask`) and broadcast notifications
/// (`notify`). **Serializing concurrent calls is the implementation's job**:
/// `ask` presents only one request at a time, and the next one gets its turn
/// only after the previous reply — e.g. a human can only approve one by one.
///
/// # Example
///
/// Implementing a custom channel: just answer "how to send a question" and "how to send a notification".
///
/// ```rust
/// # extern crate molo_agent as molo;
/// use molo::{ChannelError, MessageChannel};
///
/// // A custom channel that echoes the question back as the reply and drops notifications.
/// struct EchoChannel;
///
/// #[async_trait::async_trait]
/// impl MessageChannel for EchoChannel {
///     async fn ask(&self, message: &str) -> Result<String, ChannelError> {
///         Ok(format!("echo: {message}"))
///     }
///
///     async fn notify(&self, message: &str) -> Result<(), ChannelError> {
///         Ok(())
///     }
/// }
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), molo::ChannelError> {
/// let channel = EchoChannel;
/// assert_eq!(channel.ask("hello").await?, "echo: hello");
/// # Ok(())
/// # }
/// ```
#[async_trait::async_trait]
pub trait MessageChannel: Send + Sync {
    /// Sends a question and waits for a reply (request-response).
    ///
    /// Subsequent `ask` calls queue up until the previous reply is returned (serialization is the implementation's
    /// responsibility).
    async fn ask(&self, message: &str) -> Result<String, ChannelError>;

    /// A one-way notification that does not wait for a reply (broadcast to all receivers).
    async fn notify(&self, message: &str) -> Result<(), ChannelError>;
}

/// The reason a message channel failed.
///
/// When each variant is triggered:
///
/// - [`ChannelError::Io`] — underlying read/write failure (e.g. a terminal IO error);
/// - [`ChannelError::Closed`] — the channel is closed: the peer has been dropped, or input ended
///   (e.g. Ctrl-D in a terminal); after it closes, every subsequent call on the channel returns the same error;
/// - [`ChannelError::NoReply`] — [`IncomingMessage::reply`] was called on a notification message
///   that does not expect a reply;
/// - [`ChannelError::NotSupported`] — the current implementation does not support this operation (broadcast / watch
///   don't support `ask`).
///
/// The enum is `#[non_exhaustive]` (reserved for extension): matches must include a wildcard arm.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ChannelError {
    /// Underlying IO failure (reading or writing the channel failed).
    #[error("channel io error: {0}")]
    Io(String),
    /// Input has been closed (e.g. user pressed Ctrl-D); no reply is available.
    #[error("channel closed")]
    Closed,
    /// There are currently no receivers on the channel (e.g. `notify` on a watch channel with no subscribers).
    /// The channel itself is not closed: subscribing again restores operation.
    #[error("channel has no receivers")]
    NoReceiver,
    /// `reply` was called on a message (a notification) that has no reply channel.
    #[error("message does not expect a reply")]
    NoReply,
    /// This channel implementation does not support the operation (e.g. broadcast / watch don't support `ask`).
    #[error("operation not supported by this channel: {0}")]
    NotSupported(String),
}

impl From<std::io::Error> for ChannelError {
    fn from(err: std::io::Error) -> Self {
        // Carry only the error text; the prefix is added uniformly by the Io variant's Display, so implementers don't
        // prepend the prefix again and produce a doubled "channel io error: channel io error: ..." message.
        ChannelError::Io(err.to_string())
    }
}

/// A message in the queue: questions carry a reply channel, notifications don't.
#[derive(Debug)]
struct Envelope {
    message: String,
    reply: Option<tokio::sync::oneshot::Sender<String>>,
}

/// A message received from the channel: the text content, plus a reply slot that only questions have.
///
/// Returned by each channel's receive method. Check [`IncomingMessage::wants_reply`] first —
/// for question messages, send the reply back with [`IncomingMessage::reply`]; notification messages have no reply slot.
///
/// See [`MpscChannel`] for a complete usage example.
#[derive(Debug)]
pub struct IncomingMessage {
    text: String,
    reply_tx: Option<tokio::sync::oneshot::Sender<String>>,
}

impl IncomingMessage {
    /// The message text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Whether this is a question (expects a reply); notifications return `false`.
    ///
    /// Only messages that return `true` may call [`IncomingMessage::reply`].
    pub fn wants_reply(&self) -> bool {
        self.reply_tx.is_some()
    }

    /// Sends back a reply; the message is consumed in the process, so it can only be replied to once.
    ///
    /// # Errors
    ///
    /// - [`ChannelError::NoReply`] — this is a notification message and does not expect a reply;
    /// - [`ChannelError::Closed`] — the asker has left (e.g. `ask` was cancelled or timed out),
    ///   so nobody will receive the reply.
    pub fn reply(self, answer: String) -> Result<(), ChannelError> {
        match self.reply_tx {
            Some(tx) => tx.send(answer).map_err(|_| ChannelError::Closed),
            None => Err(ChannelError::NoReply),
        }
    }
}

mod broadcast;
#[cfg(feature = "cli-channel")]
mod cli;
mod mpsc;
mod watch;

pub use broadcast::{BroadcastChannel, BroadcastReceiver};
#[cfg(feature = "cli-channel")]
pub use cli::CliMessageChannel;
pub use mpsc::MpscChannel;
pub use watch::{WatchChannel, WatchReceiver};
