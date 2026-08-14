//! Observation channels: the Agent publishes process events internally, and the environment / UI side subscribes.
//!
//! Division of labor in the channel family: conversation channels
//! [`MessageChannel`](crate::message_channel::MessageChannel) carry request-response and notifications
//! between humans and Agents (`ask` / `notify`); observation channels carry the process events produced by the
//! Agent runtime (`publish` / `subscribe`) for the environment / UI / observability side to subscribe to,
//! with no reply expected.
//!
//! Two implementations, chosen by the number of subscribers:
//! - [`BroadcastEventChannel`] — multiple subscribers share the event stream; slow subscribers drop the oldest,
//!   and events published before subscribing are never received;
//! - [`MpscEventChannel`] — a single subscriber; strictly ordered and lossless within capacity,
//!   dropping new events when full.
//!
//! # Publishing is always non-blocking
//!
//! The Agent loop must not be held up by slow observers: when nobody is subscribed or the buffer is full,
//! events are **silently dropped** — events are process snapshots, missed is missed; no replay, no
//! backpressure, and publish calls return synchronously.
//!
//! The payload is uniformly `Arc<dyn AgentEvent>`: each Agent defines its own event type (e.g.
//! [`ReActEvent`](crate::agent::ReActEvent)) and plugs in by implementing [`AgentEvent`] with it;
//! consumers downcast via `as_any` to a known type for precise handling, and fall back to
//! [`name`](AgentEvent::name) to display unknown types.
//!
//! # Choosing an implementation
//!
//! - multiple subscribers (UI + logs + metrics from one source) → [`BroadcastEventChannel`];
//! - a single subscriber (one consumer with exclusive access) → [`MpscEventChannel`].
//!
//! Both can drop events (broadcast drops the oldest, a single queue drops new ones when full) — that is a
//! deliberate trade-off of observation semantics; [`EventChannelStats`] reports drop / lag counters for
//! diagnosis, but interactions that need reliable delivery (approval, confirmation) go through
//! [`MessageChannel`](crate::message_channel::MessageChannel), and side-effect audit belongs to the harness
//! layer.

mod broadcast;
mod mpsc;

pub use broadcast::BroadcastEventChannel;
pub use mpsc::MpscEventChannel;

use crate::agent::AgentEvent;
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Diagnostic counters for best-effort event channels.
///
/// These counters are observability signals only; they do not provide
/// backpressure, replay, or reliable audit semantics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventChannelStats {
    /// Publish attempts.
    pub published: u64,
    /// Events accepted by the channel implementation.
    pub delivered: u64,
    /// Events dropped because no subscriber was available.
    pub dropped_no_subscribers: u64,
    /// Events dropped because a bounded queue was full.
    pub dropped_full: u64,
    /// Events skipped by lagging broadcast receivers.
    pub lagged: u64,
    /// Current subscriber count when known.
    pub subscribers: usize,
}

/// The unified receive-end interface: the environment side takes subscribed events out one by one.
///
/// [`recv`](EventReceiver::recv) returning `None` means the channel is closed (all senders dropped) and the
/// event stream has ended; further calls keep returning `None`. Each implementation's drop semantics
/// (broadcast drops the oldest / a single queue drops new ones when full) are handled transparently behind
/// the interface: `recv` never errors, it only yields the events that remain. Dropping the receive end stops
/// consumption — other subscribers of a broadcast channel are unaffected; unconsumed events of a single-queue
/// channel are dropped along with the receive end.
///
/// The interface expresses asynchrony explicitly with `BoxFuture`: `async fn` cannot be a trait-object
/// method, and this type circulates exactly as a trait object (see [`EventChannel::subscribe`]).
///
/// # Example
///
/// After all senders are dropped, `recv` returns `None` and the event stream ends:
///
/// ```rust
/// # extern crate molo_agent as molo;
/// # #[tokio::main]
/// # async fn main() {
/// use molo::event_channel::{EventChannel, MpscEventChannel};
///
/// let mut rx = {
///     let channel = MpscEventChannel::new(8);
///     channel.subscribe()
/// };
///
/// // The channel goes out of scope here (its only sender disappears) → the stream ends.
/// assert!(rx.recv().await.is_none());
/// # }
/// ```
pub trait EventReceiver: Send {
    /// Takes the next event; `None` means the channel is closed (all senders dropped) and the stream has ended.
    fn recv(&mut self) -> BoxFuture<'_, Option<Arc<dyn AgentEvent>>>;
}

/// The observation channel abstraction: the Agent publishes events internally, and the environment side subscribes.
///
/// Both implementations behave the same on the publish side (non-blocking, silent drop); the difference is on
/// the subscribe side, see the docs of [`BroadcastEventChannel`] and [`MpscEventChannel`] respectively. For
/// the Agent-side wiring, see [`ReActAgent::with_event_channel`](crate::agent::ReActAgent::with_event_channel):
/// one mount, events of many runs go to the same channel, and each run ends with a `RunEnded` event.
///
/// # Example
///
/// Publish and subscribe through trait objects (the agent side holds `Arc<dyn EventChannel>`, the
/// environment side holds `Box<dyn EventReceiver>`):
///
/// ```rust
/// # extern crate molo_agent as molo;
/// # #[tokio::main]
/// # async fn main() {
/// use molo::agent::ReActEvent;
/// use molo::event_channel::{BroadcastEventChannel, EventChannel};
/// use std::sync::Arc;
///
/// // Environment side: create the channel, subscribe, then hand the channel to the Agent side.
/// let channel: Arc<dyn EventChannel> = Arc::new(BroadcastEventChannel::new(64));
/// let mut rx = channel.subscribe();
///
/// // Publish from the Agent side; never blocks, silently drops when full or without subscribers.
/// channel.publish(Arc::new(ReActEvent::RunStarted {
///     run_id: "r1".into(),
///     input: "hello".into(),
/// }));
///
/// // The environment side consumes one by one; typed handling via AgentEvent::as_any.
/// let Some(event) = rx.recv().await else {
///     panic!("expected one published event");
/// };
/// assert_eq!(event.name(), "run.started");
/// # }
/// ```
pub trait EventChannel: Send + Sync {
    /// Publishes an event (**non-blocking**; silently dropped when there are no subscribers or it is full).
    fn publish(&self, event: Arc<dyn AgentEvent>);

    /// Subscribes a receive end.
    fn subscribe(&self) -> Box<dyn EventReceiver>;

    /// Returns best-effort diagnostic counters.
    fn stats(&self) -> EventChannelStats {
        EventChannelStats::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentEvent, ReActEvent};

    /// Test event: uses `ReActEvent::RunStarted` as the payload.
    fn ev(n: &str) -> Arc<dyn AgentEvent> {
        Arc::new(ReActEvent::RunStarted {
            run_id: "test".into(),
            input: n.into(),
        })
    }

    /// Used through a trait object: the agent side holds `Arc<dyn EventChannel>` to publish,
    /// and the environment side subscribes.
    #[tokio::test]
    async fn trait_object_usage() {
        let channel: Arc<dyn EventChannel> = Arc::new(BroadcastEventChannel::new(8));
        let mut rx = channel.subscribe();
        channel.publish(ev("1"));
        match rx
            .recv()
            .await
            .unwrap()
            .as_any()
            .downcast_ref::<ReActEvent>()
        {
            Some(ReActEvent::RunStarted { input, .. }) => assert_eq!(input.as_text(), Some("1")),
            _ => panic!("test event must be RunStarted"),
        }
    }
}
