//! The single-queue implementation: one subscriber, strictly ordered and lossless within capacity,
//! dropping new events when full.
//!
//! Fits exactly one consumer consuming all events in order (e.g. a single UI panel). The publish side is
//! equally non-blocking: new events are silently dropped when the queue is full, and enqueued events
//! are unaffected.
//!
//! Two key differences from the broadcast implementation:
//! - `subscribe` can only be called once; subscribing again is a programming error (signalled with a panic);
//! - the receive end starts buffering as soon as the channel is created, so within capacity,
//!   **events published before subscribing are still received** — the broadcast implementation misses
//!   events published before subscribing.
//!
//! For a comparison with [`BroadcastEventChannel`](crate::event_channel::BroadcastEventChannel),
//! see the [event channel module](crate::event_channel).

use super::{AgentEvent, EventChannel, EventReceiver};
use futures::future::BoxFuture;
use std::sync::Arc;

/// A single-queue event channel: one subscriber, strictly ordered and lossless within capacity,
/// dropping new events when full.
///
/// # Example
///
/// ```rust
/// # #[tokio::main]
/// # async fn main() {
/// use molo::agent::ReActEvent;
/// use molo::event_channel::{EventChannel, MpscEventChannel};
/// use std::sync::Arc;
///
/// let channel = MpscEventChannel::new(16);
///
/// // Within capacity, events published before subscribing are still received (the receive end buffers
/// // from channel creation on).
/// channel.publish(Arc::new(ReActEvent::RunStarted {
///     run_id: "r1".into(),
///     input: "hello".into(),
/// }));
///
/// let mut rx = channel.subscribe();
/// let event = rx.recv().await.unwrap();
/// assert_eq!(event.name(), "run.started");
/// # }
/// ```
///
/// # Panics
///
/// Calling [`subscribe`](EventChannel::subscribe) a second time on the same channel: this is a
/// single-consumer channel, so subscribing twice is a programming error — the receive end is handed out
/// only once, and later subscribe requests cannot get it; this is signalled with a panic rather than
/// silently misbehaving.
///
/// # Choosing an implementation
///
/// When multiple subscribers are needed, or events published before subscribing don't matter, use
/// [`BroadcastEventChannel`](crate::event_channel::BroadcastEventChannel); this channel guarantees events
/// stay ordered and lossless within capacity and are received even when published before subscribing,
/// which suits a single consumer.
#[derive(Debug)]
pub struct MpscEventChannel {
    tx: tokio::sync::mpsc::Sender<Arc<dyn AgentEvent>>,
    rx: std::sync::Mutex<Option<tokio::sync::mpsc::Receiver<Arc<dyn AgentEvent>>>>,
}

impl Default for MpscEventChannel {
    fn default() -> Self {
        Self::new(256)
    }
}

impl MpscEventChannel {
    /// The queue capacity (number of events buffered).
    ///
    /// New events published when full are silently dropped; enqueued events are unaffected.
    ///
    /// # Panics
    ///
    /// Panics when `capacity == 0` (tokio mpsc does not allow zero-capacity channels);
    /// use 1 for unbuffered semantics.
    pub fn new(capacity: usize) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel(capacity);
        Self {
            tx,
            rx: std::sync::Mutex::new(Some(rx)),
        }
    }
}

impl EventChannel for MpscEventChannel {
    fn publish(&self, event: Arc<dyn AgentEvent>) {
        // Full / channel closed → silently drop: observation semantics; the publish side never blocks
        // and never errors.
        let _ = self.tx.try_send(event);
    }

    /// Subscribes to the event stream (single consumer).
    ///
    /// # Panics
    ///
    /// The receive end is handed out only once: a second `subscribe` call (double subscription) panics;
    /// concurrent `subscribe` calls are serialized by the mutex. For multiple consumers, use
    /// [`BroadcastEventChannel`](crate::event_channel::BroadcastEventChannel) instead.
    fn subscribe(&self) -> Box<dyn EventReceiver> {
        // The receive end is handed out only once: `Option::take` leaves nothing for a repeated subscribe,
        // which triggers the panic below; the mutex serializes concurrent subscribe calls (no contention
        // on the happy path).
        let mut guard = self
            .rx
            .lock()
            .expect("MpscEventChannel is single-consumer; subscribe may only be called once");
        let rx = guard
            .take()
            .expect("MpscEventChannel is single-consumer; subscribe may only be called once");
        Box::new(MpscEventReceiver { rx })
    }
}

/// The single-queue receive end: wraps a tokio `mpsc::Receiver` and consumes the events that remain, in order.
struct MpscEventReceiver {
    rx: tokio::sync::mpsc::Receiver<Arc<dyn AgentEvent>>,
}

impl EventReceiver for MpscEventReceiver {
    fn recv(&mut self) -> BoxFuture<'_, Option<Arc<dyn AgentEvent>>> {
        Box::pin(async move { self.rx.recv().await })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentEvent, ReActEvent};
    use futures::FutureExt;

    /// Test event: uses `ReActEvent::RunStarted` as the payload.
    fn ev(n: &str) -> Arc<dyn AgentEvent> {
        Arc::new(ReActEvent::RunStarted {
            run_id: "test".into(),
            input: n.into(),
        })
    }

    fn input_of(ev: &dyn AgentEvent) -> &str {
        match ev.as_any().downcast_ref::<ReActEvent>() {
            Some(ReActEvent::RunStarted { input, .. }) => {
                input.as_text().expect("test event uses text input")
            }
            _ => panic!("test event must be RunStarted"),
        }
    }

    /// Within capacity, events published before subscribing are received (the receive end buffers
    /// from channel creation on).
    #[tokio::test]
    async fn publish_subscribe_buffered() {
        let ch = MpscEventChannel::new(16);
        ch.publish(ev("1"));
        ch.publish(ev("2"));
        let mut rx = ch.subscribe();
        assert_eq!(input_of(&*rx.recv().await.unwrap()), "1");
        assert_eq!(input_of(&*rx.recv().await.unwrap()), "2");
    }

    /// Single consumer: subscribing twice is a programming error and panics.
    #[tokio::test]
    #[should_panic(expected = "MpscEventChannel is single-consumer")]
    async fn subscribe_twice_panics() {
        let ch = MpscEventChannel::new(8);
        let _rx = ch.subscribe();
        let _rx2 = ch.subscribe();
    }

    /// New events are dropped when the queue is full (those within capacity are kept).
    #[tokio::test]
    async fn full_drops_new() {
        let ch = MpscEventChannel::new(2);
        ch.publish(ev("1"));
        ch.publish(ev("2"));
        ch.publish(ev("3")); // full, dropped
        let mut rx = ch.subscribe();
        assert_eq!(input_of(&*rx.recv().await.unwrap()), "1");
        assert_eq!(input_of(&*rx.recv().await.unwrap()), "2");
        assert!(rx.recv().now_or_never().is_none()); // the 3rd was dropped; queue empty
    }

    /// The stream ends after all senders are dropped (buffered events are still receivable).
    #[tokio::test]
    async fn closed_ends_stream() {
        let mut rx = {
            let ch = MpscEventChannel::new(16);
            ch.publish(ev("1"));
            ch.subscribe()
        };
        assert_eq!(input_of(&*rx.recv().await.unwrap()), "1");
        assert!(rx.recv().await.is_none());
    }
}
