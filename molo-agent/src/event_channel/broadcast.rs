//! The broadcast implementation: multiple subscribers share the event stream; slow subscribers drop the oldest,
//! and events published before subscribing are never received.
//!
//! Observation semantics — missed is missed: publishers are never slowed down, and each subscriber catches up
//! on its own. When a subscriber consumes slower than the publish rate, the oldest events in the shared ring
//! buffer are pushed out by new ones; that subscriber skips the pushed-out events (`recv` does not error)
//! and keeps receiving the newest.
//!
//! Unlike the single-queue implementation (drops new when full), here the **oldest** are dropped: multiple
//! subscribers share one buffer, so a first-in-first-out queue per subscriber is impossible.
//!
//! For a comparison with [`MpscEventChannel`](crate::event_channel::MpscEventChannel),
//! see the [event channel module](crate::event_channel).

use super::{AgentEvent, EventChannel, EventChannelStats, EventReceiver};
use futures::future::BoxFuture;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A broadcast event channel: multiple subscribers, each consuming independently; slow subscribers drop the oldest.
///
/// # Example
///
/// ```rust
/// # extern crate molo_agent as molo;
/// # #[tokio::main]
/// # async fn main() {
/// use molo::agent::ReActEvent;
/// use molo::event_channel::{BroadcastEventChannel, EventChannel};
/// use std::sync::Arc;
///
/// let channel = BroadcastEventChannel::new(16);
/// let mut rx1 = channel.subscribe();
/// let mut rx2 = channel.subscribe();
///
/// channel.publish(Arc::new(ReActEvent::RunStarted {
///     run_id: "r1".into(),
///     input: "hello".into(),
/// }));
///
/// // Each subscriber receives its own copy of the same event.
/// let Some(event1) = rx1.recv().await else {
///     panic!("expected event for first subscriber");
/// };
/// let Some(event2) = rx2.recv().await else {
///     panic!("expected event for second subscriber");
/// };
/// assert_eq!(event1.name(), "run.started");
/// assert_eq!(event2.name(), "run.started");
/// # }
/// ```
///
/// # Choosing an implementation
///
/// Pick this channel when multiple subscribers consume from one source (UI + logs + metrics); for a single
/// consumer needing strict ordering and no loss within capacity (including events published before
/// subscribing), use [`MpscEventChannel`](crate::event_channel::MpscEventChannel) instead.
#[derive(Debug, Clone)]
pub struct BroadcastEventChannel {
    tx: tokio::sync::broadcast::Sender<Arc<dyn AgentEvent>>,
    stats: Arc<BroadcastStats>,
}

#[derive(Debug, Default)]
struct BroadcastStats {
    published: AtomicU64,
    delivered: AtomicU64,
    dropped_no_subscribers: AtomicU64,
    lagged: AtomicU64,
}

impl Default for BroadcastEventChannel {
    fn default() -> Self {
        Self::new(256)
    }
}

impl BroadcastEventChannel {
    /// The ring-buffer capacity (number of events buffered).
    ///
    /// When full, new events push out the oldest; subscribers that cannot keep up skip the pushed-out
    /// events and still get the newest.
    ///
    /// # Panics
    ///
    /// Panics when `capacity == 0` (tokio broadcast does not allow zero-capacity channels);
    /// use 1 for unbuffered semantics.
    pub fn new(capacity: usize) -> Self {
        let (tx, _rx) = tokio::sync::broadcast::channel(capacity);
        Self {
            tx,
            stats: Arc::new(BroadcastStats::default()),
        }
    }
}

impl EventChannel for BroadcastEventChannel {
    fn publish(&self, event: Arc<dyn AgentEvent>) {
        self.stats.published.fetch_add(1, Ordering::Relaxed);
        // send returns Err when there are no subscribers — observation semantics; silently drop.
        match self.tx.send(event) {
            Ok(receivers) => {
                self.stats
                    .delivered
                    .fetch_add(receivers as u64, Ordering::Relaxed);
            }
            Err(_) => {
                self.stats
                    .dropped_no_subscribers
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn subscribe(&self) -> Box<dyn EventReceiver> {
        Box::new(BroadcastEventReceiver {
            rx: self.tx.subscribe(),
            stats: self.stats.clone(),
        })
    }

    fn stats(&self) -> EventChannelStats {
        EventChannelStats {
            published: self.stats.published.load(Ordering::Relaxed),
            delivered: self.stats.delivered.load(Ordering::Relaxed),
            dropped_no_subscribers: self.stats.dropped_no_subscribers.load(Ordering::Relaxed),
            dropped_full: 0,
            lagged: self.stats.lagged.load(Ordering::Relaxed),
            subscribers: self.tx.receiver_count(),
        }
    }
}

/// The broadcast receive end: wraps a tokio `broadcast::Receiver`; skips when the buffer pushed it out
/// (Lagged), and the stream ends after all senders are dropped.
struct BroadcastEventReceiver {
    rx: tokio::sync::broadcast::Receiver<Arc<dyn AgentEvent>>,
    stats: Arc<BroadcastStats>,
}

impl EventReceiver for BroadcastEventReceiver {
    fn recv(&mut self) -> BoxFuture<'_, Option<Arc<dyn AgentEvent>>> {
        Box::pin(async move {
            loop {
                match self.rx.recv().await {
                    Ok(event) => return Some(event),
                    // A slow subscriber was skipped (Lagged): missed is missed; keep waiting for the next one.
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        self.stats.lagged.fetch_add(n, Ordering::Relaxed);
                        continue;
                    }
                    // All senders have been dropped: the stream ends.
                    Err(_) => return None,
                }
            }
        })
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

    #[tokio::test]
    async fn publish_subscribe() {
        let ch = BroadcastEventChannel::new(16);
        let mut rx = ch.subscribe();
        ch.publish(ev("1"));
        ch.publish(ev("2"));
        ch.publish(ev("3"));
        assert_eq!(input_of(&*rx.recv().await.unwrap()), "1");
        assert_eq!(input_of(&*rx.recv().await.unwrap()), "2");
        assert_eq!(input_of(&*rx.recv().await.unwrap()), "3");
    }

    #[tokio::test]
    async fn multiple_subscribers_each_receive_all() {
        let ch = BroadcastEventChannel::new(16);
        let mut rx1 = ch.subscribe();
        let mut rx2 = ch.subscribe();
        ch.publish(ev("1"));
        ch.publish(ev("2"));
        assert_eq!(input_of(&*rx1.recv().await.unwrap()), "1");
        assert_eq!(input_of(&*rx2.recv().await.unwrap()), "1");
        assert_eq!(input_of(&*rx1.recv().await.unwrap()), "2");
        assert_eq!(input_of(&*rx2.recv().await.unwrap()), "2");
    }

    /// A slow subscriber: when it does not consume, the buffer is pushed out by new events, it gets
    /// Lagged and skips, and eventually receives the newest event.
    #[tokio::test]
    async fn lag_drops_oldest() {
        let ch = BroadcastEventChannel::new(2);
        let mut rx = ch.subscribe();
        // Capacity 2, publish 3 without consuming: the earliest is pushed out.
        ch.publish(ev("1"));
        ch.publish(ev("2"));
        ch.publish(ev("3"));
        // Lagged skip; what follows is still-retained events, and the newest always arrives.
        let first = input_of(&*rx.recv().await.unwrap()).to_string();
        let second = input_of(&*rx.recv().await.unwrap()).to_string();
        assert_eq!(second, "3"); // the newest always arrives
        assert!(first == "2" || first == "3"); // only the oldest can have been skipped
        assert!(ch.stats().lagged >= 1);
    }

    /// Events published before subscribing are missed (the cursor only aligns at subscribe time).
    #[tokio::test]
    async fn events_before_subscribe_missed() {
        let ch = BroadcastEventChannel::new(16);
        ch.publish(ev("1"));
        let stats = ch.stats();
        assert_eq!(stats.published, 1);
        assert_eq!(stats.dropped_no_subscribers, 1);
        let mut rx = ch.subscribe();
        assert!(rx.recv().now_or_never().is_none());
        assert_eq!(ch.stats().subscribers, 1);
    }

    /// The stream ends after all senders are dropped (buffered events are still receivable).
    #[tokio::test]
    async fn closed_ends_stream() {
        let mut rx = {
            let ch = BroadcastEventChannel::new(16);
            // Subscribe first, then publish (broadcast's send drops immediately when there are no subscribers).
            let rx = ch.subscribe();
            ch.publish(ev("1"));
            rx
        };
        assert_eq!(input_of(&*rx.recv().await.unwrap()), "1");
        assert!(rx.recv().await.is_none());
    }
}
