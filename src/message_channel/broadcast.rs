//! A one-to-many broadcast notification channel: an implementation based on tokio broadcast.

use super::{ChannelError, IncomingMessage, MessageChannel};
use tokio::sync::{Mutex, broadcast};

/// A one-to-many broadcast channel: messages are broadcast to all subscribers ([`BroadcastChannel::subscribe`]).
///
/// Every `notify` enters each subscriber's own message queue. The queue is bounded (capacity given at
/// construction): when consumption is slower than sending, old messages from the lagged period are dropped,
/// and `recv` goes straight to the newest message it can get — dropped messages only affect slow subscribers,
/// never fast ones.
///
/// - [`MessageChannel::notify`] — broadcast a notification; every subscriber receives it;
/// - [`MessageChannel::ask`] — not supported (broadcast has no "reply" concept), returns
///   [`ChannelError::NotSupported`];
/// - receive side: [`BroadcastReceiver::recv`] takes the next broadcast.
///
/// Implements [`Clone`] — every clone is a sender of the same channel, and any clone's `notify`
/// broadcasts to all subscribers.
///
/// For one-to-one dialogue, use [`MpscChannel`](crate::MpscChannel).
///
/// # Example
///
/// ```rust
/// use molo::{BroadcastChannel, MessageChannel};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), molo::ChannelError> {
/// let announcements = BroadcastChannel::new(16);
/// let sub_a = announcements.subscribe();
/// let sub_b = announcements.subscribe();
///
/// announcements.notify("release shipped").await?;
/// assert_eq!(sub_a.recv().await?.text(), "release shipped");
/// assert_eq!(sub_b.recv().await?.text(), "release shipped");
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct BroadcastChannel {
    tx: broadcast::Sender<String>,
}

impl Default for BroadcastChannel {
    /// Default capacity is 256; when full, the oldest message is dropped in favor of new ones.
    ///
    /// Matches the default capacity of the event channel [`BroadcastEventChannel`](crate::BroadcastEventChannel),
    /// so one consumer loop can handle both kinds of channels together.
    fn default() -> Self {
        Self::new(256)
    }
}

/// The receive end of a broadcast channel: one per subscriber, each consuming independently.
///
/// Concurrent `recv` calls on the same receive end are serialized by an internal lock. When consumption is slower
/// than sending, the queue drops old messages; see the capacity notes on
/// [`BroadcastChannel`](crate::BroadcastChannel).
#[derive(Debug)]
pub struct BroadcastReceiver {
    rx: Mutex<broadcast::Receiver<String>>,
}

impl BroadcastChannel {
    /// Creates a broadcast channel; `capacity` is the message buffer limit per subscriber.
    ///
    /// # Panics
    ///
    /// Panics when `capacity == 0`.
    pub fn new(capacity: usize) -> Self {
        let (tx, _rx) = broadcast::channel(capacity);
        Self { tx }
    }

    /// Subscribes to the broadcast; each subscriber gets an independent receive end.
    pub fn subscribe(&self) -> BroadcastReceiver {
        BroadcastReceiver {
            rx: Mutex::new(self.tx.subscribe()),
        }
    }
}

impl BroadcastReceiver {
    /// Takes the next broadcast; when consumption is slower than sending, old messages are dropped and the
    /// newest available one is taken directly.
    ///
    /// Lagged drops are not surfaced as errors: slow subscribers automatically skip the lost messages and keep receiving.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::Closed`] when all senders have been dropped (the channel is closed).
    ///
    /// # Cancellation semantics
    ///
    /// Cancelled while waiting: the broadcast message stays in the queue, and the next `recv` still gets it
    /// (as long as it has not been pushed out of the queue by newer messages).
    pub async fn recv(&self) -> Result<IncomingMessage, ChannelError> {
        let mut rx = self.rx.lock().await;
        loop {
            match rx.recv().await {
                Ok(message) => {
                    return Ok(IncomingMessage {
                        text: message,
                        reply_tx: None,
                    });
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => return Err(ChannelError::Closed),
            }
        }
    }
}

#[async_trait::async_trait]
impl MessageChannel for BroadcastChannel {
    async fn ask(&self, _message: &str) -> Result<String, ChannelError> {
        Err(ChannelError::NotSupported(
            "broadcast channel does not support ask".into(),
        ))
    }

    async fn notify(&self, message: &str) -> Result<(), ChannelError> {
        // send returns Err when there are no active subscribers — a transient "nobody is listening to the
        // broadcast" state. The sender is still valid, and receivers that subscribe later get subsequent messages
        // as usual: silently drop, matching the event channel's "drop when nobody listens" publish behavior.
        let _ = self.tx.send(message.to_string());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn notify_reaches_all_subscribers() {
        let channel = BroadcastChannel::new(16);
        let sub_a = channel.subscribe();
        let sub_b = channel.subscribe();
        channel.notify("start").await.unwrap();
        let (a, b) = tokio::join!(sub_a.recv(), sub_b.recv());
        let msg_a = a.unwrap();
        let msg_b = b.unwrap();
        assert_eq!(msg_a.text(), "start");
        assert_eq!(msg_b.text(), "start");
        // Broadcast messages have no reply slot.
        assert!(!msg_a.wants_reply());
    }

    #[tokio::test]
    async fn ask_not_supported() {
        let channel = BroadcastChannel::new(16);
        assert!(matches!(
            channel.ask("q").await,
            Err(ChannelError::NotSupported(_))
        ));
    }

    #[tokio::test]
    async fn notify_without_subscribers_succeeds() {
        // Having no subscribers is a recoverable transient "nobody is listening" state, not a broken channel —
        // so this succeeds silently; receivers that subscribe later get subsequent messages as usual.
        let channel = BroadcastChannel::new(16);
        channel.notify("n").await.unwrap();
        let sub = channel.subscribe();
        channel.notify("next").await.unwrap();
        assert_eq!(sub.recv().await.unwrap().text(), "next");
    }
}
