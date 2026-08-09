//! A latest-value change notification channel: an implementation based on tokio watch.

use super::{ChannelError, IncomingMessage, MessageChannel};
use tokio::sync::{Mutex, watch};

/// Observes changes of the latest state value: holds and updates one value, and all observers
/// ([`WatchChannel::subscribe`]) are notified when it changes.
///
/// The semantics are "latest value", not a message queue:
///
/// - observers are only guaranteed to eventually see the latest value, not every intermediate change — values
///   overwritten between two `notify` calls are never re-sent;
/// - unbounded: a new value simply replaces the old one; there is no queue capacity, and no backpressure;
/// - observers are independent of each other; how fast one consumes does not affect the others.
///
/// Driven by the [`MessageChannel`] interface:
///
/// - [`MessageChannel::notify`] — update the value to the latest message; all observers get notified;
/// - [`MessageChannel::ask`] — not supported (watch has no "reply" concept), returns
///   [`ChannelError::NotSupported`];
/// - receive side: [`WatchReceiver::recv`] waits for a change and takes the latest value.
///
/// Implements [`Clone`] — every clone is a sender of the same channel, and any clone's `notify`
/// notifies all observers.
///
/// # Example
///
/// ```rust
/// use molo::{MessageChannel, WatchChannel};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), molo::ChannelError> {
/// let status = WatchChannel::new();
/// let sub_a = status.subscribe();
/// let sub_b = status.subscribe();
///
/// status.notify("ready").await?;
/// assert_eq!(sub_a.recv().await?.text(), "ready");
/// assert_eq!(sub_b.recv().await?.text(), "ready");
/// # Ok(())
/// # }
/// ```
///
/// # Notes
///
/// - the initial value is an empty string: before the first `notify`, `recv` keeps waiting;
/// - when there are no observers (never `subscribe`d, or all observers have been dropped), `notify`
///   returns [`ChannelError::NoReceiver`](crate::ChannelError::NoReceiver) — unlike
///   [`BroadcastChannel`](crate::BroadcastChannel), which succeeds silently even with nobody listening,
///   so `subscribe` first, then `notify`; the channel itself is not closed, and `notify` works again
///   after re-subscribing.
#[derive(Debug, Clone)]
pub struct WatchChannel {
    tx: watch::Sender<String>,
}

/// The receive end of a watch channel: observes value changes.
///
/// Does not implement [`Clone`] — a single receive end can only be driven by one waiter at a time
/// (serialized by an internal lock). For multiple observers, call [`subscribe`](WatchChannel::subscribe)
/// on the [`WatchChannel`] separately for each.
#[derive(Debug)]
pub struct WatchReceiver {
    rx: Mutex<watch::Receiver<String>>,
}

impl WatchChannel {
    /// Creates a watch channel whose initial value is an empty string.
    ///
    /// Before the first `notify`, a subscriber's `recv` keeps waiting.
    pub fn new() -> Self {
        let (tx, _rx) = watch::channel(String::new());
        Self { tx }
    }

    /// Subscribes to value changes; each observer gets an independent receive end, unaffected by the others.
    pub fn subscribe(&self) -> WatchReceiver {
        WatchReceiver {
            rx: Mutex::new(self.tx.subscribe()),
        }
    }
}

impl Default for WatchChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl WatchReceiver {
    /// Waits for a change and takes the latest value.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::Closed`] when the sender ([`WatchChannel`]) has been dropped.
    ///
    /// # Cancellation semantics
    ///
    /// Being cancelled while waiting has no side effects: watch is not a queue and has no "take one item" concept,
    /// the next `recv` starts from the latest value and keeps waiting for the next change.
    pub async fn recv(&self) -> Result<IncomingMessage, ChannelError> {
        let mut rx = self.rx.lock().await;
        rx.changed().await.map_err(|_| ChannelError::Closed)?;
        let message = rx.borrow().clone();
        Ok(IncomingMessage {
            text: message,
            reply_tx: None,
        })
    }
}

#[async_trait::async_trait]
impl MessageChannel for WatchChannel {
    async fn ask(&self, _message: &str) -> Result<String, ChannelError> {
        Err(ChannelError::NotSupported(
            "watch channel does not support ask".into(),
        ))
    }

    async fn notify(&self, message: &str) -> Result<(), ChannelError> {
        // watch's send only fails when there are no receivers: the channel is not closed and re-subscribing
        // restores it, so use NoReceiver rather than the semantically misleading Closed.
        self.tx
            .send(message.to_string())
            .map_err(|_| ChannelError::NoReceiver)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn notify_then_recv_latest() {
        let channel = WatchChannel::new();
        let receiver = channel.subscribe();
        // Alternate notify / recv: each recv waits for one change and takes the latest value.
        channel.notify("v1").await.unwrap();
        assert_eq!(receiver.recv().await.unwrap().text(), "v1");
        channel.notify("v2").await.unwrap();
        assert_eq!(receiver.recv().await.unwrap().text(), "v2");
    }

    #[tokio::test]
    async fn notify_before_recv_wakes_all_observers() {
        let channel = WatchChannel::new();
        let sub_a = channel.subscribe();
        let sub_b = channel.subscribe();
        channel.notify("start").await.unwrap();
        let (a, b) = tokio::join!(sub_a.recv(), sub_b.recv());
        assert_eq!(a.unwrap().text(), "start");
        assert_eq!(b.unwrap().text(), "start");
    }

    #[tokio::test]
    async fn ask_not_supported() {
        let channel = WatchChannel::new();
        assert!(matches!(
            channel.ask("q").await,
            Err(ChannelError::NotSupported(_))
        ));
    }
}
