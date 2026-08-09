//! A one-to-one Agent dialogue channel: an implementation based on a tokio mpsc queue.
//!
//! Each side of the dialogue holds one end of [`MpscChannel::pair`]; replies are bound to requests one-to-one,
//! so concurrent questioning in both directions never mismatches.

use super::{ChannelError, Envelope, IncomingMessage, MessageChannel};
use tokio::sync::{Mutex, mpsc, oneshot};

/// A one-to-one dialogue channel between two Agents (or a program and a human).
///
/// [`MpscChannel::pair`] creates two interconnected ends; each dialogue side holds one, and their behavior is fully
/// symmetric: either end can ask, notify, and receive messages.
///
/// - [`MessageChannel::ask`] — send a question and wait for a reply. Each question carries its own reply channel,
///   so replies go straight back to the asker, independent of queue order — the two sides can ask each other
///   concurrently without mismatches;
/// - [`MessageChannel::notify`] — send a one-way notification (no reply expected);
/// - the receive side is driven by the caller: [`MpscChannel::recv`] takes one message, and questions are
///   answered with [`IncomingMessage::reply`].
///
/// Messages are delivered through a bounded queue (capacity 64): when the peer consumes slowly,
/// `ask` / `notify` wait, and a full queue applies backpressure. When the peer channel has been dropped,
/// `ask` / `notify` / `recv` all return [`ChannelError::Closed`].
///
/// # Example
///
/// ```rust
/// use molo::{MessageChannel, MpscChannel};
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), molo::ChannelError> {
/// let (agent_a, agent_b) = MpscChannel::pair();
///
/// // A asks; B replies after receiving the question. Replies are bound to requests one-to-one, independent of queue order.
/// let question = agent_a.ask("What is today's task?");
/// let answer_side = async {
///     let incoming = agent_b.recv().await?;
///     incoming.reply("write documentation".into())?;
///     Ok::<(), molo::ChannelError>(())
/// };
/// let (answer, _) = tokio::join!(question, answer_side);
/// assert_eq!(answer?, "write documentation");
/// # Ok(())
/// # }
/// ```
///
/// # Cancellation semantics
///
/// - `ask` cancelled while sending (e.g. a timeout): the question never reaches the peer (the message was not yet
///   enqueued);
/// - `ask` cancelled after sending, while waiting for the reply: the reply channel is dropped along with it, so the
///   peer's [`IncomingMessage::reply`] yields [`ChannelError::Closed`] — the peer can tell from this that the
///   asker has left;
/// - `recv` cancelled while waiting: the message stays in the queue, and the next `recv` still gets it;
/// - concurrent `recv` calls on the same channel are serialized by an internal lock, so there is only one waiter
///   at a time.
///
/// For one-to-many scenarios, combine several `pair`s, or switch to
/// [`BroadcastChannel`](crate::BroadcastChannel) for one-way broadcasting.
///
/// # No built-in timeout
///
/// `ask` has no built-in timeout: if the peer does not `reply` after receiving the question, `ask` waits forever.
/// For a timeout, wrap it with `tokio::time::timeout` — see the cancellation semantics above (`ask` cancelled
/// after sending, while waiting for the reply, makes the peer's `reply` return [`ChannelError::Closed`]).
#[derive(Debug)]
pub struct MpscChannel {
    tx: mpsc::Sender<Envelope>,
    rx: Mutex<mpsc::Receiver<Envelope>>,
}

/// Default queue capacity for [`pair`](MpscChannel::pair).
const DEFAULT_CAPACITY: usize = 64;

impl MpscChannel {
    /// Creates a pair of interconnected channels (queue capacity 64); each dialogue side holds one end.
    ///
    /// The two ends behave symmetrically: either end can `ask` / `notify` / `recv`.
    pub fn pair() -> (MpscChannel, MpscChannel) {
        Self::pair_with_capacity(DEFAULT_CAPACITY)
    }

    /// Creates a pair of interconnected channels with a given queue capacity (the backpressure point:
    /// `ask` / `notify` wait for the peer to consume when full); everything else behaves like [`pair`](MpscChannel::pair).
    pub fn pair_with_capacity(capacity: usize) -> (MpscChannel, MpscChannel) {
        let (tx_a, rx_a) = mpsc::channel(capacity);
        let (tx_b, rx_b) = mpsc::channel(capacity);
        (
            MpscChannel {
                tx: tx_a,
                rx: Mutex::new(rx_b),
            },
            MpscChannel {
                tx: tx_b,
                rx: Mutex::new(rx_a),
            },
        )
    }

    /// Takes one message sent by the peer; waits asynchronously when the queue is empty.
    ///
    /// # Errors
    ///
    /// Returns [`ChannelError::Closed`] when the peer channel has been dropped (no senders remain).
    ///
    /// # Cancellation semantics
    ///
    /// Being cancelled while waiting does not lose messages: the message stays in the queue, and the next `recv`
    /// still gets it.
    pub async fn recv(&self) -> Result<IncomingMessage, ChannelError> {
        let mut rx = self.rx.lock().await;
        let envelope = rx.recv().await.ok_or(ChannelError::Closed)?;
        Ok(IncomingMessage {
            text: envelope.message,
            reply_tx: envelope.reply,
        })
    }
}

#[async_trait::async_trait]
impl MessageChannel for MpscChannel {
    async fn ask(&self, message: &str) -> Result<String, ChannelError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(Envelope {
                message: message.to_string(),
                reply: Some(reply_tx),
            })
            .await
            .map_err(|_| ChannelError::Closed)?;
        reply_rx.await.map_err(|_| ChannelError::Closed)
    }

    async fn notify(&self, message: &str) -> Result<(), ChannelError> {
        self.tx
            .send(Envelope {
                message: message.to_string(),
                reply: None,
            })
            .await
            .map_err(|_| ChannelError::Closed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn ask_reply_bound() {
        let (a, b) = MpscChannel::pair();
        let ask = a.ask("q");
        let b_side = async {
            let incoming = b.recv().await.unwrap();
            assert_eq!(incoming.text(), "q");
            assert!(incoming.wants_reply());
            incoming.reply("ans".into())
        };
        // ask and recv run concurrently: ask's future only sends when polled.
        let (answer, _) = tokio::join!(ask, b_side);
        assert_eq!(answer.unwrap(), "ans");
    }

    #[tokio::test]
    async fn concurrent_bidirectional_ask_replies_stay_bound() {
        let (a, b) = MpscChannel::pair();
        let a_ask = a.ask("q_from_a");
        let b_ask = b.ask("q_from_b");
        let a_side = async {
            let incoming = a.recv().await.unwrap();
            assert_eq!(incoming.text(), "q_from_b");
            incoming.reply("ans_to_b".into())
        };
        let b_side = async {
            let incoming = b.recv().await.unwrap();
            assert_eq!(incoming.text(), "q_from_a");
            incoming.reply("ans_to_a".into())
        };
        // Concurrent bidirectional ask: the oneshot binding keeps replies with their own asks, independent of queue order.
        let (r_a, r_b, _a_side, _b_side) = tokio::join!(a_ask, b_ask, a_side, b_side);
        assert_eq!(r_a.unwrap(), "ans_to_a");
        assert_eq!(r_b.unwrap(), "ans_to_b");
    }

    #[tokio::test]
    async fn notify_has_no_reply() {
        let (a, b) = MpscChannel::pair();
        let notify = a.notify("n");
        let b_side = async {
            let incoming = b.recv().await.unwrap();
            assert_eq!(incoming.text(), "n");
            assert!(!incoming.wants_reply());
            incoming.reply("x".into())
        };
        let (n_res, b_res) = tokio::join!(notify, b_side);
        n_res.unwrap();
        assert!(matches!(b_res.unwrap_err(), ChannelError::NoReply));
    }

    #[tokio::test]
    async fn ask_when_peer_dropped_returns_closed() {
        let (a, _b) = MpscChannel::pair();
        drop(_b); // peer is closed
        assert!(matches!(a.ask("q").await, Err(ChannelError::Closed)));
    }
}
