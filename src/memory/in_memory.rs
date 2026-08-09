//! In-memory implementation: stores all messages verbatim.
//!
//! [`InMemoryMemory`] is the simplest [`Memory`] implementation: `record`
//! appends, `context` returns a full clone, with no trimming or summarization —
//! suitable for small conversations that need no budget control.

use super::{Memory, MemoryError};
use crate::message::Message;

/// In-memory implementation: stores all messages verbatim.
///
/// The simplest [`Memory`] implementation: all messages are kept in memory with
/// no trimming, no summarization, and no persistence; `context()` returns a
/// full clone. `Default` constructs an empty session.
///
/// # Comparison
///
/// Compared to [`WindowMemory`](crate::memory::WindowMemory): the latter trims
/// to a recent window on retrieval, suiting long conversations (the model's
/// context window is finite); this implementation has no budget control —
/// messages only grow — suiting small conversations that need the full history.
///
/// # Example
///
/// ```rust
/// # #[tokio::main]
/// # async fn main() -> Result<(), molo::memory::MemoryError> {
/// use molo::memory::{InMemoryMemory, Memory};
///
/// let mut memory = InMemoryMemory::default();
/// memory.record(molo::Message::system("setup")).await?;
/// memory.record(molo::Message::user("hello")).await?;
/// memory.record(molo::Message::assistant("hello!")).await?;
///
/// let context = memory.context().await?;
/// assert_eq!(context.len(), 3);
/// assert_eq!(context[0], molo::Message::System("setup".into()));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct InMemoryMemory {
    messages: Vec<Message>,
}

#[async_trait::async_trait]
impl Memory for InMemoryMemory {
    async fn record(&mut self, message: Message) -> Result<(), MemoryError> {
        self.messages.push(message);
        Ok(())
    }

    async fn context(&self) -> Result<Vec<Message>, MemoryError> {
        Ok(self.messages.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn record_preserves_order() {
        let mut memory = InMemoryMemory::default();
        memory.record(Message::system("setup")).await.unwrap();
        memory.record(Message::user("hello")).await.unwrap();
        memory.record(Message::assistant("hello!")).await.unwrap();

        let context = memory.context().await.unwrap();
        assert_eq!(context.len(), 3);
        assert_eq!(context[0], Message::System("setup".into()));
        assert_eq!(context[1], Message::user("hello"));
        assert_eq!(context[2], Message::assistant("hello!"));
    }

    #[tokio::test]
    async fn context_is_a_copy() {
        let mut memory = InMemoryMemory::default();
        memory.record(Message::user("a")).await.unwrap();

        // Mutating the returned context does not affect Memory's internals.
        let mut context = memory.context().await.unwrap();
        context.push(Message::assistant("b"));

        assert_eq!(memory.context().await.unwrap().len(), 1);
    }
}
