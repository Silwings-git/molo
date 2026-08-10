//! Window Memory example: demonstrates WindowMemory's window trimming
//! (default, projected) and plugging in a custom trim strategy (materialized
//! compression).
//!
//! This example is **self-contained**, needs no API key, just run:
//! `cargo run --example window_memory`
//!
//! It demonstrates three ways to use the component:
//! - Budget trimming (max_tokens / max_rounds): trims by whole rounds when
//!   over budget, keeping at least the most recent round (the system prompt
//!   belongs to the Agent layer; Memory does not manage System);
//! - Lossless projection: raise the budget and the trimmed history is
//!   immediately restored;
//! - Custom strategy injection: a summarizer strategy declares replace:true
//!   (materialized), one compression covers many rounds, and when over budget
//!   again it takes "the last compression result + new messages" as input
//!   (chained).
//!
//! The conversation loop wired to a real model is in
//! `examples/window_memory_agent.rs`.

use std::sync::Arc;

use molo::{
    Budget, Memory, MemoryError, Message, TokenCounter, TrimResult, TrimStrategy, WindowMemory,
};

/// Demo summarizer strategy: keeps the most recent round and replaces earlier
/// messages with one summary (fixed text; a real one is LLM-generated).
///
/// `replace: true` = materialized — the result is written back to storage, the
/// replaced old messages are no longer kept, and later fetches are zero
/// recompute; this is the recommended semantics for heavy operations like LLM
/// summarization.
#[derive(Debug, Default)]
struct Summarizer;

#[async_trait::async_trait]
impl TrimStrategy for Summarizer {
    async fn trim(
        &self,
        messages: &[Message],
        _budget: &Budget,
        _counter: &dyn TokenCounter,
    ) -> Result<TrimResult, MemoryError> {
        // Keep the most recent round; replace earlier messages with one summary message.
        // The summary uses the System role: this avoids consecutive User messages after the replacement (wire alternation constraint).
        let result = match messages.iter().rposition(|m| matches!(m, Message::User(_))) {
            Some(pos) => {
                let mut result = Vec::with_capacity(messages.len() - pos + 1);
                result.push(Message::system("summary of the earlier context"));
                result.extend_from_slice(&messages[pos..]);
                result
            }
            None => messages.to_vec(),
        };
        Ok(TrimResult {
            messages: result,
            replace: true,
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Default window (projected): budget 30 tokens (CJK 1 char = 1 token, ASCII 4 chars = 1 token);
    //    fetch after recording 4 rounds — the earliest rounds are trimmed whole, keeping at least the most recent one.
    let mut memory = WindowMemory::new(30);
    for i in 1..=4 {
        memory
            .record(Message::user(format!("question for round {i}")))
            .await?;
        memory
            .record(Message::assistant(format!("reply for round {i}")))
            .await?;
    }
    let context = memory.context().await?;
    println!(
        "1. default window (budget 30): full 8 messages → {} after trimming",
        context.len()
    );
    for message in &context {
        println!("   {message:?}");
    }

    // 2. Lossless projection: raise the budget and the trimmed history is immediately restored (storage was never touched).
    memory.set_max_tokens(1000);
    println!(
        "2. budget raised: back to the full {} messages (lossless projection; history restored)",
        memory.context().await?.len()
    );
    memory.set_max_tokens(30);

    // 3. Round window: max_rounds keeps only the most recent 2 rounds (independent of token counting).
    let mut memory = WindowMemory::new(1000).with_max_rounds(2);
    for i in 1..=4 {
        memory
            .record(Message::user(format!("question for round {i}")))
            .await?;
        memory
            .record(Message::assistant(format!("reply for round {i}")))
            .await?;
    }
    println!(
        "3. max_rounds(2): full 8 messages → the most recent 2 rounds {}",
        memory.context().await?.len()
    );

    // 4. Custom strategy (materialized compression): inject a summarizer strategy; when over
    //    budget, old messages are replaced by the summary; subsequent under-budget fetches are
    //    zero recompute; when over budget again, the input is "the last compression result +
    //    new messages" (chained compression).
    let mut memory = WindowMemory::new(30).with_strategy(Arc::new(Summarizer));
    for i in 1..=4 {
        memory
            .record(Message::user(format!("question for round {i}")))
            .await?;
        memory
            .record(Message::assistant(format!("reply for round {i}")))
            .await?;
    }
    let compressed = memory.context().await?;
    println!(
        "4. summarizer strategy injected: {} after compression (materialized; old messages are not kept)",
        compressed.len()
    );
    for message in &compressed {
        println!("   {message:?}");
    }
    // After materialization, under budget: the next fetch is a direct clone, zero recompute.
    let again = memory.context().await?;
    println!(
        "   → fetched again: {} (under budget, zero recompute)",
        again.len()
    );
    // Appending new messages breaks the budget → chained compression: input = the last compression result + new messages.
    memory.record(Message::user("question for round 5")).await?;
    memory
        .record(Message::assistant("reply for round 5"))
        .await?;
    let context = memory.context().await?;
    println!(
        "   → compressed again after appending round 5: {} (input includes the previous summary)",
        context.len()
    );

    Ok(())
}
