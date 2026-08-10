//! Summary compression example: demonstrates
//! [`SummarizeStrategy`] — old messages over budget are handed to the
//! summarizer model, compressed into one System summary message, and written
//! back as a materialized value.
//!
//! This example is **self-contained**, needs no API key, just run:
//! `cargo run --example summarize`
//!
//! Each stage prints a "memory view": storage size, budget usage (token
//! counting matches WindowMemory's default), and the compression result. It
//! demonstrates:
//! - Basic compression: when over budget, the most recent rounds keep their
//!   original text and earlier messages are replaced by a summary;
//! - Materialized zero-recompute: after compression, fetches that stay under
//!   budget no longer call the summarizer model;
//! - Incremental compression: when over budget again, the previous summary is
//!   merged into the new summary as input.
//!
//! To wire up a real model, swap [`FakeProvider`] for
//! [`OpenAiProvider`](molo::OpenAiProvider) (optionally wrapped in
//! [`RetryProvider`](molo::RetryProvider)); the example injects summary
//! replies with a fake Provider to demonstrate the full flow. Real agent
//! integration is in `examples/summarize_agent.rs`.

use std::sync::Arc;

use molo::memory::{CharTokenCounter, Memory, SummarizeStrategy, TokenCounter, WindowMemory};
use molo::{ContentBlock, FakeProvider, FakeReply, Message};

/// Total tokens of a message sequence, counting the same way as WindowMemory.
async fn count_tokens(messages: &[Message]) -> Result<usize, molo::memory::MemoryError> {
    let counter = CharTokenCounter;
    let mut total = 0usize;
    for message in messages {
        match message {
            Message::System(s) => total += counter.count(s).await?,
            Message::User(blocks) => {
                for block in blocks {
                    let ContentBlock::Text(t) = block;
                    total += counter.count(t).await?;
                }
            }
            Message::Assistant {
                content,
                reasoning,
                tool_calls,
            } => {
                total += counter.count(content).await?;
                if let Some(r) = reasoning {
                    total += counter.count(r).await?;
                }
                for tc in tool_calls {
                    total += counter.count(&tc.arguments).await?;
                }
            }
            Message::ToolResult { content, .. } => total += counter.count(content).await?,
        }
    }
    Ok(total)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Summarizer model = FakeProvider, with a two-round script (one reply for the first compression / one for the incremental compression).
    let fake = Arc::new(FakeProvider::new([
        FakeReply::Text("A and B have been answered".into()),
        FakeReply::Text("A and B have been answered, and C has been answered too".into()),
    ]));

    // Budget: 30 tokens; the full 4 rounds cost 48 > 30, triggering compression.
    let mut memory =
        WindowMemory::new(30).with_strategy(Arc::new(SummarizeStrategy::new(fake.clone())));
    for i in 1..=4 {
        memory
            .record(Message::user(format!("question for round {i}")))
            .await?;
        memory
            .record(Message::assistant(format!("reply for round {i}")))
            .await?;
    }
    let full_tokens = count_tokens(&[
        Message::user("question for round 1"),
        Message::assistant("reply for round 1"),
    ])
    .await?;
    println!(
        "0. recorded 4 rounds of chat: 8 messages ≈ {} tokens (budget 30) → over budget; the first fetch triggers compression",
        full_tokens * 4
    );

    let context = memory.context().await?;
    let tokens = count_tokens(&context).await?;
    println!(
        "1. first compression: 8 messages → {} ≈ {tokens} tokens (summary + most recent round)",
        context.len()
    );
    for message in &context {
        println!("   {message:?}");
    }
    assert_eq!(context.len(), 3);
    assert!(matches!(context[0], Message::System(_)));
    assert_eq!(context[1], Message::user("question for round 4"));

    // After compression, [summary, most recent round] ≈ 26 tokens ≤ 30: the next fetch is a direct clone, zero recompute.
    let again = memory.context().await?;
    let again_tokens = count_tokens(&again).await?;
    println!(
        "2. fetched {} again, under budget ≈ {again_tokens} tokens (materialized, zero recompute — the summarizer model is not called again)",
        again.len()
    );
    assert_eq!(again, context);
    assert_eq!(
        fake.requests().len(),
        1,
        "after compression, under-budget fetches must not call the summarizer model again"
    );

    // Append rounds 5 and 6 (≈ 50 > 30): compress again — the old summary is merged into the new one as input.
    for i in 5..=6 {
        memory
            .record(Message::user(format!("question for round {i}")))
            .await?;
        memory
            .record(Message::assistant(format!("reply for round {i}")))
            .await?;
    }
    let context = memory.context().await?;
    let tokens = count_tokens(&context).await?;
    println!(
        "3. compressed again after appending rounds 5 and 6: {} ≈ {tokens} tokens (old summary merged into the new one)",
        context.len()
    );
    for message in &context {
        println!("   {message:?}");
    }
    assert_eq!(context.len(), 3);
    assert_eq!(context[1], Message::user("question for round 6"));
    assert_eq!(
        fake.requests().len(),
        2,
        "each compression calls the summarizer model once"
    );

    // The second compression's request input: verify the old summary was merged in (incremental compression).
    let requests = fake.requests();
    let Message::User(blocks) = &requests[1].messages[1] else {
        unreachable!("summary input is a user message");
    };
    let ContentBlock::Text(text) = &blocks[0];
    assert!(
        text.contains("A and B have been answered"),
        "incremental compression must carry the previous summary"
    );
    println!(
        "4. the summarizer model received 2 requests in total; the second request's input contains the previous summary (incremental merge)"
    );

    Ok(())
}
