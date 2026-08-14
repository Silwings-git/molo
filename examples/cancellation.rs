//! Cooperative cancellation demo: stop the reply mid-stream, then keep
//! chatting with the same Agent afterwards.
//!
//! Demonstrates two cancellation scenarios for interactive applications using
//! [`RunContext`](molo::RunContext):
//!
//! 1. **Stop mid-reply**: "press Esc" during streaming (the example simulates
//!    the keypress with a delay) — deltas already emitted are kept, the stream
//!    ends with [`MessageChunk::Cancelled`](molo::MessageChunk::Cancelled)
//!    instead of producing `Done`;
//! 2. **Continue after stopping**: run the same Agent again with a fresh
//!    token; cancellation leaves no residue.
//!
//! # Cancellation semantics
//!
//! - The cancellation source is a
//!   [`CancellationToken`](molo::CancellationToken): any task may hold the
//!   same token; calling `cancel()` requests a stop;
//! - Cancellation is cooperative: the Agent responds to the token at safe
//!   checkpoints, and content already emitted is not rolled back;
//! - Cancellation applies per run: each run uses its own token, so cancelling
//!   only affects the current run; the next round continues with a fresh token,
//!   and multi-turn sessions are unaffected by residue.
//!
//! This example is self-contained: a slow-streaming Provider simulates "still
//! generating", with no real API dependency.
//! Run: `cargo run --example cancellation`

use futures::StreamExt;
use molo::agent::{Agent, MessageChunk};
use molo::provider::{
    ChatRequest, ChatResponse, FinishReason, Provider, ProviderError, ProviderRequestContext,
    StreamEvent,
};
use molo::{CancellationToken, RunContext, RunRequest};
use std::io::Write;
use std::time::Duration;

/// Slow-streaming Provider: emits a text delta every 40ms to simulate token-by-token LLM generation.
struct SlowProvider;

#[async_trait::async_trait]
impl Provider for SlowProvider {
    async fn chat_with_context(
        &self,
        _request: ChatRequest,
        _context: &ProviderRequestContext,
    ) -> Result<ChatResponse, ProviderError> {
        unreachable!("example uses streaming path only")
    }

    async fn stream_chat_with_context(
        &self,
        _request: ChatRequest,
        context: &ProviderRequestContext,
    ) -> Result<
        futures::stream::BoxStream<'static, Result<StreamEvent, ProviderError>>,
        ProviderError,
    > {
        let context = context.clone();
        Ok(Box::pin(async_stream::stream! {
            for chunk in ["Hello", ",", " this", " is", " a", " long", " reply"] {
                if context.is_cancelled() {
                    yield Err(ProviderError::Cancelled);
                    return;
                }
                yield Ok(StreamEvent::Delta(chunk.into()));
                tokio::time::sleep(Duration::from_millis(40)).await;
            }
            yield Ok(StreamEvent::Done { reason: FinishReason::Stop, usage: None });
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = molo::react_agent!(SlowProvider, "You are a helpful assistant");

    // ---- Part 1: stop mid-reply (simulating the user pressing Esc later) ----
    println!("user: say something");
    let token = CancellationToken::new();

    // A separate "Esc listener": another task holds the same token and can cancel at any time.
    tokio::spawn({
        let token = token.clone();
        async move {
            tokio::time::sleep(Duration::from_millis(150)).await;
            println!("  [Esc] stop the current reply");
            token.cancel();
        }
    });

    let mut stream = agent
        .run_stream_request_with_context(
            RunRequest::text("say something"),
            RunContext::generated().with_cancellation(token),
        )
        .await?;
    let mut got = String::new();
    while let Some(event) = stream.next().await {
        match event? {
            MessageChunk::Delta(delta) => {
                print!("{delta}");
                std::io::stdout().flush()?;
                got.push_str(&delta);
            }
            MessageChunk::ToolCall { .. } | MessageChunk::ToolResult { .. } => {}
            MessageChunk::Done(_) => {
                println!();
                break;
            }
            MessageChunk::Cancelled => {
                println!("\n  [stopped]");
                break;
            }
            // Unknown variant (reserved for non_exhaustive extensions): silently ignore
            _ => {}
        }
    }
    println!("  partial reply emitted so far: {got:?}");
    drop(stream); // release the borrow of the agent

    // ---- Part 2: after stopping, type new content and continue with the same agent ----
    // Cancellation leaves no residue: run again with a fresh token; memory/history is intact (messages recorded in the previous round are kept).
    println!("\nuser: continue");
    let token = CancellationToken::new();
    let mut stream = agent
        .run_stream_request_with_context(
            RunRequest::text("continue"),
            RunContext::generated().with_cancellation(token),
        )
        .await?;
    while let Some(event) = stream.next().await {
        match event? {
            MessageChunk::Delta(delta) => {
                print!("{delta}");
                std::io::stdout().flush()?;
            }
            MessageChunk::ToolCall { .. } | MessageChunk::ToolResult { .. } => {}
            MessageChunk::Done(_) => {
                println!();
                break;
            }
            MessageChunk::Cancelled => {
                println!("\n  [stopped]");
                break;
            }
            // Unknown variant (reserved for non_exhaustive extensions): silently ignore
            _ => {}
        }
    }

    Ok(())
}
