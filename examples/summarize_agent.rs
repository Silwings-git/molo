//! Summarize Agent example (streaming): ReActAgent + SummarizeStrategy compress
//! the context with summaries, and replies are presented as streaming chunks.
//!
//! This example demonstrates an **assembly pattern** — the reasoning loop is
//! carried by [`ReActAgent`](molo::ReActAgent) (no hand-written loop), and
//! summarization takes effect through an injected Memory: the loop only
//! depends on [`Memory`](molo::Memory), so swapping the implementation swaps
//! the strategy; replies are streamed via
//! [`Agent::run_stream`](molo::Agent::run_stream) with text printed chunk by
//! chunk, and the turn ends with
//! [`MessageChunk::Done`](molo::MessageChunk) carrying this turn's summary
//! (rounds / tool_calls / usage).
//!
//! What happens in memory is visible through tracing logs (the example has a
//! subscriber installed):
//! - One `agent.run` span (INFO) per chat round, carrying `run.id`;
//! - When compression happens,
//!   [`SummarizeStrategy`](molo::SummarizeStrategy) emits an info log:
//!   `context summarized by LLM` (`compressed` = number of compressed
//!   entries / `kept` = number of kept entries / `summary_tokens` = summary
//!   token count) — when compression happens and its scale are immediately
//!   clear; a summary failure that degrades is logged as warn;
//! - The turn-end summary shows this round's token usage (prompt /
//!   completion).
//!
//! For a complete view of the memory with per-message detail and token usage,
//! use the self-contained example `examples/summarize.rs` (fake Provider) or
//! the hand-written-loop example `examples/window_memory_agent.rs`; this
//! example focuses on ReActAgent assembly.
//!
//! Highlights:
//! - With a small budget (MOLO_MAX_TOKENS, default 400), a few turns trigger
//!   compression — watch for the compression log (the compressed count
//!   changes);
//! - Experiment: first let the model remember your name, keep chatting until
//!   the compression log appears, then ask it your name — the summary keeps
//!   the key points, so it should still answer (compare with window
//!   trimming: what is trimmed is lost);
//! - The summary output cap (MOLO_SUMMARY_MAX_TOKENS) also reserves budget
//!   for the kept rounds: make it smaller to leave more room for recent
//!   rounds, larger for a more detailed summary.
//!
//! On startup the example reads configuration from `.env` (copy `.example.env`
//! to `.env` and fill in real values); environment variables can also override
//! directly:
//! - MOLO_API_KEY             : API key; may be left empty for local endpoints without
//!   auth (e.g. Ollama)
//! - MOLO_BASE_URL            : OpenAI-compatible endpoint, default https://api.openai.com/v1
//! - MOLO_MODEL               : model name, default gpt-4o-mini
//! - MOLO_MAX_TOKENS          : context budget, default 400 (lower triggers compression sooner)
//! - MOLO_SUMMARY_MAX_TOKENS  : summary output cap, default 150 (also the kept-rounds reserve)
//!
//! Run: `cargo run --example summarize_agent`
//! Type exit / quit / Ctrl-D to quit.

use std::io::Write;
use std::sync::Arc;

use futures::stream::StreamExt;

use molo::agent::{Agent, MessageChunk};
use molo::memory::{SummarizeStrategy, WindowMemory};
use molo::{OpenAiProvider, ReActAgent, ToolRegistry};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Observability: show only molo's INFO-level spans and logs — one agent.run
    // span per chat round; when compression happens, SummarizeStrategy's log
    // lines carry the agent.run context.
    tracing_subscriber::fmt()
        .with_env_filter("molo=info")
        .without_time()
        .init();

    dotenvy::dotenv().ok(); // load .env; silently ignore if missing

    let base_url =
        std::env::var("MOLO_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let api_key = std::env::var("MOLO_API_KEY").unwrap_or_default();
    let model = std::env::var("MOLO_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
    let max_tokens: usize = std::env::var("MOLO_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400);
    let summary_max_tokens: u32 = std::env::var("MOLO_SUMMARY_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(150);

    let provider = OpenAiProvider::new(base_url, api_key, model);

    // Assembly: ReActAgent carries the loop; the Memory injects the summarization strategy
    // (the summarizer reuses a clone of the provider — the strategy only depends on
    // [`Provider`](molo::Provider); in production you can swap in a dedicated summarizer
    // model and customize the summary instructions with with_prompt).
    let memory = WindowMemory::new(max_tokens).with_strategy(Arc::new(
        SummarizeStrategy::new(provider.clone()).with_summary_max_tokens(summary_max_tokens),
    ));
    let mut agent = ReActAgent::new(
        provider,
        ToolRegistry::new(),
        "You are a helpful assistant. The conversation may be long; early content will be compressed into summaries, but the key points will be kept.",
    )
    .with_memory(memory);

    println!(
        "Memory budget: {max_tokens} tokens; summary output cap: {summary_max_tokens} tokens.\n\
         When compression happens the log prints: context summarized by LLM (compressed / kept / summary_tokens).\n"
    );

    let mut input = String::new();
    loop {
        input.clear();
        println!("user:");
        let read = std::io::stdin().read_line(&mut input)?;
        if read == 0 {
            break; // Ctrl-D
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "exit" || input == "quit" {
            break;
        }

        // Streaming chat: recording, fetching, chatting, and summary compression all happen
        // inside run_stream; compression is visible in the logs, and here we only consume
        // message chunks for display.
        let mut stream = agent.run_stream(input).await?;
        let mut prefix_printed = false;
        while let Some(event) = stream.next().await {
            match event? {
                MessageChunk::Delta(delta) => {
                    if !prefix_printed {
                        print!("assistant: ");
                        std::io::stdout().flush()?;
                        prefix_printed = true;
                    }
                    print!("{delta}");
                    std::io::stdout().flush()?;
                }
                MessageChunk::ToolCall {
                    id, name, arguments,
                } => {
                    println!("\n  → calling {name}(#{id}), arguments: {arguments}");
                }
                MessageChunk::ToolResult { id, name, content } => {
                    println!("  → {name}(#{id}) returned: {content}");
                }
                MessageChunk::Done(summary) => {
                    println!(
                        "\n  — run summary: rounds={} tool_calls={} usage={}/{} tokens",
                        summary.rounds,
                        summary.tool_calls,
                        summary.usage.prompt_tokens,
                        summary.usage.completion_tokens,
                    );
                    break;
                }
                MessageChunk::Cancelled => {
                    println!("\n[cancelled]");
                    break;
                }
                // Unknown variant (reserved for non_exhaustive extensions): silently ignore
                _ => {}
            }
        }
        println!();
    }

    Ok(())
}
