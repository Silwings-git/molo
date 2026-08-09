//! Window Memory Agent example: the reasoning loop manages context with
//! [`WindowMemory`](molo::WindowMemory).
//!
//! The only difference from `memory_agent`: the Memory is swapped to
//! [`WindowMemory`](molo::WindowMemory) — the loop only depends on
//! [`Memory`](molo::Memory), so swapping the implementation swaps the
//! strategy; when over budget the loop still notices nothing, and the context
//! is trimmed by whole rounds (the earliest conversation disappears).
//!
//! Highlights:
//! - The context size is printed every round; with a small budget
//!   (MOLO_MAX_TOKENS, default 200), a few turns trigger trimming, showing the
//!   "context trimmed" notice;
//! - Experiment: first let the model remember your name, keep chatting until
//!   the trimming notice appears, then ask it your name — it will not be able
//!   to answer (window semantics: the trimmed early conversation is no longer
//!   in the context);
//! - Contrast: with [`InMemoryMemory`](molo::InMemoryMemory), nothing is ever
//!   trimmed (all memory is kept).
//!
//! This example sets no system prompt: the system prompt belongs to the Agent
//! layer (`AgentConfig::system_prompt`), and Memory does not manage System
//! (window trimming makes no special case for it) — for identity setup, a
//! hand-written loop can still `record(Message::system(...))` directly, but it
//! is treated like any other message.
//!
//! On startup the example reads configuration from `.env` (copy `.example.env`
//! to `.env` and fill in real values); environment variables can also override
//! directly:
//! - MOLO_API_KEY   : API key; may be left empty for local endpoints without
//!   auth (e.g. Ollama)
//! - MOLO_BASE_URL  : OpenAI-compatible endpoint, default https://api.openai.com/v1
//! - MOLO_MODEL     : model name, default gpt-4o-mini
//! - MOLO_MAX_TOKENS: window budget, default 200 (lower triggers trimming sooner)
//!
//! Run: `cargo run --example window_memory_agent`
//! Type exit / quit / Ctrl-D to quit.

use molo::{ChatRequest, Memory, Message, Provider, WindowMemory};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok(); // load .env; silently ignore if missing

    let base_url =
        std::env::var("MOLO_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let api_key = std::env::var("MOLO_API_KEY").unwrap_or_default();
    let model = std::env::var("MOLO_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
    let max_tokens: usize = std::env::var("MOLO_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);

    let provider = molo::provider::OpenAiProvider::new(base_url, api_key, model);

    // Window Memory: returns everything as-is within budget; trims by whole rounds when over
    // budget (the earliest conversation disappears). The loop code is identical to memory_agent
    // line by line; the only difference is the Memory implementation.
    let mut memory = WindowMemory::new(max_tokens);

    // Messages recorded so far: compared with the context length to detect trimming.
    let mut recorded = 0usize;
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

        // Each turn: record the user input → fetch the context and run a chat → record the assistant reply.
        memory.record(Message::user(input)).await?;
        recorded += 1;
        let context = memory.context().await?;
        if context.len() < recorded {
            println!(
                "   [context trimmed: {recorded} recorded → {} returned; the earliest conversation was dropped]",
                context.len()
            );
        } else {
            println!("   [context {} messages, within budget]", context.len());
        }
        let response = provider
            .chat(ChatRequest {
                messages: context,
                tools: vec![],
                ..Default::default()
            })
            .await?;
        // In tool-free chat, the model's reply is always a single Assistant text message.
        let reply = match response.message {
            Message::Assistant { content, .. } => content,
            _ => unreachable!("reply must be an Assistant message in tool-free chat"),
        };
        println!("assistant: {reply}");
        memory.record(Message::assistant(reply)).await?;
        recorded += 1;
    }

    Ok(())
}
