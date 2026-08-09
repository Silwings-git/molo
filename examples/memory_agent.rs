//! Memory Agent example: the reasoning loop manages session context through
//! Memory.
//!
//! The only difference from `examples/chat.rs`: the loop no longer holds its
//! own `Vec<Message>`; instead it hands each turn's messages to
//! [`InMemoryMemory`](molo::InMemoryMemory) and fetches the context from
//! [`context`](molo::Memory::context) before each chat — the loop only depends
//! on the Memory interface, does not care how messages are stored, and swapping
//! the Memory implementation (window trimming, persistence, etc.) requires no
//! loop changes.
//!
//! On startup the example reads configuration from `.env` (copy `.example.env`
//! to `.env` and fill in real values); environment variables can also override
//! directly:
//! - MOLO_API_KEY  : API key; may be left empty for local endpoints without
//!   auth (e.g. Ollama)
//! - MOLO_BASE_URL : OpenAI-compatible endpoint, default https://api.openai.com/v1
//! - MOLO_MODEL    : model name, default gpt-4o-mini
//!
//! Run: `cargo run --example memory_agent`
//! Type exit / quit / Ctrl-D to quit.

use molo::{ChatRequest, InMemoryMemory, Memory, Message, Provider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok(); // load .env; silently ignore if missing

    let base_url =
        std::env::var("MOLO_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let api_key = std::env::var("MOLO_API_KEY").unwrap_or_default();
    let model = std::env::var("MOLO_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

    let provider = molo::provider::OpenAiProvider::new(base_url, api_key, model);

    // Session context is held by Memory; the loop only interacts with it and never touches the message list directly.
    let mut memory = InMemoryMemory::default();
    memory
        .record(Message::system("You are a helpful assistant."))
        .await?;

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
        let response = provider
            .chat(ChatRequest {
                messages: memory.context().await?,
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
    }

    Ok(())
}
