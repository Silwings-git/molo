//! Command-line chat example: uses [`Provider::chat`](molo::Provider::chat)
//! directly for a full conversation — ask, wait for the full reply, merge the
//! reply into history, and repeat (non-streaming).
//!
//! On startup the example reads configuration from `.env` (copy `.example.env`
//! to `.env` and fill in real values); environment variables can also override
//! directly:
//! - MOLO_API_KEY  : API key; may be left empty for local endpoints without
//!   auth (e.g. Ollama)
//! - MOLO_BASE_URL : OpenAI-compatible endpoint, default https://api.openai.com/v1
//! - MOLO_MODEL    : model name, default gpt-4o-mini
//!
//! To print reply deltas as they arrive, use the streaming version
//! `examples/chat_stream.rs`.
//!
//! Run: `cargo run --example chat`
//! Type exit / quit / Ctrl-D to quit.

use molo::provider::OpenAiProvider;
use molo::{ChatRequest, Message, Provider};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok(); // load .env; silently ignore if missing

    let base_url =
        std::env::var("MOLO_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let api_key = std::env::var("MOLO_API_KEY").unwrap_or_default();
    let model = std::env::var("MOLO_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

    let provider = OpenAiProvider::new(base_url, api_key, model);

    let mut messages = vec![Message::system("You are a helpful assistant.")];
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

        messages.push(Message::user(input));
        let response = provider
            .chat(ChatRequest {
                messages: messages.clone(),
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
        messages.push(Message::assistant(reply));
    }

    Ok(())
}
