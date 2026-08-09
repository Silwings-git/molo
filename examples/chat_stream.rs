//! Streaming command-line chat example: consumes the
//! [`StreamEvent`](molo::StreamEvent) event stream via
//! [`Provider::stream_chat`](molo::Provider::stream_chat) — reply deltas are
//! printed as they arrive instead of waiting for the full reply.
//!
//! Environment variables are the same as in examples/chat.rs
//! (MOLO_API_KEY / MOLO_BASE_URL / MOLO_MODEL).
//!
//! Run: `cargo run --example chat_stream`
//! Type exit / quit / Ctrl-D to quit.

use std::io::Write;

use futures::StreamExt;
use molo::provider::OpenAiProvider;
use molo::{ChatRequest, Message, Provider, StreamEvent};

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
        print!("assistant: ");
        std::io::stdout().flush()?;

        let mut reply = String::new();
        let mut stream = provider
            .stream_chat(ChatRequest {
                messages: messages.clone(),
                tools: vec![],
                ..Default::default()
            })
            .await?;
        while let Some(event) = stream.next().await {
            match event? {
                StreamEvent::Delta(delta) => {
                    print!("{delta}");
                    std::io::stdout().flush()?;
                    reply.push_str(&delta);
                }
                // Plain-text chat provides no tools, so the model will not request tool calls; reasoning is not displayed.
                StreamEvent::ToolCall { .. } => {}
                StreamEvent::Reasoning(_) => {}
                StreamEvent::Done { .. } => break,
                // Unknown variant (reserved for non_exhaustive extensions): silently ignore
                _ => {}
            }
        }
        println!();
        messages.push(Message::assistant(reply));
    }

    Ok(())
}
