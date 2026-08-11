//! Multimodal (image input) example: sends a user message containing an
//! image file to a multimodal model via the OpenAI-compatible endpoint.
//!
//! On startup the example reads configuration from `.env` (copy `.example.env`
//! to `.env` and fill in real values); environment variables can also override
//! directly:
//! - MOLO_API_KEY  : API key; may be left empty for local endpoints without
//!   auth (e.g. Ollama)
//! - MOLO_BASE_URL : OpenAI-compatible endpoint, default https://api.openai.com/v1
//! - MOLO_MODEL    : multimodal model name, default gpt-4o-mini
//!
//! The example takes the image path as a command-line argument; the MIME type
//! is inferred from the file extension.
//!
//! Run:
//! `cargo run --example multimodal -- /path/to/photo.png`
//! `cargo run --example multimodal -- /path/to/photo.png --stream`

use futures::StreamExt;
use molo::provider::OpenAiProvider;
use molo::{ChatRequest, ContentBlock, ImageContent, Message, Provider, StreamEvent};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok(); // load .env; silently ignore if missing

    // The first non-flag argument is the image path; the rest of the
    // message is a fixed question for this example.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(image_path) = args.first().filter(|a| !a.starts_with("--")) else {
        eprintln!("usage: cargo run --example multimodal -- <image path> [--stream]");
        std::process::exit(2);
    };
    let stream = args.iter().any(|a| a == "--stream");

    let base_url =
        std::env::var("MOLO_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let api_key = std::env::var("MOLO_API_KEY").unwrap_or_default();
    let model = std::env::var("MOLO_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

    let provider = OpenAiProvider::new(base_url, api_key, model);

    // Read the image file and infer the MIME type from the extension.
    let bytes = std::fs::read(image_path)?;
    let mime_type = image_mime_type(image_path);
    println!("sending image ({mime_type}, {} bytes)", bytes.len());

    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user_blocks(vec![
            ContentBlock::Text("Describe this image in detail.".into()),
            ContentBlock::Image(ImageContent::new(mime_type, bytes)),
        ]),
    ];

    let request = ChatRequest {
        messages,
        tools: vec![],
        ..Default::default()
    };

    if stream {
        let mut stream = provider.stream_chat(request).await?;
        while let Some(event) = stream.next().await {
            match event? {
                // Reasoning fragments are rendered on their own line (they
                // can interleave with content in the wire order).
                StreamEvent::Reasoning(r) => println!("[thinking] {r}"),
                StreamEvent::Delta(t) => print!("{t}"),
                StreamEvent::Done { .. } => println!(),
                _ => {}
            }
        }
    } else {
        let response = provider.chat(request).await?;
        let Message::Assistant { content, .. } = &response.message else {
            unreachable!("reply must be an Assistant message in tool-free chat");
        };
        println!("assistant: {content}");
    }

    Ok(())
}

/// Infers the MIME type from a file extension; defaults to `image/png` when
/// the extension is unknown.
fn image_mime_type(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("png") => "image/png",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("bmp") => "image/bmp",
        _ => "image/png",
    }
}
