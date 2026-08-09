//! Real Provider usage example: prints this run's execution summary at the end
//! of each chat turn.
//!
//! Drives a [`ReActAgent`](molo::ReActAgent) with a real
//! [`OpenAiProvider`](molo::provider::OpenAiProvider) and **prints this run's
//! execution summary at the end of every turn**
//! ([`RunSummary`](molo::RunSummary)): chat rounds, tool executions, and token
//! usage (prompt / completion / total).
//!
//! The usage data path: non-streaming responses carry usage natively; streaming
//! enables it via `stream_options.include_usage`, with the final chunk carrying
//! the totals — ReActAgent accumulates them round by round and delivers them
//! with [`MessageChunk::Done`](molo::MessageChunk::Done) (carrying the
//! RunSummary).
//!
//! On startup the example reads configuration from `.env` (copy `.example.env`
//! to `.env` and fill in real values); environment variables can also override
//! directly:
//! - MOLO_API_KEY  : API key; may be left empty for local endpoints without
//!   auth (e.g. Ollama)
//! - MOLO_BASE_URL : OpenAI-compatible endpoint, default https://api.openai.com/v1
//! - MOLO_MODEL    : model name, default gpt-4o-mini
//!
//! Run: `cargo run --example usage`
//! Try asking: "What is (1 + 2) * 3?" — the model will request the calculator
//! tool, and at the end you can see the accumulated usage of the two rounds
//! (tool round + direct-answer round).
//!
//! Note: `run` (non-streaming) returns text only, with no usage summary; for
//! usage stats, use `run_stream` — the summary arrives with the `Done
//! (RunSummary)` at the end of the stream.

use molo::agent::{Agent, MessageChunk};
use molo::provider::OpenAiProvider;
use molo::react_agent;
use molo::tool::{SharedState, Tool, ToolError, ToolSchema};

use futures::stream::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use std::io::Write;

/// Arguments for the calculator tool.
#[derive(Debug, Deserialize, JsonSchema)]
struct CalcArgs {
    /// The math expression to evaluate, e.g. "1 + 2 * 3".
    #[schemars(description = "The math expression to evaluate, e.g. \"1 + 2 * 3\"")]
    expression: String,
}

/// A tool that evaluates math expressions.
struct Calculator;

#[async_trait::async_trait]
impl Tool for Calculator {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "calculator".into(),
            description: "Evaluates a math expression; supports basic arithmetic and parentheses, e.g. \"(1 + 2) * 3\".".into(),
            parameters: serde_json::to_value(schemars::schema_for!(CalcArgs))
                .expect("tool schema must serialize"),
        }
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        _state: &SharedState,
    ) -> Result<String, ToolError> {
        let args: CalcArgs = serde_json::from_value(arguments)?;
        let value =
            evalexpr::eval(&args.expression).map_err(|e| ToolError::Execution(e.to_string()))?;
        Ok(value.to_string())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok(); // load .env; silently ignore if missing

    let base_url =
        std::env::var("MOLO_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let api_key = std::env::var("MOLO_API_KEY").unwrap_or_default();
    let model = std::env::var("MOLO_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

    let provider = OpenAiProvider::new(base_url, api_key, model);
    // Convenient assembly: the `react_agent!` macro auto-boxes and registers the tool list; Memory defaults to a bounded window
    // (WindowMemory, 128k token budget, auto-trims the oldest rounds in long sessions).
    let mut agent = react_agent!(
        provider,
        [Calculator],
        "You are a helpful assistant. Use the calculator tool for calculations instead of doing math in your head.",
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

        // Consume the event stream: text printed token by token, tool process presented as events;
        // when `Done(RunSummary)` arrives, print this run's usage summary.
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
                    id,
                    name,
                    arguments,
                } => {
                    println!("\n  → calling {name}(#{id}), arguments: {arguments}");
                }
                MessageChunk::ToolResult { id, name, content } => {
                    println!("  → {name}(#{id}) returned: {content}");
                }
                MessageChunk::Done(summary) => {
                    println!("\n--- this run's execution summary ---");
                    println!(
                        "rounds: {} (tool executions: {})",
                        summary.rounds, summary.tool_calls
                    );
                    println!(
                        "usage: prompt {} / completion {} / total {} tokens",
                        summary.usage.prompt_tokens,
                        summary.usage.completion_tokens,
                        summary.usage.total_tokens,
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
