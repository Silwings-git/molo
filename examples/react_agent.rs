//! ReActAgent example: assembling and using the framework's built-in classic
//! Agent loop.
//!
//! [`ReActAgent`](molo::ReActAgent) is a general implementation of the classic
//! reasoning loop: record the user input → run a chat turn → execute tool
//! calls the model requests and feed results back → until the model answers
//! directly. Assembly = the three pieces (Provider / Memory / ToolRegistry)
//! via constructor arguments + optional behavior (system prompt / max tool
//! rounds) via [`AgentConfig`](molo::AgentConfig).
//!
//! On startup the example reads configuration from `.env` (copy `.example.env`
//! to `.env` and fill in real values); environment variables can also override
//! directly:
//! - MOLO_API_KEY  : API key; may be left empty for local endpoints without
//!   auth (e.g. Ollama)
//! - MOLO_BASE_URL : OpenAI-compatible endpoint, default https://api.openai.com/v1
//! - MOLO_MODEL    : model name, default gpt-4o-mini
//!
//! Run mode (selected by the first command-line argument):
//! - `cargo run --example react_agent -- stream` (default): streaming chat,
//!   text printed token by token, tool process presented as
//!   [`MessageChunk`](molo::MessageChunk) events;
//! - `cargo run --example react_agent -- chat`: non-streaming chat, the whole
//!   turn returned at once.
//!
//! Try asking: "What is (1 + 2) * 3?"; the model will request the calculator
//! tool, the Agent executes it and feeds the result back, and the model gives
//! the final answer based on it.
//!
//! Difference from `examples/agent.rs`: agent.rs shows the application
//! implementing [`Agent`](molo::Agent) itself (its own loop shape); this
//! example uses the framework's built-in ReActAgent directly, with the round
//! limit / cooperative cancellation / Usage aggregation / event channel / Trace
//! all built in.
//!
//! # Macro assembly overview ([`react_agent!`](molo::react_agent), self-contained, no real API needed)
//!
//! ```rust
//! use molo::provider::{FakeProvider, FakeReply};
//! use molo::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolResult, ToolSchema};
//! use molo::{react_agent, ToolRegistry};
//!
//! /// Demo tool: returns the arguments as-is (ignores the shared state).
//! struct Echo;
//! #[async_trait::async_trait]
//! impl Tool for Echo {
//!     fn schema(&self) -> ToolSchema {
//!         ToolSchema::new("echo", "Echoes back", serde_json::json!({}))
//!     }
//!     async fn call(&self, arguments: serde_json::Value, _context: ToolContext<'_>) -> Result<ToolResult, ToolError> {
//!         Ok(ToolOutput::text(arguments.to_string()).into())
//!     }
//! }
//!
//! fn fake() -> FakeProvider {
//!     FakeProvider::new([FakeReply::Text("hi".into())])
//! }
//!
//! // Six assembly shapes (construction only; the full run is in main):
//! let a1 = react_agent!(fake());                                // no tools, no system prompt
//! let a2 = react_agent!(fake(), "You are an assistant");         // no tools, with system prompt
//! let a3 = react_agent!(fake(), [Echo]);                        // tool list, no system prompt
//! let a4 = react_agent!(fake(), [Echo], "You are an assistant"); // tool list, with system prompt
//! let mut registry = ToolRegistry::new();
//! registry.register(Echo);
//! let a5 = react_agent!(fake(), registry.clone());              // existing registry, no system prompt
//! let a6 = react_agent!(fake(), registry, "You are an assistant"); // existing registry, with system prompt (the registry can be reused/cloned)
//! # let _ = (a1, a2, a3, a4, a5, a6);
//! ```
//!
//! The system prompt is optional (omitted = none); the tool list supports
//! heterogeneous types (the macro auto-boxes and registers them); to pass a
//! system prompt held in a variable, write
//! `react_agent!(provider, [], system_var)`. Shared state (flow between tools /
//! application reads and writes across runs / sharing across agents) is in
//! `examples/shared_state.rs`.

use molo::agent::{Agent, MessageChunk};
use molo::provider::OpenAiProvider;
use molo::react_agent;
use molo::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolResult, ToolSchema};

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
        ToolSchema::new(
            "calculator",
            "Evaluates a math expression; supports basic arithmetic and parentheses, e.g. \"(1 + 2) * 3\".",
            serde_json::to_value(schemars::schema_for!(CalcArgs))
                .expect("tool schema must serialize"),
        )
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        _context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let args: CalcArgs = serde_json::from_value(arguments)?;
        let value =
            evalexpr::eval(&args.expression).map_err(|e| ToolError::Execution(e.to_string()))?;
        Ok(ToolOutput::text(value.to_string()).into())
    }
}

/// How the chat turn is run (ReActAgent's two entry points).
enum Mode {
    /// [`Agent::run_stream`](Agent::run_stream): text printed token by token, tool process presented as events.
    Stream,
    /// [`Agent::run`](Agent::run): the whole turn returned at once.
    Chat,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok(); // load .env; silently ignore if missing

    let base_url =
        std::env::var("MOLO_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let api_key = std::env::var("MOLO_API_KEY").unwrap_or_default();
    let model = std::env::var("MOLO_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
    let mode = match std::env::args().nth(1).as_deref() {
        Some("chat") => Mode::Chat,
        _ => Mode::Stream,
    };

    let provider = OpenAiProvider::new(base_url, api_key, model);

    // Convenient assembly: the `react_agent!` macro auto-boxes and registers the tool list
    // (heterogeneous tools), creating a ToolRegistry internally; the equivalent is
    // `ReActAgent::new(provider, registry, system_prompt)`.
    // Memory defaults to a bounded window (WindowMemory, 128k token budget, auto-trims the
    // oldest rounds in long sessions); use with_memory to swap in a custom Memory.
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

        // The reasoning loop lives inside ReActAgent; here we only consume events for display.
        match mode {
            Mode::Stream => {
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
                        MessageChunk::Done(_) => break,
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
            Mode::Chat => {
                let answer = agent.run(input).await?;
                if !answer.is_empty() {
                    println!("assistant: {answer}");
                }
            }
        }
    }

    Ok(())
}
