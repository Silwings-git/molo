//! EventChannel Agent example (real model): subscribes to the observation
//! channel and renders the Agent's process in real time.
//!
//! Pairs with `examples/event_channel.rs` (self-contained) — this example is
//! driven by a real Provider and demonstrates **decoupled observation**: the
//! subscription task runs concurrently with the run — the Agent runs in the
//! background, events are pushed to the channel in real time, and the UI (a
//! terminal here) renders them one by one without blocking each other.
//!
//! Highlights:
//! - Text deltas / reasoning / tool started-completed (including success and
//!   failure) / run summary all arrive via the channel; `run_stream` only
//!   drives the run (its output is ignored — text is already rendered in the
//!   `Delta` events);
//! - Multi-turn sessions: the channel **persists**, `RunEnded` marks each
//!   turn's boundary, and the Agent's memory is kept across turns;
//! - Swap in [`MpscEventChannel`](molo::MpscEventChannel) (single consumer) to
//!   experience the same API.
//!
//! On startup the example reads configuration from `.env` (copy `.example.env`
//! to `.env` and fill in real values); environment variables can also override
//! directly:
//! - MOLO_API_KEY  : API key; may be left empty for local endpoints without
//!   auth (e.g. Ollama)
//! - MOLO_BASE_URL : OpenAI-compatible endpoint, default https://api.openai.com/v1
//! - MOLO_MODEL    : model name, default gpt-4o-mini
//!
//! Run: `cargo run --example event_channel_agent`
//! Try asking: "What is (1 + 2) * 3?" — the model will request the calculator
//! tool, and in the event stream you can see tool started → completed → final
//! summary.
//! Type exit / quit / Ctrl-D to quit.

use molo::AgentError;
use molo::agent::{Agent, AgentEvent, MessageChunk, ReActEvent};
use molo::event_channel::{BroadcastEventChannel, EventChannel};
use molo::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolResult, ToolSchema};

use futures::StreamExt;
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

/// UI-style rendering: a single downcast to the `ReActEvent` enum, rendered by
/// an exhaustive match (isomorphic to `examples/event_channel.rs`); unknown
/// event types fall back to name().
fn render(event: &dyn AgentEvent) {
    match event.as_any().downcast_ref::<ReActEvent>() {
        // run_id is the correlation key aligning the event stream with observed data: trace spans carry a run.id attribute with the same value.
        Some(ReActEvent::RunStarted { run_id, input }) => {
            println!("▶ run started [{run_id}]: {input:?}")
        }
        Some(ReActEvent::Delta { text }) => print!("{text}"), // text delta, printed as it arrives
        Some(ReActEvent::Reasoning { text }) => println!("\n  [reasoning] {text}"),
        Some(ReActEvent::ToolStarted {
            id,
            name,
            arguments,
        }) => {
            println!("  → tool {name} started (call {id}): arguments {arguments}")
        }
        Some(ReActEvent::ToolCompleted { name, result, .. }) => match result {
            Ok(text) => println!("  ✓ tool {name} completed: {text}"),
            Err(err) => println!("  ✗ tool {name} failed: {err}"),
        },
        Some(ReActEvent::RunEnded { summary, error }) => {
            let outcome = match error {
                None => "ended normally",
                Some(AgentError::Cancelled) => "cancelled",
                Some(_) => "errored",
            };
            println!(
                "\n■ run {outcome}: {} rounds / {} tool calls",
                summary.rounds, summary.tool_calls
            );
        }
        None => println!("  [unknown event {}]", event.name()),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok(); // load .env; silently ignore if missing

    let base_url =
        std::env::var("MOLO_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let api_key = std::env::var("MOLO_API_KEY").unwrap_or_default();
    let model = std::env::var("MOLO_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());
    let provider = molo::provider::OpenAiProvider::new(base_url, api_key, model);

    let mut registry = molo::ToolRegistry::new();
    registry.register(Calculator);

    // The environment creates the channel → injects it into the agent → subscribes (the channel goes into the agent; the receiver stays on the environment side).
    let channel = BroadcastEventChannel::new(256);
    let mut rx = channel.subscribe();
    let mut agent = molo::react_agent!(provider, registry, "You are a helpful assistant")
        .with_event_channel(channel);

    // The subscription task runs concurrently with the run: the agent runs in the background, events are pushed in real time, and the UI renders them one by one.
    let render_task = tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            render(&*event);
            let _ = std::io::stdout().flush();
        }
    });

    let mut input = String::new();
    loop {
        input.clear();
        println!("\nuser:");
        let read = std::io::stdin().read_line(&mut input)?;
        if read == 0 {
            break; // Ctrl-D
        }
        let input = input.trim();
        if input.is_empty() || matches!(input, "exit" | "quit") {
            break;
        }

        // The stream only drives the run (text is already rendered in the channel's Delta events); Done closes it out.
        let mut stream = agent.run_stream(input).await?;
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(MessageChunk::Done(_) | MessageChunk::Cancelled) => break,
                Ok(_) => {}
                Err(e) => return Err(e.into()),
            }
        }
    }

    // Session over: drop the agent (the channel's last holder) → the channel closes → the render task wraps up.
    drop(agent);
    render_task.await?;
    Ok(())
}
