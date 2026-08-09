//! Trace example: rendering the execution chain to the console.
//!
//! molo's loops emit tracing spans at fixed points (`agent.run` /
//! `llm_request` / `tool`, two levels, grouped by the round attribute);
//! whether and how much is recorded is entirely decided by the **user-side
//! subscriber** — this example renders the console with `tracing-subscriber`;
//! production can swap in `tracing-opentelemetry` to bridge OTLP (Langfuse /
//! Honeycomb / a self-hosted collector). Without a subscriber, spans cost
//! almost nothing, and the trace can be ignored entirely (the same idea as a
//! Java framework logging internally while users filter or ignore by level).
//!
//! All spans carry a `run.id` attribute: the same id carried by `RunStarted`
//! in the event stream — the correlation key between observed data and
//! business events. Levels: INFO shows only the run skeleton; debug collects
//! all details (the filtering syntax is in the EnvFilter in main).
//!
//! This example is **self-contained**, needs no API key, just run:
//! `cargo run --example trace`

use molo::agent::{Agent, MessageChunk};
use molo::provider::{FakeProvider, FakeReply};
use molo::tool::{ToolError, ToolRegistry};
use molo::{ToolCall, Usage, react_agent};

use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::FmtSpan;

/// Addition arguments (struct argument: field descriptions go into the JSON Schema).
#[derive(Debug, Deserialize, JsonSchema)]
struct AddArgs {
    /// The augend.
    a: i32,
    /// The addend.
    b: i32,
}

/// Addition tool (macro-defined; arguments are always an object on the wire).
#[molo::tool(description = "Adds two integers")]
async fn add(args: AddArgs) -> Result<String, ToolError> {
    Ok((args.a + args.b).to_string())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // User-chosen subscriber: console rendering, capturing all molo debug-level spans
    // (llm_request / tool are DEBUG, agent.run is INFO; switch to "molo=info" to see only
    // the run skeleton, or "off" to ignore entirely). fmt does not print spans by default,
    // so span events must be enabled explicitly (FmtSpan::FULL = enter + exit + close +
    // timing); for tree-shaped rendering, swap in tracing-tree.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("molo=debug"))
        .with_span_events(FmtSpan::FULL)
        .init();

    let mut registry = ToolRegistry::new();
    registry.register(Add);

    // Script: the first run has two rounds (tool round + direct answer), the second is one streaming round,
    // the third is one round; usage is injected to show both the observability side (span fields)
    // and the business side (Done's RunSummary).
    let fake = FakeProvider::new([
        FakeReply::WithUsage {
            reply: Box::new(FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "add".into(),
                    arguments: r#"{"a":1,"b":2}"#.into(),
                }],
            }),
            usage: Usage::new(10, 2),
        },
        FakeReply::text_with_usage("the answer is 3", Usage::new(20, 5)),
        FakeReply::text_with_usage("calculated again", Usage::new(5, 3)),
        FakeReply::text_with_usage("anything else", Usage::new(2, 1)),
    ]);
    let mut agent = react_agent!(fake, registry, "You are a math assistant");

    // Non-streaming run: one agent.run for the whole call, containing two llm_request
    // spans (tool round + answer round), distinguished by the round attribute.
    println!("== non-streaming run ==");
    agent.run("what is 1+2?").await?;

    // Streaming run: the same span structure; agent.run covers the whole stream consumption, with text dispatched token by token.
    println!("== streaming run ==");
    let mut stream = agent.run_stream("calculate again").await?;
    while let Some(event) = stream.next().await {
        if let MessageChunk::Delta(text) = event? {
            print!("{text}");
        }
    }
    println!();
    drop(stream); // release the borrow of the agent so the next run can start

    // Third run: note the agent.run run.id increments (same source as the event stream's RunStarted).
    println!("== third run ==");
    agent.run("anything else?").await?;
    Ok(())
}
