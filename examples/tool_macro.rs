//! Tool macro example: define a tool in one shot with
//! [`#[molo::tool]`](molo::tool).
//!
//! Writing a tool by hand takes about 25 lines (struct + schema() + call()
//! boilerplate); `#[molo::tool]` generates all of it from an async function
//! (struct + schema + call, including argument parsing and error conversion).
//! The generated shape:
//! - Arguments: 0 or 1 of any type (a primitive or a
//!   `#[derive(JsonSchema)]` struct) + an optional trailing `&SharedState`
//!   (the macro recognizes it by name and injects it automatically);
//! - Returns `Result<String, ToolError>`;
//! - Attributes: `description` (required), `name` (optional, defaults to the
//!   function name).
//!
//! This example is **self-contained**, needs no API key, just run:
//! `cargo run --example tool_macro`

use molo::agent::{Agent, MessageChunk};
use molo::provider::{FakeProvider, FakeReply};
use molo::tool::{SharedState, ToolError};
use molo::{ToolCall, ToolRegistry, react_agent};

use futures::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;

/// Argument struct (field descriptions go into the JSON Schema, giving the same prompt quality as hand-written).
#[derive(Debug, Deserialize, JsonSchema)]
struct CalcArgs {
    /// The math expression to evaluate, e.g. "1 + 2 * 3".
    #[schemars(description = "The math expression to evaluate, e.g. \"1 + 2 * 3\"")]
    expression: String,
}

/// Struct argument: one macro line generates the tool (registered name = function name).
#[molo::tool(description = "Evaluates a math expression; supports basic arithmetic and parentheses")]
async fn calculator(args: CalcArgs) -> Result<String, ToolError> {
    let value =
        evalexpr::eval(&args.expression).map_err(|e| ToolError::Execution(e.to_string()))?;
    Ok(value.to_string())
}

/// Primitive argument: automatically wrapped into an object schema; the call reads the corresponding field.
#[molo::tool(description = "Greets the user")]
async fn hello(name: String) -> Result<String, ToolError> {
    Ok(format!("Hello, {name}!"))
}

/// No arguments.
#[molo::tool(description = "Returns a fixed demo text")]
async fn ping() -> Result<String, ToolError> {
    Ok("pong".into())
}

/// Shared state: the macro recognizes `&SharedState` by name and injects it; the tool reads and writes it directly.
#[molo::tool(description = "Accumulates the call count and returns the current counter")]
async fn counter(state: &SharedState) -> Result<String, ToolError> {
    state.with_mut::<usize>(|n| *n += 1);
    Ok(format!("count={}", state.get::<usize>().unwrap_or(0)))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Shared state: held by the application and mounted on the agent (the counter tool accesses the same instance through its parameter).
    let state = SharedState::new();
    state.insert(0usize);

    // Macro-generated structs register directly; the react_agent! tool-list arm works the same way (heterogeneous).
    let mut registry = ToolRegistry::new();
    registry
        .register(Calculator)
        .register(Hello)
        .register(Ping)
        .register(Counter);

    // Script: the model first requests calculator / counter, then hello, and finally answers directly.
    let fake = FakeProvider::new([
        FakeReply::ToolCalls {
            content: "".into(),
            calls: vec![
                ToolCall {
                    id: "c1".into(),
                    name: "calculator".into(),
                    arguments: r#"{"expression":"(1 + 2) * 3"}"#.into(),
                },
                ToolCall {
                    id: "c2".into(),
                    name: "counter".into(),
                    arguments: "{}".into(),
                },
            ],
        },
        FakeReply::ToolCalls {
            content: "".into(),
            calls: vec![ToolCall {
                id: "c3".into(),
                name: "hello".into(),
                arguments: r#"{"name":"molo"}"#.into(),
            }],
        },
        FakeReply::Text("task complete".into()),
    ]);
    let mut agent = react_agent!(fake, registry, "You are an assistant").with_state(state);

    // Streaming consumption: tool process shown one by one.
    let mut stream = agent.run_stream("start the task").await?;
    while let Some(event) = stream.next().await {
        match event? {
            MessageChunk::ToolResult { name, content, .. } => println!("→ tool {name}: {content}"),
            MessageChunk::Delta(delta) => println!("→ reply: {delta}"),
            _ => {}
        }
    }
    Ok(())
}
