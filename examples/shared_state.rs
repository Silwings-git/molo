//! SharedState example: three ways to use shared state.
//!
//! [`SharedState`](molo::SharedState) is a type-safe heterogeneous container:
//! values are stored and retrieved by type, and multiple values can coexist.
//! Tools receive it **injected at call time** via the `state` parameter of
//! [`Tool::call`](molo::Tool::call) — the caller holds the state and passes it
//! on every call; an Agent mounts it via
//! [`with_state`](molo::agent::ReActAgent::with_state), and the application
//! reads and writes it across runs through `agent.state`. This example
//! demonstrates:
//! - Flow between tools: multiple tools in the same round share one instance
//!   (counter accumulation, shared session info);
//! - Application reads/writes across runs: the application updates the state,
//!   and tools in the next round see the new value;
//! - Sharing across agents: cloning is cheap (Arc), so multiple Agents can
//!   hold the same instance.
//!
//! This example is **self-contained**, needs no API key, just run:
//! `cargo run --example shared_state`

use molo::agent::{Agent, MessageChunk};
use molo::provider::{FakeProvider, FakeReply};
use molo::tool::{SharedState, Tool, ToolError, ToolSchema};
use molo::{ToolCall, ToolRegistry, react_agent};

use futures::StreamExt;

/// Session info (an application-defined type stored in the shared state).
#[derive(Debug, Clone, PartialEq)]
struct Session {
    user: String,
}

/// Counter tool: increments the counter in the shared state by 1 on every call and returns the current value.
struct Counter;

#[async_trait::async_trait]
impl Tool for Counter {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "counter".into(),
            description: "Accumulates the call count and returns the current counter.".into(),
            parameters: serde_json::json!({}),
        }
    }

    async fn call(
        &self,
        _arguments: serde_json::Value,
        state: &SharedState,
    ) -> Result<String, ToolError> {
        state.with_mut::<usize>(|n| *n += 1);
        Ok(format!("count={}", state.get::<usize>().unwrap_or(0)))
    }
}

/// Session tool: reads the session info from the shared state (shares the same instance as counter).
struct SessionTool;

#[async_trait::async_trait]
impl Tool for SessionTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "session".into(),
            description: "Returns the current session user.".into(),
            parameters: serde_json::json!({}),
        }
    }

    async fn call(
        &self,
        _arguments: serde_json::Value,
        state: &SharedState,
    ) -> Result<String, ToolError> {
        Ok(match state.get::<Session>() {
            Some(s) => format!("user={}", s.user),
            None => "user=unknown".into(),
        })
    }
}

/// Consumes one turn's streaming events and prints tool process and the final reply.
async fn run_and_show(
    agent: &mut molo::ReActAgent,
    input: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut stream = agent.run_stream(input).await?;
    while let Some(event) = stream.next().await {
        match event? {
            MessageChunk::ToolResult { name, content, .. } => {
                println!("   → tool {name}: {content}")
            }
            MessageChunk::Delta(delta) => println!("   → reply: {delta}"),
            _ => {}
        }
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Shared state: held by the application; tools and Agents read and write through the same instance (Clone is cheap, Arc).
    let state = SharedState::new();
    state.insert(Session {
        user: "alice".into(),
    });
    state.insert(0usize);

    let mut registry = ToolRegistry::new();
    registry.register(Counter).register(SessionTool);

    // 1. Flow between tools: in the same round, counter writes the count and session reads the session (same instance).
    let fake = FakeProvider::new([
        FakeReply::ToolCalls {
            content: "".into(),
            calls: vec![
                ToolCall {
                    id: "c1".into(),
                    name: "counter".into(),
                    arguments: "{}".into(),
                },
                ToolCall {
                    id: "c2".into(),
                    name: "session".into(),
                    arguments: "{}".into(),
                },
            ],
        },
        FakeReply::Text("task complete".into()),
    ]);
    let mut agent = react_agent!(fake, registry, "You are an assistant").with_state(state.clone());
    println!("1. flow between tools (two tools in one round sharing the state):");
    run_and_show(&mut agent, "execute the task").await?;
    println!(
        "   read from the application side: count={} (written by counter)",
        state.get::<usize>().unwrap_or(0)
    );

    // 2. Application reads/writes across runs: the application updates the session, and tools in the next round see the new value.
    state.insert(Session { user: "bob".into() });
    let fake = FakeProvider::new([
        FakeReply::ToolCalls {
            content: "".into(),
            calls: vec![ToolCall {
                id: "c3".into(),
                name: "session".into(),
                arguments: "{}".into(),
            }],
        },
        FakeReply::Text("understood".into()),
    ]);
    let mut session_registry = ToolRegistry::new();
    session_registry.register(SessionTool);
    let mut agent = react_agent!(fake, session_registry, "You are an assistant").with_state(state.clone());
    println!("2. application writes across runs (a new agent reuses the same state):");
    run_and_show(&mut agent, "query the current user").await?;

    // 3. Sharing across agents: another agent holds the same instance; the application reads the final state directly.
    println!(
        "3. final state on the application side: user={:?}, count={} (counter was called once in section 1; the count is kept)",
        state.get::<Session>().unwrap(),
        state.get::<usize>().unwrap_or(0)
    );
    println!("   → shared by cloning: any tool / Agent holding state.clone() shares the same instance");

    Ok(())
}
