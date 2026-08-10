//! EventChannel example: observation channel — the Agent pushes process events
//! internally, and the outside subscribes.
//!
//! The channel family: [`MessageChannel`](molo::MessageChannel) = conversation
//! (ask/notify, between humans and Agents);
//! [`EventChannel`](molo::event_channel::EventChannel) = observation
//! (publish/subscribe, subscribed by the environment / UI). After a ReActAgent
//! mounts a channel via
//! [`with_event_channel`](molo::agent::ReActAgent::with_event_channel), both
//! streaming and non-streaming runs push their own event sets: `RunStarted` /
//! `Delta` / `Reasoning` / `ToolStarted` / `ToolCompleted` / `RunEnded` (each
//! Agent defines its own event types; the framework does not anticipate the
//! variants).
//!
//! # Choosing: BroadcastEventChannel vs. MpscEventChannel
//!
//! - [`BroadcastEventChannel`](molo::event_channel::BroadcastEventChannel):
//!   multiple subscribers, one copy each; slow subscribers skip missed events
//!   (drop the oldest, get the newest);
//! - [`MpscEventChannel`](molo::MpscEventChannel): single consumer, strictly
//!   ordered and lossless within capacity; drops new events when full. Both
//!   share the same publish-side API — swapping is a one-line change.
//!
//! This example is self-contained (driven by FakeProvider), needs no API key,
//! just run:
//! `cargo run --example event_channel`
//!
//! Demonstrates:
//! 1. Mount BroadcastEventChannel → subscribe → UI-style consumption
//!    (`as_any().downcast_ref` handles known events precisely, unknown events
//!    fall back to `name()`);
//! 2. Tool success / failure is distinguished in `ToolCompleted.result`
//!    (Ok/Err, so a UI can render green/red);
//! 3. `RunEnded` carries the run summary and outcome (normal / error);
//! 4. MpscEventChannel (single consumer) has the same API; swapping is a
//!    one-line change.

use molo::AgentError;
use molo::agent::{Agent, AgentEvent, ReActEvent};
use molo::event_channel::{BroadcastEventChannel, EventChannel};
use molo::tool::{SharedState, Tool, ToolError, ToolSchema};

/// Echo tool (success path).
struct Echo;

#[async_trait::async_trait]
impl Tool for Echo {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "echo".into(),
            description: "Echoes text back".into(),
            parameters: serde_json::json!({}),
        }
    }
    async fn call(
        &self,
        _arguments: serde_json::Value,
        _state: &SharedState,
    ) -> Result<String, ToolError> {
        Ok("Echo: hello".into())
    }
}

/// A tool that always fails (failure path).
struct Boom;

#[async_trait::async_trait]
impl Tool for Boom {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "boom".into(),
            description: "A tool that always fails".into(),
            parameters: serde_json::json!({}),
        }
    }
    async fn call(
        &self,
        _arguments: serde_json::Value,
        _state: &SharedState,
    ) -> Result<String, ToolError> {
        Err(ToolError::Execution("internal error".into()))
    }
}

/// UI-style consumption: a single downcast to the `ReActEvent` enum, rendered
/// by an exhaustive match; event types of other Agents (unknown) fall back to
/// name().
fn render(event: &dyn AgentEvent) {
    match event.as_any().downcast_ref::<ReActEvent>() {
        // run_id is the correlation key aligning the event stream with observed data: trace spans carry a run.id attribute with the same value.
        Some(ReActEvent::RunStarted { run_id, input }) => {
            println!("▶ run started [{run_id}]: {input}")
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
                "■ run {outcome}: {} rounds / {} tool calls",
                summary.rounds, summary.tool_calls
            );
        }
        None => println!("  [unknown event {}]", event.name()),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Tools: echo (success) + boom (failure) — two calls in the same round, one succeeds and one fails.
    let mut registry = molo::ToolRegistry::new();
    registry.register(Echo).register(Boom);

    let fake = molo::FakeProvider::new([
        molo::FakeReply::ToolCalls {
            content: "Let me try these two tools".into(),
            calls: vec![
                molo::ToolCall {
                    id: "c1".into(),
                    name: "echo".into(),
                    arguments: r#"{"text":"hello"}"#.into(),
                },
                molo::ToolCall {
                    id: "c2".into(),
                    name: "boom".into(),
                    arguments: "{}".into(),
                },
            ],
        },
        molo::FakeReply::Text("OK, echo succeeded but boom errored".into()),
    ]);

    // Environment side: create the channel → inject it into the agent → subscribe.
    let channel = BroadcastEventChannel::new(256);
    let mut rx = channel.subscribe();
    let mut agent =
        molo::react_agent!(fake, registry, "You are an assistant").with_event_channel(channel);

    println!("== subscribing to the EventChannel to observe a run ==");
    agent.run("execute the two tools").await?;

    // After the run, drop the agent (the channel's last holder) so the final events (including RunEnded) become available.
    drop(agent);
    while let Some(event) = rx.recv().await {
        render(&*event);
    }
    println!();

    // MpscEventChannel: single consumer, same API (also non-blocking on the publish side).
    let channel = molo::MpscEventChannel::new(64);
    let mut rx = channel.subscribe();
    let mut agent = molo::react_agent!(
        molo::FakeProvider::new([molo::FakeReply::Text("hello".into())]),
        "You are an assistant",
    )
    .with_event_channel(channel);
    agent.run("are you there").await?;
    drop(agent);
    println!(
        "== MpscEventChannel (single consumer, strictly ordered and lossless within capacity) =="
    );
    while let Some(event) = rx.recv().await {
        println!("  [{}]", event.name());
    }
    Ok(())
}
