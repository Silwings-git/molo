//! MessageChannel example: demonstrates four channel implementations
//! (Cli / Mpsc / Broadcast / Watch) and the injection shape of "a tool holding
//! the channel when constructed".
//!
//! # Choosing: the four channels
//!
//! - [`CliMessageChannel`](molo::CliMessageChannel): request-response where
//!   the responder is a human; used for tools to confirm with the user;
//! - [`MpscChannel`](molo::MpscChannel): one-on-one Q&A where the responder is
//!   a program (Agent); see `examples/mpsc_agent.rs`;
//! - [`BroadcastChannel`](molo::BroadcastChannel): one-to-many broadcast, no
//!   reply awaited; see `examples/broadcast_agent.rs`;
//! - [`WatchChannel`](molo::WatchChannel): observes changes to the latest
//!   value; takes the newest value rather than all.
//!
//! The first two sections need your terminal input; the later ones complete in
//! memory automatically. The full assembly wired to a real model (the model
//! requests the confirm tool → the tool asks the user) is in
//! `examples/confirm_agent.rs`.
//!
//! This example is self-contained, needs no API key, just run:
//! `cargo run --example message_channel`

use molo::{
    BroadcastChannel, CliMessageChannel, MessageChannel, MpscChannel, RunContext, SharedState,
    Tool, ToolContext, ToolError, ToolOutput, ToolResult, ToolSchema, WatchChannel,
};
use schemars::JsonSchema;
use serde::Deserialize;
use std::sync::Arc;

/// Arguments for the confirm tool.
#[derive(Debug, Deserialize, JsonSchema)]
struct ConfirmArgs {
    /// Description of the operation to confirm, e.g. "delete file x".
    #[schemars(description = "Description of the operation to confirm, e.g. \"delete file x\"")]
    operation: String,
}

/// Confirmation tool: asks the user for confirmation before executing a dangerous operation (the tool holds the channel when constructed).
///
/// Pause-resume happens inside `call`'s `ask`: the model requests this tool →
/// the tool asks the user → the user replies → the tool returns; the Agent
/// loop is completely unaware of the pause.
struct ConfirmTool {
    channel: Arc<dyn MessageChannel>,
}

#[async_trait::async_trait]
impl Tool for ConfirmTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "confirm",
            "Confirms with the user before executing a dangerous operation; only continue if the user replies yes.",
            serde_json::to_value(schemars::schema_for!(ConfirmArgs))
                .expect("tool schema must serialize"),
        )
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        _context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let args: ConfirmArgs = serde_json::from_value(arguments)?;
        let answer = self
            .channel
            .ask(&format!(
                "Confirm executing the operation \"{}\"? Reply yes or no",
                args.operation
            ))
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?;
        if answer == "yes" {
            Ok(ToolOutput::text("Confirmed, continue executing.").into())
        } else {
            Ok(ToolOutput::text("The user declined; the operation was cancelled.").into())
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The channel instance must be shared: multiple tools / calls use the same instance (wrapping stdin separately would make them swallow each other's input).
    let channel: Arc<dyn MessageChannel> = Arc::new(CliMessageChannel::new());

    // 1. notify: one-way notification, no reply awaited (broadcast shape).
    channel.notify("Starting the task...").await?;

    // 2. ask: request-response; with multiple concurrent asks, the channel guarantees only one request is presented at a time.
    let answer = channel
        .ask("Execute a dangerous operation? Reply yes or no")
        .await?;
    println!("you replied: {answer}");

    // 3. The injection shape of a tool holding the channel: ConfirmTool asks internally; execution is pause-resume.
    let confirm = ConfirmTool {
        channel: channel.clone(),
    };
    let run = RunContext::new("message-channel-example");
    let state = SharedState::new();
    let result = confirm
        .call(
            serde_json::json!({ "operation": "delete the entire project directory" }),
            ToolContext::new(&run, &state, "manual-call", "confirm"),
        )
        .await?;
    println!("tool result: {result}");

    // 4. MpscChannel: one-on-one conversation between agents (queue delivery + replies bound to requests).
    let (agent_a, agent_b) = MpscChannel::pair();
    let ask = agent_a.ask("help me calculate 1+2");
    let b_side = async {
        let incoming = agent_b.recv().await?;
        println!("agent B received the question: {}", incoming.text());
        incoming.reply("3".into())
    };
    let (answer, b_result) = tokio::join!(ask, b_side);
    b_result?;
    println!("agent A received the reply: {}", answer?);

    // 5. BroadcastChannel: 1-to-n broadcast notification (every subscriber receives its own copy).
    let broadcast = BroadcastChannel::new(16);
    let worker_a = broadcast.subscribe();
    let worker_b = broadcast.subscribe();
    broadcast
        .notify("all workers, the task is starting")
        .await?;
    let (msg_a, msg_b) = tokio::join!(worker_a.recv(), worker_b.recv());
    println!("worker A received: {}", msg_a?.text());
    println!("worker B received: {}", msg_b?.text());

    // 6. WatchChannel: change notification for the latest value (the observer waits for a change and takes the newest value).
    let watch = WatchChannel::new();
    let observer = watch.subscribe();
    watch.notify("status: running").await?;
    println!("observer received: {}", observer.recv().await?.text());

    Ok(())
}
