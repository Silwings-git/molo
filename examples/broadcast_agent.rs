//! BroadcastAgent example: a real model + tool-call loop + broadcast
//! notifications via BroadcastChannel.
//!
//! Demonstrates: the main Agent's notify_workers tool broadcasts a message to
//! all subscribers through a [`BroadcastChannel`](molo::BroadcastChannel)
//! (programmatic workers, each receiving independently) — one-way notification,
//! no reply is awaited.
//!
//! # Choosing: when to use broadcast
//!
//! - Notification without waiting for a reply, one copy to every subscriber →
//!   broadcast (BroadcastChannel);
//! - One-on-one question answering (e.g. consulting an expert) → see
//!   `examples/mpsc_agent.rs`;
//! - Waiting for human confirmation → see `examples/confirm_agent.rs`.
//!
//! On startup the example reads configuration from `.env`
//! (MOLO_API_KEY / MOLO_BASE_URL / MOLO_MODEL), same as
//! `examples/tool_agent.rs`; run:
//! `cargo run --example broadcast_agent`
//!
//! Try asking: "Notify all workers to prepare for a new release";
//! you will see the main Agent call notify_workers, and both workers each
//! receive the broadcast.

use futures::stream::StreamExt;
use molo::provider::OpenAiProvider;
use molo::tool::{SharedState, Tool, ToolError, ToolSchema};
use molo::{
    BroadcastChannel, BroadcastReceiver, ChatRequest, Message, MessageChannel, Provider,
    StreamEvent, ToolCall, ToolRegistry,
};
use schemars::JsonSchema;
use serde::Deserialize;
use std::io::Write;

/// Arguments for the notify_workers tool.
#[derive(Debug, Deserialize, JsonSchema)]
struct NotifyArgs {
    /// The message to broadcast to all workers.
    #[schemars(description = "The message to broadcast to all workers")]
    message: String,
}

/// Broadcast notification tool: broadcasts a message to all subscribers through a BroadcastChannel (one-way, no reply awaited).
struct NotifyWorkersTool {
    channel: BroadcastChannel,
}

#[async_trait::async_trait]
impl Tool for NotifyWorkersTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "notify_workers".into(),
            description: "Broadcasts a message to all workers; returns success when broadcast."
                .into(),
            parameters: serde_json::to_value(schemars::schema_for!(NotifyArgs))
                .expect("tool schema must serialize"),
        }
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        _state: &SharedState,
    ) -> Result<String, ToolError> {
        let args: NotifyArgs = serde_json::from_value(arguments)?;
        self.channel
            .notify(&args.message)
            .await
            .map(|_| "Broadcast successful, all workers received the message.".to_string())
            .map_err(|e| ToolError::Execution(e.to_string()))
    }
}

/// The result of one chat turn: text reply and tool calls, with reasoning already attached to the corresponding message.
struct Turn {
    messages: Vec<Message>,
}

/// How the chat turn is run.
enum Mode {
    /// Provider::stream_chat: text printed token by token, tool calls received as a whole at the end of the turn.
    Stream,
    /// Provider::chat: the full turn reply returned at once.
    Chat,
}

/// Maximum number of consecutive tool rounds, preventing an infinite loop of repeated tool requests.
const MAX_TOOL_ROUNDS: usize = 10;

/// Programmatic worker: keeps receiving broadcasts and prints them (the example's "subscriber", not a model).
async fn worker(name: &'static str, receiver: BroadcastReceiver) {
    while let Ok(incoming) = receiver.recv().await {
        println!("  [{name}] received: {}", incoming.text());
    }
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

    // Broadcast channel + two worker subscribers.
    let broadcast = BroadcastChannel::new(16);
    let worker_a = broadcast.subscribe();
    let worker_b = broadcast.subscribe();
    tokio::spawn(worker("worker A", worker_a));
    tokio::spawn(worker("worker B", worker_b));

    // Main agent assembly: calculator + notify_workers (embedding a BroadcastChannel).
    let mut registry = ToolRegistry::new();
    registry
        .register(Calculator)
        .register(NotifyWorkersTool { channel: broadcast });
    let tool_schemas = registry.schemas();

    let mut messages = vec![Message::system(
        "You are a release-coordination assistant. Use the calculator tool when \
         you need to calculate; use the notify_workers tool when you need to \
         notify all workers.",
    )];
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

        // Reasoning loop: run a chat turn, execute tool calls the model requests and feed results back, until the model answers with text directly.
        let mut tool_rounds = 0;
        loop {
            tool_rounds += 1;
            let turn = run_turn(&provider, &messages, &tool_schemas, &mode).await?;

            // Whether this turn requested tools (all requests of a turn are in a single Assistant message).
            let calls: Vec<ToolCall> = turn
                .messages
                .iter()
                .flat_map(|m| match m {
                    Message::Assistant { tool_calls, .. } => tool_calls.to_vec(),
                    _ => Vec::new(),
                })
                .collect();

            if calls.is_empty() {
                // The model answered directly: merge into history (including reasoning) and wait for the next user input.
                messages.extend(turn.messages);
                break;
            }
            if tool_rounds >= MAX_TOOL_ROUNDS {
                println!("(reached max tool rounds {MAX_TOOL_ROUNDS}, stopping tool calls)");
                messages.extend(turn.messages);
                break;
            }

            // Tool round: merge the whole turn's messages (including reasoning) into history, then feed each execution result back right after.
            messages.extend(turn.messages);
            for call in calls {
                println!("  → calling {}, arguments: {}", call.name, call.arguments);
                // Err's Display is "error as text"; still record it and feed it back as usual.
                let content = registry
                    .call(&call.name, &call.arguments, &SharedState::new())
                    .await
                    .unwrap_or_else(|e| e.to_string());
                println!("  → {} returned: {content}", call.name);
                messages.push(Message::ToolResult {
                    id: call.id,
                    content,
                });
            }
        }
    }

    Ok(())
}

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

/// Run one chat turn and return this turn's messages (reasoning attached to the message); the `assistant: ` prefix is printed here.
async fn run_turn(
    provider: &OpenAiProvider,
    messages: &[Message],
    tools: &[ToolSchema],
    mode: &Mode,
) -> Result<Turn, Box<dyn std::error::Error>> {
    let request = ChatRequest {
        messages: messages.to_vec(),
        tools: tools.to_vec(),
        ..Default::default()
    };
    match mode {
        Mode::Stream => {
            let mut stream = provider.stream_chat(request).await?;
            let mut text = String::new();
            let mut reasoning = String::new();
            let mut calls = Vec::new();
            // The prefix is delayed until the first text delta: pure tool turns do not show an empty "assistant: ".
            let mut prefix_printed = false;
            while let Some(event) = stream.next().await {
                match event? {
                    StreamEvent::Delta(delta) => {
                        if !prefix_printed {
                            print!("assistant: ");
                            std::io::stdout().flush()?;
                            prefix_printed = true;
                        }
                        print!("{delta}");
                        std::io::stdout().flush()?;
                        text.push_str(&delta);
                    }
                    StreamEvent::Reasoning(chunk) => reasoning.push_str(&chunk),
                    StreamEvent::ToolCall {
                        id,
                        name,
                        arguments,
                    } => {
                        calls.push(ToolCall {
                            id,
                            name,
                            arguments,
                        });
                    }
                    StreamEvent::Done { .. } => {}
                    // Unknown variant (reserved for non_exhaustive extensions): silently ignore
                    _ => {}
                }
            }
            println!();
            Ok(Turn {
                messages: assemble_turn(text, reasoning, calls),
            })
        }
        Mode::Chat => {
            let response = provider.chat(request).await?;
            // This turn's reply is always a single Assistant message; text is printed, and reasoning and tool calls are returned with the message.
            let (content, reasoning, tool_calls) = match response.message {
                Message::Assistant {
                    content,
                    reasoning,
                    tool_calls,
                } => (content, reasoning, tool_calls),
                _ => unreachable!("the reply must be an Assistant message by contract"),
            };
            if !content.is_empty() {
                println!("assistant: {content}");
            }
            Ok(Turn {
                messages: vec![Message::Assistant {
                    content,
                    reasoning,
                    tool_calls,
                }],
            })
        }
    }
}

/// Assemble one turn's streaming events into messages: text, reasoning, and
/// tool calls are combined into **one** Assistant message, mirroring the wire
/// structure so the Provider carries it back unchanged in the history.
fn assemble_turn(text: String, reasoning: String, calls: Vec<ToolCall>) -> Vec<Message> {
    let mut messages = Vec::new();
    if !text.is_empty() || !calls.is_empty() || !reasoning.is_empty() {
        messages.push(Message::Assistant {
            content: text,
            reasoning: (!reasoning.is_empty()).then_some(reasoning),
            tool_calls: calls,
        });
    }
    messages
}
