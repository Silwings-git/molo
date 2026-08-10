//! Confirmation Agent example: a full assembly of a real model + ToolRegistry +
//! MessageChannel.
//!
//! Demonstrates how molo's building blocks combine into an Agent with "human
//! confirmation":
//! - [`ToolRegistry`](molo::ToolRegistry): holds tools and executes them by
//!   name;
//! - [`MessageChannel`](molo::MessageChannel) (Cli implementation): the tool
//!   `ask`s the user internally — when the model requests the confirm tool, the
//!   Agent loop pauses **without noticing**, waits for the user's terminal
//!   reply, then resumes and feeds the result back to the model.
//!
//! The reasoning loop is the classic "chat → execute tools and feed results
//! back → until the model answers directly", with tool execution going through
//! the ToolRegistry.
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
//! - `cargo run --example confirm_agent -- stream` (default): streaming chat,
//!   text printed token by token;
//! - `cargo run --example confirm_agent -- chat`: non-streaming chat, the whole
//!   turn returned at once.
//!
//! Try asking: "Deleting the entire project directory is a dangerous operation;
//! confirm with the user first, then execute it";
//! the model will request the confirm tool, and you continue after typing
//! yes / no in the terminal.

use futures::stream::StreamExt;
use molo::provider::OpenAiProvider;
use molo::tool::{SharedState, Tool, ToolError, ToolSchema};
use molo::{
    ChatRequest, CliMessageChannel, Message, MessageChannel, Provider, StreamEvent, ToolCall,
    ToolRegistry,
};
use schemars::JsonSchema;
use serde::Deserialize;
use std::io::Write;
use std::sync::Arc;

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
/// the tool asks the user → the user replies → the tool returns. The Agent
/// loop is completely unaware of the pause.
struct ConfirmTool {
    channel: Arc<dyn MessageChannel>,
}

#[async_trait::async_trait]
impl Tool for ConfirmTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "confirm".into(),
            description: "Confirms with the user before executing a dangerous operation; only continue if the user replies yes, otherwise cancel.".into(),
            parameters: serde_json::to_value(schemars::schema_for!(ConfirmArgs))
                .expect("tool schema must serialize"),
        }
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        _state: &SharedState,
    ) -> Result<String, ToolError> {
        let args: ConfirmArgs = serde_json::from_value(arguments)?;
        let answer = self
            .channel
            .ask(&format!(
                "Confirm executing \"{}\"? Reply yes or no",
                args.operation
            ))
            .await
            .map_err(|e| ToolError::Execution(e.to_string()))?;
        if answer == "yes" {
            Ok("The user confirmed; you may proceed.".into())
        } else {
            Ok("The user declined; do not execute.".into())
        }
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

    // Assembly: ToolRegistry holds the tools; ConfirmTool embeds a MessageChannel (Cli).
    // The channel instance is shared: ConfirmTool and any future tools that need to ask share the same one.
    let channel: Arc<dyn MessageChannel> = Arc::new(CliMessageChannel::new());
    let mut registry = ToolRegistry::new();
    registry.register(Calculator).register(ConfirmTool {
        channel: channel.clone(),
    });
    let tool_schemas = registry.schemas();

    let mut messages = vec![Message::system(
        "You are a helpful assistant. Use the calculator tool for calculations \
         instead of doing math in your head; before executing a dangerous \
         operation, you must call the confirm tool to seek the user's \
         confirmation.",
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
