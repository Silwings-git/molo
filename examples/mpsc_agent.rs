//! MpscAgent example: a real model + tool-call loop + two-Agent conversation
//! over MpscChannel.
//!
//! Demonstrates: the main Agent's ask_expert tool sends the question through
//! an [`MpscChannel`](molo::MpscChannel) to **another real-model Agent** (an
//! expert with its own loop); the expert's answer comes back through the
//! channel and the main loop continues without noticing.
//!
//! # Choosing: who is on the other end of the channel
//!
//! - The responder is a human (CLI): see `examples/confirm_agent.rs`;
//! - The responder is a model (expert): this example.
//!   MpscChannel is just a conversation pipe; what is plugged into each end is
//!   up to the application.
//!
//! On startup the example reads configuration from `.env`
//! (MOLO_API_KEY / MOLO_BASE_URL / MOLO_MODEL), same as
//! `examples/tool_agent.rs`; run:
//! `cargo run --example mpsc_agent`
//!
//! Try asking: "Why is the sky blue? Please consult the expert with the
//! ask_expert tool";
//! you will see the main Agent call ask_expert → the expert Agent reasons on
//! its own → the reply comes back to the main Agent.

use futures::stream::StreamExt;
use molo::provider::OpenAiProvider;
use molo::tool::{SharedState, Tool, ToolContext, ToolError, ToolOutput, ToolResult, ToolSchema};
use molo::{
    ChatRequest, Message, MessageChannel, MpscChannel, Provider, StreamEvent, ToolCall,
    ToolRegistry,
};
use schemars::JsonSchema;
use serde::Deserialize;
use std::io::Write;
use std::sync::Arc;

/// Arguments for the ask_expert tool.
#[derive(Debug, Deserialize, JsonSchema)]
struct AskArgs {
    /// The question to ask the expert.
    #[schemars(description = "The question to ask the expert")]
    question: String,
}

/// Expert-consulting tool: sends the question to the expert agent through an MpscChannel and waits for its answer.
struct AskExpertTool {
    channel: Arc<dyn MessageChannel>,
}

#[async_trait::async_trait]
impl Tool for AskExpertTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "ask_expert",
            "Hands the question to the expert agent for an answer; returns the expert's reply.",
            serde_json::to_value(schemars::schema_for!(AskArgs))
                .expect("tool schema must serialize"),
        )
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        _context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let args: AskArgs = serde_json::from_value(arguments)?;
        self.channel
            .ask(&args.question)
            .await
            .map(ToolOutput::text)
            .map(Into::into)
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

    let provider = OpenAiProvider::new(base_url.clone(), api_key.clone(), model.clone());

    // Expert agent: an independent real-model loop that answers questions as they arrive (no tool calls).
    let (expert_ask, expert_side) = MpscChannel::pair();
    let expert_provider = OpenAiProvider::new(base_url, api_key, model);
    tokio::spawn(async move { expert_loop(&expert_provider, expert_side).await });

    // Main agent assembly: calculator + ask_expert (embedding an MpscChannel).
    let mut registry = ToolRegistry::new();
    registry.register(Calculator).register(AskExpertTool {
        channel: Arc::new(expert_ask),
    });
    let tool_schemas = registry.schemas();
    let run_context = molo::RunContext::new("mpsc-agent-example");
    let state = SharedState::new();

    let mut messages = vec![Message::system(
        "You are a helpful assistant. Use the calculator tool when you need to \
         calculate; when you run into a specialized question (science, history, \
         etc.), use the ask_expert tool to consult the expert.",
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
                    .call(&call, &run_context, &state)
                    .await
                    .map(|result| result.to_string())
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

/// Expert Agent loop: receive a question → answer with a real model → reply through the channel, and repeat.
async fn expert_loop(
    provider: &OpenAiProvider,
    channel: MpscChannel,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut messages = vec![Message::system(
        "You are a knowledgeable expert; answer the user's questions directly and do not use any tools.",
    )];
    loop {
        let incoming = channel.recv().await?;
        println!("  [expert] received question: {}", incoming.text());
        messages.push(Message::user(incoming.text()));
        let response = provider
            .chat(ChatRequest {
                messages: messages.clone(),
                ..Default::default()
            })
            .await?;
        let content = match response.message {
            Message::Assistant { content, .. } => content,
            _ => unreachable!("the reply must be an Assistant message by contract"),
        };
        println!("  [expert] answer: {content}");
        messages.push(Message::Assistant {
            content: content.clone(),
            reasoning: None,
            tool_calls: vec![],
        });
        incoming.reply(content)?;
    }
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
