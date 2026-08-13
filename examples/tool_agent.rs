//! Tool Agent example: demonstrates tool calls and the
//! "model requests a tool → execute → feed the result back" loop.
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
//! - `cargo run --example tool_agent -- stream` (default): streaming chat,
//!   text printed token by token, showing Provider::stream_chat's event stream
//!   and streaming tool calls;
//! - `cargo run --example tool_agent -- chat`: non-streaming chat, the whole
//!   turn returned at once, showing Provider::chat's tool calls.
//!
//! Both modes share the same tools, reasoning loop, and result feeding; only
//! the "start a chat turn" path differs. Try asking: "What is (1 + 2) * 3?";
//! the model will request the calculator tool, the Agent executes it and feeds
//! the result back, and the model gives the final answer based on it.
//!
//! The reasoning loop in this example (chat → execute tool calls and feed
//! back → until the model answers directly) is a **reference implementation**:
//! the framework ships the same semantics in
//! [`ReActAgent`](molo::ReActAgent) (see `examples/react_agent.rs`), with a
//! round limit / cooperative cancellation / Usage aggregation / event channel /
//! Trace included; prefer the built-in implementation when you need these.

use molo::provider::OpenAiProvider;
use molo::tool::{SharedState, Tool, ToolContext, ToolError, ToolOutput, ToolResult, ToolSchema};
use molo::{ChatRequest, Message, Provider, RunContext, StreamEvent, ToolCall};

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

    // The Agent holds the tool list; the model sees them through their schemas.
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(Calculator)];
    let tool_schemas: Vec<ToolSchema> = tools.iter().map(|t| t.schema()).collect();

    let mut messages = vec![Message::system(
        "You are a helpful assistant. Use the calculator tool for calculations instead of doing math in your head.",
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
                let content = run_tool(&tools, &call.name, &call.arguments).await;
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

/// Find the tool by name and execute it; a missing tool or an execution failure is fed back as text so the model can decide the next step.
async fn run_tool(tools: &[Box<dyn Tool>], name: &str, arguments: &str) -> String {
    let Some(tool) = tools.iter().find(|t| t.schema().name == name) else {
        return format!("tool not found: {name}");
    };
    let args = match serde_json::from_str(arguments) {
        Ok(value) => value,
        Err(e) => return format!("arguments are not valid JSON: {e}"),
    };
    let run = RunContext::new("tool-agent-example");
    let state = SharedState::new();
    let context = ToolContext::new(&run, &state, "manual-call", name);
    match tool.call(args, context).await {
        Ok(result) => result.to_string(),
        Err(e) => format!("tool error: {e}"),
    }
}
