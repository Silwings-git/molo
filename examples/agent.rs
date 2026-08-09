//! Agent example: a hand-written reasoning loop — implementing the
//! [`Agent`](molo::Agent) trait on the application side.
//!
//! For most use cases, the framework's built-in [`ReActAgent`](molo::ReActAgent)
//! is enough (see `examples/react_agent.rs`); this example demonstrates a
//! hand-written loop: "record the user input → run a chat turn → execute the
//! tool calls the model requests and feed the results back → until the model
//! answers directly", and overrides the streaming
//! [`run_stream`](molo::Agent::run_stream) to show token-by-token output and
//! tool-process events.
//!
//! # Choosing: hand-written loop vs. the built-in ReActAgent
//!
//! - Built-in ReActAgent: comes with a tool-round limit, cooperative
//!   cancellation, Usage aggregation, an event channel, and Trace — ready to
//!   use out of the box in most scenarios;
//! - Hand-written loop (this example): full control over the reasoning
//!   process, but you must implement all of the above yourself;
//! - How to decide: if you need any of those capabilities, prefer the built-in
//!   implementation rather than rewriting the loop. How to assemble the
//!   Provider / Memory / Tool and the system prompt is up to the application.
//!
//! On startup the example reads configuration from `.env` (copy `.example.env`
//! to `.env` and fill in real values); environment variables can also override
//! directly:
//! - MOLO_API_KEY  : API key; may be left empty for local endpoints without
//!   auth (e.g. Ollama)
//! - MOLO_BASE_URL : OpenAI-compatible endpoint, default https://api.openai.com/v1
//! - MOLO_MODEL    : model name, default gpt-4o-mini
//!
//! Run: `cargo run --example agent`
//! Type exit / quit / Ctrl-D to quit.
//! Try asking: "What is (1 + 2) * 3?"; the model will request the calculator
//! tool, the Agent executes it and feeds the result back, and the model gives
//! the final answer based on it.

use std::collections::VecDeque;
use std::io::Write;

use futures::StreamExt;
use futures::stream::BoxStream;
use molo::agent::{Agent, AgentError};
use molo::memory::InMemoryMemory;
use molo::provider::OpenAiProvider;
use molo::tool::{SharedState, Tool, ToolError, ToolSchema};
use molo::{ChatRequest, Memory, Message, MessageChunk, Provider, StreamEvent, ToolCall};

use schemars::JsonSchema;
use serde::Deserialize;

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

/// The hand-written Agent in this example: a tool reasoning loop.
///
/// Loop shape: chat → execute tool calls and feed the results back → until the
/// model answers directly; a failed tool call does not abort the loop — the
/// error is returned as text so the model can decide what to do next;
/// `max_tool_rounds` prevents an infinite loop of repeated tool requests.
struct CalculatorAgent {
    provider: Box<dyn Provider>,
    memory: Box<dyn Memory>,
    tools: Vec<Box<dyn Tool>>,
    max_tool_rounds: usize,
}

#[async_trait::async_trait]
impl Agent for CalculatorAgent {
    async fn run(&mut self, input: &str) -> Result<String, AgentError> {
        self.memory.record(Message::user(input)).await?;

        // Tool definitions are the same every round, so compute them once; the model decides whether to call a tool based on them.
        let schemas: Vec<ToolSchema> = self.tools.iter().map(|t| t.schema()).collect();

        for _ in 0..self.max_tool_rounds {
            let response = self
                .provider
                .chat(ChatRequest {
                    messages: self.memory.context().await?,
                    tools: schemas.clone(),
                    ..Default::default()
                })
                .await?;

            // This round's reply is always a single Assistant message (Provider contract);
            // text, reasoning, and tool calls stay in the same message (wire constraint:
            // multiple tool calls in the same round are not split apart).
            let Message::Assistant {
                content,
                reasoning,
                tool_calls,
            } = response.message
            else {
                unreachable!("the reply must be an Assistant message by contract")
            };

            if !content.is_empty() || reasoning.is_some() || !tool_calls.is_empty() {
                self.memory
                    .record(Message::Assistant {
                        content: content.clone(),
                        reasoning,
                        tool_calls: tool_calls.clone(),
                    })
                    .await?;
            }

            if tool_calls.is_empty() {
                return Ok(content); // the model answered directly
            }

            // Tool round: execute every requested call, then feed each result back right after.
            for call in tool_calls {
                let content = self.run_tool(&call.name, &call.arguments).await;
                self.memory
                    .record(Message::ToolResult {
                        id: call.id,
                        content,
                    })
                    .await?;
            }
        }

        Err(AgentError::TooManyToolRounds(self.max_tool_rounds))
    }

    async fn run_stream<'a>(
        &'a mut self,
        input: &'a str,
    ) -> Result<BoxStream<'a, Result<MessageChunk, AgentError>>, AgentError> {
        self.memory.record(Message::user(input)).await?;
        let schemas: Vec<ToolSchema> = self.tools.iter().map(|t| t.schema()).collect();
        let max_rounds = self.max_tool_rounds;

        // Streaming state machine: one poll advances one step — queued events are
        // dispatched one by one; when the queue is empty, start a streaming chat round
        // and enqueue this round's events (text deltas / tool calls / tool results).
        // State = (agent borrow, rounds completed, events pending dispatch, finished).
        let stream = futures::stream::unfold(
            (
                self,
                0usize,
                VecDeque::<Result<MessageChunk, AgentError>>::new(),
                false,
            ),
            // The move closure holds schemas / max_rounds; each round first clones its own
            // schemas so captured variables are not moved across rounds.
            move |(agent, rounds, mut pending, finished)| {
                let schemas = schemas.clone();
                async move {
                    if let Some(event) = pending.pop_front() {
                        return Some((event, (agent, rounds, pending, finished)));
                    }
                    // After a normal finish (Done dispatched) or an error: the stream ends.
                    if finished {
                        return None;
                    }
                    // Reached the max tool rounds: end with an error event.
                    if rounds >= max_rounds {
                        return Some((
                            Err(AgentError::TooManyToolRounds(max_rounds)),
                            (agent, rounds, pending, true),
                        ));
                    }

                    let messages = match agent.memory.context().await {
                        Ok(messages) => messages,
                        Err(e) => {
                            return Some((
                                Err(AgentError::Memory(e)),
                                (agent, rounds, pending, true),
                            ));
                        }
                    };
                    let mut provider_stream = match agent
                        .provider
                        .stream_chat(ChatRequest {
                            messages,
                            tools: schemas,
                            ..Default::default()
                        })
                        .await
                    {
                        Ok(stream) => stream,
                        Err(e) => {
                            return Some((
                                Err(AgentError::Provider(e)),
                                (agent, rounds, pending, true),
                            ));
                        }
                    };

                    // Consume this round's full event stream and assemble it into a single Assistant record.
                    let mut text = String::new();
                    let mut reasoning = String::new();
                    let mut calls = Vec::new();
                    while let Some(event) = provider_stream.next().await {
                        match event {
                            Ok(StreamEvent::Delta(delta)) => text.push_str(&delta),
                            Ok(StreamEvent::Reasoning(chunk)) => reasoning.push_str(&chunk),
                            Ok(StreamEvent::ToolCall {
                                id,
                                name,
                                arguments,
                            }) => {
                                calls.push(ToolCall {
                                    id,
                                    name,
                                    arguments,
                                });
                            }
                            Ok(StreamEvent::Done { .. }) => {}
                            // Unknown variant (reserved for non_exhaustive extensions): silently ignore
                            Ok(_) => {}
                            Err(e) => {
                                return Some((
                                    Err(AgentError::Provider(e)),
                                    (agent, rounds + 1, pending, true),
                                ));
                            }
                        }
                    }

                    if let Err(e) = agent
                        .memory
                        .record(Message::Assistant {
                            content: text.clone(),
                            reasoning: (!reasoning.is_empty()).then_some(reasoning),
                            tool_calls: calls.clone(),
                        })
                        .await
                    {
                        return Some((
                            Err(AgentError::Memory(e)),
                            (agent, rounds + 1, pending, true),
                        ));
                    }

                    // Enqueue this round's events: text deltas and tool calls; tool results follow after execution.
                    if !text.is_empty() {
                        pending.push_back(Ok(MessageChunk::Delta(text)));
                    }
                    for call in &calls {
                        pending.push_back(Ok(MessageChunk::ToolCall {
                            id: call.id.clone(),
                            name: call.name.clone(),
                            arguments: call.arguments.clone(),
                        }));
                    }

                    if calls.is_empty() {
                        // The model answered directly: dispatch Done, then the stream ends.
                        pending.push_back(Ok(MessageChunk::Done(molo::RunSummary::default())));
                        return Some((
                            pending.pop_front().expect("just enqueued"),
                            (agent, rounds + 1, pending, true),
                        ));
                    }

                    // Tool round: execute the calls one by one, feeding each result back right after.
                    for call in calls {
                        let content = agent.run_tool(&call.name, &call.arguments).await;
                        pending.push_back(Ok(MessageChunk::ToolResult {
                            id: call.id,
                            name: call.name,
                            content,
                        }));
                    }
                    Some((
                        pending.pop_front().expect("just enqueued"),
                        (agent, rounds + 1, pending, false),
                    ))
                }
            },
        );
        Ok(Box::pin(stream))
    }
}

impl CalculatorAgent {
    /// Find the tool by name and execute it; a missing tool, arguments that are
    /// not valid JSON, and execution failures all return a text error instead of
    /// aborting the loop, leaving the next step to the model.
    async fn run_tool(&self, name: &str, arguments: &str) -> String {
        let Some(tool) = self.tools.iter().find(|t| t.schema().name == name) else {
            return format!("tool not found: {name}");
        };
        let args = match serde_json::from_str(arguments) {
            Ok(value) => value,
            Err(e) => return format!("arguments are not valid JSON: {e}"),
        };
        match tool.call(args, &SharedState::new()).await {
            Ok(text) => text,
            Err(e) => format!("tool error: {e}"),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenvy::dotenv().ok(); // load .env; silently ignore if missing

    let base_url =
        std::env::var("MOLO_BASE_URL").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let api_key = std::env::var("MOLO_API_KEY").unwrap_or_default();
    let model = std::env::var("MOLO_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

    let provider = OpenAiProvider::new(base_url, api_key, model);

    // The Agent holds the tool list; the model sees them through their schemas.
    let tools: Vec<Box<dyn Tool>> = vec![Box::new(Calculator)];

    // Assembly: provider + memory + tools + system prompt (implementation detail, decided by the application).
    let mut memory = InMemoryMemory::default();
    memory
        .record(Message::system(
            "You are a helpful assistant. Use the calculator tool for calculations instead of doing math in your head.",
        ))
        .await?;

    let mut agent = CalculatorAgent {
        provider: Box::new(provider),
        memory: Box::new(memory),
        tools,
        max_tool_rounds: 10,
    };

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

        // Streaming event-driven: print text deltas as they arrive; tool events go on their own lines.
        match agent.run_stream(input).await {
            Ok(mut stream) => {
                let mut prefix_printed = false;
                while let Some(event) = stream.next().await {
                    match event {
                        Ok(MessageChunk::Delta(delta)) => {
                            if !prefix_printed {
                                print!("assistant: ");
                                std::io::stdout().flush()?;
                                prefix_printed = true;
                            }
                            print!("{delta}");
                            std::io::stdout().flush()?;
                        }
                        Ok(MessageChunk::ToolCall {
                            name, arguments, ..
                        }) => {
                            println!("\n  → calling {name}, arguments: {arguments}");
                        }
                        Ok(MessageChunk::ToolResult { name, content, .. }) => {
                            println!("  → {name} returned: {content}");
                        }
                        Ok(MessageChunk::Done(_)) => break,
                        Ok(MessageChunk::Cancelled) => {
                            println!("\n[cancelled]");
                            break;
                        }
                        // Unknown variant (reserved for non_exhaustive extensions): silently ignore
                        Ok(_) => {}
                        Err(e) => println!("\nerror: {e}"),
                    }
                }
                println!();
            }
            Err(e) => println!("error: {e}"),
        }
    }

    Ok(())
}
