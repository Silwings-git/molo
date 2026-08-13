//! Fake Provider example: test your own Agent loop without depending on a real
//! API.
//!
//! The framework exposes [`FakeProvider`](molo::FakeProvider) as a test aid: a
//! script = the sequence of per-round replies, consumed in order by
//! `chat` / `stream_chat`; when the script is exhausted it errors instead of
//! replaying. This example drives a minimal tool-loop Agent with it:
//! - Script injection: the model first requests a tool call → the Agent
//!   executes and feeds back the result → the model answers directly;
//! - Request recording: `requests()` asserts what the Agent sent to the model
//!   (tool results have been fed back);
//! - Streaming delivery: `stream_chat` and `chat` share the same script and
//!   semantics;
//! - Error injection and script exhaustion: errors in a failed round propagate
//!   out; after exhaustion, one more run fails explicitly right away.
//!
//! # When to use FakeProvider
//!
//! - Writing tests and local verification: no API key needed, fast and
//!   reproducible;
//! - Driving a custom Agent loop: the script precisely controls each step's
//!   reply, so loop behavior is assertable (a mock will not catch loop bugs
//!   like "missing tool result" for you — correctness relies on `requests()` to
//!   verify the messages sent to the model);
//! - Injecting errors: send the loop down the failure path and verify error
//!   handling.
//!
//! This example is self-contained, needs no API key, just run:
//! `cargo run --example fake_provider`

use futures::StreamExt;
use molo::agent::{Agent, AgentError};
use molo::memory::InMemoryMemory;
use molo::provider::{ChatRequest, FakeProvider, FakeReply, Provider, ProviderError};
use molo::tool::{SharedState, Tool, ToolContext, ToolError, ToolOutput, ToolResult, ToolSchema};
use molo::{
    Memory, Message, RunContext, RunMetadata, RunOutput, RunRequest, RunSummary, ToolCall,
    ToolRegistry,
};

/// Demo tool: returns the input as-is, simulating any programmable tool.
struct Echo;

#[async_trait::async_trait]
impl Tool for Echo {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "echo",
            "Returns the input text as-is.",
            serde_json::json!({}),
        )
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        _context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        Ok(ToolOutput::text(format!("echo: {arguments}")).into())
    }
}

/// Demo Agent: a minimal tool reasoning loop (non-streaming) — chat → execute
/// tools and feed back → until the model answers directly.
///
/// Holds the concrete [`FakeProvider`] type directly — after testing the loop
/// we need `requests()` for assertions, and a concrete type keeps the fake
/// scoped to the test; real assemblies inject `Box<dyn Provider>`, and tests
/// can inject a fake in the same shape. Tool execution goes through the
/// standard [`ToolRegistry`](molo::ToolRegistry) (name lookup + argument
/// parsing + error-as-text).
struct SimpleAgent {
    provider: FakeProvider,
    memory: InMemoryMemory,
    tools: ToolRegistry,
    max_tool_rounds: usize,
}

impl SimpleAgent {
    fn new(provider: FakeProvider, tools: ToolRegistry) -> Self {
        Self {
            provider,
            memory: InMemoryMemory::default(),
            tools,
            max_tool_rounds: 5,
        }
    }
}

#[async_trait::async_trait]
impl Agent for SimpleAgent {
    /// Reasoning loop: run a chat turn → execute tool calls the model requests and feed results back → until the model answers directly.
    /// A failed tool call does not abort the loop; the error is returned as text so the model can decide what to do next.
    async fn run_request_with_context(
        &mut self,
        request: RunRequest,
        context: RunContext,
    ) -> Result<RunOutput, AgentError> {
        self.memory.record(request.input.into_message()).await?;

        // Tool definitions are the same every round, so compute them once; the model decides whether to call a tool based on them.
        let schemas = self.tools.schemas();
        let state = SharedState::new();

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
            // text and tool calls stay in the same message, with execution results fed back right after as ToolResult.
            let Message::Assistant {
                content,
                tool_calls,
                ..
            } = response.message
            else {
                unreachable!("the reply must be an Assistant message by contract")
            };
            if !content.is_empty() || !tool_calls.is_empty() {
                self.memory
                    .record(Message::Assistant {
                        content: content.clone(),
                        reasoning: None,
                        tool_calls: tool_calls.clone(),
                    })
                    .await?;
            }

            if tool_calls.is_empty() {
                return Ok(run_output(context.run_id, content)); // the model answered directly
            }

            // Tool round: execute every requested call, then feed each result back right after;
            // execution details (name lookup / argument parsing / error-as-text) are handled by the ToolRegistry.
            for call in tool_calls {
                // Err's Display is "error as text"; still record it and feed it back as usual.
                let content = self
                    .tools
                    .call(&call, &context, &state)
                    .await
                    .map(|result| result.to_string())
                    .unwrap_or_else(|e| e.to_string());
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
}

fn run_output(run_id: String, answer: String) -> RunOutput {
    RunOutput {
        run_id,
        answer: answer.clone(),
        summary: RunSummary::default(),
        final_message: Message::assistant(answer),
        artifacts: Vec::new(),
        metadata: RunMetadata::new(),
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Script injection drives a tool round: the model first requests echo (script turn 1), then answers directly (script turn 2).
    let fake = FakeProvider::new([
        FakeReply::ToolCalls {
            content: String::new(),
            calls: vec![ToolCall {
                id: "c1".into(),
                name: "echo".into(),
                arguments: r#"{"text":"hello"}"#.into(),
            }],
        },
        FakeReply::Text("Task complete".into()),
    ]);
    let mut tools = ToolRegistry::new();
    tools.register(Echo);
    let mut agent = SimpleAgent::new(fake, tools);
    let answer = agent.run("say something to echo").await?;
    println!("1. tool round → final answer: {answer}");

    // 2. Request recording: assert that the second request fed the tool result back to the model —
    //    this is the only way to verify "the loop did the right thing" with a fake (a real API
    //    would return 400 on a missing tool result; a mock would not, so rely on requests()).
    let requests = agent.provider.requests();
    println!("2. received {} chat requests in total:", requests.len());
    for (i, request) in requests.iter().enumerate() {
        let roles: Vec<&str> = request
            .messages
            .iter()
            .map(|m| match m {
                Message::System(_) => "system",
                Message::User(_) => "user",
                Message::Assistant { .. } => "assistant",
                Message::ToolResult { .. } => "tool",
            })
            .collect();
        println!(
            "   request {}: {} messages [{roles:?}]",
            i + 1,
            request.messages.len()
        );
    }
    let tool_result_passed_back = requests[1]
        .messages
        .iter()
        .any(|m| matches!(m, Message::ToolResult { content, .. } if content.contains("echo")));
    println!("   → tool result fed back to the model: {tool_result_passed_back}");

    // 3. Streaming delivery: same script, same semantics; the event stream = deltas + a closer.
    let fake = FakeProvider::new([FakeReply::TextWithReasoning {
        content: "answer".into(),
        reasoning: "thinking".into(),
    }]);
    println!("3. stream_chat event stream:");
    let mut stream = fake.stream_chat(ChatRequest::default()).await?;
    while let Some(event) = stream.next().await {
        println!("   {event:?}");
    }

    // 4. Error injection: this round fails, and the Agent propagates the Provider error as-is.
    let fake = FakeProvider::new([FakeReply::Error(ProviderError::Api {
        status: 429,
        message: "rate limited".into(),
    })]);
    let mut agent = SimpleAgent::new(fake, ToolRegistry::new());
    match agent.run("ask a question").await {
        Err(AgentError::Provider(e)) => println!("4. error round: Agent propagated {e}"),
        Ok(_) => println!("4. unexpected success (an error round should fail the run)"),
        Err(e) => println!("4. unexpected error: {e}"),
    }

    // 5. Script exhaustion: no replaying; one more run fails explicitly right away.
    let fake = FakeProvider::new([FakeReply::Text("only one round".into())]);
    let mut agent = SimpleAgent::new(fake, ToolRegistry::new());
    agent.run("first question").await?;
    match agent.run("second question").await {
        Err(AgentError::Provider(e)) => println!("5. script exhausted: {e}"),
        Ok(_) => println!("5. unexpected success (script exhaustion should error)"),
        Err(e) => println!("5. unexpected error: {e}"),
    }

    Ok(())
}
