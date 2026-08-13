//! Sub-agent example: the main agent delegates to sub-agents through tools,
//! plus follow-up conversations with a named sub-agent pool.
//!
//! Delegation options ([`SubAgentTool`](molo::agent::SubAgentTool) has three
//! shapes + the [`SubAgentPool`](molo::agent::SubAgentPool) pool):
//! 1. **Persistent expert** (`from_agent`): a sub-agent instance is
//!    pre-constructed, and repeated calls continue the same session (context
//!    accumulates) — good for repeatedly consulting a "resident expert";
//! 2. **Dynamic delegation** (`from_factory`): each call creates a transient
//!    sub-agent via a factory, discarded after the run — good for one-off
//!    sub-tasks;
//! 3. **Convenience shape** (`from_react` / `spawn_react`): the model (main
//!    agent) defines the sub-agent's **system prompt and task** in the call
//!    arguments, and a standard ReAct sub-agent is assembled on the spot;
//! 4. **Named pool**: the main agent creates and names sub-agents, keeping
//!    them after the task finishes; later the user can hand a new task to the
//!    same sub-agent with `@name`, and it continues with its previous memory
//!    (`@name` parsing and presentation is the application's job; this example
//!    wraps the pool into three tools — `spawn_agent` / `send_agent` /
//!    `list_agents` — that the model calls directly in the conversation: when
//!    the user says "hand this code to @Xiaohong to review", the model calls
//!    `send_agent`). `spawn_agent` uses the convenience shape: the system
//!    prompt and task in the arguments are defined by the model.
//!
//! A sub-agent only needs to implement [`Agent`](molo::Agent) — this example
//! uses the built-in [`ReActAgent`](molo::ReActAgent); an application's own
//! loop types work too.
//!
//! Startup is the same as `examples/react_agent.rs`: reads
//! `MOLO_API_KEY` / `MOLO_BASE_URL` / `MOLO_MODEL` from `.env`; the `stream`
//! (default) or `chat` argument selects the conversation mode.
//!
//! # Assembly overview (self-contained, no real API needed)
//!
//! ```rust
//! use molo::agent::{Agent, AgentError, SubAgentTool};
//! use molo::provider::{FakeProvider, FakeReply};
//! use molo::tool::{ToolError, ToolRegistry, ToolResult};
//! use molo::react_agent;
//!
//! /// Demo sub-agent: echoes the input back as-is.
//! struct Echo;
//! #[molo::async_trait]
//! impl Agent for Echo {
//!     async fn run(&mut self, input: &str) -> Result<String, AgentError> {
//!         Ok(format!("echo: {input}"))
//!     }
//! }
//!
//! // Persistent shape: a resident expert; repeated calls continue the same session
//! let persistent = SubAgentTool::from_agent(
//!     "consult",
//!     "Hand the question to the resident expert and return its answer",
//!     serde_json::json!({ "type": "object", "properties": {} }),
//!     Box::new(Echo),
//! );
//!
//! // Dynamic shape: a fresh instance per call via a factory, discarded after the run
//! let dynamic = SubAgentTool::from_factory(
//!     "delegate",
//!     "Delegate the sub-task to a one-off sub-agent and return its conclusion",
//!     serde_json::json!({ "type": "object", "properties": {} }),
//!     |_| -> Result<Box<dyn Agent + Send>, ToolError> { Ok(Box::new(Echo)) },
//! );
//!
//! // Convenience shape: the model defines the sub-agent's system prompt and task in the arguments
//! let react = SubAgentTool::from_react(
//!     "spawn",
//!     "Create a standard sub-agent: the arguments include system_prompt (the sub-agent's system prompt) and task (the task)",
//!     FakeProvider::new([FakeReply::Text("sub reply".into())]),
//!     ToolRegistry::new(),
//!     serde_json::json!({ "type": "object", "properties": {
//!         "system_prompt": { "type": "string" },
//!         "task": { "type": "string" },
//!     } }),
//! );
//!
//! // Registering into the main agent's registry completes the assembly; the main loop needs no changes
//! let mut registry = ToolRegistry::new();
//! registry.register(persistent).register(dynamic).register(react);
//! let agent = react_agent!(
//!     FakeProvider::new([FakeReply::Text("OK".into())]),
//!     registry,
//!     "You are the main agent; delegate when you need an expert",
//! );
//! # let _ = agent;
//! ```

use molo::agent::{Agent, MessageChunk, ReActAgent, SubAgentPool};
use molo::provider::OpenAiProvider;
use molo::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolResult, ToolSchema};

use futures::stream::StreamExt;
use schemars::JsonSchema;
use serde::Deserialize;
use std::io::Write;

/// Arguments for the spawn_agent tool: name + system prompt + the first task.
#[derive(Debug, Deserialize, JsonSchema)]
struct SpawnArgs {
    /// The sub-agent's name (later follow-ups use send_agent with this name).
    name: String,
    /// The sub-agent's system prompt (the main agent defines its role and behavior).
    system_prompt: String,
    /// The first task (executed by the sub-agent in its own context).
    task: String,
}

/// Arguments for the send_agent tool: name + a new task.
#[derive(Debug, Deserialize, JsonSchema)]
struct SendArgs {
    /// The name of a previously created sub-agent.
    name: String,
    /// The new task handed to it (it answers with its previous memory).
    message: String,
}

/// spawn_agent tool: creates and names a sub-agent and immediately runs the task.
///
/// Holds connection parameters instead of a provider instance: each sub-agent
/// creates its own loop and connection.
struct SpawnAgent {
    pool: SubAgentPool,
    base_url: String,
    api_key: String,
    model: String,
}

#[async_trait::async_trait]
impl Tool for SpawnAgent {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "spawn_agent",
            "Creates and names a sub-agent and immediately runs the task; afterwards send_agent can continue the conversation by name",
            serde_json::to_value(schemars::schema_for!(SpawnArgs))
                .expect("tool schema must serialize"),
        )
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        _context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let args: SpawnArgs = serde_json::from_value(arguments)?;
        // Convenience shape: the system prompt and task are defined by the model (main agent), assembled as a standard ReAct agent.
        let reply = self
            .pool
            .spawn_react(
                &args.name,
                OpenAiProvider::new(
                    self.base_url.clone(),
                    self.api_key.clone(),
                    self.model.clone(),
                ),
                ToolRegistry::new(),
                &args.system_prompt,
                &args.task,
            )
            .await
            .map_err(ToolError::from)?;
        Ok(ToolOutput::text(reply).into())
    }
}

/// send_agent tool: hands a new task to a previously created sub-agent by name (continuing its session).
struct SendAgent {
    pool: SubAgentPool,
}

#[async_trait::async_trait]
impl Tool for SendAgent {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "send_agent",
            "Hands a new task to a previously created sub-agent (by name); it answers with its previous memory; if the name does not exist, spawn_agent first",
            serde_json::to_value(schemars::schema_for!(SendArgs))
                .expect("tool schema must serialize"),
        )
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        _context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let args: SendArgs = serde_json::from_value(arguments)?;
        let reply = self
            .pool
            .send(&args.name, &args.message)
            .await
            .map_err(ToolError::from)?;
        Ok(ToolOutput::text(reply).into())
    }
}

/// list_agents tool: lists the names of created sub-agents (for model queries / UI display).
struct ListAgents {
    pool: SubAgentPool,
}

#[async_trait::async_trait]
impl Tool for ListAgents {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "list_agents",
            "Lists the names of all created sub-agents",
            serde_json::json!({ "type": "object", "properties": {} }),
        )
    }

    async fn call(
        &self,
        _arguments: serde_json::Value,
        _context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let names = self.pool.names().await;
        Ok(ToolOutput::text(names.join(", ")).into())
    }
}

/// How the chat turn is run (same as the react_agent example).
enum Mode {
    Stream,
    Chat,
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

    // A named pool + three pool tools (the application layer wraps the pool into tools; @name UI parsing is the application's job)
    let pool = SubAgentPool::new();
    let mut registry = ToolRegistry::new();
    registry.register(SpawnAgent {
        pool: pool.clone(),
        base_url: base_url.clone(),
        api_key: api_key.clone(),
        model: model.clone(),
    });
    registry.register(SendAgent { pool: pool.clone() });
    registry.register(ListAgents { pool });

    let mut agent = ReActAgent::new(
        OpenAiProvider::new(base_url, api_key, model),
        registry,
        "You are the main agent. Delegate sub-agents when specialized capability \
         is needed: create with spawn_agent (give a name and a task); when the \
         user mentions @name or a follow-up is needed, use send_agent; when \
         unsure of a name, list_agents first.",
    );

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

        match mode {
            Mode::Stream => {
                let mut stream = agent.run_stream(input).await?;
                let mut prefix_printed = false;
                while let Some(event) = stream.next().await {
                    match event? {
                        MessageChunk::Delta(delta) => {
                            if !prefix_printed {
                                print!("assistant: ");
                                std::io::stdout().flush()?;
                                prefix_printed = true;
                            }
                            print!("{delta}");
                            std::io::stdout().flush()?;
                        }
                        MessageChunk::ToolCall {
                            id,
                            name,
                            arguments,
                        } => {
                            println!("\n  → calling {name}(#{id}), arguments: {arguments}");
                        }
                        MessageChunk::ToolResult { id, name, content } => {
                            println!("  → {name}(#{id}) returned: {content}");
                        }
                        MessageChunk::Done(_) => break,
                        MessageChunk::Cancelled => {
                            println!("\n[cancelled]");
                            break;
                        }
                        _ => {}
                    }
                }
                println!();
            }
            Mode::Chat => {
                let answer = agent.run(input).await?;
                if !answer.is_empty() {
                    println!("assistant: {answer}");
                }
            }
        }
    }

    Ok(())
}
