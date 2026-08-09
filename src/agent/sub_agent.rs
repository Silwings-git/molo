//! Sub-agent parts: wrap another reasoning loop as a tool so the current
//! loop can delegate to it.
//!
//! Two parts:
//! - [`SubAgentTool`] — a sub-agent as a tool. Register it into the main
//!   loop's [`ToolRegistry`](crate::tool::ToolRegistry) and the assembly is
//!   complete; the main loop gains delegation with zero changes;
//! - [`SubAgentPool`] — a pool of named sub-agents. Create and name them
//!   dynamically, then address them by name to continue their conversation
//!   (a task agent is not discarded when its task ends; you can come back
//!   and continue talking anytime).
//!
//! Both depend only on the [`Agent`] trait and `Send`, not on any concrete
//! loop implementation — application-written loop types work as sub-agents
//! too; the built-in [`ReActAgent`] is just the most common implementation.
//!
//! Typical assembly (from the main agent's perspective):
//!
//! ```text
//! main agent (any Agent implementation)
//!   └─ ToolRegistry
//!        ├─ SubAgentTool (persistent / dynamic)   ← model delegates on demand
//!        └─ pool tools (spawn / send / list, wrapping SubAgentPool as
//!            tools at the application layer; "@name" addressing on the UI
//!            also resolves at the application layer)
//! ```

use crate::agent::{Agent, ReActAgent};
use crate::provider::Provider;
use crate::tool::{SharedState, Tool, ToolError, ToolRegistry, ToolSchema};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Factory for the dynamic form: receives this call's arguments and returns
/// a newly created sub-agent.
type SubAgentFactory =
    Box<dyn Fn(serde_json::Value) -> Result<Box<dyn Agent + Send>, ToolError> + Send + Sync>;

/// Provider constructor: creates a new provider instance per call (reused
/// by the convenience form).
type ProviderFactory = Box<dyn (Fn() -> Box<dyn Provider>) + Send + Sync>;

/// Where a sub-agent's instance comes from (internal representation of the
/// three forms).
enum SubAgentSource {
    /// Persistent instance: calls are serialized through an internal lock,
    /// and the conversation accumulates across calls.
    Instance(Mutex<Box<dyn Agent + Send>>),
    /// Dynamic factory: each call creates a transient sub-agent that is
    /// discarded when done; no shared state, naturally concurrent.
    Factory(SubAgentFactory),
    /// Convenience form: each call builds a standard ReAct sub-agent from
    /// the `system_prompt` / `task` fields in the arguments (the model
    /// defines the sub-agent's system prompt and task).
    React {
        /// Provider constructor for the sub-agent (captures the
        /// user-supplied Clone provider).
        make_provider: ProviderFactory,
        /// The sub-agent's tool set.
        tools: ToolRegistry,
    },
}

/// A sub-agent as a tool: wrap another reasoning loop as a [`Tool`]; once
/// registered into the main loop's
/// [`ToolRegistry`](crate::tool::ToolRegistry), the model can delegate on
/// demand during conversation.
///
/// Two forms, chosen by conversation need:
/// - [`from_agent`](SubAgentTool::from_agent): **persistent** — holds the
///   sub-agent instance and continues the same conversation across calls
///   (context accumulates), suited to repeatedly consulting a "resident
///   expert";
/// - [`from_factory`](SubAgentTool::from_factory): **dynamic** — each call
///   creates a transient sub-agent through the factory, discarded when done,
///   suited to one-shot delegation; the factory sees this call's arguments
///   and can freely assemble the sub-agent (type / tool subset / system
///   prompt).
///
/// The sub-agent only needs to implement the [`Agent`] trait and be `Send`.
/// On call, the model's arguments are passed to the sub-agent as their JSON
/// text form, and the sub-agent's answer text is returned as the tool result
/// to the main loop. Calls to the same tool instance are serialized
/// (internal lock); different tool instances run in parallel.
///
/// Cancellation semantics: a sub-agent run receives no cancellation signal
/// (tool calls have no cancellation parameter); when the main loop is
/// cancelled, unfinished sub-agent calls abort as their future is dropped
/// (non-cooperative), and already-recorded messages are kept.
///
/// # Examples
///
/// Any type implementing the [`Agent`] trait can be a sub-agent — this
/// example uses a minimal application-written type (independent of the
/// built-in loops):
///
/// ```
/// use molo::agent::{Agent, AgentError, SubAgentTool};
/// use molo::tool::{SharedState, Tool};
/// use serde_json::json;
///
/// /// Demo sub-agent: echoes the input back verbatim.
/// struct Echo;
/// #[molo::async_trait]
/// impl Agent for Echo {
///     async fn run(&mut self, input: &str) -> Result<String, AgentError> {
///         Ok(format!("echo: {input}"))
///     }
/// }
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), molo::tool::ToolError> {
/// let tool = SubAgentTool::from_agent(
///     "echo",
///     "Hand the content to the echo sub-agent",
///     json!({ "type": "object", "properties": { "message": { "type": "string" } } }),
///     Box::new(Echo),
/// );
///
/// let result = tool
///     .call(json!({ "message": "Hello" }), &SharedState::default())
///     .await?;
/// assert_eq!(result, "echo: {\"message\":\"Hello\"}");
/// # Ok(())
/// # }
/// ```
pub struct SubAgentTool {
    schema: ToolSchema,
    source: SubAgentSource,
}

impl SubAgentTool {
    /// Persistent form: holds the sub-agent instance and continues the same
    /// conversation across calls.
    ///
    /// # Parameters
    ///
    /// - `name` / `description`: the name and purpose exposed to the model
    ///   (the model's basis for choosing the tool);
    /// - `parameters`: the JSON Schema for the sub-task input (prefer
    ///   generating it with `schemars::schema_for!` from a serde struct);
    /// - `agent`: the sub-agent instance (implements the [`Agent`] trait
    ///   and is `Send`).
    pub fn from_agent(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
        agent: Box<dyn Agent + Send>,
    ) -> Self {
        Self {
            schema: ToolSchema {
                name: name.into(),
                description: description.into(),
                parameters,
            },
            source: SubAgentSource::Instance(Mutex::new(agent)),
        }
    }

    /// Dynamic form: each call creates a transient sub-agent through the
    /// factory, discarded when done.
    ///
    /// The factory receives this call's arguments (JSON) and can assemble
    /// the sub-agent accordingly — choosing the type, trimming the tool
    /// subset, or writing a system prompt; failing to parse the arguments
    /// returns [`ToolError::InvalidArguments`].
    ///
    /// # Parameters
    ///
    /// - `name` / `description`: the name and purpose exposed to the model
    ///   (the model's basis for choosing the tool);
    /// - `parameters`: the JSON Schema for the sub-task input (prefer
    ///   generating it with `schemars::schema_for!` from a serde struct);
    /// - `factory`: the factory run on each call, returning the new
    ///   sub-agent.
    pub fn from_factory(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
        factory: impl Fn(serde_json::Value) -> Result<Box<dyn Agent + Send>, ToolError>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            schema: ToolSchema {
                name: name.into(),
                description: description.into(),
                parameters,
            },
            source: SubAgentSource::Factory(Box::new(factory)),
        }
    }

    /// Convenience form: each call builds a standard ReAct sub-agent from
    /// the `system_prompt` / `task` fields in the model's arguments — the
    /// model (main agent) defines the sub-agent's system prompt and task at
    /// call time, no hand-written factory needed.
    ///
    /// The parameters schema should include two fields (both degrade
    /// gracefully when missing): `system_prompt` (the sub-agent's system
    /// prompt, empty when missing) and `task` (the sub-agent's task, empty
    /// input when missing). What's passed to the sub-agent as input is the
    /// text of the `task` field (not the whole argument JSON).
    ///
    /// # Parameters
    ///
    /// - `name` / `description`: the name and purpose exposed to the model
    ///   (the model's basis for choosing the tool);
    /// - `provider`: the sub-agent's provider (**must be `Clone`** — every
    ///   call builds an independent loop); sharing one provider instance
    ///   between the main and sub agents is fine;
    /// - `tools`: the sub-agent's tool set;
    /// - `parameters`: the JSON Schema for the sub-task input.
    ///
    /// # Examples
    ///
    /// ```
    /// use molo::agent::SubAgentTool;
    /// use molo::provider::{FakeProvider, FakeReply};
    /// use molo::tool::{SharedState, Tool, ToolRegistry};
    /// use serde_json::json;
    ///
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), molo::tool::ToolError> {
    /// let fake = FakeProvider::new([FakeReply::Text("sub answer".into())]);
    /// let tool = SubAgentTool::from_react(
    ///     "delegate",
    ///     "Delegate a sub-task, defining the sub-agent's system prompt and task",
    ///     fake,
    ///     ToolRegistry::new(),
    ///     json!({ "type": "object", "properties": {
    ///         "system_prompt": { "type": "string" },
    ///         "task": { "type": "string" },
    ///     } }),
    /// );
    ///
    /// // The model defines the system prompt and task → a new sub-agent is
    /// // created and run
    /// let result = tool
    ///     .call(
    ///         json!({ "system_prompt": "You are a reviewer", "task": "Review this code" }),
    ///         &SharedState::default(),
    ///     )
    ///     .await?;
    /// assert_eq!(result, "sub answer");
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_react(
        name: impl Into<String>,
        description: impl Into<String>,
        provider: impl Provider + Clone + 'static,
        tools: ToolRegistry,
        parameters: serde_json::Value,
    ) -> Self {
        let make_provider: ProviderFactory =
            Box::new(move || Box::new(provider.clone()) as Box<dyn Provider>);
        Self {
            schema: ToolSchema {
                name: name.into(),
                description: description.into(),
                parameters,
            },
            source: SubAgentSource::React {
                make_provider,
                tools,
            },
        }
    }
}

impl fmt::Debug for SubAgentTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubAgentTool")
            .field("name", &self.schema.name)
            .finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl Tool for SubAgentTool {
    fn schema(&self) -> ToolSchema {
        self.schema.clone()
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        _state: &SharedState,
    ) -> Result<String, ToolError> {
        match &self.source {
            SubAgentSource::Instance(agent) => {
                let mut guard = agent.lock().await;
                run_sub_agent(&mut **guard, &arguments).await
            }
            SubAgentSource::Factory(factory) => {
                let mut agent = factory(arguments.clone())?;
                run_sub_agent(&mut *agent, &arguments).await
            }
            SubAgentSource::React {
                make_provider,
                tools,
            } => {
                // The model defines the system prompt and task: the task
                // field text is the input (not the whole JSON).
                let system_prompt = arguments
                    .get("system_prompt")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let task = arguments.get("task").and_then(|v| v.as_str()).unwrap_or("");
                let mut agent = ReActAgent::new((make_provider)(), tools.clone(), system_prompt);
                agent
                    .run(task)
                    .await
                    .map_err(|e| ToolError::Execution(e.to_string()))
            }
        }
    }
}

/// Run a sub-agent once: the arguments, as JSON text, become the
/// sub-agent's input; a sub-agent failure is mapped to an execution error
/// (carrying the original error text, which the main loop can read and
/// continue from).
async fn run_sub_agent(
    agent: &mut (dyn Agent + Send),
    arguments: &serde_json::Value,
) -> Result<String, ToolError> {
    agent
        .run(&arguments.to_string())
        .await
        .map_err(|e| ToolError::Execution(e.to_string()))
}

/// Addressing errors for the named sub-agent pool.
///
/// # Examples
///
/// ```
/// use molo::agent::PoolError;
///
/// let err = PoolError::NotFound("ghost".into());
/// assert_eq!(err.to_string(), "no such sub agent: 'ghost'");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum PoolError {
    /// The name already exists at creation time (not overwritten, to avoid
    /// accidental damage).
    #[error("sub agent '{0}' already exists")]
    Duplicate(String),
    /// The name doesn't exist when addressing.
    #[error("no such sub agent: '{0}'")]
    NotFound(String),
    /// The factory failed to create the sub-agent.
    #[error("sub agent factory failed: {0}")]
    Factory(String),
    /// The sub-agent run failed (with the original error text).
    #[error("sub agent run failed: {0}")]
    Run(String),
}

impl From<PoolError> for ToolError {
    fn from(err: PoolError) -> Self {
        ToolError::Execution(err.to_string())
    }
}

/// A single named slot in the pool: a sub-agent instance + a serialization
/// lock (calls to the same name queue up).
type AgentSlot = Arc<Mutex<Box<dyn Agent + Send>>>;

/// A pool of named sub-agents: create and name sub-agents dynamically, then
/// address them by name to continue their conversations.
///
/// Unlike the one-shot [`SubAgentTool::from_factory`], the pool keeps the
/// instances it creates — they are not discarded when their task ends; the
/// user (or the model) can hand a new task to the same sub-agent by name
/// later, and it picks up the conversation with its prior memory. Suited to
/// "the main agent distributes multiple tasks, each with its own
/// independent-context sub-agent, and continues by name afterwards".
///
/// The pool is a shared part: cloning is cheap (internal `Arc`), and
/// multiple loops / threads can hold the same pool; calls to the same
/// sub-agent name are serialized (inner lock), different names run in
/// parallel.
///
/// Cancellation semantics: same as [`SubAgentTool`] — sub-agent runs
/// receive no cancellation signal; when the caller is cancelled, unfinished
/// runs abort as their future is dropped.
///
/// # Examples
///
/// Create and name a sub-agent, run its first task immediately; continue by
/// name afterwards:
///
/// ```
/// use molo::agent::{Agent, AgentError, SubAgentPool};
///
/// /// Demo sub-agent: echoes the input back verbatim.
/// struct Echo;
/// #[molo::async_trait]
/// impl Agent for Echo {
///     async fn run(&mut self, input: &str) -> Result<String, AgentError> {
///         Ok(format!("echo: {input}"))
///     }
/// }
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), molo::agent::PoolError> {
/// let pool = SubAgentPool::new();
///
/// // Create a named sub-agent and run its first task (the reply returns to
/// // the caller)
/// let reply = pool
///     .spawn(
///         "red",
///         || -> Result<Box<dyn Agent + Send>, molo::tool::ToolError> { Ok(Box::new(Echo)) },
///         "task one",
///     )
///     .await?;
/// assert_eq!(reply, "echo: task one");
///
/// // Continue by name: the same instance, with its prior memory
/// let reply = pool.send("red", "task two").await?;
/// assert_eq!(reply, "echo: task two");
///
/// // Name enumeration and probing (UI list / model queries)
/// assert!(pool.contains("red").await);
/// assert_eq!(pool.names().await, vec!["red".to_string()]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Default)]
pub struct SubAgentPool {
    agents: Arc<Mutex<HashMap<String, AgentSlot>>>,
}

impl SubAgentPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create and name a sub-agent, running its first task immediately.
    ///
    /// The factory only builds the sub-agent (type / tool subset / system
    /// prompt, all up to you); once built, the instance goes into the pool
    /// before running — even if the first task fails, the sub-agent stays in
    /// the pool and can be continued or retried via
    /// [`send`](SubAgentPool::send).
    ///
    /// # Errors
    ///
    /// - The name already exists: returns
    ///   [`PoolError::Duplicate`](PoolError::Duplicate);
    /// - The factory failed: returns
    ///   [`PoolError::Factory`](PoolError::Factory);
    /// - The first task failed: returns [`PoolError::Run`](PoolError::Run)
    ///   (the instance is kept).
    pub async fn spawn(
        &self,
        name: &str,
        factory: impl FnOnce() -> Result<Box<dyn Agent + Send>, ToolError>,
        input: &str,
    ) -> Result<String, PoolError> {
        let slot = {
            let mut agents = self.agents.lock().await;
            if agents.contains_key(name) {
                return Err(PoolError::Duplicate(name.to_string()));
            }
            let slot = Arc::new(Mutex::new(
                factory().map_err(|e| PoolError::Factory(e.to_string()))?,
            ));
            agents.insert(name.to_string(), slot.clone());
            slot
        };
        let mut guard = slot.lock().await;
        guard
            .run(input)
            .await
            .map_err(|e| PoolError::Run(e.to_string()))
    }

    /// Convenience form: create and name a standard ReAct sub-agent (with a
    /// given system prompt), running its first task immediately; afterwards
    /// you can continue by name via [`send`](SubAgentPool::send).
    ///
    /// The only difference from [`spawn`](SubAgentPool::spawn) is how the
    /// sub-agent is built — here the system prompt and task are given and
    /// handled by the standard ReAct loop; the provider doesn't need to be
    /// `Clone` (a named creation calls the factory only once).
    ///
    /// # Errors
    ///
    /// Same as [`spawn`](SubAgentPool::spawn): duplicate name
    /// [`PoolError::Duplicate`](PoolError::Duplicate); first-task failure
    /// [`PoolError::Run`](PoolError::Run) (the instance is kept, and the
    /// conversation can continue).
    pub async fn spawn_react(
        &self,
        name: &str,
        provider: impl Provider + 'static,
        tools: ToolRegistry,
        system_prompt: &str,
        task: &str,
    ) -> Result<String, PoolError> {
        self.spawn(
            name,
            move || -> Result<Box<dyn Agent + Send>, ToolError> {
                Ok(Box::new(ReActAgent::new(provider, tools, system_prompt)))
            },
            task,
        )
        .await
    }

    /// Address by name: hand a new task to an already-created sub-agent,
    /// continuing its conversation (with its prior memory).
    ///
    /// # Errors
    ///
    /// Returns [`PoolError::NotFound`](PoolError::NotFound) when the name
    /// doesn't exist; [`PoolError::Run`](PoolError::Run) when the run fails.
    pub async fn send(&self, name: &str, input: &str) -> Result<String, PoolError> {
        let slot = {
            let agents = self.agents.lock().await;
            agents
                .get(name)
                .cloned()
                .ok_or_else(|| PoolError::NotFound(name.to_string()))?
        };
        let mut guard = slot.lock().await;
        guard
            .run(input)
            .await
            .map_err(|e| PoolError::Run(e.to_string()))
    }

    /// Whether the name already exists (duplicate check before creation /
    /// UI probing).
    pub async fn contains(&self, name: &str) -> bool {
        self.agents.lock().await.contains_key(name)
    }

    /// All names in the pool (sorted), for UI lists or model queries.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() {
    /// use molo::agent::SubAgentPool;
    ///
    /// let pool = SubAgentPool::new();
    /// assert!(pool.names().await.is_empty());
    /// # }
    /// ```
    pub async fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.agents.lock().await.keys().cloned().collect();
        names.sort();
        names
    }
}

impl fmt::Debug for SubAgentPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubAgentPool").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentError, ReActAgent};
    use crate::message::{ContentBlock, Message, ToolCall};
    use crate::provider::{FakeProvider, FakeReply, ProviderError};
    use crate::tool::ToolRegistry;
    use serde_json::json;

    /// Whether a User message contains the given text (message content is
    /// content blocks).
    fn user_has_text(msg: &Message, text: &str) -> bool {
        matches!(
            msg,
            Message::User(blocks)
                if blocks.iter().any(|b| matches!(b, ContentBlock::Text(t) if t == text))
        )
    }

    /// Test stub: records every input, used to assert whether the instance
    /// continues and whether inputs arrive as arguments.
    #[derive(Default)]
    struct RecordingAgent {
        seen: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl Agent for RecordingAgent {
        async fn run(&mut self, input: &str) -> Result<String, AgentError> {
            self.seen.lock().await.push(input.to_string());
            Ok(format!("processed: {input}"))
        }
    }

    /// Test stub: fails on first run, succeeds afterwards (verifies
    /// "failures keep the agent, ready to continue").
    struct FlakyAgent {
        failed_once: bool,
    }

    #[async_trait::async_trait]
    impl Agent for FlakyAgent {
        async fn run(&mut self, _input: &str) -> Result<String, AgentError> {
            if !self.failed_once {
                self.failed_once = true;
                return Err(AgentError::Provider(ProviderError::Api {
                    status: 0,
                    message: "boom".into(),
                }));
            }
            Ok("recovered".into())
        }
    }

    #[test]
    fn schema_passthrough() {
        let tool = SubAgentTool::from_agent(
            "consult",
            "Consult a sub-agent",
            json!({ "type": "object", "properties": { "q": { "type": "string" } } }),
            Box::new(RecordingAgent::default()),
        );
        let schema = tool.schema();
        assert_eq!(schema.name, "consult");
        assert_eq!(schema.description, "Consult a sub-agent");
        assert_eq!(schema.parameters["properties"]["q"]["type"], "string");
    }

    #[tokio::test]
    async fn persistent_agent_continues_session_across_calls() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let tool = SubAgentTool::from_agent(
            "consult",
            "Consult a sub-agent",
            json!({}),
            Box::new(RecordingAgent { seen: seen.clone() }),
        );

        tool.call(json!({ "q": 1 }), &SharedState::default())
            .await
            .unwrap();
        tool.call(json!({ "q": 2 }), &SharedState::default())
            .await
            .unwrap();

        // Same instance continues: both inputs are recorded, and the second
        // input is the second call's arguments.
        let all = seen.lock().await;
        assert_eq!(all.len(), 2);
        assert!(all[0].contains("1"));
        assert!(all[1].contains("2"));
    }

    #[tokio::test]
    async fn dynamic_factory_fresh_agent_per_call() {
        let spawns = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let spawns_factory = spawns.clone();
        let tool =
            SubAgentTool::from_factory("delegate", "One-shot delegation", json!({}), move |_args| {
                spawns_factory.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(Box::new(RecordingAgent::default()))
            });

        tool.call(json!({ "q": 1 }), &SharedState::default())
            .await
            .unwrap();
        tool.call(json!({ "q": 2 }), &SharedState::default())
            .await
            .unwrap();

        // Every call creates a new instance through the factory (the old
        // instance is discarded when the call ends — no state to continue).
        assert_eq!(spawns.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn factory_failure_maps_to_invalid_arguments() {
        let tool = SubAgentTool::from_factory("delegate", "One-shot delegation", json!({}), |_| {
            Err(ToolError::InvalidArguments("bad kind".into()))
        });
        let err = tool
            .call(json!({}), &SharedState::default())
            .await
            .unwrap_err();
        assert_eq!(err, ToolError::InvalidArguments("bad kind".into()));
    }

    #[tokio::test]
    async fn sub_agent_failure_maps_to_execution() {
        let tool = SubAgentTool::from_agent(
            "flaky",
            "A sub-agent that fails",
            json!({}),
            Box::new(FlakyAgent { failed_once: false }),
        );
        let err = tool
            .call(json!({}), &SharedState::default())
            .await
            .unwrap_err();
        match err {
            ToolError::Execution(msg) => assert!(msg.contains("boom")),
            other => panic!("expected Execution, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn pool_spawn_then_send_continues_session() {
        let pool = SubAgentPool::new();
        let seen = Arc::new(Mutex::new(Vec::new()));

        let spawn_factory = {
            let seen = seen.clone();
            move || -> Result<Box<dyn Agent + Send>, ToolError> {
                Ok(Box::new(RecordingAgent { seen }))
            }
        };
        let reply = pool.spawn("red", spawn_factory, "task one").await.unwrap();
        assert_eq!(reply, "processed: task one");

        let reply = pool.send("red", "task two").await.unwrap();
        assert_eq!(reply, "processed: task two");

        // Same instance continues: both inputs recorded, and the first input
        // is spawn's task.
        let all = seen.lock().await;
        assert_eq!(all.len(), 2);
        assert!(all[0].contains("task one"));
    }

    #[tokio::test]
    async fn pool_rejects_duplicate_names() {
        let pool = SubAgentPool::new();
        pool.spawn(
            "a",
            || -> Result<Box<dyn Agent + Send>, ToolError> {
                Ok(Box::new(RecordingAgent::default()))
            },
            "x",
        )
        .await
        .unwrap();
        let err = pool
            .spawn(
                "a",
                || -> Result<Box<dyn Agent + Send>, ToolError> {
                    Ok(Box::new(RecordingAgent::default()))
                },
                "y",
            )
            .await
            .unwrap_err();
        assert_eq!(err, PoolError::Duplicate("a".into()));
    }

    #[tokio::test]
    async fn pool_send_unknown_name_errors() {
        let pool = SubAgentPool::new();
        let err = pool.send("ghost", "hi").await.unwrap_err();
        assert_eq!(err, PoolError::NotFound("ghost".into()));
    }

    #[tokio::test]
    async fn pool_keeps_failed_agent_for_retry() {
        let pool = SubAgentPool::new();
        let err = pool
            .spawn(
                "flaky",
                || -> Result<Box<dyn Agent + Send>, ToolError> {
                    Ok(Box::new(FlakyAgent { failed_once: false }))
                },
                "first task",
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("boom"));

        // After the failure the instance is still in the pool; the
        // conversation can continue (retry).
        let reply = pool.send("flaky", "try again").await.unwrap();
        assert_eq!(reply, "recovered");
    }

    #[tokio::test]
    async fn pool_names_sorted_and_contains() {
        let pool = SubAgentPool::new();
        pool.spawn(
            "b",
            || -> Result<Box<dyn Agent + Send>, ToolError> {
                Ok(Box::new(RecordingAgent::default()))
            },
            "x",
        )
        .await
        .unwrap();
        pool.spawn(
            "a",
            || -> Result<Box<dyn Agent + Send>, ToolError> {
                Ok(Box::new(RecordingAgent::default()))
            },
            "x",
        )
        .await
        .unwrap();

        assert_eq!(pool.names().await, vec!["a".to_string(), "b".to_string()]);
        assert!(pool.contains("a").await);
        assert!(!pool.contains("c").await);
    }

    #[tokio::test]
    async fn concurrent_sends_to_same_name_serialize() {
        let pool = SubAgentPool::new();
        pool.spawn(
            "solo",
            || -> Result<Box<dyn Agent + Send>, ToolError> {
                Ok(Box::new(RecordingAgent::default()))
            },
            "x",
        )
        .await
        .unwrap();

        // Two concurrent sends to the same name: the inner lock serializes
        // them; both succeed and both inputs are recorded.
        let (a, b) = tokio::join!(pool.send("solo", "one"), pool.send("solo", "two"));
        assert_eq!(a.unwrap(), "processed: one");
        assert_eq!(b.unwrap(), "processed: two");
    }

    #[tokio::test]
    async fn nested_parent_calls_subagent() {
        // Sub-agent: independent loop, answers directly.
        let sub = ReActAgent::new(
            FakeProvider::new([FakeReply::Text("sub conclusion".into())]),
            ToolRegistry::new(),
            "sub-agent",
        );
        let sub_tool = SubAgentTool::from_agent(
            "consult",
            "Consult a sub-agent",
            json!({ "type": "object", "properties": {} }),
            Box::new(sub),
        );
        let mut registry = ToolRegistry::new();
        registry.register(sub_tool);

        // Parent agent: the first-round request calls consult; after
        // receiving the result, the second round gives the final answer.
        let parent = ReActAgent::new(
            FakeProvider::new([
                FakeReply::ToolCalls {
                    content: String::new(),
                    calls: vec![ToolCall {
                        id: "t1".into(),
                        name: "consult".into(),
                        arguments: "{\"q\":\"x\"}".into(),
                    }],
                },
                FakeReply::Text("Overall conclusion".into()),
            ]),
            registry,
            "parent agent",
        );
        let mut parent = parent;
        let answer = parent.run("go").await.unwrap();
        assert_eq!(answer, "Overall conclusion");
    }

    #[tokio::test]
    async fn from_react_system_prompt_and_task_reach_sub_agent() {
        // Arc shares one instance: the sub-agent's request history can be
        // asserted (FakeProvider's deep-copying Clone would leave records in
        // copies; the test stub uses the Arc form).
        let fake = Arc::new(FakeProvider::new([FakeReply::Text("sub answer".into())]));
        let tool = SubAgentTool::from_react(
            "delegate",
            "Delegate a sub-task",
            fake.clone(),
            ToolRegistry::new(),
            json!({}),
        );

        let result = tool
            .call(
                json!({ "system_prompt": "You are a reviewer", "task": "Review this code" }),
                &SharedState::default(),
            )
            .await
            .unwrap();
        assert_eq!(result, "sub answer");

        // The system prompt enters the sub-agent's request (System message);
        // the input is the task field text.
        let reqs = fake.requests();
        assert_eq!(reqs.len(), 1);
        assert!(
            reqs[0]
                .messages
                .iter()
                .any(|m| matches!(m, Message::System(s) if s.contains("You are a reviewer")))
        );
        assert!(
            reqs[0]
                .messages
                .iter()
                .any(|m| user_has_text(m, "Review this code"))
        );
    }

    #[tokio::test]
    async fn from_react_missing_fields_default_to_empty() {
        let fake = Arc::new(FakeProvider::new([FakeReply::Text("sub answer".into())]));
        let tool = SubAgentTool::from_react(
            "delegate",
            "Delegate a sub-task",
            fake.clone(),
            ToolRegistry::new(),
            json!({}),
        );

        // No system_prompt / task fields: empty system prompt (no System
        // message assembled), empty input.
        tool.call(json!({}), &SharedState::default()).await.unwrap();
        let reqs = fake.requests();
        assert!(
            reqs[0]
                .messages
                .iter()
                .all(|m| !matches!(m, Message::System(_)))
        );
        assert!(reqs[0].messages.iter().any(|m| user_has_text(m, "")));
    }

    #[tokio::test]
    async fn spawn_react_then_send_continues_session() {
        let pool = SubAgentPool::new();
        let fake = Arc::new(FakeProvider::new([
            FakeReply::Text("answer one".into()),
            FakeReply::Text("answer two".into()),
        ]));

        let reply = pool
            .spawn_react(
                "red",
                fake.clone(),
                ToolRegistry::new(),
                "review expert",
                "task one",
            )
            .await
            .unwrap();
        assert_eq!(reply, "answer one");
        pool.send("red", "task two").await.unwrap();

        // Continue: the second request carries the first conversation (same
        // instance, session continues).
        let reqs = fake.requests();
        assert_eq!(reqs.len(), 2);
        assert!(
            reqs[1]
                .messages
                .iter()
                .any(|m| matches!(m, Message::System(s) if s.contains("review expert")))
        );
        assert!(reqs[1].messages.iter().any(|m| user_has_text(m, "task one")));
        assert!(reqs[1].messages.iter().any(|m| user_has_text(m, "task two")));
    }
}
