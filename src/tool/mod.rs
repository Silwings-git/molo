//! Tool: external capabilities an agent can invoke.
//!
//! This file is where Tool is defined: the [`Tool`] trait defines the
//! interface agents use to execute tools, [`ToolSchema`] describes the
//! definition exposed to the model, [`ToolError`] describes why a call
//! fails; it contains no concrete tools — those are implemented by agent
//! applications (see `examples/tool_agent.rs`).
//!
//! Companion components:
//! - [`ToolRegistry`] — holds tools, looks up and executes by name; the
//!   unified entry point between the agent loop and tools;
//! - [`SharedState`] — a container for cross-tool shared state, injected
//!   via the `state` parameter of [`Tool::call`].

pub use registry::{MissingTools, RegistryError, ToolRegistry};
pub use shared_state::SharedState;

use serde::{Deserialize, Serialize};

/// The definition of a tool as exposed to the model.
///
/// The model decides whether to call the tool and how to generate arguments
/// from `name` / `description` / `parameters`; Provider implementations map
/// it to the vendor's wire format.
///
/// # Example
///
/// ```
/// use molo::tool::ToolSchema;
/// use serde_json::json;
///
/// let schema = ToolSchema {
///     name: "get_weather".into(),
///     description: "Get the weather for a given city".into(),
///     parameters: json!({
///         "type": "object",
///         "properties": {
///             "city": { "type": "string", "description": "City name" }
///         },
///         "required": ["city"]
///     }),
/// };
///
/// assert_eq!(schema.name, "get_weather");
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Tool name (the basis on which the model selects a tool).
    pub name: String,
    /// Tool description (the basis on which the model understands the
    /// tool's purpose).
    pub description: String,
    /// JSON Schema for the arguments, preferably generated from a serde
    /// struct with `schemars::schema_for!`.
    pub parameters: serde_json::Value,
}

/// A tool an agent can invoke.
///
/// A tool has two perspectives:
/// - [`Tool::schema`] — the model perspective — tells the model what the
///   tool is and what its arguments look like;
/// - [`Tool::call`] — the execution perspective — actually runs the
///   model-provided arguments and returns a text result for the model.
///
/// Implementations must be `Send + Sync`: the agent loop may execute tools
/// concurrently on any thread. Tools that need to flow / share custom
/// content across tools read and write the `state` parameter in `call`;
/// tools that do not can ignore it (`_state`).
///
/// # Example
///
/// ```
/// use molo::tool::{SharedState, Tool, ToolError, ToolSchema};
/// use serde_json::json;
///
/// // A demo tool: returns a fixed time.
/// struct TimeTool;
///
/// #[molo::async_trait]
/// impl Tool for TimeTool {
///     fn schema(&self) -> ToolSchema {
///         ToolSchema {
///             name: "time".into(),
///             description: "Return the current time".into(),
///             parameters: json!({ "type": "object", "properties": {} }),
///         }
///     }
///
///     async fn call(
///         &self,
///         _arguments: serde_json::Value,
///         _state: &SharedState,
///     ) -> Result<String, ToolError> {
///         Ok("12:00".into())
///     }
/// }
///
/// let tool = TimeTool;
/// assert_eq!(tool.schema().name, "time");
/// ```
#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    /// Model perspective: this tool's definition.
    fn schema(&self) -> ToolSchema;

    /// Execution perspective: run this tool.
    ///
    /// `arguments` is the model-generated arguments JSON (parsed uniformly
    /// by the agent loop); `state` is the current agent's shared state (a
    /// type-safe heterogeneous container, see [`SharedState`]) — tools
    /// that need to flow / share custom content across tools read and
    /// write it; tools that do not can ignore the parameter (`_state`).
    /// Returns result text, which the agent passes back to the model as
    /// [`ToolResult`](crate::Message::ToolResult).
    async fn call(
        &self,
        arguments: serde_json::Value,
        state: &SharedState,
    ) -> Result<String, ToolError>;

    /// Whether the tool result is protected: marked on record, exempt from
    /// window trimming.
    ///
    /// Used for persistent behavioral guidance such as skill bodies — if a
    /// tool result is trimmed by the window, the model silently degrades
    /// (it keeps running but loses the specialized instructions, with no
    /// visible error). Tools returning `true` have their results recorded
    /// via `Memory::record_protected` and are never trimmed by window
    /// Memory; the default is no protection.
    fn protected_output(&self) -> bool {
        false
    }
}

/// Reasons a tool call fails.
///
/// `#[non_exhaustive]` ensures future error categories are not breaking
/// changes; external crates should match with a wildcard arm to stay
/// compatible with variants added in later versions.
///
/// # Example
///
/// ```
/// use molo::tool::ToolError;
///
/// let err = ToolError::InvalidArguments("missing field city".into());
/// assert_eq!(err.to_string(), "invalid arguments: missing field city");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ToolError {
    /// The model-provided arguments do not match the tool's arguments schema.
    #[error("invalid arguments: {0}")]
    InvalidArguments(String),
    /// The tool failed while executing.
    ///
    /// The Display text carries no "tool " prefix: the error type name
    /// already conveys the domain, avoiding a doubled prefix like
    /// "tool error: tool ..." after being wrapped by
    /// [`RegistryError::Execution`](crate::tool::RegistryError).
    #[error("execution failed: {0}")]
    Execution(String),
}

impl From<serde_json::Error> for ToolError {
    fn from(err: serde_json::Error) -> Self {
        ToolError::InvalidArguments(err.to_string())
    }
}

mod registry;
mod shared_state;

#[cfg(test)]
mod macro_tests {
    // Functional tests for the `#[molo::tool]` macro: generated Tool
    // implementations register, execute, and produce correct schemas.

    use super::{SharedState, ToolError, ToolRegistry};
    use schemars::JsonSchema;
    use serde::Deserialize;

    /// struct arguments (with field descriptions, carried into the JSON Schema).
    #[derive(Debug, Deserialize, JsonSchema)]
    struct CalcArgs {
        /// The math expression to evaluate, e.g. "1 + 2 * 3".
        #[schemars(description = "The math expression to evaluate, e.g. \"1 + 2 * 3\"")]
        expression: String,
    }

    #[molo::tool(description = "Evaluate a math expression")]
    async fn calculator(args: CalcArgs) -> Result<String, ToolError> {
        Ok(format!("calc:{}", args.expression))
    }

    /// Primitive argument type.
    #[molo::tool(description = "Say hello")]
    async fn hello(name: String) -> Result<String, ToolError> {
        Ok(format!("hello,{name}"))
    }

    /// Zero arguments: parameters is an empty object schema.
    #[molo::tool(description = "Return the current Unix timestamp")]
    async fn now() -> Result<String, ToolError> {
        Ok("now".into())
    }

    /// Shared state: the macro recognizes `&SharedState` by name and
    /// injects it.
    #[molo::tool(description = "Count calls, returning the current count")]
    async fn counter(state: &SharedState) -> Result<String, ToolError> {
        state.with_mut::<usize>(|n| *n += 1);
        Ok(format!("count={}", state.get::<usize>().unwrap_or(0)))
    }

    /// Business arguments plus shared state together.
    #[molo::tool(description = "Record an action, returning the current count")]
    async fn record(action: String, state: &SharedState) -> Result<String, ToolError> {
        state.with_mut::<usize>(|n| *n += 1);
        Ok(format!(
            "recorded {action}, count={}",
            state.get::<usize>().unwrap_or(0)
        ))
    }

    /// The name attribute overrides the default function name.
    #[molo::tool(name = "renamed", description = "Rename demo")]
    async fn original_name() -> Result<String, ToolError> {
        Ok("renamed".into())
    }

    fn registry() -> ToolRegistry {
        let mut registry = ToolRegistry::new();
        registry
            .register(Calculator)
            .register(Hello)
            .register(Now)
            .register(Counter)
            .register(Record)
            .register(OriginalName);
        registry
    }

    /// The generated struct registers; the registration name = default
    /// function name / name attribute.
    #[test]
    fn macro_generates_registrable_tool() {
        let registry = registry();
        assert_eq!(
            registry.names(),
            vec!["calculator", "hello", "now", "counter", "record", "renamed"]
        );
    }

    /// schema: name / description come from the attributes; parameters
    /// come from the argument type's JSON Schema.
    #[test]
    fn macro_generates_correct_schema() {
        let registry = registry();

        // struct arguments: field descriptions enter the schema.
        let calc_schema = registry
            .get("calculator")
            .expect("calculator registered")
            .schema();
        assert_eq!(calc_schema.name, "calculator");
        assert_eq!(calc_schema.description, "Evaluate a math expression");
        let params = calc_schema.parameters;
        assert_eq!(params["type"], "object");
        assert_eq!(
            params["properties"]["expression"]["description"],
            "The math expression to evaluate, e.g. \"1 + 2 * 3\""
        );
        // draft-07 meta fields ($schema / title) stripped: acceptable to
        // strictly validating endpoints.
        assert!(params.get("$schema").is_none());
        assert!(params.get("title").is_none());

        // Primitive argument: String → string.
        let hello_schema = registry.get("hello").expect("hello registered").schema();
        assert_eq!(
            hello_schema.parameters["properties"]["name"]["type"],
            "string"
        );

        // Zero arguments: empty object schema.
        let now_schema = registry.get("now").expect("now registered").schema();
        assert_eq!(
            now_schema.parameters,
            serde_json::json!({ "type": "object", "properties": {} })
        );
    }

    /// Execution: argument parsing / error conversion are handled by the
    /// macro-generated call.
    #[tokio::test]
    async fn macro_generates_working_call() {
        let registry = registry();
        let state = SharedState::new();

        assert_eq!(
            registry
                .call("calculator", r#"{"expression":"1+1"}"#, &state)
                .await
                .unwrap(),
            "calc:1+1"
        );
        assert_eq!(
            registry
                .call("hello", r#"{"name":"molo"}"#, &state)
                .await
                .unwrap(),
            "hello,molo"
        );
        assert_eq!(registry.call("now", "{}", &state).await.unwrap(), "now");

        // Invalid arguments: Err's Display is the "error-to-text"
        // (macro-generated from_value + From<serde_json::Error>).
        let err = registry
            .call("calculator", "not-json", &state)
            .await
            .unwrap_err();
        assert!(err.to_string().starts_with("invalid arguments:"));

        // Shared state: the caller-provided instance is injected
        // (consecutive calls on the same instance accumulate the count).
        state.insert(0usize);
        assert_eq!(
            registry.call("counter", "{}", &state).await.unwrap(),
            "count=1"
        );
        assert_eq!(
            registry.call("counter", "{}", &state).await.unwrap(),
            "count=2"
        );
        assert_eq!(
            registry
                .call("record", r#"{"action":"x"}"#, &state)
                .await
                .unwrap(),
            "recorded x, count=3"
        );
    }

    /// The generated struct works directly with ReActAgent assembly (the
    /// react_agent! macro's tool list argument).
    #[tokio::test]
    async fn macro_tool_works_in_agent() {
        use crate::agent::Agent;
        use crate::{FakeProvider, FakeReply, ToolCall};

        let fake = FakeProvider::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "hello".into(),
                    arguments: r#"{"name":"molo"}"#.into(),
                }],
            },
            FakeReply::Text("done".into()),
        ]);
        let mut agent = crate::react_agent!(fake, [Hello], "You are an assistant");
        assert_eq!(agent.run("hi").await.unwrap(), "done");
    }
}
