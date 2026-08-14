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
//!   via [`ToolContext`] on [`Tool::call`].

pub use registry::{MissingTools, RegistryError, ToolRegistry};
pub use shared_state::SharedState;

use crate::effect::{DisplayOutput, EffectRequest, RiskLevel};
use crate::run::{Artifact, RunContext, RunMetadata};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::time::Duration;

/// Namespace assigned to a tool by the host application or extension layer.
///
/// The provider-facing tool name is still a single unique string. The
/// namespace is host-facing metadata used for extension unload, policy,
/// audit, and debugging.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolNamespace {
    /// Namespace kind.
    pub kind: ToolNamespaceKind,
    /// Stable host-assigned namespace id.
    pub id: String,
}

impl ToolNamespace {
    /// Constructs a namespace from a kind and stable id.
    pub fn new(kind: ToolNamespaceKind, id: impl Into<String>) -> Self {
        Self {
            kind,
            id: id.into(),
        }
    }

    /// Namespace for local application tools registered without extension
    /// source metadata.
    pub fn local() -> Self {
        Self::new(ToolNamespaceKind::Local, "local")
    }

    /// Namespace for tools discovered from one MCP server.
    pub fn mcp_server(id: impl Into<String>) -> Self {
        Self::new(ToolNamespaceKind::McpServer, id)
    }

    /// Namespace for tools exposed by one skill layer.
    pub fn skill_layer(id: impl Into<String>) -> Self {
        Self::new(ToolNamespaceKind::SkillLayer, id)
    }
}

impl fmt::Display for ToolNamespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}:{}", self.kind, self.id)
    }
}

/// Kind of tool namespace.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolNamespaceKind {
    /// Local application-owned tools.
    Local,
    /// Tools discovered from an MCP server.
    McpServer,
    /// Tools exposed by an Agent Skills layer.
    SkillLayer,
    /// Tools exposed by a sub-agent.
    SubAgent,
    /// Application-specific namespace kind.
    Custom(String),
}

/// Trust level assigned to a tool source.
///
/// This value is a policy input only. It does not grant permission and must
/// not be used to bypass harness governance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolTrustLevel {
    /// Host-owned trusted code.
    Trusted,
    /// Project-local source selected by the host.
    Project,
    /// User-installed extension source.
    UserInstalled,
    /// External process or service.
    External,
    /// Untrusted source.
    Untrusted,
}

/// Host-facing metadata describing where a provider-visible tool came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolSource {
    /// Source namespace.
    pub namespace: ToolNamespace,
    /// Raw source name before provider-facing disambiguation.
    pub raw_name: String,
    /// Provider-facing display name registered in [`ToolRegistry`].
    pub display_name: String,
    /// Source trust level.
    pub trust: ToolTrustLevel,
    /// Host/application metadata.
    pub metadata: RunMetadata,
}

impl ToolSource {
    /// Constructs source metadata with external trust by default.
    pub fn new(
        namespace: ToolNamespace,
        raw_name: impl Into<String>,
        display_name: impl Into<String>,
    ) -> Self {
        Self {
            namespace,
            raw_name: raw_name.into(),
            display_name: display_name.into(),
            trust: ToolTrustLevel::External,
            metadata: RunMetadata::new(),
        }
    }

    /// Constructs source metadata for a local application tool.
    pub fn local(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            namespace: ToolNamespace::local(),
            raw_name: name.clone(),
            display_name: name,
            trust: ToolTrustLevel::Trusted,
            metadata: RunMetadata::new(),
        }
    }

    /// Sets the trust level.
    pub fn with_trust(mut self, trust: ToolTrustLevel) -> Self {
        self.trust = trust;
        self
    }

    /// Sets source metadata.
    pub fn with_metadata(mut self, metadata: RunMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// The definition of a tool.
///
/// The model decides whether to call the tool and how to generate arguments
/// from `name` / `description` / `parameters`; Provider implementations map
/// those three fields to the vendor's wire format. [`ToolPolicy`] and
/// [`ToolSchema::metadata`] are framework-facing metadata and are not sent to
/// providers unless a provider adapter explicitly supports such annotations.
///
/// # Example
///
/// ```
/// use molo::tool::ToolSchema;
/// use serde_json::json;
///
/// let schema = ToolSchema::new(
///     "get_weather",
///     "Get the weather for a given city",
///     json!({
///         "type": "object",
///         "properties": {
///             "city": { "type": "string", "description": "City name" }
///         },
///         "required": ["city"]
///     }),
/// );
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
    /// Framework-facing policy declaration.
    pub policy: ToolPolicy,
    /// Framework/application metadata.
    pub metadata: RunMetadata,
}

impl ToolSchema {
    /// Constructs a tool schema with default policy and no metadata.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            policy: ToolPolicy::default(),
            metadata: RunMetadata::new(),
        }
    }

    /// Sets framework-facing policy metadata.
    pub fn with_policy(mut self, policy: ToolPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Sets framework/application metadata.
    pub fn with_metadata(mut self, metadata: RunMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Tool policy metadata declared by the tool author.
///
/// This is an input to registry events and harness policy; it is not an
/// authorization decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolPolicy {
    /// Declared side-effect level.
    pub side_effects: SideEffectLevel,
    /// Default risk declaration.
    pub risk: RiskLevel,
    /// Whether the tool author recommends confirmation before execution.
    pub requires_confirmation: bool,
    /// Suggested timeout for tool/effect execution.
    pub timeout: Option<Duration>,
    /// Default memory policy for this tool's model-visible output.
    pub memory_policy: ToolMemoryPolicy,
}

impl Default for ToolPolicy {
    fn default() -> Self {
        Self {
            side_effects: SideEffectLevel::Pure,
            risk: RiskLevel::Low,
            requires_confirmation: false,
            timeout: None,
            memory_policy: ToolMemoryPolicy::Normal,
        }
    }
}

/// Declared side-effect level for a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SideEffectLevel {
    /// Pure computation.
    Pure,
    /// Reads host/application state but does not write it.
    ReadOnly,
    /// Writes host/application state.
    Write,
    /// Interacts with an external system.
    External,
}

/// Memory handling policy for model-visible tool/effect output.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolMemoryPolicy {
    /// Record normally.
    #[default]
    Normal,
    /// Record as protected memory when supported by the memory implementation.
    Protected,
}

impl ToolMemoryPolicy {
    /// Whether the output should be recorded as protected memory.
    pub fn is_protected(self) -> bool {
        matches!(self, Self::Protected)
    }
}

/// Model-visible output produced by a tool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Text visible to the model through [`Message::ToolResult`](crate::Message::ToolResult).
    pub content: String,
    /// Optional host/UI display output.
    pub display: Option<DisplayOutput>,
    /// Artifact handles produced by the tool.
    pub artifacts: Vec<Artifact>,
    /// Memory policy for the model-visible content.
    pub memory_policy: ToolMemoryPolicy,
    /// Framework/application metadata.
    pub metadata: RunMetadata,
}

impl ToolOutput {
    /// Constructs plain text model-visible output.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            display: None,
            artifacts: Vec::new(),
            memory_policy: ToolMemoryPolicy::Normal,
            metadata: RunMetadata::new(),
        }
    }

    /// Sets host/UI display output.
    pub fn with_display(mut self, display: DisplayOutput) -> Self {
        self.display = Some(display);
        self
    }

    /// Sets artifact handles.
    pub fn with_artifacts(mut self, artifacts: Vec<Artifact>) -> Self {
        self.artifacts = artifacts;
        self
    }

    /// Sets memory policy.
    pub fn with_memory_policy(mut self, policy: ToolMemoryPolicy) -> Self {
        self.memory_policy = policy;
        self
    }

    /// Sets framework/application metadata.
    pub fn with_metadata(mut self, metadata: RunMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

impl From<String> for ToolOutput {
    fn from(content: String) -> Self {
        Self::text(content)
    }
}

impl From<&str> for ToolOutput {
    fn from(content: &str) -> Self {
        Self::text(content)
    }
}

/// Result of a tool call.
///
/// Pure or low-risk work can return [`ToolResult::Output`]. Side-effecting
/// tools should return [`ToolResult::Effect`], allowing an outer harness to
/// govern and execute the requested side effect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ToolResult {
    /// Immediate model-visible output.
    Output(ToolOutput),
    /// Side-effect request for an outer harness.
    Effect(EffectRequest),
}

impl ToolResult {
    /// Returns model-visible text for immediate output results.
    ///
    /// Effect results return `None` because the side effect has not executed.
    pub fn output_content(&self) -> Option<&str> {
        match self {
            Self::Output(output) => Some(&output.content),
            Self::Effect(_) => None,
        }
    }

    /// Returns model-visible text, or an empty string for effect requests
    /// that have not executed yet.
    pub fn content_or_empty(&self) -> &str {
        self.output_content().unwrap_or("")
    }
}

impl std::fmt::Display for ToolResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Output(output) => f.write_str(&output.content),
            Self::Effect(request) => {
                write!(
                    f,
                    "effect request: {} ({})",
                    request.description, request.id
                )
            }
        }
    }
}

impl From<ToolOutput> for ToolResult {
    fn from(output: ToolOutput) -> Self {
        Self::Output(output)
    }
}

impl From<String> for ToolResult {
    fn from(content: String) -> Self {
        Self::Output(ToolOutput::text(content))
    }
}

impl From<&str> for ToolResult {
    fn from(content: &str) -> Self {
        Self::Output(ToolOutput::text(content))
    }
}

impl PartialEq<str> for ToolResult {
    fn eq(&self, other: &str) -> bool {
        self.output_content() == Some(other)
    }
}

impl PartialEq<&str> for ToolResult {
    fn eq(&self, other: &&str) -> bool {
        self == *other
    }
}

impl PartialEq<ToolResult> for str {
    fn eq(&self, other: &ToolResult) -> bool {
        other == self
    }
}

impl PartialEq<ToolResult> for &str {
    fn eq(&self, other: &ToolResult) -> bool {
        other == *self
    }
}

impl PartialEq<String> for ToolResult {
    fn eq(&self, other: &String) -> bool {
        self == other.as_str()
    }
}

impl PartialEq<ToolResult> for String {
    fn eq(&self, other: &ToolResult) -> bool {
        other == self
    }
}

/// Context passed to a tool call.
#[derive(Debug, Clone, Copy)]
pub struct ToolContext<'a> {
    /// Run execution context.
    pub run: &'a RunContext,
    /// Shared cross-tool state.
    pub state: &'a SharedState,
    /// Source model tool-call id.
    pub tool_call_id: &'a str,
    /// Tool name used for this call.
    pub tool_name: &'a str,
}

impl<'a> ToolContext<'a> {
    /// Constructs tool-call context.
    pub fn new(
        run: &'a RunContext,
        state: &'a SharedState,
        tool_call_id: &'a str,
        tool_name: &'a str,
    ) -> Self {
        Self {
            run,
            state,
            tool_call_id,
            tool_name,
        }
    }
}

/// A tool an agent can invoke.
///
/// A tool has two perspectives:
/// - [`Tool::schema`] — the model perspective — tells the model what the
///   tool is and what its arguments look like;
/// - [`Tool::call`] — parses model-provided arguments into an immediate
///   [`ToolOutput`] or an [`EffectRequest`] to be executed by an outer
///   harness.
///
/// Implementations must be `Send + Sync`: the agent loop may execute tools
/// concurrently on any thread. Tools that need to flow / share custom
/// content across tools read and write [`ToolContext::state`];
/// tools that do not can ignore it (`_state`).
///
/// # Example
///
/// ```
/// use molo::tool::{Tool, ToolContext, ToolError, ToolOutput, ToolResult, ToolSchema};
/// use serde_json::json;
///
/// // A demo tool: returns a fixed time.
/// struct TimeTool;
///
/// #[molo::async_trait]
/// impl Tool for TimeTool {
///     fn schema(&self) -> ToolSchema {
///         ToolSchema::new(
///             "time",
///             "Return the current time",
///             json!({ "type": "object", "properties": {} }),
///         )
///     }
///
///     async fn call(
///         &self,
///         _arguments: serde_json::Value,
///         _context: ToolContext<'_>,
///     ) -> Result<ToolResult, ToolError> {
///         Ok(ToolOutput::text("12:00").into())
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
    /// `arguments` is the model-generated arguments JSON parsed by the
    /// registry. `context` carries the run context, source tool-call id/name,
    /// and the agent's shared state.
    async fn call(
        &self,
        arguments: serde_json::Value,
        context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError>;
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

#[cfg(all(test, feature = "macros"))]
mod macro_tests {
    // Functional tests for the `#[molo::tool]` macro: generated Tool
    // implementations register, execute, and produce correct schemas.

    use super::{SharedState, ToolError, ToolRegistry};
    use crate::RunContext;
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
                .call_named(
                    "calculator",
                    r#"{"expression":"1+1"}"#,
                    &RunContext::new("macro-test"),
                    &state,
                )
                .await
                .unwrap(),
            "calc:1+1"
        );
        assert_eq!(
            registry
                .call_named(
                    "hello",
                    r#"{"name":"molo"}"#,
                    &RunContext::new("macro-test"),
                    &state,
                )
                .await
                .unwrap(),
            "hello,molo"
        );
        assert_eq!(
            registry
                .call_named("now", "{}", &RunContext::new("macro-test"), &state)
                .await
                .unwrap(),
            "now"
        );

        // Invalid arguments: Err's Display is the "error-to-text"
        // (macro-generated from_value + From<serde_json::Error>).
        let err = registry
            .call_named(
                "calculator",
                "not-json",
                &RunContext::new("macro-test"),
                &state,
            )
            .await
            .unwrap_err();
        assert!(err.to_string().starts_with("invalid arguments:"));

        // Shared state: the caller-provided instance is injected
        // (consecutive calls on the same instance accumulate the count).
        state.insert(0usize);
        assert_eq!(
            registry
                .call_named("counter", "{}", &RunContext::new("macro-test"), &state)
                .await
                .unwrap(),
            "count=1"
        );
        assert_eq!(
            registry
                .call_named("counter", "{}", &RunContext::new("macro-test"), &state)
                .await
                .unwrap(),
            "count=2"
        );
        assert_eq!(
            registry
                .call_named(
                    "record",
                    r#"{"action":"x"}"#,
                    &RunContext::new("macro-test"),
                    &state,
                )
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
