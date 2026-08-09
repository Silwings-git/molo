//! ToolRegistry: the registry component — holds tools, responsible for
//! lookup and execution by name.
//!
//! This is the unified entry point where the agent loop executes tools; a
//! single call completes four steps: "lookup → argument parsing →
//! execution → error classification"; classification is carried by
//! [`RegistryError`], and the "error-to-text passed back to the model"
//! semantics live in the error type's Display.

use futures::FutureExt;
use indexmap::IndexMap;
use std::any::Any;
use std::collections::HashSet;
use std::fmt;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use super::{SharedState, Tool, ToolError, ToolSchema};

/// Extract the message text from a panic payload (to pass back to the
/// model); unknown types fall back to "unknown panic".
///
/// The text is truncated to 500 characters to keep an over-long panic from
/// blowing up the text passed back to the model.
fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        return (*s).chars().take(500).collect();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.chars().take(500).collect();
    }
    "unknown panic".into()
}

/// Tool registry: holds tools, responsible for lookup and execution by
/// name.
///
/// ```
/// # #[tokio::main]
/// # async fn main() -> Result<(), molo::RegistryError> {
/// use molo::tool::{SharedState, Tool, ToolError, ToolRegistry, ToolSchema};
/// use serde_json::json;
///
/// struct Calculator;
/// #[molo::async_trait]
/// impl Tool for Calculator {
///     fn schema(&self) -> ToolSchema {
///         ToolSchema {
///             name: "calculator".into(),
///             description: "Calculate".into(),
///             parameters: json!({ "type": "object", "properties": {} }),
///         }
///     }
///     async fn call(
///         &self,
///         _arguments: serde_json::Value,
///         _state: &SharedState,
///     ) -> Result<String, ToolError> {
///         Ok("42".into())
///     }
/// }
///
/// let mut registry = ToolRegistry::new();
/// registry.register(Calculator);
/// let state = SharedState::new();
/// // The model requests a tool by name; error classification (not
/// // registered / args not JSON / execution failed) rides along with Err.
/// let result = registry.call("calculator", "{}", &state).await?;
/// assert_eq!(result, "42");
/// // Allowlist subset: the sub-registry shares the same tool instances as
/// // the main registry.
/// let sub = registry.subset(&["calculator"]).unwrap();
/// assert_eq!(sub.names(), vec!["calculator"]);
/// # Ok(())
/// # }
/// ```
///
/// - **same-named tools: later registration replaces** — registering a
///   same-named tool replaces it in place, so the registry never holds
///   duplicates (the semantics of updating a registered tool, no new
///   entry, stable order);
/// - **internally held as a single `IndexMap<String, Arc<dyn Tool>>`**:
///   O(1) lookup by name while preserving registration order (order
///   affects how the model chooses tools on the wire);
/// - **`Arc<dyn Tool>` sharing**: tool instances can be shared across
///   registries — the sub-registry produced by
///   [`subset`](ToolRegistry::subset) shares the same tool instances as
///   the main registry (the scenario where a main agent creates
///   sub-agents with a restricted tool set);
/// - **`call` returns `Result<String, RegistryError>`** — classification
///   rides along with `Err` (tool not found / args not JSON / execution
///   failed), and `Err`'s Display is the "error-to-text" the agent loop
///   can pass straight back to the model (see [`RegistryError`]); callers
///   that need to bypass the registry's argument parsing can grab the
///   tool directly with [`get`](ToolRegistry::get);
/// - **`Debug` prints the registration-name list** (in registration
///   order, handy for debugging).
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: IndexMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            tools: IndexMap::new(),
        }
    }

    /// Register a tool; returns `self` for chaining.
    ///
    /// When a same-named tool is registered again, the later one
    /// **replaces** the earlier (keeping its original position), which
    /// fits updating a registered tool with a new instance.
    pub fn register(&mut self, tool: impl Tool + 'static) -> &mut Self {
        self.tools.insert(tool.schema().name, Arc::new(tool));
        self
    }

    /// Names of currently registered tools, in registration order
    /// (same-named tools already deduplicated).
    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Remove a registered tool; returns `true` when removed, `false` when
    /// the tool does not exist.
    ///
    /// For swapping tool sets at runtime (e.g. removing framework-injected
    /// tools when switching assembly modes); remaining tools keep their
    /// registration order.
    pub fn remove(&mut self, name: &str) -> bool {
        self.tools.shift_remove(name).is_some()
    }

    /// Bulk-trim in place by name: removes tools whose names fail `keep`,
    /// returning the removed tool names (in original registration order);
    /// remaining tools keep their registration order.
    ///
    /// Complements [`subset`](ToolRegistry::subset): subset leaves this
    /// table untouched and produces an allowlist sub-table sharing tool
    /// instances with the main table; this method mutates this table in
    /// place, suitable for bulk-removing tools at runtime — e.g. clearing
    /// all tools of an MCP server by its namespace prefix when unloading
    /// it:
    ///
    /// ```
    /// use molo::tool::{SharedState, Tool, ToolError, ToolRegistry, ToolSchema};
    /// use serde_json::json;
    ///
    /// struct Named(&'static str);
    /// #[molo::async_trait]
    /// impl Tool for Named {
    ///     fn schema(&self) -> ToolSchema {
    ///         ToolSchema {
    ///             name: self.0.into(),
    ///             description: self.0.into(),
    ///             parameters: json!({}),
    ///         }
    ///     }
    ///     async fn call(
    ///         &self,
    ///         _arguments: serde_json::Value,
    ///         _state: &SharedState,
    ///     ) -> Result<String, ToolError> {
    ///         Ok("ok".into())
    ///     }
    /// }
    ///
    /// let mut registry = ToolRegistry::new();
    /// registry
    ///     .register(Named("filesystem__read_file"))
    ///     .register(Named("filesystem__list_dir"))
    ///     .register(Named("calculator"));
    /// // Strip all tools of an MCP server (their names carry the
    /// // "filesystem__" prefix):
    /// let removed = registry.retain(|name| !name.starts_with("filesystem__"));
    /// assert_eq!(removed, ["filesystem__read_file", "filesystem__list_dir"]);
    /// assert_eq!(registry.names(), ["calculator"]);
    /// ```
    pub fn retain(&mut self, mut keep: impl FnMut(&str) -> bool) -> Vec<String> {
        let mut removed = Vec::new();
        self.tools.retain(|name, _| {
            if keep(name) {
                true
            } else {
                removed.push(name.clone());
                false
            }
        });
        removed
    }

    /// All tools' definitions for the model, in registration order (no
    /// duplicates, guaranteed at registration).
    pub fn schemas(&self) -> Vec<ToolSchema> {
        self.tools.values().map(|t| t.schema()).collect()
    }

    /// Get a tool reference by name, bypassing the registry's argument
    /// parsing to call [`Tool::call`] directly; returns `None` when not
    /// registered.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    /// Look up and execute a tool by name.
    ///
    /// A single call completes three steps — "lookup → argument parsing →
    /// execution"; failure classifications are described by
    /// [`RegistryError`]:
    /// - tool not registered → [`RegistryError::NotFound`];
    /// - arguments not valid JSON, or valid JSON with a mismatched
    ///   structure (missing fields, etc.) →
    ///   [`RegistryError::InvalidArguments`];
    /// - tool execution failed → [`RegistryError::Execution`](RegistryError::Execution)
    ///   (the underlying [`ToolError`] is reachable via `source()`).
    ///
    /// `state` is injected into the tool at call time (see
    /// [`SharedState`]); the agent loop passes its own state straight
    /// through, so tools read and write the caller-provided instance.
    ///
    /// Tool panics do not escape this method: the panic is caught and
    /// converted into [`RegistryError::Execution`](RegistryError::Execution),
    /// with the message carrying the tool name and panic content, for the
    /// caller to pass back to the model.
    ///
    /// # Errors
    ///
    /// The three failure classes are described above; `Err`'s **Display is
    /// the "error-to-text"** — the agent loop passes `e.to_string()` back
    /// to the model as a ToolResult, and the text is directly readable by
    /// the model.
    pub async fn call(
        &self,
        name: &str,
        arguments: &str,
        state: &SharedState,
    ) -> Result<String, RegistryError> {
        let Some(tool) = self.tools.get(name) else {
            return Err(RegistryError::NotFound(name.to_string()));
        };
        let args = match serde_json::from_str(arguments) {
            Ok(value) => value,
            Err(e) => return Err(RegistryError::InvalidArguments(e.to_string())),
        };
        // The tool is user code, and panics are inputs an LLM-generated
        // argument could trigger: catch them as execution errors passed
        // back to the model instead of letting panics escape the framework.
        let result = AssertUnwindSafe(tool.call(args, state))
            .catch_unwind()
            .await
            .map_err(|payload| {
                // Explicit as_ref to &dyn Any: passing the Box's reference
                // directly would break downcast after deref coercion; take
                // the &dyn Any itself.
                let message = panic_message(payload.as_ref());
                RegistryError::Execution {
                    name: name.to_string(),
                    source: ToolError::Execution(format!("panicked: {message}")),
                }
            })?;
        // The non-panic path carries the tool name too (symmetric with the
        // panic-capture path); the tool's own error is preserved via
        // source, and Display is produced uniformly by this variant.
        // Structurally mismatched arguments (valid JSON but missing fields
        // / wrong types) count as argument errors: passed through as
        // InvalidArguments, so the model sees "invalid arguments: …"
        // rather than an execution failure with a "tool error:" prefix.
        result.map_err(|e| match e {
            ToolError::InvalidArguments(msg) => RegistryError::InvalidArguments(msg),
            other => RegistryError::Execution {
                name: name.to_string(),
                source: other,
            },
        })
    }

    /// Trim a sub-registry by name (an allowlist) sharing the same tool
    /// instances as the main registry.
    ///
    /// Used to restrict a sub-agent's tool set: when the main agent
    /// creates a sub-agent, this method trims an allowlisted registry, and
    /// both tables share the same tool instances (consistent state).
    /// The sub-registry keeps the main registry's registration order.
    ///
    /// # Errors
    ///
    /// When an allowlisted name is not found in the main registry,
    /// [`MissingTools`] is returned; what to do with the missing list
    /// (error / warn / silent) is the caller's decision — the library
    /// does not choose for the caller.
    pub fn subset(&self, names: &[&str]) -> Result<ToolRegistry, MissingTools> {
        let wanted: HashSet<&str> = names.iter().copied().collect();
        let mut tools = IndexMap::new();
        let mut found = HashSet::new();
        for (name, tool) in &self.tools {
            if wanted.contains(name.as_str()) {
                found.insert(name.clone());
                tools.insert(name.clone(), tool.clone());
            }
        }
        let missing: Vec<String> = names
            .iter()
            .filter(|n| !found.contains(**n))
            .map(|n| (*n).to_string())
            .collect();
        if missing.is_empty() {
            Ok(ToolRegistry { tools })
        } else {
            Err(MissingTools { names: missing })
        }
    }
}

impl fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.names()).finish()
    }
}

/// Tool names in a [`ToolRegistry::subset`] allowlist that do not exist in
/// the main registry.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("tools not found in registry: {}", self.names.join(", "))]
pub struct MissingTools {
    names: Vec<String>,
}

/// Reasons a tool execution fails (registry level, defined by the
/// framework).
///
/// All three failure classes are framework-component behavior, not
/// knowledge of user tools; the rich errors of user tools are already
/// finalized at the [`ToolError`] boundary and reachable via
/// [`RegistryError::Execution`]'s `source()`.
///
/// `Err`'s **Display is the "error-to-text"**: the agent loop passes
/// `e.to_string()` back to the model as a ToolResult, and the text is
/// directly readable by the model.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum RegistryError {
    /// The tool is not registered.
    #[error("tool not found: {0}")]
    NotFound(String),
    /// The model-provided arguments are invalid: not valid JSON, or valid
    /// JSON with a mismatched structure (missing fields / wrong types,
    /// rejected by the tool itself as
    /// [`ToolError::InvalidArguments`]).
    #[error("invalid arguments: {0}")]
    InvalidArguments(String),
    /// The tool execution failed; carries the tool name (for locating the
    /// failure with multiple tools, same as the panic path) and the source
    /// error.
    ///
    /// Display starts with a single "tool error" prefix: the source error
    /// (`ToolError`) itself carries no "tool" prefix, so concatenating
    /// them yields exactly one "tool", never a doubled prefix like
    /// "tool error: tool ...".
    #[error("tool error: {name} failed: {source}")]
    Execution {
        /// The name of the tool that failed.
        name: String,
        /// The underlying source error; also reachable via `source()`.
        #[source]
        source: ToolError,
    },
}

impl MissingTools {
    /// The list of missing tool names.
    ///
    /// # Example
    ///
    /// ```
    /// use molo::tool::ToolRegistry;
    ///
    /// let registry = ToolRegistry::new();
    /// let missing = registry.subset(&["weather", "stock"]).unwrap_err();
    /// assert_eq!(missing.names(), &["weather", "stock"]);
    /// ```
    pub fn names(&self) -> &[String] {
        &self.names
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolError;
    use std::error::Error;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Test tool: returns fixed text by name; fails to execute when `fail`
    /// is set.
    struct FakeTool {
        name: &'static str,
        output: &'static str,
        fail: bool,
    }

    #[async_trait::async_trait]
    impl Tool for FakeTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: self.name.into(),
                description: self.output.into(),
                parameters: serde_json::json!({}),
            }
        }

        async fn call(
            &self,
            _arguments: serde_json::Value,
            _state: &SharedState,
        ) -> Result<String, ToolError> {
            if self.fail {
                Err(ToolError::Execution("boom".into()))
            } else {
                Ok(self.output.into())
            }
        }
    }

    /// Test tool: increments a shared counter on each call, used to verify
    /// that a subset shares the same instance as the main registry.
    struct CountingTool {
        name: &'static str,
        calls: Arc<AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl Tool for CountingTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: self.name.into(),
                description: "counts calls".into(),
                parameters: serde_json::json!({}),
            }
        }

        async fn call(
            &self,
            _arguments: serde_json::Value,
            _state: &SharedState,
        ) -> Result<String, ToolError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok("ok".into())
        }
    }

    fn echo(name: &'static str) -> FakeTool {
        FakeTool {
            name,
            output: name,
            fail: false,
        }
    }

    /// Registers search / calculator, with search registered twice (same
    /// name, later registration replaces).
    fn registry() -> ToolRegistry {
        let mut r = ToolRegistry::new();
        r.register(echo("search"))
            .register(echo("calculator"))
            .register(echo("search"));
        r
    }

    #[test]
    fn names_in_registration_order_dedup() {
        assert_eq!(registry().names(), vec!["search", "calculator"]);
    }

    #[test]
    fn schemas_in_registration_order() {
        let schemas = registry().schemas();
        assert_eq!(schemas.len(), 2);
        assert_eq!(schemas[0].name, "search");
        assert_eq!(schemas[1].name, "calculator");
    }

    #[tokio::test]
    async fn register_duplicate_replaces() {
        let mut r = ToolRegistry::new();
        r.register(FakeTool {
            name: "a",
            output: "first",
            fail: false,
        })
        .register(FakeTool {
            name: "a",
            output: "second",
            fail: false,
        });
        assert_eq!(r.names(), vec!["a"]);
        assert_eq!(r.schemas()[0].description, "second");
        assert_eq!(
            r.call("a", "{}", &SharedState::new()).await.unwrap(),
            "second"
        );
    }

    #[tokio::test]
    async fn get_returns_tool_with_error_semantics() {
        let mut r = ToolRegistry::new();
        r.register(FakeTool {
            name: "a",
            output: "",
            fail: true,
        });
        // Bypass call: invoke Tool::call directly, keeping the error
        // semantics (Execution) in the Result.
        let tool = r.get("a").expect("registered");
        let result = tool.call(serde_json::json!({}), &SharedState::new()).await;
        assert!(matches!(result, Err(ToolError::Execution(_))));
        assert!(r.get("nope").is_none());
    }

    #[tokio::test]
    async fn call_succeeds() {
        assert_eq!(
            registry()
                .call("calculator", "{}", &SharedState::new())
                .await
                .unwrap(),
            "calculator"
        );
    }

    #[tokio::test]
    async fn call_unknown_tool_returns_not_found() {
        // Classification rides along with Err; Display is the
        // "error-to-text".
        let err = registry()
            .call("nope", "{}", &SharedState::new())
            .await
            .unwrap_err();
        assert!(matches!(&err, RegistryError::NotFound(name) if name == "nope"));
        assert_eq!(err.to_string(), "tool not found: nope");
    }

    #[tokio::test]
    async fn call_invalid_json_returns_invalid_arguments() {
        let err = registry()
            .call("calculator", "not-json", &SharedState::new())
            .await
            .unwrap_err();
        assert!(matches!(err, RegistryError::InvalidArguments(_)));
        assert!(err.to_string().starts_with("invalid arguments:"));
    }

    #[tokio::test]
    async fn call_structural_error_returns_invalid_arguments_not_execution() {
        // Valid JSON with a mismatched structure (missing required field):
        // the tool rejects with InvalidArguments, classified as
        // InvalidArguments rather than Execution — an argument semantics
        // error, not an execution failure.
        struct StrictTool;
        #[async_trait::async_trait]
        impl Tool for StrictTool {
            fn schema(&self) -> ToolSchema {
                ToolSchema {
                    name: "strict".into(),
                    description: "requires a".into(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": { "a": { "type": "integer" } },
                        "required": ["a"],
                    }),
                }
            }
            async fn call(
                &self,
                arguments: serde_json::Value,
                _state: &SharedState,
            ) -> Result<String, ToolError> {
                let a = arguments
                    .get("a")
                    .ok_or_else(|| ToolError::InvalidArguments("missing field `a`".into()))?;
                Ok(a.to_string())
            }
        }

        let mut r = ToolRegistry::new();
        r.register(StrictTool);
        let err = r
            .call("strict", r#"{"b":1}"#, &SharedState::new())
            .await
            .unwrap_err();
        assert!(matches!(&err, RegistryError::InvalidArguments(msg) if msg == "missing field `a`"));
        assert_eq!(err.to_string(), "invalid arguments: missing field `a`");
    }

    #[tokio::test]
    async fn call_execution_error_returns_execution_with_source() {
        let mut r = ToolRegistry::new();
        r.register(FakeTool {
            name: "broken",
            output: "",
            fail: true,
        });
        // Classification preserved; Display carries the tool name with a
        // single "tool error" prefix ("tool error: broken failed: ...");
        // source reaches the underlying ToolError.
        let err = r
            .call("broken", "{}", &SharedState::new())
            .await
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "tool error: broken failed: execution failed: boom"
        );
        assert!(matches!(err.source(), Some(e) if e.to_string() == "execution failed: boom"));
    }

    /// Tool panics do not escape the framework: captured as an execution
    /// error (text passed back to the model, loop continues).
    #[tokio::test]
    async fn tool_panic_is_captured_as_execution_error() {
        struct PanickingTool;
        #[async_trait::async_trait]
        impl Tool for PanickingTool {
            fn schema(&self) -> ToolSchema {
                ToolSchema {
                    name: "panic".into(),
                    description: "panics".into(),
                    parameters: serde_json::json!({}),
                }
            }
            async fn call(
                &self,
                _arguments: serde_json::Value,
                _state: &SharedState,
            ) -> Result<String, ToolError> {
                panic!("boom")
            }
        }

        let mut r = ToolRegistry::new();
        r.register(PanickingTool);
        let err = r
            .call("panic", "{}", &SharedState::new())
            .await
            .unwrap_err();
        // Message carries the tool name + panic content, so the model /
        // user knows which tool crashed.
        assert!(
            matches!(err, RegistryError::Execution { name, source: ToolError::Execution(msg) }
            if name == "panic" && msg.contains("panicked") && msg.contains("boom"))
        );
    }

    #[test]
    fn retain_removes_by_prefix_in_registration_order() {
        // MCP server tools carry a "{server_name}__" prefix: bulk-remove
        // by prefix.
        let mut r = ToolRegistry::new();
        r.register(echo("fs__read"))
            .register(echo("fs__write"))
            .register(echo("calc"));
        let removed = r.retain(|name| !name.starts_with("fs__"));
        assert_eq!(removed, ["fs__read", "fs__write"]);
        assert_eq!(r.names(), ["calc"]);
    }

    #[test]
    fn retain_keep_all_returns_empty() {
        let mut r = registry();
        assert!(r.retain(|_| true).is_empty());
        assert_eq!(r.names(), ["search", "calculator"]);
    }

    #[test]
    fn subset_keeps_registration_order() {
        // Arguments in random order, result still follows the main
        // registration order.
        let sub = registry().subset(&["calculator", "search"]).unwrap();
        assert_eq!(sub.names(), vec!["search", "calculator"]);
    }

    #[tokio::test]
    async fn subset_duplicate_name_takes_latest() {
        let mut r = ToolRegistry::new();
        r.register(FakeTool {
            name: "a",
            output: "first",
            fail: false,
        })
        .register(FakeTool {
            name: "a",
            output: "second",
            fail: false,
        });
        let sub = r.subset(&["a"]).unwrap();
        assert_eq!(sub.names(), vec!["a"]);
        assert_eq!(
            sub.call("a", "{}", &SharedState::new()).await.unwrap(),
            "second"
        );
    }

    #[test]
    fn subset_missing_names_error_with_list() {
        let err = registry()
            .subset(&["search", "nope", "calculator", "also-nope"])
            .unwrap_err();
        assert_eq!(err.names(), &["nope", "also-nope"]);
    }

    #[tokio::test]
    async fn subset_shares_tool_instances() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut r = ToolRegistry::new();
        r.register(CountingTool {
            name: "counter",
            calls: calls.clone(),
        });
        let sub = r.subset(&["counter"]).unwrap();
        // Parent and child each execute once; sharing one instance yields
        // count 2, not two independent instances.
        r.call("counter", "{}", &SharedState::new()).await.unwrap();
        sub.call("counter", "{}", &SharedState::new())
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    /// Clone: the cloned registry shares the same tool instances as the
    /// original (Arc values, cheap to clone).
    #[tokio::test]
    async fn clone_shares_tool_instances() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut r = ToolRegistry::new();
        r.register(CountingTool {
            name: "counter",
            calls: calls.clone(),
        });
        let r2 = r.clone();
        r.call("counter", "{}", &SharedState::new()).await.unwrap();
        r2.call("counter", "{}", &SharedState::new()).await.unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    /// state is passed through call: the tool reads exactly the
    /// caller-provided instance at call time.
    #[tokio::test]
    async fn call_passes_shared_state_to_tool() {
        struct StateTool;
        #[async_trait::async_trait]
        impl Tool for StateTool {
            fn schema(&self) -> ToolSchema {
                ToolSchema {
                    name: "state_tool".into(),
                    description: "read and write shared state".into(),
                    parameters: serde_json::json!({}),
                }
            }
            async fn call(
                &self,
                _arguments: serde_json::Value,
                state: &SharedState,
            ) -> Result<String, ToolError> {
                state.with_mut::<usize>(|n| *n += 1);
                Ok(format!("count={}", state.get::<usize>().unwrap_or(0)))
            }
        }

        let state = SharedState::new();
        state.insert(0usize);
        let mut r = ToolRegistry::new();
        r.register(StateTool);

        // Consecutive calls on the same instance: count accumulates
        // (proving the tool reads and writes the caller-provided
        // instance).
        assert_eq!(r.call("state_tool", "{}", &state).await.unwrap(), "count=1");
        assert_eq!(r.call("state_tool", "{}", &state).await.unwrap(), "count=2");
    }
}
