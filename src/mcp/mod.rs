//! MCP client adapter — brings tools exposed by external MCP servers into
//! molo.
//!
//! Usage in three steps: construct an [`McpClient`] → pull tools via
//! [`tools`](McpClient::tools) → register each into the
//! [`ToolRegistry`](crate::tool::ToolRegistry); the agent then uses them like
//! any ordinary tool, unaware of protocol differences:
//!
//! ```text
//! let mut client = McpClient::from_command("filesystem", command);
//! let mut registry = ToolRegistry::new();
//! for tool in client.tools().await? {
//!     registry.register(tool);          // display name like "filesystem__read_file"
//! }
//! let agent = ReActAgent::new(provider, registry, "system prompt");
//! ```
//!
//! # Protocol background
//!
//! The newer MCP protocol is stateless: there is no more initialize handshake,
//! capability negotiation, or sessions; the version and capabilities ride
//! along with every request. stdio and Streamable HTTP are the only active
//! transports. This component connects in the stateless shape and falls back
//! to legacy servers via the `Auto` lifecycle (when discovery fails and the
//! server proves to be legacy, it automatically falls back to the old
//! handshake).
//!
//! This component only does **client-side consumption** (wiring external MCP
//! server tools into molo); the server side (exposing molo tools as an MCP
//! server) is out of scope. Resource / prompt protocol capabilities are also
//! not wired in — this component focuses on tool integration.

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::process::Command as StdCommand;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use rmcp::model::{CallToolRequestParams, ContentBlock, ProtocolVersion};
use rmcp::service::{ClientCacheConfig, ClientInitializeError, RoleClient, RunningService};
use rmcp::transport::child_process::TokioChildProcess;
use rmcp::transport::streamable_http_client::StreamableHttpClientWorker;
use rmcp::{ClientLifecycleMode, ClientServiceExt};
use serde::{Deserialize, Serialize};

use crate::effect::{DisplayFormat, DisplayOutput, EffectKind, EffectRequest, RiskLevel};
#[cfg(feature = "harness")]
use crate::harness::{
    ClassifiedEffect, EffectExecutor, ExecutionError, ExecutionPolicy, NetworkPolicy,
    PolicyDecision, PolicyEngine, RawEffectOutput, SandboxPolicy,
};
use crate::run::{RunContext, RunMetadata};
use crate::tool::{
    SideEffectLevel, Tool, ToolContext, ToolError, ToolNamespace, ToolOutput, ToolPolicy,
    ToolResult, ToolSchema, ToolSource, ToolTrustLevel,
};

/// Connection shape: captured at construction, used at `connect()` time.
#[derive(Debug)]
enum ConnSpec {
    /// stdio child process: program + args, rebuildable (auto-reconnects after
    /// cleanup).
    Args { program: String, args: Vec<String> },
    /// stdio child process: a full `Command` (env / cwd etc.), one-shot —
    /// consumed at connect time; reconnecting requires rebuilding the
    /// McpClient (see from_command_configured).
    Command(StdCommand),
    /// A consumed one-shot command (reconnection gives guidance).
    Consumed,
    /// Streamable HTTP: server URL, reusable.
    Url(String),
}

/// Default timeout for the whole connect flow (child process startup +
/// protocol handshake).
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Default timeout for a single tools/list request.
const DEFAULT_LIST_TIMEOUT: Duration = Duration::from_secs(10);
/// Default timeout for a single tools/call invocation.
const DEFAULT_CALL_TIMEOUT: Duration = Duration::from_secs(60);

/// Stable host-assigned MCP server id.
///
/// MCP server-reported names are not globally unique. Hosts should choose a
/// stable id for policy, audit, registry namespace, and effect payloads.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct McpServerId(String);

impl McpServerId {
    /// Constructs an MCP server id.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns the server id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for McpServerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for McpServerId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<&str> for McpServerId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

/// MCP tool id scoped to one server.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct McpToolId {
    /// Host-assigned MCP server id.
    pub server: McpServerId,
    /// Raw tool name exposed by that server.
    pub raw_name: String,
}

impl McpToolId {
    /// Constructs a scoped MCP tool id.
    pub fn new(server: impl Into<McpServerId>, raw_name: impl Into<String>) -> Self {
        Self {
            server: server.into(),
            raw_name: raw_name.into(),
        }
    }
}

/// Cache hint for an MCP tool catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpCacheHint {
    /// Cache time-to-live in milliseconds, when supplied by the server or
    /// host adapter.
    pub ttl_ms: Option<u64>,
}

/// Description of one MCP tool discovered from a server.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpToolDescriptor {
    /// Scoped MCP tool id.
    pub id: McpToolId,
    /// Provider-facing, globally disambiguated tool name.
    pub display_name: String,
    /// Tool description.
    pub description: String,
    /// Input JSON Schema.
    pub input_schema: serde_json::Value,
    /// Output JSON Schema, when the server supplies one.
    pub output_schema: Option<serde_json::Value>,
    /// Server-supplied annotations. Treat as untrusted unless the server is
    /// explicitly trusted by host policy.
    pub annotations: serde_json::Value,
    /// Stable digest of the input schema used to detect catalog drift.
    pub schema_digest: String,
    /// Optional cache hint.
    pub cache: Option<McpCacheHint>,
    /// Host/application metadata.
    pub metadata: RunMetadata,
}

impl McpToolDescriptor {
    /// Source metadata suitable for [`ToolRegistry`](crate::ToolRegistry)
    /// source-aware registration.
    pub fn source(&self, trust: ToolTrustLevel) -> ToolSource {
        ToolSource::new(
            ToolNamespace::mcp_server(self.id.server.as_str()),
            self.id.raw_name.clone(),
            self.display_name.clone(),
        )
        .with_trust(trust)
        .with_metadata(self.metadata.clone())
    }
}

/// Snapshot of a server's MCP tool catalog.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpToolCatalog {
    /// Host-assigned server id.
    pub server: McpServerId,
    /// Tools in deterministic server/list order.
    pub tools: Vec<McpToolDescriptor>,
    /// Fetch time.
    pub fetched_at: SystemTime,
    /// Expiration time, when known.
    pub expires_at: Option<SystemTime>,
    /// Host/application metadata.
    pub metadata: RunMetadata,
}

/// MCP tool execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum McpToolMode {
    /// Tool calls the server directly inside [`Tool::call`].
    DirectOutput,
    /// Tool only emits an [`EffectRequest`] for a harness to execute.
    GovernedEffect,
}

/// MCP client adapter: connects to an MCP server and converts its tools into
/// molo tools.
///
/// This type is a **converter**, not a long-lived handle: once connected, each
/// generated [`McpTool`] holds its own connection reference, and the
/// `McpClient` itself can be dropped after assembly; the connection is kept
/// alive by the tools and closes automatically when the last tool is dropped.
///
/// Connections are **lazy**: construction initiates nothing; the first
/// [`tools`](McpClient::tools) call connects automatically; an explicit
/// [`connect`](McpClient::connect) is provided for startup pre-flight checks.
/// Every [`tools`](McpClient::tools) call re-pulls the tool list — tool
/// changes on the server during the connection take effect automatically, with
/// no cache staleness.
///
/// The assembly methods (`with_*`) take `&mut self` (unlike the consuming
/// style used elsewhere in the framework): the instance is still needed after
/// assembly (connect, pull tools), and the mutating style keeps ownership of
/// it; the cost is that chaining on temporaries is not possible.
///
/// # Assembly example (real server)
///
/// A complete runnable example lives in `examples/mcp.rs` (self-contained: the
/// example forks a child process that acts as the server).
///
/// # Panics
///
/// Methods on this type never panic; network / protocol failures uniformly go
/// through [`McpError`].
pub struct McpClient {
    server_name: String,
    spec: ConnSpec,
    prefix: bool,
    running: Option<Arc<RunningService<RoleClient, ()>>>,
    /// Timeout for the whole connect flow (child process startup + protocol
    /// handshake); 10s by default.
    connect_timeout: Duration,
    /// Timeout for a single tools/list request; 10s by default.
    list_timeout: Duration,
    /// Timeout for a single tools/call invocation; 60s by default.
    call_timeout: Duration,
}

impl McpClient {
    /// Connects via a stdio child process (program + args); `server_name` also
    /// serves as the namespace prefix for tool names (see
    /// [`with_name_prefix`](McpClient::with_name_prefix)).
    ///
    /// Stored as "program + args", so it can auto-reconnect after `cleanup`;
    /// the process uses piped stdio by default. Use
    /// [`from_command_configured`](McpClient::from_command_configured) when a
    /// full env / cwd configuration is needed.
    ///
    /// ```
    /// use molo::McpClient;
    ///
    /// let client = McpClient::from_command(
    ///     "filesystem",
    ///     "npx",
    ///     ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    /// );
    /// assert_eq!(client.server_name(), "filesystem");
    /// ```
    pub fn from_command(
        server_name: impl Into<String>,
        program: impl Into<String>,
        args: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            server_name: server_name.into(),
            spec: ConnSpec::Args {
                program: program.into(),
                args: args.into_iter().map(Into::into).collect(),
            },
            prefix: true,
            running: None,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            list_timeout: DEFAULT_LIST_TIMEOUT,
            call_timeout: DEFAULT_CALL_TIMEOUT,
        }
    }

    /// Connects via a stdio child process (full `Command` configuration);
    /// `server_name` semantics are the same as
    /// [`from_command`](McpClient::from_command).
    ///
    /// `command` can configure environment variables, working directory, etc.
    /// (the process uses piped stdio by default); note it is **one-shot**: it
    /// is consumed at connect time, and reconnecting after `cleanup` returns
    /// [`McpError::Connect`] suggesting a fresh McpClient — use the
    /// program + args form of [`from_command`](McpClient::from_command) when
    /// reconnection matters.
    pub fn from_command_configured(server_name: impl Into<String>, command: StdCommand) -> Self {
        Self {
            server_name: server_name.into(),
            spec: ConnSpec::Command(command),
            prefix: true,
            running: None,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            list_timeout: DEFAULT_LIST_TIMEOUT,
            call_timeout: DEFAULT_CALL_TIMEOUT,
        }
    }

    /// Connects via Streamable HTTP; `server_name` semantics are the same as
    /// [`from_command`](McpClient::from_command).
    ///
    /// The URL is resolved at [`connect`](McpClient::connect) time; an invalid
    /// URL is returned as [`McpError::Connect`].
    ///
    /// ```
    /// use molo::McpClient;
    ///
    /// let client = McpClient::from_url("weather", "https://example.com/mcp");
    /// assert_eq!(client.server_name(), "weather");
    /// ```
    pub fn from_url(server_name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            server_name: server_name.into(),
            spec: ConnSpec::Url(url.into()),
            prefix: true,
            running: None,
            connect_timeout: DEFAULT_CONNECT_TIMEOUT,
            list_timeout: DEFAULT_LIST_TIMEOUT,
            call_timeout: DEFAULT_CALL_TIMEOUT,
        }
    }

    /// Namespace-prefix switch for tool names (on by default).
    ///
    /// When on, the tool display name is `{server_name}__{raw tool name}`
    /// (e.g., `filesystem__read_file`), avoiding name collisions across
    /// multiple servers; when off, the tool name is the server's raw name — if
    /// it collides with an already-registered tool, `ToolRegistry` uses
    /// "last registration wins" semantics (silent overwrite), at the caller's
    /// own risk.
    ///
    /// Only affects the display names produced by **subsequent**
    /// [`tools`](McpClient::tools) calls; already-created [`McpTool`]s are
    /// unchanged.
    pub fn with_name_prefix(&mut self, enabled: bool) -> &mut Self {
        self.prefix = enabled;
        self
    }

    /// Timeout for the whole connect flow (default 10s): child process startup +
    /// protocol handshake. A timeout returns [`McpError::Connect`] with
    /// "timed out" in the message. Increase it for slow startup scenarios such
    /// as npx's first-run package fetch.
    pub fn with_connect_timeout(&mut self, timeout: Duration) -> &mut Self {
        self.connect_timeout = timeout;
        self
    }

    /// Timeout for a single tools/list request (default 10s).
    pub fn with_list_timeout(&mut self, timeout: Duration) -> &mut Self {
        self.list_timeout = timeout;
        self
    }

    /// Timeout for a single tools/call invocation (default 60s): when the
    /// server hangs or the network goes black-hole, the tool call terminates
    /// with a [`ToolError::Execution`] timeout instead of blocking the
    /// inference loop forever.
    pub fn with_call_timeout(&mut self, timeout: Duration) -> &mut Self {
        self.call_timeout = timeout;
        self
    }

    /// The server name (also used as the namespace prefix).
    pub fn server_name(&self) -> &str {
        &self.server_name
    }

    /// Establishes the connection explicitly; idempotent (returns immediately
    /// when already connected).
    ///
    /// For startup pre-flight checks (exposing configuration errors such as
    /// child process startup failure or an unreachable URL as early as
    /// possible); normally no explicit call is needed —
    /// [`tools`](McpClient::tools) connects automatically.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::Connect`] with the concrete reason when the child
    /// process fails to start, the URL is invalid or unreachable, or the
    /// protocol handshake fails.
    pub async fn connect(&mut self) -> Result<(), McpError> {
        if self.running.is_some() {
            return Ok(());
        }
        // Take the server name first: the match holds a mutable borrow of
        // self.spec, and error construction needs the name.
        let server_name = self.server_name.clone();
        let running = match std::mem::replace(&mut self.spec, ConnSpec::Consumed) {
            // Program + args form: restore the spec after connecting (success
            // or failure) so it can be reconnected repeatedly.
            ConnSpec::Args { program, args } => {
                let result = {
                    let mut command = StdCommand::new(program.clone());
                    command.args(args.clone());
                    let transport =
                        match TokioChildProcess::new(tokio::process::Command::from(command)) {
                            Ok(transport) => transport,
                            Err(e) => {
                                // spawn failed (e.g., program missing):
                                // restore the spec before returning —
                                // otherwise the spec stays Consumed and a
                                // retry after fixing the config would only get
                                // "already consumed", forcing a rebuild.
                                self.spec = ConnSpec::Args { program, args };
                                return Err(McpError::Connect {
                                    server: server_name.clone(),
                                    message: format!("spawn failed: {e}"),
                                });
                            }
                        };
                    serve_with_timeout(
                        &server_name,
                        self.connect_timeout,
                        ().serve_with_lifecycle(transport, lifecycle_mode()),
                    )
                    .await
                };
                self.spec = ConnSpec::Args { program, args };
                result
            }
            // Full-Command form: one-shot — the command is consumed with the
            // connection; reconnecting requires rebuilding the McpClient.
            ConnSpec::Command(command) => {
                let program = command.get_program().to_string_lossy().to_string();
                let transport = TokioChildProcess::new(tokio::process::Command::from(command))
                    .map_err(|e| McpError::Connect {
                        server: server_name.clone(),
                        message: format!("spawn {program}: {e}"),
                    })?;
                serve_with_timeout(
                    &server_name,
                    self.connect_timeout,
                    ().serve_with_lifecycle(transport, lifecycle_mode()),
                )
                .await
            }
            // URL form: restore the spec after connecting (success or failure)
            // so it can be reconnected repeatedly.
            ConnSpec::Url(url) => {
                let result = serve_with_timeout(
                    &server_name,
                    self.connect_timeout,
                    ().serve_with_lifecycle(
                        StreamableHttpClientWorker::<reqwest::Client>::new_simple(url.clone()),
                        lifecycle_mode(),
                    ),
                )
                .await;
                self.spec = ConnSpec::Url(url);
                result
            }
            ConnSpec::Consumed => {
                return Err(McpError::Connect {
                    server: server_name.clone(),
                    message: "stdio command already consumed; create a new McpClient to reconnect"
                        .into(),
                });
            }
        }?;
        // Disable rmcp's response cache: every tools() call re-pulls (server
        // tool changes take effect automatically, and disconnects are not
        // masked by a stale cache); rmcp enables caching by default honoring
        // the protocol-level ttlMs, which conflicts with this behavior.
        running
            .peer()
            .set_response_cache_config(ClientCacheConfig::disabled())
            .await;
        self.running = Some(Arc::new(running));
        Ok(())
    }

    /// Disconnects; idempotent.
    ///
    /// Releases the connection reference held by this component: if any
    /// generated [`McpTool`] is still referenced (e.g., still registered in a
    /// `ToolRegistry`), the tools keep the connection alive and calls work as
    /// usual; once all tools are released, the connection closes automatically
    /// (the child process terminates). A later
    /// [`tools`](McpClient::tools) call reconnects automatically.
    pub async fn cleanup(&mut self) -> Result<(), McpError> {
        self.running = None;
        Ok(())
    }

    /// Pulls the server's tool catalog; connects automatically and re-pulls
    /// on every call.
    ///
    /// Discovery is separate from tool wrapping so hosts can inspect
    /// descriptors, register source metadata, or choose governed effect tools.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::Connect`] when the automatic connect fails; returns
    /// [`McpError::ListTools`] with the concrete reason when listing tools
    /// fails.
    pub async fn tool_catalog(&mut self) -> Result<McpToolCatalog, McpError> {
        let tools = self.list_tools_raw().await?;
        Ok(McpToolCatalog {
            server: McpServerId::new(self.server_name.clone()),
            tools: tools
                .into_iter()
                .map(|tool| descriptor_from_tool(&self.server_name, tool, self.prefix))
                .collect(),
            fetched_at: SystemTime::now(),
            expires_at: None,
            metadata: RunMetadata::new(),
        })
    }

    /// Pulls all tools from the server and converts them into direct molo
    /// tools; connects automatically and re-pulls on every call.
    ///
    /// The returned [`McpDirectTool`]s can be registered directly into a
    /// `ToolRegistry` (or cloned to share across multiple registries); tool
    /// names follow [`with_name_prefix`](McpClient::with_name_prefix).
    ///
    /// This is the direct convenience path: each tool calls the MCP server
    /// inside [`Tool::call`] and therefore does not pass through harness
    /// policy, approval, sandbox/network policy, audit, or transcript. Use
    /// [`effect_tools`](Self::effect_tools) with `mcp + harness` for
    /// production side-effect governance.
    ///
    /// # Example
    ///
    /// A full connect-and-register example lives in `examples/mcp.rs` (embeds
    /// a minimal MCP server that the child process spawns itself, no external
    /// services needed).
    ///
    /// # Errors
    ///
    /// Returns [`McpError::Connect`] when the automatic connect fails; returns
    /// [`McpError::ListTools`] with the concrete reason when listing tools
    /// fails (protocol-level error).
    pub async fn tools(&mut self) -> Result<Vec<McpDirectTool>, McpError> {
        self.connect().await?;
        let Some(running) = self.running.clone() else {
            return Err(McpError::ListTools {
                server: self.server_name.clone(),
                message: "no active connection".into(),
            });
        };
        Ok(self
            .tool_catalog()
            .await?
            .tools
            .into_iter()
            .map(|descriptor| {
                McpDirectTool::new(descriptor, self.call_timeout, Arc::clone(&running))
            })
            .collect())
    }

    /// Pulls all tools from the server and converts them into governed effect
    /// tools.
    ///
    /// These tools do not hold an MCP connection and do not call the server
    /// directly. A call returns [`ToolResult::Effect`] with
    /// [`EffectKind::Mcp`], to be executed by a harness with an
    /// [`McpEffectExecutor`].
    #[cfg(feature = "harness")]
    pub async fn effect_tools(&mut self) -> Result<Vec<McpEffectTool>, McpError> {
        Ok(self
            .tool_catalog()
            .await?
            .tools
            .into_iter()
            .map(McpEffectTool::new)
            .collect())
    }

    async fn list_tools_raw(&mut self) -> Result<Vec<rmcp::model::Tool>, McpError> {
        self.connect().await?;
        let Some(running) = self.running.clone() else {
            return Err(McpError::ListTools {
                server: self.server_name.clone(),
                message: "no active connection".into(),
            });
        };
        let tools = tokio::time::timeout(self.list_timeout, running.peer().list_all_tools())
            .await
            .map_err(|_| McpError::ListTools {
                server: self.server_name.clone(),
                message: "list tools timed out".into(),
            })?
            .map_err(|e| McpError::ListTools {
                server: self.server_name.clone(),
                message: e.to_string(),
            })?;
        // Upper bound: a malicious/compromised server could expose an enormous
        // number of tools, all registered into the context.
        if tools.len() > MAX_MCP_TOOLS {
            return Err(McpError::ListTools {
                server: self.server_name.clone(),
                message: format!(
                    "server exposes too many tools ({} > {MAX_MCP_TOOLS})",
                    tools.len()
                ),
            });
        }
        Ok(tools)
    }
}

impl std::fmt::Debug for McpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpClient")
            .field("server_name", &self.server_name)
            .field("spec", &self.spec)
            .field("prefix", &self.prefix)
            .field("connected", &self.running.is_some())
            .finish()
    }
}

/// Direct adapter tool produced by [`McpClient::tools`]: implements molo's
/// [`Tool`] trait and proxies calls to the MCP server.
///
/// # Safety Boundary
///
/// This direct path calls the MCP server inside [`Tool::call`]. It does not
/// pass through the harness lifecycle, so production applications with
/// external or side-effecting servers should prefer [`McpEffectTool`] and
/// [`McpEffectExecutor`] with the `mcp + harness` features enabled.
///
/// Holds a connection reference (Arc) internally and is Clone-able — register
/// it in multiple registries (or share between a main agent and sub-agents)
/// sharing the same connection and state.
#[derive(Clone)]
pub struct McpDirectTool {
    /// Display name (per the prefix switch; visible to the model).
    name: String,
    /// Host-assigned server id.
    server_id: McpServerId,
    /// The raw name on the server (used for forwarding requests).
    raw_name: String,
    /// Tool description.
    description: String,
    /// Parameter JSON Schema (passed through verbatim).
    parameters: serde_json::Value,
    /// Catalog schema digest.
    schema_digest: String,
    /// Source annotations.
    annotations: serde_json::Value,
    /// Timeout for a single call (from the producing McpClient's config).
    call_timeout: Duration,
    /// Connection handle.
    peer: Arc<RunningService<RoleClient, ()>>,
}

impl McpDirectTool {
    /// Builds a direct adapter tool from a descriptor.
    fn new(
        descriptor: McpToolDescriptor,
        call_timeout: Duration,
        peer: Arc<RunningService<RoleClient, ()>>,
    ) -> Self {
        Self {
            name: descriptor.display_name,
            server_id: descriptor.id.server,
            raw_name: descriptor.id.raw_name,
            description: descriptor.description,
            parameters: descriptor.input_schema,
            schema_digest: descriptor.schema_digest,
            annotations: descriptor.annotations,
            call_timeout,
            peer,
        }
    }

    /// Source metadata for registering this direct tool in a source-aware
    /// [`ToolRegistry`](crate::ToolRegistry).
    pub fn source(&self) -> ToolSource {
        ToolSource::new(
            ToolNamespace::mcp_server(self.server_id.as_str()),
            self.raw_name.clone(),
            self.name.clone(),
        )
        .with_trust(ToolTrustLevel::External)
    }
}

/// Backward-compatible alias for the direct MCP tool wrapper.
///
/// New code should prefer [`McpDirectTool`] for prototype/direct execution or
/// [`McpEffectTool`] for harness-governed execution.
pub type McpTool = McpDirectTool;

/// Mapping result of a server tool description: display name / raw name /
/// description / parameter Schema.
#[cfg(test)]
struct MappedTool {
    name: String,
    raw_name: String,
    description: String,
    parameters: serde_json::Value,
}

/// Pure mapping: server tool description → [`MappedTool`].
#[cfg(test)]
fn map_tool(server_name: &str, tool: rmcp::model::Tool, prefix: bool) -> MappedTool {
    let raw_name = tool.name.to_string();
    MappedTool {
        name: tool_display_name(server_name, &raw_name, prefix),
        raw_name,
        description: tool.description.unwrap_or_default().to_string(),
        parameters: serde_json::Value::Object((*tool.input_schema).clone()),
    }
}

fn descriptor_from_tool(
    server_name: &str,
    tool: rmcp::model::Tool,
    prefix: bool,
) -> McpToolDescriptor {
    let raw_name = tool.name.to_string();
    let input_schema = serde_json::Value::Object((*tool.input_schema).clone());
    let output_schema = tool
        .output_schema
        .map(|schema| serde_json::Value::Object((*schema).clone()));
    let annotations = tool
        .annotations
        .and_then(|annotations| serde_json::to_value(annotations).ok())
        .unwrap_or(serde_json::Value::Null);
    McpToolDescriptor {
        id: McpToolId::new(server_name, raw_name.clone()),
        display_name: tool_display_name(server_name, &raw_name, prefix),
        description: tool.description.unwrap_or_default().to_string(),
        schema_digest: schema_digest(&input_schema),
        input_schema,
        output_schema,
        annotations,
        cache: None,
        metadata: RunMetadata::new(),
    }
}

fn schema_digest(schema: &serde_json::Value) -> String {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let text = serde_json::to_string(schema).unwrap_or_else(|_| schema.to_string());
    let mut hash = FNV_OFFSET;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("fnv64:{hash:016x}")
}

#[cfg(feature = "harness")]
fn risk_rank(risk: RiskLevel) -> u8 {
    match risk {
        RiskLevel::Low => 0,
        RiskLevel::Medium => 1,
        RiskLevel::High => 2,
        RiskLevel::Critical => 3,
    }
}

impl std::fmt::Debug for McpDirectTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpDirectTool")
            .field("server_id", &self.server_id)
            .field("name", &self.name)
            .field("raw_name", &self.raw_name)
            .finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl Tool for McpDirectTool {
    fn schema(&self) -> ToolSchema {
        let mut metadata = RunMetadata::new();
        metadata.insert(
            "mcp_server_id".to_string(),
            serde_json::json!(self.server_id.as_str()),
        );
        metadata.insert(
            "mcp_raw_tool_name".to_string(),
            serde_json::json!(self.raw_name),
        );
        metadata.insert(
            "mcp_schema_digest".to_string(),
            serde_json::json!(self.schema_digest),
        );
        metadata.insert("mcp_annotations".to_string(), self.annotations.clone());
        ToolSchema::new(
            self.name.clone(),
            self.description.clone(),
            self.parameters.clone(),
        )
        .with_policy(ToolPolicy {
            side_effects: SideEffectLevel::External,
            risk: RiskLevel::Medium,
            timeout: Some(self.call_timeout),
            ..Default::default()
        })
        .with_metadata(metadata)
    }

    /// Proxies a call: forwards arguments via `tools/call` and joins the
    /// result content blocks into text.
    ///
    /// Both the server-returned **tool-level error** (`CallToolResult::error`)
    /// and **protocol-level errors** (transport / JSON-RPC failures) map to
    /// [`ToolError::Execution`] with the server text in the message, relayed
    /// back to the model by the agent loop; non-text content blocks (images
    /// etc.) render as placeholder descriptions.
    async fn call(
        &self,
        arguments: serde_json::Value,
        _context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let Some(args) = arguments.as_object() else {
            return Err(ToolError::InvalidArguments(
                "mcp tool arguments must be a JSON object".into(),
            ));
        };
        let params = CallToolRequestParams::new(self.raw_name.clone()).with_arguments(args.clone());
        // Call timeout: never block the inference loop forever when the server
        // hangs or the network goes black-hole.
        let result = tokio::time::timeout(self.call_timeout, self.peer.peer().call_tool(params))
            .await
            .map_err(|_| ToolError::Execution("mcp tool call timed out".into()))?
            .map_err(|e| ToolError::Execution(format!("mcp tool call failed: {e}")))?;
        let mut text = content_to_text(&result.content);
        // Structured results of the newer protocol (`structured_content`): a
        // standalone JSON block appended after the text, not silently dropped
        // (so the model can access the full structured data).
        if let Some(structured) = &result.structured_content {
            let rendered = serde_json::to_string_pretty(structured).map_err(|e| {
                ToolError::Execution(format!("structured content serialization failed: {e}"))
            })?;
            text = format!("{text}\n[structured]\n{rendered}");
        }
        if result.is_error.unwrap_or(false) {
            let message = if text.is_empty() {
                "mcp tool reported an error".to_string()
            } else {
                text
            };
            Err(ToolError::Execution(message))
        } else {
            Ok(ToolOutput::text(text).into())
        }
    }
}

/// Payload carried inside an [`EffectKind::Mcp`] request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpCallPayload {
    /// Target server id. Executors must resolve this against host-owned
    /// client configuration, not against model-provided URLs or commands.
    pub server_id: McpServerId,
    /// Raw MCP tool name on the target server.
    pub tool_name: String,
    /// Provider-facing display name that produced the call.
    pub display_name: String,
    /// Tool arguments.
    pub arguments: serde_json::Value,
    /// Catalog schema digest observed at assembly time.
    pub schema_digest: Option<String>,
    /// MCP protocol version, when known.
    pub protocol_version: Option<String>,
    /// Multi-round tool-result input responses, when supported by the host.
    pub input_responses: Option<serde_json::Value>,
    /// Multi-round tool-result request state, when supported by the host.
    pub request_state: Option<String>,
    /// Host/application metadata.
    pub metadata: RunMetadata,
}

impl McpCallPayload {
    /// Constructs an MCP call payload.
    pub fn new(
        id: McpToolId,
        display_name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self {
            server_id: id.server,
            tool_name: id.raw_name,
            display_name: display_name.into(),
            arguments,
            schema_digest: None,
            protocol_version: None,
            input_responses: None,
            request_state: None,
            metadata: RunMetadata::new(),
        }
    }

    /// Sets the catalog schema digest.
    pub fn with_schema_digest(mut self, digest: impl Into<String>) -> Self {
        self.schema_digest = Some(digest.into());
        self
    }

    /// Sets host/application metadata.
    pub fn with_metadata(mut self, metadata: RunMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Converts this payload into a harness effect request.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::InvalidPayload`] if the payload cannot be
    /// serialized.
    pub fn into_effect(self) -> Result<EffectRequest, McpError> {
        let description = format!(
            "call MCP tool {} on server {}",
            self.tool_name, self.server_id
        );
        let metadata = self.metadata.clone();
        let request = EffectRequest::new(
            EffectKind::Mcp,
            description,
            serde_json::to_value(&self).map_err(|e| McpError::InvalidPayload(e.to_string()))?,
        )
        .with_risk(RiskLevel::Medium)
        .with_metadata(metadata);
        Ok(request)
    }

    /// Decodes an MCP call payload from an effect request.
    ///
    /// # Errors
    ///
    /// Returns [`McpError::InvalidPayload`] when the effect kind is not MCP or
    /// the payload cannot be decoded.
    pub fn from_effect(request: &EffectRequest) -> Result<Self, McpError> {
        if request.kind != EffectKind::Mcp {
            return Err(McpError::InvalidPayload(format!(
                "expected EffectKind::Mcp, got {:?}",
                request.kind
            )));
        }
        serde_json::from_value(request.payload.clone())
            .map_err(|e| McpError::InvalidPayload(e.to_string()))
    }
}

/// MCP tool wrapper for harness-governed execution.
///
/// Calls to this tool do not contact the MCP server. They return
/// [`ToolResult::Effect`] so an outer harness can apply policy, approval,
/// network/sandbox limits, audit, and transcript recording before an
/// [`McpEffectExecutor`] performs `tools/call`.
#[cfg(feature = "harness")]
#[derive(Debug, Clone)]
pub struct McpEffectTool {
    descriptor: McpToolDescriptor,
    risk: RiskLevel,
    timeout: Option<Duration>,
}

#[cfg(feature = "harness")]
impl McpEffectTool {
    /// Constructs a governed MCP effect tool from a catalog descriptor.
    pub fn new(descriptor: McpToolDescriptor) -> Self {
        Self {
            descriptor,
            risk: RiskLevel::Medium,
            timeout: None,
        }
    }

    /// Sets the request-declared risk for generated effects.
    pub fn with_risk(mut self, risk: RiskLevel) -> Self {
        self.risk = risk;
        self
    }

    /// Sets the request timeout suggestion for generated effects.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Source metadata for registry source-aware registration.
    pub fn source(&self) -> ToolSource {
        self.descriptor.source(ToolTrustLevel::External)
    }
}

#[cfg(feature = "harness")]
#[async_trait::async_trait]
impl Tool for McpEffectTool {
    fn schema(&self) -> ToolSchema {
        let mut metadata = self.descriptor.metadata.clone();
        metadata.insert(
            "mcp_server_id".to_string(),
            serde_json::json!(self.descriptor.id.server.as_str()),
        );
        metadata.insert(
            "mcp_raw_tool_name".to_string(),
            serde_json::json!(self.descriptor.id.raw_name),
        );
        metadata.insert(
            "mcp_schema_digest".to_string(),
            serde_json::json!(self.descriptor.schema_digest),
        );
        ToolSchema::new(
            self.descriptor.display_name.clone(),
            self.descriptor.description.clone(),
            self.descriptor.input_schema.clone(),
        )
        .with_policy(ToolPolicy {
            side_effects: SideEffectLevel::External,
            risk: self.risk,
            timeout: self.timeout,
            ..Default::default()
        })
        .with_metadata(metadata)
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        if !arguments.is_object() {
            return Err(ToolError::InvalidArguments(
                "mcp tool arguments must be a JSON object".into(),
            ));
        }
        let mut metadata = self.descriptor.metadata.clone();
        metadata.insert(
            "source_tool_call_id".to_string(),
            serde_json::json!(context.tool_call_id),
        );
        metadata.insert(
            "source_tool_name".to_string(),
            serde_json::json!(context.tool_name),
        );
        let payload = McpCallPayload::new(
            self.descriptor.id.clone(),
            self.descriptor.display_name.clone(),
            arguments,
        )
        .with_schema_digest(self.descriptor.schema_digest.clone())
        .with_metadata(metadata);
        let mut request = payload
            .into_effect()
            .map_err(|e| ToolError::Execution(e.to_string()))?
            .with_source(context.tool_call_id, context.tool_name)
            .with_risk(self.risk);
        if let Some(timeout) = self.timeout {
            request = request.with_timeout(timeout);
        }
        Ok(ToolResult::Effect(request))
    }
}

/// Output returned by a host-owned MCP client provider.
#[cfg(feature = "harness")]
#[derive(Debug, Clone, PartialEq)]
pub struct McpToolCallOutput {
    /// Model-visible text content.
    pub content: String,
    /// Structured MCP content, when present.
    pub structured_content: Option<serde_json::Value>,
    /// Whether the MCP server reported a tool-level error.
    pub is_error: bool,
    /// Host/application metadata.
    pub metadata: RunMetadata,
}

#[cfg(feature = "harness")]
impl McpToolCallOutput {
    /// Constructs a successful text MCP output.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            structured_content: None,
            is_error: false,
            metadata: RunMetadata::new(),
        }
    }

    /// Marks this output as a tool-level error.
    pub fn into_error(mut self) -> Self {
        self.is_error = true;
        self
    }
}

/// Host-owned MCP client provider used by [`McpEffectExecutor`].
///
/// Implementations resolve `server_id` against configured clients. They must
/// not accept arbitrary URLs, commands, or credentials from
/// [`McpCallPayload`].
#[cfg(feature = "harness")]
#[async_trait::async_trait]
pub trait McpClientProvider: Send + Sync {
    /// Calls one MCP tool under a timeout selected by the harness/executor.
    async fn call_tool(
        &self,
        payload: &McpCallPayload,
        timeout: Duration,
        context: &RunContext,
    ) -> Result<McpToolCallOutput, McpError>;
}

/// Effect executor for [`EffectKind::Mcp`] requests.
#[cfg(feature = "harness")]
#[derive(Debug, Clone)]
pub struct McpEffectExecutor<C> {
    clients: C,
}

#[cfg(feature = "harness")]
impl<C> McpEffectExecutor<C> {
    /// Constructs an MCP effect executor from a host-owned client provider.
    pub fn new(clients: C) -> Self {
        Self { clients }
    }
}

#[cfg(feature = "harness")]
#[async_trait::async_trait]
impl<C> EffectExecutor for McpEffectExecutor<C>
where
    C: McpClientProvider,
{
    async fn execute(
        &self,
        request: &EffectRequest,
        policy: &ExecutionPolicy,
        context: &RunContext,
    ) -> Result<RawEffectOutput, ExecutionError> {
        let payload = McpCallPayload::from_effect(request)
            .map_err(|e| ExecutionError::Failed(e.to_string()))?;
        let timeout = policy.timeout.unwrap_or(DEFAULT_CALL_TIMEOUT);
        let output = self
            .clients
            .call_tool(&payload, timeout, context)
            .await
            .map_err(|e| ExecutionError::Failed(e.to_string()))?;
        let mut text = output.content;
        if let Some(structured) = &output.structured_content {
            let rendered = serde_json::to_string_pretty(structured)
                .map_err(|e| ExecutionError::Failed(e.to_string()))?;
            text = format!("{text}\n[structured]\n{rendered}");
        }
        if output.is_error {
            return Err(ExecutionError::Failed(if text.is_empty() {
                "mcp tool reported an error".to_string()
            } else {
                text
            }));
        }
        Ok(RawEffectOutput::text(text)
            .with_display(DisplayOutput::new(
                DisplayFormat::PlainText,
                "MCP tool call completed",
            ))
            .with_metadata(output.metadata))
    }
}

/// Policy for one MCP server.
#[cfg(feature = "harness")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpServerPolicy {
    /// Server id.
    pub server_id: McpServerId,
    /// Server trust level.
    pub trust: ToolTrustLevel,
    /// Allowed raw tool names. `None` means all tools are allowed unless
    /// denied explicitly.
    pub allowed_tools: Option<Vec<String>>,
    /// Denied raw tool names.
    pub denied_tools: Vec<String>,
    /// Minimum risk for this server.
    pub default_risk: RiskLevel,
    /// Whether every allowed call still requires approval.
    pub require_approval: bool,
    /// Network policy expected for this server.
    pub network: NetworkPolicy,
    /// Sandbox policy expected for this server.
    pub sandbox: SandboxPolicy,
}

#[cfg(feature = "harness")]
impl McpServerPolicy {
    /// Constructs a policy that allows tools for one server with approval.
    pub fn requiring_approval(server_id: impl Into<McpServerId>) -> Self {
        Self {
            server_id: server_id.into(),
            trust: ToolTrustLevel::External,
            allowed_tools: None,
            denied_tools: Vec::new(),
            default_risk: RiskLevel::Medium,
            require_approval: true,
            network: NetworkPolicy::Deny,
            sandbox: SandboxPolicy::ReadOnly,
        }
    }

    /// Constructs a policy that denies every tool for one server.
    pub fn deny_all(server_id: impl Into<McpServerId>) -> Self {
        Self {
            allowed_tools: Some(Vec::new()),
            ..Self::requiring_approval(server_id)
        }
    }
}

/// MCP permission bridge usable as a harness [`PolicyEngine`].
#[cfg(feature = "harness")]
#[derive(Debug, Clone)]
pub struct McpPermissionBridge {
    server_policies: HashMap<McpServerId, McpServerPolicy>,
    default_policy: McpServerPolicy,
}

#[cfg(feature = "harness")]
impl McpPermissionBridge {
    /// Constructs a bridge with unknown servers denied by default.
    pub fn new() -> Self {
        Self {
            server_policies: HashMap::new(),
            default_policy: McpServerPolicy::deny_all("__unknown__"),
        }
    }

    /// Adds or replaces one server policy.
    pub fn with_server_policy(mut self, policy: McpServerPolicy) -> Self {
        self.server_policies
            .insert(policy.server_id.clone(), policy);
        self
    }

    fn policy_for(&self, server_id: &McpServerId) -> &McpServerPolicy {
        self.server_policies
            .get(server_id)
            .unwrap_or(&self.default_policy)
    }
}

#[cfg(feature = "harness")]
impl Default for McpPermissionBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "harness")]
#[async_trait::async_trait]
impl PolicyEngine for McpPermissionBridge {
    async fn evaluate(
        &self,
        effect: &ClassifiedEffect,
        _context: &RunContext,
    ) -> Result<PolicyDecision, crate::harness::HarnessError> {
        if effect.request.kind != EffectKind::Mcp {
            return Ok(PolicyDecision::Allow);
        }
        let payload = McpCallPayload::from_effect(&effect.request).map_err(|e| {
            crate::harness::HarnessError::Policy(format!("invalid MCP payload: {e}"))
        })?;
        let policy = self.policy_for(&payload.server_id);
        if policy.server_id != payload.server_id {
            return Ok(PolicyDecision::Deny {
                reason: format!("unknown MCP server: {}", payload.server_id),
            });
        }
        if policy
            .denied_tools
            .iter()
            .any(|tool| tool == &payload.tool_name)
        {
            return Ok(PolicyDecision::Deny {
                reason: format!("MCP tool denied: {}", payload.tool_name),
            });
        }
        if let Some(allowed) = &policy.allowed_tools
            && !allowed.iter().any(|tool| tool == &payload.tool_name)
        {
            return Ok(PolicyDecision::Deny {
                reason: format!("MCP tool not allowed: {}", payload.tool_name),
            });
        }
        if policy.require_approval || risk_rank(effect.effective_risk) >= risk_rank(RiskLevel::High)
        {
            return Ok(PolicyDecision::RequireApproval {
                reason: format!(
                    "MCP call requires approval: {}::{}",
                    payload.server_id, payload.tool_name
                ),
            });
        }
        Ok(PolicyDecision::Allow)
    }
}

/// Assembly-time errors (connect / list tools); call-time errors go through
/// [`ToolError::Execution`].
///
/// Callers usually only handle this error at assembly time; tool execution
/// failures are relayed back to the model as text by the agent loop without
/// aborting the loop.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum McpError {
    /// Child process startup failure / invalid or unreachable URL / protocol
    /// handshake failure; carries the server name to identify the target in
    /// multi-server setups.
    #[error("mcp connect failed (server {server}): {message}")]
    Connect {
        /// The target server name (the server_name passed at construction).
        server: String,
        /// Failure reason.
        message: String,
    },
    /// Listing tools failed (protocol-level error); carries the server name.
    #[error("mcp list tools failed (server {server}): {message}")]
    ListTools {
        /// The target server name (the server_name passed at construction).
        server: String,
        /// Failure reason.
        message: String,
    },
    /// MCP effect payload is malformed or not an MCP request.
    #[error("invalid mcp payload: {0}")]
    InvalidPayload(String),
    /// Calling an MCP tool failed at protocol/transport level.
    #[error("mcp tool call failed (server {server}, tool {tool}): {message}")]
    CallTool {
        /// Target server id.
        server: String,
        /// Raw tool name.
        tool: String,
        /// Failure reason.
        message: String,
    },
}

/// Starts the connection lifecycle with a timeout (handshake / initialization
/// after spawn): a timeout returns [`McpError::Connect`] with "timed out" in
/// the message for easy diagnosis.
async fn serve_with_timeout(
    server_name: &str,
    timeout: Duration,
    serve: impl Future<Output = Result<RunningService<RoleClient, ()>, ClientInitializeError>>,
) -> Result<RunningService<RoleClient, ()>, McpError> {
    match tokio::time::timeout(timeout, serve).await {
        Ok(result) => result.map_err(|e| McpError::Connect {
            server: server_name.to_string(),
            message: e.to_string(),
        }),
        Err(_) => Err(McpError::Connect {
            server: server_name.to_string(),
            message: "connect timed out".into(),
        }),
    }
}

/// Prefers the newer stateless protocol; falls back to the legacy handshake
/// when `Auto` discovery fails.
fn lifecycle_mode() -> ClientLifecycleMode {
    ClientLifecycleMode::Auto {
        preferred_versions: vec![ProtocolVersion::V_2026_07_28],
        legacy_version: Some(ProtocolVersion::V_2025_11_25),
    }
}

/// Tool display name: `{server_name}__{raw}` (prefix on) or the raw name
/// (prefix off).
fn tool_display_name(server_name: &str, raw: &str, prefix: bool) -> String {
    if prefix {
        format!("{server_name}__{raw}")
    } else {
        raw.to_string()
    }
}

/// Size limit (bytes) of a single MCP tool call result text: the result enters
/// the model context, and a malicious/compromised server could use it to blow
/// up the context. Results over the limit are truncated and ended with a
/// placeholder marker.
const MAX_TOOL_RESULT_BYTES: usize = 1024 * 1024;

/// Upper bound on the number of tools pulled per `tools()` call: all tools get
/// registered into the ToolRegistry and enter the model context, so they must
/// be bounded.
const MAX_MCP_TOOLS: usize = 512;

/// Renders result content blocks as text: text blocks are joined, non-text
/// blocks output placeholder descriptions; when the total length exceeds
/// [`MAX_TOOL_RESULT_BYTES`], it is truncated and ended with a marker.
fn content_to_text(blocks: &[ContentBlock]) -> String {
    let mut out = String::new();
    for (i, block) in blocks.iter().enumerate() {
        let part = match block {
            ContentBlock::Text(text) => text.text.clone(),
            ContentBlock::Image(image) => format!("[image: {}]", image.mime_type),
            ContentBlock::Audio(audio) => format!("[audio: {}]", audio.mime_type),
            ContentBlock::Resource(_) => "[resource]".into(),
            ContentBlock::ResourceLink(_) => "[resource link]".into(),
            // ContentBlock is #[non_exhaustive]: external crates cannot detect
            // new variants at compile time — output a placeholder and warn,
            // never silently swallow.
            _ => {
                #[cfg(feature = "tracing")]
                tracing::warn!("mcp tool result contains an unknown content block");
                "[content]".into()
            }
        };
        if out.len() + part.len() + usize::from(i > 0) > MAX_TOOL_RESULT_BYTES {
            out.push_str("[truncated: result exceeds size limit]");
            return out;
        }
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&part);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{SharedState, ToolContext, ToolRegistry};
    use rmcp::model::{
        CallToolResponse, CallToolResult, ListToolsResult, PaginatedRequestParams,
        ServerCapabilities, ServerInfo, Tool as RmcpTool,
    };
    use rmcp::service::{RequestContext, RoleServer};
    use rmcp::transport::stdio;
    use rmcp::{ErrorData, ServerHandler, serve_server};
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn call_mcp_tool(
        tool: &McpTool,
        arguments: serde_json::Value,
    ) -> Result<String, ToolError> {
        let run = crate::RunContext::new("mcp-tool-test");
        let state = SharedState::new();
        let result = tool
            .call(
                arguments,
                ToolContext::new(&run, &state, "call-mcp", &tool.schema().name),
            )
            .await?;
        Ok(result.to_string())
    }

    /// Streamable HTTP success path: starts a minimal stateless MCP-over-HTTP
    /// server locally (hand-written JSON-RPC responses, no new dependencies),
    /// connects via `from_url`, pulls tools with `tools()`, and calls one —
    /// covering the second active transport besides stdio.
    #[tokio::test]
    async fn http_roundtrip_list_and_call_tools() {
        use std::time::Duration;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Hand-written minimal server: stateless protocol (server/discover +
        // tools/list + tools/call), accepts in a loop (the client may open a
        // new connection per request), and reads in a loop within a
        // connection.
        let server = tokio::spawn(async move {
            loop {
                let (mut socket, _) = match listener.accept().await {
                    Ok(s) => s,
                    Err(_) => break,
                };
                tokio::spawn(async move {
                    let mut buf = [0u8; 64 * 1024];
                    loop {
                        let n = match socket.read(&mut buf).await {
                            Ok(0) => break,
                            Ok(n) => n,
                            Err(_) => break,
                        };
                        let req = String::from_utf8_lossy(&buf[..n]).to_string();
                        // The request body comes after the last blank line (a
                        // single read usually contains the headers + a small
                        // body).
                        let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
                        let Ok(value) = serde_json::from_str::<serde_json::Value>(body) else {
                            break;
                        };
                        let id = value["id"].clone();
                        let result = match value["method"].as_str() {
                            Some("server/discover") => json!({
                                "resultType": "complete",
                                "supportedVersions": ["2026-07-28"],
                                "capabilities": { "tools": {} },
                                "ttlMs": 0,
                                "cacheScope": "public",
                            }),
                            Some("tools/list") => json!({
                                "tools": [{
                                    "name": "echo",
                                    "description": "echo",
                                    "inputSchema": { "type": "object", "properties": {} },
                                }],
                                "nextCursor": null,
                                "ttlMs": 0,
                                "cacheScope": "public",
                            }),
                            Some("tools/call") => json!({
                                "resultType": "complete",
                                "content": [{ "type": "text", "text": "pong" }],
                            }),
                            _ => json!({}),
                        };
                        let resp_body =
                            serde_json::json!({ "jsonrpc": "2.0", "id": id, "result": result })
                                .to_string();
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                            resp_body.len(),
                            resp_body
                        );
                        if socket.write_all(resp.as_bytes()).await.is_err() {
                            break;
                        }
                    }
                });
            }
        });

        let mut client = McpClient::from_url("http-server", format!("http://{addr}"));
        let tools = tokio::time::timeout(Duration::from_secs(10), client.tools())
            .await
            .expect("tools() timed out")
            .unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].schema().name, "http-server__echo");

        let text =
            tokio::time::timeout(Duration::from_secs(10), call_mcp_tool(&tools[0], json!({})))
                .await
                .expect("call timed out")
                .unwrap();
        assert_eq!(text, "pong");
        server.abort();
    }

    /// HTTP mock server (test-only): invokes `respond` (method name, params)
    /// for each JSON-RPC request → the result JSON of the response; `None` =
    /// keep the connection open without responding (simulating a hung server
    /// for timeout tests).
    fn spawn_http_mock(
        respond: impl Fn(&str, &serde_json::Value) -> Option<serde_json::Value> + Send + Sync + 'static,
    ) -> (std::net::SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let respond = Arc::new(respond);
        let server = tokio::spawn(async move {
            let listener = tokio::net::TcpListener::from_std(listener).unwrap();
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                let respond = Arc::clone(&respond);
                tokio::spawn(async move {
                    let mut buf = [0u8; 8192];
                    loop {
                        let n = match socket.read(&mut buf).await {
                            Ok(0) | Err(_) => break,
                            Ok(n) => n,
                        };
                        // The request body comes after the last blank line (a
                        // single read usually contains the headers + a small
                        // body).
                        let req = String::from_utf8_lossy(&buf[..n]).to_string();
                        let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
                        let request: serde_json::Value = match serde_json::from_str(body) {
                            Ok(v) => v,
                            Err(_) => break,
                        };
                        let method = request["method"].as_str().unwrap_or_default();
                        let params = request
                            .get("params")
                            .cloned()
                            .unwrap_or(serde_json::Value::Null);
                        let Some(result) = respond(method, &params) else {
                            // Simulate a hung server: keep the connection
                            // open, never respond, and wait for the client
                            // timeout.
                            tokio::time::sleep(Duration::from_secs(60)).await;
                            break;
                        };
                        let resp_body = serde_json::json!({
                            "jsonrpc": "2.0",
                            "id": request["id"].clone(),
                            "result": result,
                        })
                        .to_string();
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                            resp_body.len(),
                            resp_body
                        );
                        if socket.write_all(resp.as_bytes()).await.is_err() {
                            break;
                        }
                    }
                });
            }
        });
        (addr, server)
    }

    #[tokio::test]
    async fn structured_content_is_rendered_not_dropped() {
        // Newer-protocol structured results (structured_content): joined into
        // the result text, not silently dropped.
        let (addr, server) = spawn_http_mock(|method, _| match method {
            "server/discover" => Some(json!({
                "resultType": "complete",
                "supportedVersions": ["2026-07-28"],
                "capabilities": { "tools": {} },
                "ttlMs": 0,
                "cacheScope": "public",
            })),
            "tools/list" => Some(json!({
                "tools": [{
                    "name": "summary",
                    "description": "structured",
                    "inputSchema": { "type": "object", "properties": {} },
                }],
                "nextCursor": null,
                "ttlMs": 0,
                "cacheScope": "public",
            })),
            "tools/call" => Some(json!({
                "resultType": "complete",
                "content": [{ "type": "text", "text": "hello" }],
                // wire key is camelCase (rmcp model rename_all = "camelCase").
                "structuredContent": { "answer": 42 },
            })),
            _ => Some(json!({})),
        });
        let mut client = McpClient::from_url("s", format!("http://{addr}"));
        let tools = client.tools().await.unwrap();
        let text = call_mcp_tool(&tools[0], json!({})).await.unwrap();
        assert!(text.starts_with("hello"));
        assert!(text.contains("[structured]"));
        assert!(text.contains("\"answer\""));
        server.abort();
    }

    #[tokio::test]
    async fn call_tool_hangs_returns_timeout_error() {
        // The server never responds to tools/call: the call terminates with an
        // Execution timeout after call_timeout instead of blocking forever.
        let (addr, server) = spawn_http_mock(|method, _| match method {
            "server/discover" => Some(json!({
                "resultType": "complete",
                "supportedVersions": ["2026-07-28"],
                "capabilities": { "tools": {} },
                "ttlMs": 0,
                "cacheScope": "public",
            })),
            "tools/list" => Some(json!({
                "tools": [{
                    "name": "hang",
                    "description": "hangs",
                    "inputSchema": { "type": "object", "properties": {} },
                }],
                "nextCursor": null,
                "ttlMs": 0,
                "cacheScope": "public",
            })),
            "tools/call" => None,
            _ => Some(json!({})),
        });
        let mut client = McpClient::from_url("s", format!("http://{addr}"));
        client.with_call_timeout(Duration::from_millis(100));
        let tools = client.tools().await.unwrap();
        let err = call_mcp_tool(&tools[0], json!({})).await.unwrap_err();
        assert!(matches!(&err, ToolError::Execution(msg) if msg == "mcp tool call timed out"));
        server.abort();
    }

    /// Minimal MCP server (shared shape for tests / examples): two tools —
    /// `echo` returns its text verbatim, `fail` always fails at tool level.
    #[derive(Default)]
    struct FakeServer;

    /// Tool parameter Schema helper: `rmcp::Tool::new` needs a `Map`, not a
    /// `Value`.
    fn tool_schema(value: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        value
            .as_object()
            .expect("tool schema must be an object")
            .clone()
    }

    impl ServerHandler for FakeServer {
        fn get_info(&self) -> ServerInfo {
            ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
        }

        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            Ok(ListToolsResult {
                tools: vec![
                    RmcpTool::new(
                        "echo",
                        "returns the text field verbatim",
                        tool_schema(json!({
                            "type": "object",
                            "properties": { "text": { "type": "string" } },
                            "required": ["text"],
                        })),
                    ),
                    RmcpTool::new(
                        "fail",
                        "always fails at tool level",
                        tool_schema(json!({ "type": "object" })),
                    ),
                ],
                ..Default::default()
            })
        }

        async fn call_tool(
            &self,
            request: CallToolRequestParams,
            _context: RequestContext<RoleServer>,
        ) -> Result<CallToolResponse, ErrorData> {
            match request.name.as_ref() {
                "echo" => {
                    let text = request
                        .arguments
                        .as_ref()
                        .and_then(|args| args.get("text"))
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    Ok(CallToolResult::success(vec![ContentBlock::text(text.to_string())]).into())
                }
                "fail" => Ok(CallToolResult::error(vec![ContentBlock::text("boom")]).into()),
                name => Err(ErrorData::invalid_params(
                    format!("unknown tool: {name}"),
                    None,
                )),
            }
        }
    }

    // ---- Pure-function tests (no connection needed) ----

    #[test]
    fn display_name_with_prefix_on() {
        assert_eq!(tool_display_name("fs", "read_file", true), "fs__read_file");
    }

    #[test]
    fn display_name_with_prefix_off() {
        assert_eq!(tool_display_name("fs", "read_file", false), "read_file");
    }

    #[test]
    fn content_to_text_joins_text_blocks() {
        let blocks = vec![
            ContentBlock::text("first line"),
            ContentBlock::text("second line"),
        ];
        assert_eq!(content_to_text(&blocks), "first line\nsecond line");
    }

    #[test]
    fn content_to_text_placeholders_non_text() {
        let blocks = vec![
            ContentBlock::text("take a look at this:"),
            ContentBlock::image("base64data", "image/png"),
            ContentBlock::audio("base64data", "audio/wav"),
        ];
        assert_eq!(
            content_to_text(&blocks),
            "take a look at this:\n[image: image/png]\n[audio: audio/wav]"
        );
    }

    #[test]
    fn content_to_text_empty() {
        assert_eq!(content_to_text(&[]), "");
    }

    #[test]
    fn map_tool_keeps_prefix_and_passthrough() {
        let tool = RmcpTool::new(
            "echo",
            "description",
            tool_schema(json!({ "type": "object" })),
        );
        let mapped = map_tool("fs", tool, true);
        assert_eq!(mapped.name, "fs__echo");
        assert_eq!(mapped.raw_name, "echo");
        assert_eq!(mapped.description, "description");
        assert_eq!(mapped.parameters, json!({ "type": "object" }));
    }

    #[test]
    fn descriptor_keeps_server_scoped_identity_and_digest() {
        let tool = RmcpTool::new(
            "echo",
            "description",
            tool_schema(json!({ "type": "object" })),
        );
        let descriptor = descriptor_from_tool("fake", tool, true);

        assert_eq!(descriptor.id.server, McpServerId::new("fake"));
        assert_eq!(descriptor.id.raw_name, "echo");
        assert_eq!(descriptor.display_name, "fake__echo");
        assert!(descriptor.schema_digest.starts_with("fnv64:"));
        assert_eq!(
            descriptor.source(ToolTrustLevel::External).namespace,
            ToolNamespace::mcp_server("fake")
        );
    }

    #[cfg(feature = "harness")]
    #[tokio::test]
    async fn effect_tool_returns_mcp_effect_without_calling_server() {
        let descriptor = descriptor_from_tool(
            "fake",
            RmcpTool::new(
                "echo",
                "description",
                tool_schema(json!({ "type": "object" })),
            ),
            true,
        );
        let tool = McpEffectTool::new(descriptor.clone()).with_timeout(Duration::from_secs(5));
        let run = crate::RunContext::new("mcp-effect-test");
        let state = SharedState::new();
        let result = tool
            .call(
                json!({ "text": "hello" }),
                ToolContext::new(&run, &state, "call-1", "fake__echo"),
            )
            .await
            .unwrap();

        let ToolResult::Effect(request) = result else {
            panic!("expected effect request");
        };
        assert_eq!(request.kind, EffectKind::Mcp);
        assert_eq!(request.source.tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(request.timeout, Some(Duration::from_secs(5)));
        let payload = McpCallPayload::from_effect(&request).unwrap();
        assert_eq!(payload.server_id, descriptor.id.server);
        assert_eq!(payload.tool_name, "echo");
        assert_eq!(payload.display_name, "fake__echo");
        assert_eq!(payload.arguments, json!({ "text": "hello" }));
        assert_eq!(
            payload.schema_digest.as_deref(),
            Some(descriptor.schema_digest.as_str())
        );
    }

    #[cfg(feature = "harness")]
    #[derive(Debug)]
    struct FakeMcpProvider {
        output: McpToolCallOutput,
    }

    #[cfg(feature = "harness")]
    #[async_trait::async_trait]
    impl McpClientProvider for FakeMcpProvider {
        async fn call_tool(
            &self,
            _payload: &McpCallPayload,
            _timeout: Duration,
            _context: &RunContext,
        ) -> Result<McpToolCallOutput, McpError> {
            Ok(self.output.clone())
        }
    }

    #[cfg(feature = "harness")]
    #[tokio::test]
    async fn effect_executor_maps_success_and_tool_error() {
        let payload = McpCallPayload::new(
            McpToolId::new("fake", "echo"),
            "fake__echo",
            json!({ "text": "hello" }),
        );
        let request = payload.into_effect().unwrap();
        let policy = ExecutionPolicy {
            sandbox: SandboxPolicy::ReadOnly,
            network: NetworkPolicy::Deny,
            timeout: Some(Duration::from_secs(1)),
            output_limit: crate::harness::OutputLimit::default(),
        };
        let run = RunContext::new("mcp-executor-test");

        let executor = McpEffectExecutor::new(FakeMcpProvider {
            output: McpToolCallOutput::text("hello"),
        });
        let raw = executor.execute(&request, &policy, &run).await.unwrap();
        assert_eq!(raw.observation_for_model, "hello");

        let executor = McpEffectExecutor::new(FakeMcpProvider {
            output: McpToolCallOutput::text("boom").into_error(),
        });
        let err = executor.execute(&request, &policy, &run).await.unwrap_err();
        assert!(matches!(err, ExecutionError::Failed(message) if message == "boom"));
    }

    #[cfg(feature = "harness")]
    #[tokio::test]
    async fn permission_bridge_denies_unknown_and_filters_tools() {
        let bridge = McpPermissionBridge::new().with_server_policy(McpServerPolicy {
            server_id: McpServerId::new("fake"),
            trust: ToolTrustLevel::External,
            allowed_tools: Some(vec!["echo".to_string()]),
            denied_tools: vec!["delete".to_string()],
            default_risk: RiskLevel::Medium,
            require_approval: false,
            network: NetworkPolicy::Deny,
            sandbox: SandboxPolicy::ReadOnly,
        });
        let run = RunContext::new("mcp-policy-test");
        let allowed = McpCallPayload::new(McpToolId::new("fake", "echo"), "fake__echo", json!({}))
            .into_effect()
            .unwrap();
        let denied =
            McpCallPayload::new(McpToolId::new("fake", "delete"), "fake__delete", json!({}))
                .into_effect()
                .unwrap();
        let unknown = McpCallPayload::new(
            McpToolId::new("unknown", "echo"),
            "unknown__echo",
            json!({}),
        )
        .into_effect()
        .unwrap();

        let decision = bridge.evaluate(&classified(allowed), &run).await.unwrap();
        assert_eq!(decision, PolicyDecision::Allow);
        let decision = bridge.evaluate(&classified(denied), &run).await.unwrap();
        assert!(matches!(decision, PolicyDecision::Deny { .. }));
        let decision = bridge.evaluate(&classified(unknown), &run).await.unwrap();
        assert!(matches!(decision, PolicyDecision::Deny { .. }));
    }

    #[cfg(feature = "harness")]
    fn classified(request: EffectRequest) -> ClassifiedEffect {
        ClassifiedEffect {
            request,
            requested_risk: RiskLevel::Medium,
            effective_risk: RiskLevel::Medium,
            reasons: Vec::new(),
            metadata: RunMetadata::new(),
        }
    }

    // ---- End-to-end tests (the child process spawns itself as the MCP
    // server) ----

    /// Acts as a minimal MCP server (stdio transport, blocking until the
    /// parent closes the connection).
    ///
    /// This test is a **child-process fixture**: `#[ignore]` makes normal test
    /// runs skip it; end-to-end tests launch the current test binary with
    /// `--ignored`, and the child process runs only this test, blocking on
    /// serve. rmcp clients silently ignore non-JSON lines, so the test
    /// harness's stdout output never pollutes the protocol stream. ⚠️ Running
    /// `cargo test -- --ignored` manually makes this test block on stdin;
    /// terminate it manually if no connection arrives.
    #[tokio::test]
    #[ignore]
    async fn serve_as_fake_server() {
        // serve_server returns the RunningService once the first request
        // (discover) is handled; waiting() blocks until the connection closes
        // (the parent disconnects), serving further requests in the meantime.
        let running = serve_server(FakeServer, stdio()).await.unwrap();
        running.waiting().await.unwrap();
        // The parent has closed the connection (pipe read end closed); the
        // harness's teardown printing would EPIPE-pollute the parent's output:
        // exit directly on the happy path, bypassing the harness teardown.
        std::process::exit(0);
    }

    /// Uses the current test binary as the program with `--ignored --quiet`
    /// args: the child process runs only the ignored
    /// [`serve_as_fake_server`](self::serve_as_fake_server) test as the server
    /// (`--quiet` keeps the harness silent, so it neither pollutes the
    /// protocol stream nor produces EPIPE noise on disconnect).
    fn fake_server_command() -> (String, Vec<String>) {
        let exe = std::env::current_exe()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        (exe, vec!["--ignored".into(), "--quiet".into()])
    }

    #[tokio::test]
    async fn tools_roundtrip_echo_and_error() {
        let (program, args) = fake_server_command();
        let mut client = McpClient::from_command("fake", program, args);
        let tools = client.tools().await.unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].schema().name, "fake__echo");
        assert_eq!(tools[1].schema().name, "fake__fail");

        let echo = tools
            .iter()
            .find(|t| t.schema().name == "fake__echo")
            .unwrap();
        let result = call_mcp_tool(echo, json!({ "text": "hello" }))
            .await
            .unwrap();
        assert_eq!(result, "hello");

        // Tool-level error → ToolError::Execution with the server text in the
        // message.
        let fail = tools
            .iter()
            .find(|t| t.schema().name == "fake__fail")
            .unwrap();
        let err = call_mcp_tool(fail, json!({})).await.unwrap_err();
        assert!(matches!(&err, ToolError::Execution(msg) if msg == "boom"));

        // Arguments that are not a JSON object → InvalidArguments.
        let err = call_mcp_tool(echo, json!([1, 2])).await.unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
    }

    #[tokio::test]
    async fn registry_assembly_with_prefix_off() {
        let (program, args) = fake_server_command();
        let mut client = McpClient::from_command("fake", program, args);
        client.with_name_prefix(false);
        let mut registry = ToolRegistry::new();
        for tool in client.tools().await.unwrap() {
            registry.register(tool);
        }
        assert_eq!(registry.names(), vec!["echo", "fail"]);
        let result = registry
            .call_named(
                "echo",
                r#"{"text":"hi"}"#,
                &crate::RunContext::new("mcp-registry-test"),
                &SharedState::new(),
            )
            .await
            .unwrap();
        assert_eq!(result, "hi");
    }

    #[tokio::test]
    async fn cleanup_then_tools_reconnects() {
        let (program, args) = fake_server_command();
        let mut client = McpClient::from_command("fake", program, args);
        assert_eq!(client.tools().await.unwrap().len(), 2);
        client.cleanup().await.unwrap();
        // After disconnect, tools() reconnects automatically (the program +
        // args form is rebuildable).
        assert_eq!(client.tools().await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn connect_is_idempotent() {
        let (program, args) = fake_server_command();
        let mut client = McpClient::from_command("fake", program, args);
        client.connect().await.unwrap();
        client.connect().await.unwrap(); // already connected, no-op
        assert_eq!(client.tools().await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn args_spawn_failure_restores_spec_for_retry() {
        // The program does not exist: after a spawn failure the spec must be
        // restored — the next connect fails with the same spawn error rather
        // than "already consumed" (so it can be retried after fixing the
        // config).
        let mut client = McpClient::from_command(
            "ghost",
            "molo-no-such-program-xyz",
            std::iter::empty::<&str>(),
        );
        let err = client.connect().await.unwrap_err();
        assert!(
            matches!(&err, McpError::Connect { message, .. } if message.contains("spawn failed"))
        );
        let err2 = client.connect().await.unwrap_err();
        assert!(
            matches!(&err2, McpError::Connect { message, .. } if message.contains("spawn failed")),
            "spec must be restored after spawn failure; retry should fail with spawn error, not already consumed, got: {err2}"
        );
    }

    #[tokio::test]
    async fn configured_command_is_one_shot_and_guides_rebuild() {
        let (program, _) = fake_server_command();
        let mut command = std::process::Command::new(program);
        command.args(["--ignored", "--quiet"]);
        let mut client = McpClient::from_command_configured("fake", command);
        assert_eq!(client.tools().await.unwrap().len(), 2);
        client.cleanup().await.unwrap();
        // The full-Command form is one-shot: reconnecting errors out and
        // suggests rebuilding the McpClient.
        let err = client.tools().await.unwrap_err();
        assert!(matches!(&err, McpError::Connect { message, .. }
            if message.contains("create a new McpClient")));
    }

    #[tokio::test]
    async fn from_url_connect_failure_reports_connect_error() {
        // 127.0.0.1:1 refuses connections immediately: verifies the error
        // mapping on the URL transport path.
        let mut client = McpClient::from_url("nowhere", "http://127.0.0.1:1/mcp");
        let err = client.connect().await.unwrap_err();
        assert!(matches!(&err, McpError::Connect { server, .. } if server == "nowhere"));
        assert!(
            err.to_string()
                .starts_with("mcp connect failed (server nowhere)")
        );
    }
}
