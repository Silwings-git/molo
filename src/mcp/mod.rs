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

use std::future::Future;
use std::process::Command as StdCommand;
use std::sync::Arc;
use std::time::Duration;

use rmcp::model::{CallToolRequestParams, ContentBlock, ProtocolVersion};
use rmcp::service::{ClientCacheConfig, ClientInitializeError, RoleClient, RunningService};
use rmcp::transport::child_process::TokioChildProcess;
use rmcp::transport::streamable_http_client::StreamableHttpClientWorker;
use rmcp::{ClientLifecycleMode, ClientServiceExt};

use crate::tool::{SharedState, Tool, ToolError, ToolSchema};

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

    /// Timeout for the whole connect flow (default 10s): child process startup
    /// + protocol handshake. A timeout returns [`McpError::Connect`] with
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

    /// Pulls all tools from the server and converts them into molo tools;
    /// connects automatically and re-pulls on every call.
    ///
    /// The returned [`McpTool`]s can be registered directly into a
    /// `ToolRegistry` (or cloned to share across multiple registries); tool
    /// names follow [`with_name_prefix`](McpClient::with_name_prefix).
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
    pub async fn tools(&mut self) -> Result<Vec<McpTool>, McpError> {
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
        Ok(tools
            .into_iter()
            .map(|tool| {
                McpTool::new(
                    &self.server_name,
                    tool,
                    self.prefix,
                    self.call_timeout,
                    Arc::clone(&running),
                )
            })
            .collect())
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

/// Adapter tool produced by [`McpClient::tools`]: implements molo's [`Tool`]
/// trait and proxies calls to the MCP server.
///
/// Holds a connection reference (Arc) internally and is Clone-able — register
/// it in multiple registries (or share between a main agent and sub-agents)
/// sharing the same connection and state.
#[derive(Clone)]
pub struct McpTool {
    /// Display name (per the prefix switch; visible to the model).
    name: String,
    /// The raw name on the server (used for forwarding requests).
    raw_name: String,
    /// Tool description.
    description: String,
    /// Parameter JSON Schema (passed through verbatim).
    parameters: serde_json::Value,
    /// Timeout for a single call (from the producing McpClient's config).
    call_timeout: Duration,
    /// Connection handle.
    peer: Arc<RunningService<RoleClient, ()>>,
}

impl McpTool {
    /// Builds an adapter tool from the server-returned tool description;
    /// `prefix` decides whether the display name carries the `{server_name}__`
    /// namespace; `call_timeout` is the per-call timeout (from
    /// [`McpClient::with_call_timeout`]).
    fn new(
        server_name: &str,
        tool: rmcp::model::Tool,
        prefix: bool,
        call_timeout: Duration,
        peer: Arc<RunningService<RoleClient, ()>>,
    ) -> Self {
        let mapped = map_tool(server_name, tool, prefix);
        Self {
            name: mapped.name,
            raw_name: mapped.raw_name,
            description: mapped.description,
            parameters: mapped.parameters,
            call_timeout,
            peer,
        }
    }
}

/// Mapping result of a server tool description: display name / raw name /
/// description / parameter Schema.
struct MappedTool {
    name: String,
    raw_name: String,
    description: String,
    parameters: serde_json::Value,
}

/// Pure mapping: server tool description → [`MappedTool`].
fn map_tool(server_name: &str, tool: rmcp::model::Tool, prefix: bool) -> MappedTool {
    let raw_name = tool.name.to_string();
    MappedTool {
        name: tool_display_name(server_name, &raw_name, prefix),
        raw_name,
        description: tool.description.unwrap_or_default().to_string(),
        parameters: serde_json::Value::Object((*tool.input_schema).clone()),
    }
}

impl std::fmt::Debug for McpTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("McpTool")
            .field("name", &self.name)
            .field("raw_name", &self.raw_name)
            .finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl Tool for McpTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        }
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
        _state: &SharedState,
    ) -> Result<String, ToolError> {
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
            Ok(text)
        }
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
    use crate::tool::ToolRegistry;
    use rmcp::model::{
        CallToolResponse, CallToolResult, ListToolsResult, PaginatedRequestParams,
        ServerCapabilities, ServerInfo, Tool as RmcpTool,
    };
    use rmcp::service::{RequestContext, RoleServer};
    use rmcp::transport::stdio;
    use rmcp::{ErrorData, ServerHandler, serve_server};
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

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

        let state = crate::tool::SharedState::new();
        let text = tokio::time::timeout(Duration::from_secs(10), tools[0].call(json!({}), &state))
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
        let state = crate::tool::SharedState::new();
        let text = tools[0].call(json!({}), &state).await.unwrap();
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
        let state = crate::tool::SharedState::new();
        let err = tools[0].call(json!({}), &state).await.unwrap_err();
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
        let blocks = vec![ContentBlock::text("first line"), ContentBlock::text("second line")];
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
        let tool = RmcpTool::new("echo", "description", tool_schema(json!({ "type": "object" })));
        let mapped = map_tool("fs", tool, true);
        assert_eq!(mapped.name, "fs__echo");
        assert_eq!(mapped.raw_name, "echo");
        assert_eq!(mapped.description, "description");
        assert_eq!(mapped.parameters, json!({ "type": "object" }));
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

        let state = SharedState::new();
        let echo = tools
            .iter()
            .find(|t| t.schema().name == "fake__echo")
            .unwrap();
        let result = echo.call(json!({ "text": "hello" }), &state).await.unwrap();
        assert_eq!(result, "hello");

        // Tool-level error → ToolError::Execution with the server text in the
        // message.
        let fail = tools
            .iter()
            .find(|t| t.schema().name == "fake__fail")
            .unwrap();
        let err = fail.call(json!({}), &state).await.unwrap_err();
        assert!(matches!(&err, ToolError::Execution(msg) if msg == "boom"));

        // Arguments that are not a JSON object → InvalidArguments.
        let err = echo.call(json!([1, 2]), &state).await.unwrap_err();
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
            .call("echo", r#"{"text":"hi"}"#, &SharedState::new())
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
