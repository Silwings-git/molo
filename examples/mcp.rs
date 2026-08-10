//! MCP client adapter example: bring tools exposed by an external MCP server
//! into molo.
//!
//! This example is **self-contained**, depending on no external service: the
//! process forks itself as a child with the `--as-server` argument, acting as a
//! minimal MCP server (stdio transport, two tools); the main flow connects to
//! it with [`McpClient`] and demonstrates three paths:
//!
//! 1. `tools()` pulls the tools and registers them into the
//!    [`ToolRegistry`], with names prefixed `fake__`;
//! 2. Calls MCP tools directly (both success and tool-level failure results);
//! 3. Assembles a [`ReActAgent`] whose reasoning loop has the model request MCP
//!    tools and receive results (driven by a FakeProvider script, no real LLM
//!    needed).
//!
//! Run: `cargo run --example mcp`.

use std::sync::Arc;

use molo::tool::{SharedState, ToolRegistry};
use molo::{Agent, FakeProvider, FakeReply, McpClient, ReActAgent, ToolCall};
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ListToolsResult,
    PaginatedRequestParams, ServerCapabilities, ServerInfo, Tool as RmcpTool,
};
use rmcp::service::{RequestContext, RoleServer};
use rmcp::transport::stdio;
use rmcp::{ErrorData, serve_server};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|arg| arg == "--as-server") {
        // Child process role: act as a minimal MCP server until the parent disconnects.
        // Note: the child's stdout is the protocol channel; it must not print anything.
        return run_fake_server().await;
    }

    // ---- Assembly: connect to the MCP server and register its tools into the ToolRegistry ----
    let program = std::env::current_exe()?.to_string_lossy().into_owned();
    let mut client = McpClient::from_command("fake", program, ["--as-server"]);
    let mut registry = ToolRegistry::new();
    for tool in client.tools().await? {
        registry.register(tool);
    }
    println!("tools connected: {:?}", registry.names());

    // ---- Path 1: call MCP tools directly ----
    let state = SharedState::new();
    let text = registry
        .call("fake__echo", r#"{"text":"hello, MCP!"}"#, &state)
        .await?;
    println!("direct call to fake__echo: {text}");

    // Tool-level failure: the error text is fed back via the registry; Display is the error text.
    let err = registry
        .call("fake__fail", "{}", &state)
        .await
        .expect_err("fail tool must fail");
    println!("tool-level failure fed back: {err}");

    // ---- Path 2: call MCP tools inside the Agent's reasoning loop ----
    // FakeProvider script: the first request calls fake__echo, then closes with text after receiving the result.
    let fake = Arc::new(FakeProvider::new([
        FakeReply::ToolCalls {
            content: "let me call the external tool first".into(),
            calls: vec![ToolCall {
                id: "c1".into(),
                name: "fake__echo".into(),
                arguments: r#"{"text":"an MCP call from the model's perspective"}"#.into(),
            }],
        },
        FakeReply::Text("tool result received, task complete".into()),
    ]));
    let mut agent = ReActAgent::new(
        fake.clone(),
        registry,
        "You are an assistant that uses external tools; tool results are fed back as ToolResult messages",
    );
    let answer = agent.run("say hello with the fake__echo tool").await?;
    println!("Agent answer: {answer}");
    Ok(())
}

/// Minimal MCP server: two tools — `echo` returns text as-is, `fail` always fails at the tool level.
async fn run_fake_server() -> Result<(), Box<dyn std::error::Error>> {
    let running = serve_server(FakeServer, stdio()).await?;
    // serve_server returns after handling the first request (server/discover); then blocks until the connection closes.
    running.waiting().await?;
    Ok(())
}

/// Minimal MCP server implementation (stateless protocol shape, stdio transport).
#[derive(Default)]
struct FakeServer;

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
                    "Returns the text field as-is",
                    json!({
                        "type": "object",
                        "properties": { "text": { "type": "string" } },
                        "required": ["text"],
                    })
                    .as_object()
                    .expect("schema must be an object")
                    .clone(),
                ),
                RmcpTool::new(
                    "fail",
                    "Always fails at the tool level",
                    json!({ "type": "object" })
                        .as_object()
                        .expect("schema must be an object")
                        .clone(),
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
            "fail" => Ok(CallToolResult::error(vec![ContentBlock::text(
                "tool execution failed (demo)",
            )])
            .into()),
            name => Err(ErrorData::invalid_params(
                format!("unknown tool: {name}"),
                None,
            )),
        }
    }
}
