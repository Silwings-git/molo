use molo::{
    AlwaysAllowApprovalBroker, BasicHarness, FakeProvider, FakeReply, HarnessRuntime,
    McpCallPayload, McpClientProvider, McpEffectExecutor, McpEffectTool, McpServerId,
    McpToolCallOutput, McpToolDescriptor, McpToolId, ReActAgent, RunContext, RunMetadata,
    RunRequest, ToolRegistry,
};
use serde_json::json;

#[derive(Debug, Clone)]
struct FakeMcpClient;

#[molo::async_trait]
impl McpClientProvider for FakeMcpClient {
    async fn call_tool(
        &self,
        payload: &McpCallPayload,
        _timeout: std::time::Duration,
        _context: &RunContext,
    ) -> Result<McpToolCallOutput, molo::McpError> {
        let text = payload
            .arguments
            .get("text")
            .and_then(|value| value.as_str())
            .unwrap_or_default();
        Ok(McpToolCallOutput::text(format!("echo: {text}")))
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let descriptor = McpToolDescriptor {
        id: McpToolId::new(McpServerId::new("fake"), "echo"),
        display_name: "fake__echo".into(),
        description: "Echo text through a governed MCP effect".into(),
        input_schema: json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        }),
        output_schema: None,
        annotations: serde_json::Value::Null,
        schema_digest: "example".into(),
        cache: None,
        metadata: RunMetadata::new(),
    };
    let tool = McpEffectTool::new(descriptor.clone());

    let mut registry = ToolRegistry::new();
    registry.register_with_source(tool, descriptor.source(molo::ToolTrustLevel::External))?;

    let provider = FakeProvider::new([
        FakeReply::ToolCalls {
            content: String::new(),
            calls: vec![molo::ToolCall {
                id: "call-echo".into(),
                name: "fake__echo".into(),
                arguments: r#"{"text":"hello"}"#.into(),
            }],
        },
        FakeReply::Text("The MCP server echoed hello.".into()),
    ]);
    let harness = BasicHarness::new(
        McpEffectExecutor::new(FakeMcpClient),
        molo::DefaultPolicyEngine,
        AlwaysAllowApprovalBroker,
        molo::NoopAuditSink,
        molo::NoopTranscriptStore,
    );
    let runtime = HarnessRuntime::new(provider, harness);
    let mut kernel = ReActAgent::kernel(registry, "Use governed MCP observations.");

    let output = runtime
        .run(
            &mut kernel,
            RunRequest::text("Echo hello through fake__echo."),
            RunContext::new("mcp-governed-example"),
        )
        .await?;

    println!("{}", output.answer);
    Ok(())
}
