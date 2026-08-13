use molo::{
    AlwaysAllowApprovalBroker, BasicHarness, EffectKind, EffectRequest, FakeProvider, FakeReply,
    HarnessRuntime, RawEffectOutput, ReActAgent, RunContext, RunRequest, StaticEffectExecutor,
    Tool, ToolContext, ToolError, ToolRegistry, ToolResult, ToolSchema,
};
use serde_json::json;

struct ReadConfigTool;

#[molo::async_trait]
impl Tool for ReadConfigTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "read_config",
            "Read the application config through the outer harness",
            json!({"type": "object", "properties": {}}),
        )
    }

    async fn call(
        &self,
        _arguments: serde_json::Value,
        _context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        Ok(ToolResult::Effect(
            EffectRequest::new(
                EffectKind::ReadFile,
                "Read config",
                json!({ "path": "app.toml" }),
            )
            .with_id("read-config"),
        ))
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = FakeProvider::new([
        FakeReply::ToolCalls {
            content: String::new(),
            calls: vec![molo::ToolCall {
                id: "call-read-config".into(),
                name: "read_config".into(),
                arguments: "{}".into(),
            }],
        },
        FakeReply::Text("Config says retries = 2.".into()),
    ]);

    let executor = StaticEffectExecutor::new()
        .with_output("read-config", RawEffectOutput::text(r#"{"retries":2}"#));
    let harness = BasicHarness::new(
        executor,
        molo::DefaultPolicyEngine,
        AlwaysAllowApprovalBroker,
        molo::NoopAuditSink,
        molo::NoopTranscriptStore,
    );
    let runtime = HarnessRuntime::new(provider, harness);

    let mut registry = ToolRegistry::new();
    registry.register(ReadConfigTool);
    let mut kernel = ReActAgent::kernel(registry, "Answer from governed observations.");

    let output = runtime
        .run(
            &mut kernel,
            RunRequest::text("Read the config and summarize it."),
            RunContext::new("harness-example"),
        )
        .await?;

    println!("{}", output.answer);
    Ok(())
}
