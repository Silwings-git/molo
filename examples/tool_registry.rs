//! ToolRegistry example: demonstrates the registry's full API (new / register /
//! names / schemas / get / call / subset).
//!
//! This example is **self-contained**, needs no API key, just run:
//! `cargo run --example tool_registry`
//!
//! It shows ToolRegistry used as a component: register tools → query →
//! execute → carve out a sub-registry (the scenario where a main agent
//! restricts a sub-agent's tool scope). The tool-call loop wired to a real
//! model is in `examples/tool_agent.rs`.

use molo::{
    RunContext, SharedState, Tool, ToolCall, ToolContext, ToolError, ToolOutput, ToolRegistry,
    ToolResult, ToolSchema,
};
use schemars::JsonSchema;
use serde::Deserialize;

/// Arguments for the calculator tool.
#[derive(Debug, Deserialize, JsonSchema)]
struct CalcArgs {
    /// The math expression to evaluate, e.g. "1 + 2 * 3".
    #[schemars(description = "The math expression to evaluate, e.g. \"1 + 2 * 3\"")]
    expression: String,
}

/// A tool that evaluates math expressions.
struct Calculator;

#[async_trait::async_trait]
impl Tool for Calculator {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "calculator",
            "Evaluates a math expression; supports basic arithmetic and parentheses, e.g. \"(1 + 2) * 3\".",
            serde_json::to_value(schemars::schema_for!(CalcArgs))
                .expect("tool schema must serialize"),
        )
    }

    async fn call(
        &self,
        arguments: serde_json::Value,
        _context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let args: CalcArgs = serde_json::from_value(arguments)?;
        let value =
            evalexpr::eval(&args.expression).map_err(|e| ToolError::Execution(e.to_string()))?;
        Ok(ToolOutput::text(value.to_string()).into())
    }
}

/// A tool that returns the current Unix timestamp; no arguments.
struct CurrentTime;

#[async_trait::async_trait]
impl Tool for CurrentTime {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            "current_time",
            "Returns the current Unix timestamp in seconds.",
            serde_json::json!({ "type": "object", "properties": {} }),
        )
    }

    async fn call(
        &self,
        _arguments: serde_json::Value,
        _context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| ToolError::Execution(e.to_string()))?;
        Ok(ToolOutput::text(now.as_secs().to_string()).into())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Register: chained register; the registration order is the order returned by names / schemas.
    let mut registry = ToolRegistry::new();
    registry.register(Calculator).register(CurrentTime);
    let run = RunContext::new("tool-registry-example");
    let state = SharedState::new();

    // 2. names / schemas: in registration order.
    println!("1. registered tools: {:?}", registry.names());
    for schema in registry.schemas() {
        println!(
            "   - {}: {} - {}",
            schema.name, schema.description, schema.parameters
        );
    }

    // 3. call: normal execution; arguments are passed as JSON text on the wire; Ok = the result text.
    println!(
        "2. call calculator: {}",
        registry
            .call_named(
                "calculator",
                r#"{"expression": "(1 + 2) * 3"}"#,
                &run,
                &state
            )
            .await?
    );

    // 4. call's error paths: tool not found / arguments not JSON / execution failure → Err carries the category,
    //    and Display is "error as text" (the Agent loop feeds it back to the model as a ToolResult).
    println!(
        "3. call a nonexistent tool: {}",
        registry
            .call_named("nope", "{}", &run, &state)
            .await
            .unwrap_err()
    );
    println!(
        "4. call with non-JSON arguments:  {}",
        registry
            .call_named("calculator", "not valid JSON", &run, &state)
            .await
            .unwrap_err()
    );

    // 5. get: bypass the registry's argument parsing and call Tool::call directly
    //    (the expression field is deliberately missing here, yielding InvalidArguments instead of Execution).
    let calculator = registry
        .get("calculator")
        .expect("calculator must be registered");
    let result = calculator
        .call(
            serde_json::json!({}),
            ToolContext::new(&run, &state, "manual-call", "calculator"),
        )
        .await;
    match result {
        Ok(_) => println!("5. (unexpected success)"),
        Err(e) => println!("5. error semantics of get calling Tool::call directly: {e}"),
    }

    // 6. subset: restrict the tool scope when a main agent creates a sub-agent — the sub-registry
    //    gets only calculator and shares the same tool instances with the main registry.
    let sub = registry.subset(&["calculator"])?;
    println!("6. sub-registry (only calculator): {:?}", sub.names());
    println!(
        "   → the sub-registry can execute too: {}",
        sub.call(
            &ToolCall {
                id: "sub-call".into(),
                name: "calculator".into(),
                arguments: r#"{"expression": "7 * 6"}"#.into(),
            },
            &run,
            &state,
        )
        .await?
    );

    // 7. subset with a missing name: Err carries the missing list; the caller decides whether to error / warn / stay silent.
    match registry.subset(&["calculator", "nonexistent"]) {
        Ok(_) => {}
        Err(e) => println!("7. subset with missing names: {e}"),
    }

    // 8. Debug: prints the tool name list, handy for inspecting the registry while debugging.
    println!("8. registry debug output: {registry:?}");

    Ok(())
}
