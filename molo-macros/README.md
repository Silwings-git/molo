# molo-macros

The procedural macro crate for molo: one-shot tool definitions via `#[molo::tool]` — generates a
`Tool` implementation (struct + schema + call) from an async function.

Re-exported by `molo` (`use molo::tool`); you normally don't need to depend on this crate directly.

```rust
#[molo::tool(
    description = "Evaluate a mathematical expression",
    side_effects = "pure",
    risk = "low"
)]
async fn calculator(expression: String) -> Result<String, ToolError> {
    // ...
}
```

The macro also accepts `Result<ToolOutput, ToolError>` and
`Result<ToolResult, ToolError>`. Policy hints are declarations on
`ToolSchema`; effect execution still belongs to the outer harness/runtime.

## License

MIT OR Apache-2.0
