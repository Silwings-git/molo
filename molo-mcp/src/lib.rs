//! MCP adapter for molo.
#![warn(missing_docs)]

pub mod mcp;

pub use molo_core::{effect, run};
pub mod tool {
    //! Tool types used by MCP adapters.
    pub use molo_agent::tool::{MissingTools, RegistryError, ToolRegistry};
    pub use molo_core::tool::*;
}
#[cfg(feature = "harness")]
pub use molo_harness as harness;

pub use mcp::*;
pub use molo_core::{
    DisplayFormat, DisplayOutput, EffectKind, EffectRequest, RiskLevel, RunContext, RunMetadata,
    SharedState, Tool, ToolContext, ToolError, ToolOutput, ToolResult, ToolSchema, ToolSource,
    ToolTrustLevel,
};
#[cfg(feature = "harness")]
pub use molo_harness::{
    EffectExecutor, ExecutionError, ExecutionPolicy, HarnessError, PolicyDecision, PolicyEngine,
    RawEffectOutput,
};
