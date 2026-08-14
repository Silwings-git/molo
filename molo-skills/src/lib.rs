//! Agent Skills protocol support for molo.
#![warn(missing_docs)]

pub mod skill;

pub mod tool {
    //! Tool types used by skill tools and hosts.
    pub use molo_agent::tool::{MissingTools, RegistryError, ToolRegistry};
    pub use molo_core::tool::*;
}

pub use molo_core::{
    RunContext, RunMetadata, SharedState, Tool, ToolContext, ToolError, ToolMemoryPolicy,
};
pub use skill::*;
