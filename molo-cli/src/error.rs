use std::process::ExitCode;

/// CLI error.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CliError {
    /// Argument parsing failed.
    #[error("{0}")]
    Args(String),
    /// Runtime creation failed.
    #[error("runtime error: {0}")]
    Runtime(String),
    /// Configuration failed.
    #[error("config error: {0}")]
    Config(String),
    /// Input/output failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialization failed.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    /// Provider failed.
    #[error("provider error: {0}")]
    Provider(#[from] molo::ProviderError),
    /// Agent failed.
    #[error("agent error: {0}")]
    Agent(#[from] molo::AgentError),
    /// Harness runtime failed.
    #[error("harness runtime error: {0}")]
    HarnessRuntime(#[from] molo::HarnessRuntimeError),
    /// Harness failed.
    #[error("harness error: {0}")]
    Harness(#[from] molo::HarnessError),
    /// Coding primitive failed.
    #[error("coding error: {0}")]
    Coding(#[from] molo::CodingError),
    /// Workspace operation failed.
    #[error("workspace error: {0}")]
    Workspace(#[from] molo::WorkspaceError),
    /// Git operation failed.
    #[error("git error: {0}")]
    Git(#[from] molo::GitError),
    /// Instruction resolution failed.
    #[error("instruction error: {0}")]
    Instruction(#[from] molo::InstructionError),
    /// Context gathering failed.
    #[error("context error: {0}")]
    Context(#[from] molo::CodingContextError),
}

impl CliError {
    /// Returns the process exit code associated with this error.
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Args(_) | Self::Config(_) => 2,
            _ => 1,
        }
    }

    /// Returns the process [`ExitCode`] associated with this error.
    pub fn process_exit_code(&self) -> ExitCode {
        ExitCode::from(self.exit_code() as u8)
    }
}
