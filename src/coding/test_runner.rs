use crate::harness::ExecutionPolicy;
use crate::{RunContext, RunMetadata};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::command::{CommandExecutor, CommandOutput, CommandRequest, CommandStatus};

/// Test run request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TestRunRequest {
    /// Command to execute as a verification step.
    pub command: CommandRequest,
    /// Human-readable test target name.
    pub name: String,
}

/// Structured verification result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Test target name.
    pub name: String,
    /// Whether verification succeeded.
    pub passed: bool,
    /// Short model-visible summary.
    pub summary: String,
    /// Command output.
    pub output: Option<CommandOutput>,
    /// Host-owned metadata.
    pub metadata: RunMetadata,
}

/// Convenience adapter for test commands.
#[async_trait]
pub trait TestRunner: Send + Sync {
    /// Runs a test command through a command executor policy boundary.
    async fn run(
        &self,
        request: TestRunRequest,
        policy: &ExecutionPolicy,
        context: &RunContext,
    ) -> Result<VerificationResult, TestRunError>;
}

/// Test runner backed by a [`CommandExecutor`].
#[derive(Debug, Clone)]
pub struct CommandTestRunner<C> {
    commands: C,
}

impl<C> CommandTestRunner<C> {
    /// Constructs a command test runner.
    pub fn new(commands: C) -> Self {
        Self { commands }
    }
}

#[async_trait]
impl<C> TestRunner for CommandTestRunner<C>
where
    C: CommandExecutor,
{
    async fn run(
        &self,
        request: TestRunRequest,
        policy: &ExecutionPolicy,
        context: &RunContext,
    ) -> Result<VerificationResult, TestRunError> {
        let output = self
            .commands
            .execute(request.command, policy, context)
            .await
            .map_err(|error| TestRunError::Command {
                message: error.to_string(),
            })?;
        let passed = matches!(output.status, CommandStatus::Exited { code: 0 });
        let summary = if passed {
            format!("{} passed", request.name)
        } else {
            format!("{} failed: {:?}", request.name, output.status)
        };
        Ok(VerificationResult {
            name: request.name,
            passed,
            summary,
            output: Some(output),
            metadata: RunMetadata::new(),
        })
    }
}

/// Test run errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TestRunError {
    /// Command execution failed.
    #[error("test command error: {message}")]
    Command {
        /// Model-safe explanation.
        message: String,
    },
}
