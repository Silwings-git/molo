use crate::harness::ExecutionError;
use thiserror::Error;

use super::command::CommandError;
use super::git::GitError;
use super::instructions::InstructionError;
use super::search::SearchError;
use super::workspace::WorkspaceError;

/// Errors returned by typed coding payload adapters and executor routing.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CodingError {
    /// The effect kind does not match the expected coding payload.
    #[error("unexpected effect kind: {message}")]
    UnexpectedKind {
        /// Model-safe explanation.
        message: String,
    },
    /// The effect payload could not be encoded or decoded.
    #[error("invalid coding payload: {message}")]
    InvalidPayload {
        /// Model-safe explanation.
        message: String,
    },
    /// Workspace operation failed.
    #[error("workspace error: {0}")]
    Workspace(Box<WorkspaceError>),
    /// Command operation failed.
    #[error("command error: {0}")]
    Command(#[from] CommandError),
    /// Git inspection failed.
    #[error("git error: {0}")]
    Git(#[from] GitError),
    /// Repository search failed.
    #[error("search error: {0}")]
    Search(#[from] SearchError),
    /// Instruction resolution failed.
    #[error("instruction error: {0}")]
    Instruction(#[from] InstructionError),
}

impl From<WorkspaceError> for CodingError {
    fn from(error: WorkspaceError) -> Self {
        Self::Workspace(Box::new(error))
    }
}

impl CodingError {
    /// Builds an invalid-payload error.
    pub fn invalid_payload(message: impl Into<String>) -> Self {
        Self::InvalidPayload {
            message: message.into(),
        }
    }

    /// Builds an unexpected-kind error.
    pub fn unexpected_kind(message: impl Into<String>) -> Self {
        Self::UnexpectedKind {
            message: message.into(),
        }
    }
}

impl From<CodingError> for ExecutionError {
    fn from(error: CodingError) -> Self {
        match error {
            CodingError::Command(CommandError::TimedOut { message }) => {
                ExecutionError::TimedOut(message)
            }
            CodingError::Command(CommandError::Cancelled { message }) => {
                ExecutionError::Cancelled(message)
            }
            CodingError::Command(CommandError::UnsupportedPolicy { message }) => {
                ExecutionError::Denied(message)
            }
            CodingError::UnexpectedKind { message } => ExecutionError::Unsupported(message),
            other => ExecutionError::Failed(other.to_string()),
        }
    }
}
