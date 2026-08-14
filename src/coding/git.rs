use crate::harness::{ExecutionPolicy, NetworkPolicy, OutputLimit, SandboxPolicy};
use crate::{RunContext, RunMetadata};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use super::command::{CommandExecutor, CommandOutputLimit, CommandRequest, CommandStatus};
use super::workspace::{WorkspaceDiff, WorkspacePath};

/// Read-only git operation for typed git effects.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GitOperation {
    /// `git status --porcelain=v1 -b`.
    Status(GitStatusRequest),
    /// `git diff`.
    Diff(GitDiffRequest),
    /// Changed files derived from status.
    ChangedFiles(GitChangedFilesRequest),
    /// Current HEAD.
    Head,
}

/// Git status request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitStatusRequest {
    /// Include branch header.
    pub include_branch: bool,
}

impl Default for GitStatusRequest {
    fn default() -> Self {
        Self {
            include_branch: true,
        }
    }
}

/// Git diff request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitDiffRequest {
    /// Paths to diff. Empty means all paths.
    pub paths: Vec<WorkspacePath>,
    /// Whether staged changes should be diffed.
    pub staged: bool,
    /// Maximum diff bytes.
    pub max_bytes: usize,
}

impl Default for GitDiffRequest {
    fn default() -> Self {
        Self {
            paths: Vec::new(),
            staged: false,
            max_bytes: 256 * 1024,
        }
    }
}

/// Changed-files request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitChangedFilesRequest {
    /// Include untracked files.
    pub include_untracked: bool,
}

impl Default for GitChangedFilesRequest {
    fn default() -> Self {
        Self {
            include_untracked: true,
        }
    }
}

/// Parsed git status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitStatus {
    /// Branch header, when requested and available.
    pub branch: Option<String>,
    /// Changed files.
    pub changed_files: Vec<GitChangedFile>,
    /// Raw status text after output limiting.
    pub raw: String,
    /// Whether raw output was truncated.
    pub truncated: bool,
    /// Host-owned metadata.
    pub metadata: RunMetadata,
}

/// Changed git file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitChangedFile {
    /// Workspace path.
    pub path: WorkspacePath,
    /// Two-character porcelain status.
    pub status: String,
}

/// Current git head.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitHead {
    /// Commit hash.
    pub commit: String,
    /// Current branch name, when available.
    pub branch: Option<String>,
}

/// Read-only git inspector.
#[async_trait]
pub trait GitInspector: Send + Sync {
    /// Returns git status.
    async fn status(
        &self,
        request: GitStatusRequest,
        context: &RunContext,
    ) -> Result<GitStatus, GitError>;

    /// Returns git diff as a workspace diff summary.
    async fn diff(
        &self,
        request: GitDiffRequest,
        context: &RunContext,
    ) -> Result<WorkspaceDiff, GitError>;

    /// Returns changed files from git status.
    async fn changed_files(
        &self,
        request: GitChangedFilesRequest,
        context: &RunContext,
    ) -> Result<Vec<GitChangedFile>, GitError>;

    /// Returns current git head.
    async fn head(&self, context: &RunContext) -> Result<Option<GitHead>, GitError>;
}

/// Git inspector implemented by invoking read-only git commands.
#[derive(Debug, Clone)]
pub struct CliGitInspector<C> {
    commands: C,
    timeout: Duration,
}

impl<C> CliGitInspector<C> {
    /// Constructs a git inspector from a command executor.
    pub fn new(commands: C) -> Self {
        Self {
            commands,
            timeout: Duration::from_secs(10),
        }
    }

    /// Sets command timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[async_trait]
impl<C> GitInspector for CliGitInspector<C>
where
    C: CommandExecutor,
{
    async fn status(
        &self,
        request: GitStatusRequest,
        context: &RunContext,
    ) -> Result<GitStatus, GitError> {
        let output = self
            .run_git(["status", "--porcelain=v1", "-b"], context)
            .await?;
        let mut branch = None;
        let mut changed = Vec::new();
        for line in output.stdout.text.lines() {
            if line.starts_with("## ") {
                if request.include_branch {
                    branch = Some(line.trim_start_matches("## ").to_string());
                }
                continue;
            }
            if line.len() < 4 {
                continue;
            }
            let status = line[..2].to_string();
            let raw_path = line[3..].trim();
            let path_text = raw_path
                .rsplit_once(" -> ")
                .map(|(_, to)| to)
                .unwrap_or(raw_path)
                .trim_matches('"');
            if let Ok(path) = WorkspacePath::parse(path_text) {
                changed.push(GitChangedFile { path, status });
            }
        }
        Ok(GitStatus {
            branch,
            changed_files: changed,
            raw: output.stdout.text,
            truncated: output.truncated,
            metadata: output.metadata,
        })
    }

    async fn diff(
        &self,
        request: GitDiffRequest,
        context: &RunContext,
    ) -> Result<WorkspaceDiff, GitError> {
        let mut argv = vec!["diff".to_string(), "--no-color".to_string()];
        if request.staged {
            argv.push("--cached".to_string());
        }
        if !request.paths.is_empty() {
            argv.push("--".to_string());
            argv.extend(request.paths.iter().map(WorkspacePath::display));
        }
        let mut command = CommandRequest::new(std::iter::once("git".to_string()).chain(argv));
        command.timeout = Some(self.timeout);
        command.output_limit = CommandOutputLimit {
            stdout_bytes: request.max_bytes,
            stderr_bytes: 64 * 1024,
        };
        let output = self.run(command, context).await?;
        Ok(WorkspaceDiff {
            changed_files: request.paths,
            text: output.stdout.text,
            truncated: output.truncated,
            metadata: output.metadata,
        })
    }

    async fn changed_files(
        &self,
        request: GitChangedFilesRequest,
        context: &RunContext,
    ) -> Result<Vec<GitChangedFile>, GitError> {
        let status = self.status(GitStatusRequest::default(), context).await?;
        Ok(status
            .changed_files
            .into_iter()
            .filter(|file| request.include_untracked || file.status != "??")
            .collect())
    }

    async fn head(&self, context: &RunContext) -> Result<Option<GitHead>, GitError> {
        let commit = self
            .run_git(["rev-parse", "HEAD"], context)
            .await?
            .stdout
            .text
            .trim()
            .to_string();
        if commit.is_empty() {
            return Ok(None);
        }
        let branch_output = self.run_git(["branch", "--show-current"], context).await?;
        let branch = match branch_output.stdout.text.trim() {
            "" => None,
            branch => Some(branch.to_string()),
        };
        Ok(Some(GitHead { commit, branch }))
    }
}

impl<C> CliGitInspector<C>
where
    C: CommandExecutor,
{
    async fn run_git<I, S>(
        &self,
        args: I,
        context: &RunContext,
    ) -> Result<super::command::CommandOutput, GitError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let argv = std::iter::once("git".to_string())
            .chain(args.into_iter().map(Into::into))
            .collect::<Vec<_>>();
        self.run(CommandRequest::new(argv), context).await
    }

    async fn run(
        &self,
        mut command: CommandRequest,
        context: &RunContext,
    ) -> Result<super::command::CommandOutput, GitError> {
        command.timeout = command.timeout.or(Some(self.timeout));
        command.requested_network = Some(NetworkPolicy::Deny);
        let output = self
            .commands
            .execute(
                command,
                &ExecutionPolicy {
                    sandbox: SandboxPolicy::ReadOnly,
                    network: NetworkPolicy::Deny,
                    timeout: Some(self.timeout),
                    output_limit: OutputLimit::default(),
                },
                context,
            )
            .await
            .map_err(|error| GitError::Command {
                message: error.to_string(),
            })?;
        match output.status {
            CommandStatus::Exited { code: 0 } => Ok(output),
            _ => Err(GitError::Command {
                message: format!("git command failed: {}", output.stderr.text),
            }),
        }
    }
}

/// Git inspection errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GitError {
    /// Git command failed.
    #[error("git command error: {message}")]
    Command {
        /// Model-safe explanation.
        message: String,
    },
    /// Git output could not be parsed.
    #[error("git parse error: {message}")]
    Parse {
        /// Model-safe explanation.
        message: String,
    },
    /// Git operation is unsupported.
    #[error("unsupported git operation: {message}")]
    Unsupported {
        /// Model-safe explanation.
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_status_path() {
        let line = " M src/lib.rs";
        let status = line[..2].to_string();
        let path = WorkspacePath::parse(line[3..].trim()).unwrap();
        assert_eq!(status, " M");
        assert_eq!(path.display(), "src/lib.rs");
    }
}
