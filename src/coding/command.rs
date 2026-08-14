use crate::harness::{ExecutionPolicy, NetworkPolicy, SandboxPolicy};
use crate::{RunContext, RunMetadata};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::workspace::{Workspace, WorkspacePath};

/// Environment variable handling for command execution.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EnvPolicy {
    /// Start from an empty environment and use only `CommandRequest::env`.
    #[default]
    Empty,
    /// Inherit only listed keys from the parent process, then apply
    /// `CommandRequest::env`.
    AllowList(Vec<String>),
    /// Inherit the full parent environment. This is not the default because
    /// it can leak secrets.
    InheritAll,
}

/// PTY mode requested for a command.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PtyMode {
    /// No PTY. This is the Phase 5 baseline implementation.
    #[default]
    Disabled,
    /// Request a PTY. The baseline local executor returns unsupported.
    Enabled,
}

/// Per-stream output limit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandOutputLimit {
    /// Maximum stdout bytes returned.
    pub stdout_bytes: usize,
    /// Maximum stderr bytes returned.
    pub stderr_bytes: usize,
}

impl Default for CommandOutputLimit {
    fn default() -> Self {
        Self {
            stdout_bytes: 64 * 1024,
            stderr_bytes: 64 * 1024,
        }
    }
}

/// Command execution request.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandRequest {
    /// Program and arguments. The first element is the program. No implicit
    /// shell parsing is performed.
    pub argv: Vec<String>,
    /// Working directory inside the workspace.
    pub cwd: WorkspacePath,
    /// Explicit environment variables. Values are redacted from Debug.
    pub env: BTreeMap<String, String>,
    /// Parent environment inheritance policy.
    pub env_policy: EnvPolicy,
    /// Optional stdin bytes.
    pub stdin: Option<Vec<u8>>,
    /// PTY mode.
    pub pty: PtyMode,
    /// Request timeout.
    pub timeout: Option<Duration>,
    /// Requested sandbox policy.
    pub requested_sandbox: Option<SandboxPolicy>,
    /// Requested network policy.
    pub requested_network: Option<NetworkPolicy>,
    /// Output limits.
    pub output_limit: CommandOutputLimit,
    /// Host-owned metadata.
    pub metadata: RunMetadata,
}

impl fmt::Debug for CommandRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CommandRequest")
            .field("argv", &self.argv)
            .field("cwd", &self.cwd)
            .field("env_keys", &self.env.keys().collect::<Vec<_>>())
            .field("env_policy", &self.env_policy)
            .field("stdin_len", &self.stdin.as_ref().map(Vec::len))
            .field("pty", &self.pty)
            .field("timeout", &self.timeout)
            .field("requested_sandbox", &self.requested_sandbox)
            .field("requested_network", &self.requested_network)
            .field("output_limit", &self.output_limit)
            .field("metadata", &self.metadata)
            .finish()
    }
}

impl CommandRequest {
    /// Constructs a command request from argv.
    pub fn new<I, S>(argv: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            argv: argv.into_iter().map(Into::into).collect(),
            cwd: WorkspacePath::root(),
            env: BTreeMap::new(),
            env_policy: EnvPolicy::Empty,
            stdin: None,
            pty: PtyMode::Disabled,
            timeout: None,
            requested_sandbox: None,
            requested_network: Some(NetworkPolicy::Deny),
            output_limit: CommandOutputLimit::default(),
            metadata: RunMetadata::new(),
        }
    }

    /// Sets the working directory.
    pub fn with_cwd(mut self, cwd: WorkspacePath) -> Self {
        self.cwd = cwd;
        self
    }

    /// Sets a timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

/// Executor capability report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandExecutorCapabilities {
    /// Whether non-PTY one-shot command execution is supported.
    pub one_shot: bool,
    /// Whether PTY command execution is supported.
    pub pty: bool,
    /// Whether the executor can technically enforce sandbox policy.
    pub sandbox_enforcement: bool,
    /// Whether the executor can technically enforce network policy.
    pub network_enforcement: bool,
}

impl Default for CommandExecutorCapabilities {
    fn default() -> Self {
        Self {
            one_shot: true,
            pty: false,
            sandbox_enforcement: false,
            network_enforcement: false,
        }
    }
}

/// Text output with truncation metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutputText {
    /// UTF-8 lossy text.
    pub text: String,
    /// Original byte length.
    pub bytes: usize,
    /// Whether text was truncated.
    pub truncated: bool,
}

/// Terminal command status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CommandStatus {
    /// Process exited with an exit code.
    Exited {
        /// Exit code.
        code: i32,
    },
    /// Process terminated by signal or platform-specific status.
    Signaled {
        /// Signal or platform-specific explanation.
        signal: String,
    },
    /// Process exceeded timeout.
    TimedOut,
    /// Process was cancelled by run context.
    Cancelled,
}

/// Report describing which policies were enforced by the command executor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyEnforcementReport {
    /// Sandbox policy requested by the harness.
    pub sandbox: SandboxPolicy,
    /// Network policy requested by the harness.
    pub network: NetworkPolicy,
    /// Whether the sandbox policy was technically enforced.
    pub sandbox_enforced: bool,
    /// Whether the network policy was technically enforced.
    pub network_enforced: bool,
    /// Warnings about advisory or unsupported policy.
    pub warnings: Vec<String>,
}

/// Command output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandOutput {
    /// Terminal status.
    pub status: CommandStatus,
    /// Captured stdout.
    pub stdout: OutputText,
    /// Captured stderr.
    pub stderr: OutputText,
    /// Process duration.
    pub duration: Duration,
    /// Whether either stream was truncated.
    pub truncated: bool,
    /// Policy enforcement report.
    pub policy_enforcement: PolicyEnforcementReport,
    /// Host-owned metadata.
    pub metadata: RunMetadata,
}

/// Executes approved command requests.
#[async_trait]
pub trait CommandExecutor: Send + Sync {
    /// Returns executor capabilities.
    fn capabilities(&self) -> CommandExecutorCapabilities;

    /// Executes a command under a harness execution policy.
    async fn execute(
        &self,
        request: CommandRequest,
        policy: &ExecutionPolicy,
        context: &RunContext,
    ) -> Result<CommandOutput, CommandError>;
}

/// Local one-shot command executor.
#[derive(Debug, Clone)]
pub struct LocalCommandExecutor<W> {
    workspace: W,
    allow_advisory_policy: bool,
}

impl<W> LocalCommandExecutor<W> {
    /// Constructs a local command executor for a workspace.
    pub fn new(workspace: W) -> Self {
        Self {
            workspace,
            allow_advisory_policy: false,
        }
    }

    /// Allows policy modes that cannot be technically enforced locally to be
    /// reported as advisory instead of failing closed.
    pub fn with_advisory_policy(mut self, allow: bool) -> Self {
        self.allow_advisory_policy = allow;
        self
    }
}

#[async_trait]
impl<W> CommandExecutor for LocalCommandExecutor<W>
where
    W: Workspace,
{
    fn capabilities(&self) -> CommandExecutorCapabilities {
        CommandExecutorCapabilities::default()
    }

    async fn execute(
        &self,
        request: CommandRequest,
        policy: &ExecutionPolicy,
        context: &RunContext,
    ) -> Result<CommandOutput, CommandError> {
        if request.argv.is_empty() {
            return Err(CommandError::InvalidRequest {
                message: "argv must not be empty".to_string(),
            });
        }
        if request.pty != PtyMode::Disabled {
            return Err(CommandError::UnsupportedPolicy {
                message: "PTY command execution is not supported by LocalCommandExecutor"
                    .to_string(),
            });
        }
        let sandbox = request
            .requested_sandbox
            .clone()
            .unwrap_or_else(|| policy.sandbox.clone());
        let network = request
            .requested_network
            .clone()
            .unwrap_or_else(|| policy.network.clone());
        let mut warnings = Vec::new();
        if !self.allow_advisory_policy
            && !matches!(
                sandbox,
                SandboxPolicy::ReadOnly | SandboxPolicy::WorkspaceWrite
            )
        {
            return Err(CommandError::UnsupportedPolicy {
                message: format!("unsupported sandbox policy: {sandbox:?}"),
            });
        }
        if !self.allow_advisory_policy && !matches!(network, NetworkPolicy::Deny) {
            return Err(CommandError::UnsupportedPolicy {
                message: format!("unsupported network policy: {network:?}"),
            });
        }
        if self.allow_advisory_policy {
            warnings.push("sandbox/network policy reported as advisory".to_string());
        }

        let cwd = self
            .workspace
            .resolve(&request.cwd, super::workspace::WorkspaceAccess::List)
            .await
            .map_err(|error| CommandError::InvalidRequest {
                message: format!("invalid command cwd: {error}"),
            })?;

        let timeout = choose_timeout(request.timeout, policy.timeout, context.remaining())
            .ok_or_else(|| CommandError::InvalidRequest {
                message: "command timeout is required".to_string(),
            })?;
        if timeout.is_zero() {
            return Err(CommandError::TimedOut {
                message: "command timeout elapsed".to_string(),
            });
        }

        let mut command = tokio::process::Command::new(&request.argv[0]);
        command.args(&request.argv[1..]);
        command.current_dir(cwd.absolute);
        command.stdin(if request.stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        });
        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());
        command.kill_on_drop(true);
        apply_env(&mut command, &request.env_policy, &request.env);

        if let Some(stdin) = &request.stdin {
            command.stdin(Stdio::piped());
            command.env("MOLO_STDIN_BYTES", stdin.len().to_string());
        }

        let started = Instant::now();
        let mut child = command.spawn().map_err(|error| CommandError::Spawn {
            message: format!("failed to spawn command: {error}"),
        })?;
        if let Some(stdin) = &request.stdin
            && let Some(mut child_stdin) = child.stdin.take()
        {
            child_stdin
                .write_all(stdin)
                .await
                .map_err(|error| CommandError::Io {
                    message: format!("failed to write stdin: {error}"),
                })?;
        }

        let stdout_task = child.stdout.take().map(|mut stdout| {
            tokio::spawn(async move {
                let mut bytes = Vec::new();
                stdout.read_to_end(&mut bytes).await.map(|_| bytes)
            })
        });
        let stderr_task = child.stderr.take().map(|mut stderr| {
            tokio::spawn(async move {
                let mut bytes = Vec::new();
                stderr.read_to_end(&mut bytes).await.map(|_| bytes)
            })
        });

        let status = tokio::select! {
            _ = context.cancellation.cancelled() => {
                let _ = child.kill().await;
                return Ok(terminal_output(
                    CommandStatus::Cancelled,
                    started.elapsed(),
                    sandbox,
                    network,
                    warnings,
                    &request,
                ));
            }
            result = tokio::time::timeout(timeout, child.wait()) => {
                match result {
                    Ok(status) => status.map_err(|error| CommandError::Io {
                        message: format!("failed to wait for command: {error}"),
                    })?,
                    Err(_) => {
                        let _ = child.kill().await;
                        return Ok(terminal_output(
                            CommandStatus::TimedOut,
                            started.elapsed(),
                            sandbox,
                            network,
                            warnings,
                            &request,
                        ));
                    }
                }
            }
        };

        let stdout_bytes = join_reader(stdout_task).await?;
        let stderr_bytes = join_reader(stderr_task).await?;
        let stdout = truncate_bytes(&stdout_bytes, request.output_limit.stdout_bytes);
        let stderr = truncate_bytes(&stderr_bytes, request.output_limit.stderr_bytes);
        let truncated = stdout.truncated || stderr.truncated;
        let status = match status.code() {
            Some(code) => CommandStatus::Exited { code },
            None => CommandStatus::Signaled {
                signal: format!("{status:?}"),
            },
        };
        Ok(CommandOutput {
            status,
            stdout,
            stderr,
            duration: started.elapsed(),
            truncated,
            policy_enforcement: PolicyEnforcementReport {
                sandbox,
                network,
                sandbox_enforced: false,
                network_enforced: false,
                warnings,
            },
            metadata: command_metadata(&request),
        })
    }
}

fn apply_env(
    command: &mut tokio::process::Command,
    policy: &EnvPolicy,
    env: &BTreeMap<String, String>,
) {
    command.env_clear();
    match policy {
        EnvPolicy::Empty => {}
        EnvPolicy::AllowList(keys) => {
            for key in keys {
                if let Some(value) = std::env::var_os(key) {
                    command.env(key, value);
                }
            }
        }
        EnvPolicy::InheritAll => {
            for (key, value) in std::env::vars_os() {
                command.env(key, value);
            }
        }
    }
    for (key, value) in env {
        command.env(key, value);
    }
}

fn choose_timeout(
    request: Option<Duration>,
    policy: Option<Duration>,
    remaining: Option<Duration>,
) -> Option<Duration> {
    [request, policy, remaining]
        .into_iter()
        .flatten()
        .reduce(|left, right| left.min(right))
}

fn truncate_bytes(bytes: &[u8], max: usize) -> OutputText {
    let truncated = bytes.len() > max;
    let mut selected = bytes.to_vec();
    if truncated {
        selected.truncate(max);
    }
    OutputText {
        text: String::from_utf8_lossy(&selected).into_owned(),
        bytes: bytes.len(),
        truncated,
    }
}

async fn join_reader(
    task: Option<tokio::task::JoinHandle<std::io::Result<Vec<u8>>>>,
) -> Result<Vec<u8>, CommandError> {
    let Some(task) = task else {
        return Ok(Vec::new());
    };
    task.await
        .map_err(|error| CommandError::Io {
            message: format!("failed to join output reader: {error}"),
        })?
        .map_err(|error| CommandError::Io {
            message: format!("failed to read command output: {error}"),
        })
}

fn terminal_output(
    status: CommandStatus,
    duration: Duration,
    sandbox: SandboxPolicy,
    network: NetworkPolicy,
    warnings: Vec<String>,
    request: &CommandRequest,
) -> CommandOutput {
    CommandOutput {
        status,
        stdout: truncate_bytes(&[], request.output_limit.stdout_bytes),
        stderr: truncate_bytes(&[], request.output_limit.stderr_bytes),
        duration,
        truncated: false,
        policy_enforcement: PolicyEnforcementReport {
            sandbox,
            network,
            sandbox_enforced: false,
            network_enforced: false,
            warnings,
        },
        metadata: command_metadata(request),
    }
}

fn command_metadata(request: &CommandRequest) -> RunMetadata {
    let mut metadata = RunMetadata::new();
    metadata.insert("argv".to_string(), serde_json::json!(request.argv));
    metadata.insert(
        "env_keys".to_string(),
        serde_json::json!(request.env.keys().cloned().collect::<Vec<_>>()),
    );
    metadata
}

/// Command execution errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CommandError {
    /// Command request is invalid.
    #[error("invalid command request: {message}")]
    InvalidRequest {
        /// Model-safe explanation.
        message: String,
    },
    /// Requested policy cannot be enforced by this executor.
    #[error("unsupported command policy: {message}")]
    UnsupportedPolicy {
        /// Model-safe explanation.
        message: String,
    },
    /// Process spawn failed.
    #[error("failed to spawn command: {message}")]
    Spawn {
        /// Model-safe explanation.
        message: String,
    },
    /// Command timed out before producing output.
    #[error("command timed out: {message}")]
    TimedOut {
        /// Model-safe explanation.
        message: String,
    },
    /// Command was cancelled before producing output.
    #[error("command cancelled: {message}")]
    Cancelled {
        /// Model-safe explanation.
        message: String,
    },
    /// Output exceeded a hard limit.
    #[error("command output limit: {message}")]
    OutputLimit {
        /// Model-safe explanation.
        message: String,
    },
    /// I/O failed.
    #[error("command I/O error: {message}")]
    Io {
        /// Model-safe explanation.
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::LocalWorkspace;
    use crate::harness::OutputLimit;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("molo-command-test-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn command_debug_redacts_env_values() {
        let mut request = CommandRequest::new(["echo", "ok"]);
        request
            .env
            .insert("TOKEN".to_string(), "secret".to_string());
        let debug = format!("{request:?}");
        assert!(debug.contains("TOKEN"));
        assert!(!debug.contains("secret"));
    }

    #[tokio::test]
    async fn local_command_executes_without_shell_parsing() {
        let root = temp_dir("argv");
        let workspace = LocalWorkspace::new(&root).unwrap();
        let executor = LocalCommandExecutor::new(workspace);
        let output = executor
            .execute(
                CommandRequest::new(["printf", "%s", "a;b"]),
                &ExecutionPolicy {
                    sandbox: SandboxPolicy::ReadOnly,
                    network: NetworkPolicy::Deny,
                    timeout: Some(Duration::from_secs(5)),
                    output_limit: OutputLimit::default(),
                },
                &RunContext::new("cmd"),
            )
            .await
            .unwrap();
        assert_eq!(output.stdout.text, "a;b");
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn local_command_truncates_stdout_and_stderr_separately() {
        let root = temp_dir("truncate");
        let workspace = LocalWorkspace::new(&root).unwrap();
        let executor = LocalCommandExecutor::new(workspace);
        let mut request = CommandRequest::new(["sh", "-c", "printf 12345; printf abcde >&2"]);
        request.output_limit = CommandOutputLimit {
            stdout_bytes: 3,
            stderr_bytes: 2,
        };
        let output = executor
            .execute(
                request,
                &ExecutionPolicy {
                    sandbox: SandboxPolicy::ReadOnly,
                    network: NetworkPolicy::Deny,
                    timeout: Some(Duration::from_secs(5)),
                    output_limit: OutputLimit::default(),
                },
                &RunContext::new("cmd"),
            )
            .await
            .unwrap();
        assert_eq!(output.stdout.text, "123");
        assert_eq!(output.stderr.text, "ab");
        assert!(output.truncated);
        let _ = std::fs::remove_dir_all(root);
    }
}
