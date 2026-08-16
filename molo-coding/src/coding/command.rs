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
    /// No PTY. The local executor supports one-shot, non-interactive commands.
    #[default]
    Disabled,
    /// Request a PTY. The local executor returns unsupported when PTY support
    /// is unavailable.
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

/// How strictly policy/capability mismatches are handled.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PolicyCapabilityMode {
    /// Required sandbox or network restrictions must be technically enforced.
    #[default]
    RequireEnforced,
    /// Advisory execution may continue, but reports must say so explicitly.
    AllowAdvisory,
}

impl PolicyCapabilityMode {
    /// Returns true when advisory execution is explicitly allowed.
    pub fn allows_advisory(self) -> bool {
        matches!(self, Self::AllowAdvisory)
    }
}

/// Structured enforcement status for a policy dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PolicyEnforcementStatus {
    /// The requested policy was technically enforced by the executor backend.
    Enforced,
    /// The executor ran without technical enforcement and reported the downgrade.
    Advisory,
    /// The executor does not support this policy dimension.
    Unsupported,
    /// This policy dimension was not requested for the command.
    NotRequested,
    /// The executor could not determine whether enforcement happened.
    Unknown,
}

impl PolicyEnforcementStatus {
    /// Returns true when the status is [`PolicyEnforcementStatus::Enforced`].
    pub fn is_enforced(self) -> bool {
        matches!(self, Self::Enforced)
    }

    /// Returns true when the status is [`PolicyEnforcementStatus::Advisory`].
    pub fn is_advisory(self) -> bool {
        matches!(self, Self::Advisory)
    }
}

/// Command executor backend family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CommandExecutorBackend {
    /// Plain local process execution.
    LocalProcess,
    /// Host-provided backend outside molo's built-in executors.
    HostProvided,
    /// OS sandbox backend.
    Sandbox,
    /// Container backend.
    Container,
    /// Remote isolated worker backend.
    Remote,
    /// Application-specific backend kind.
    Custom(String),
}

/// Executor identity included in capability and execution reports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandExecutorIdentity {
    /// Executor name.
    pub name: String,
    /// Executor version, when available.
    pub version: Option<String>,
    /// Backend kind.
    pub backend: CommandExecutorBackend,
    /// Platform summary.
    pub platform: String,
    /// Host-owned metadata.
    pub metadata: RunMetadata,
}

impl CommandExecutorIdentity {
    /// Constructs an executor identity.
    pub fn new(name: impl Into<String>, backend: CommandExecutorBackend) -> Self {
        Self {
            name: name.into(),
            version: None,
            backend,
            platform: format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH),
            metadata: RunMetadata::new(),
        }
    }

    /// Constructs the identity for [`LocalCommandExecutor`].
    pub fn local_process() -> Self {
        Self::new(
            "local-command-executor",
            CommandExecutorBackend::LocalProcess,
        )
        .with_version(env!("CARGO_PKG_VERSION"))
    }

    /// Sets executor version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Sets host-owned metadata.
    pub fn with_metadata(mut self, metadata: RunMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Executor capability report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandExecutorCapabilities {
    /// Executor identity.
    pub identity: CommandExecutorIdentity,
    /// Whether non-PTY one-shot command execution is supported.
    pub one_shot: bool,
    /// Whether PTY command execution is supported.
    pub pty: bool,
    /// Whether the executor can technically enforce sandbox policy.
    ///
    /// Prefer [`CommandExecutorCapabilities::sandbox`] for new code.
    pub sandbox_enforcement: bool,
    /// Whether the executor can technically enforce network policy.
    ///
    /// Prefer [`CommandExecutorCapabilities::network`] for new code.
    pub network_enforcement: bool,
    /// Sandbox capability status for restrictive sandbox policies.
    pub sandbox: PolicyEnforcementStatus,
    /// Network capability status for restrictive network policies.
    pub network: PolicyEnforcementStatus,
    /// Whether timeout/cancellation can clean up the full process tree.
    pub process_cleanup: PolicyEnforcementStatus,
    /// Resource limit enforcement status beyond wall-time/output limits.
    pub resource_limits: PolicyEnforcementStatus,
    /// Host-owned capability metadata.
    pub metadata: RunMetadata,
}

impl Default for CommandExecutorCapabilities {
    fn default() -> Self {
        Self::local_process()
    }
}

impl CommandExecutorCapabilities {
    /// Capability report for the built-in local executor.
    pub fn local_process() -> Self {
        Self {
            identity: CommandExecutorIdentity::local_process(),
            one_shot: true,
            pty: false,
            sandbox_enforcement: false,
            network_enforcement: false,
            sandbox: PolicyEnforcementStatus::Advisory,
            network: PolicyEnforcementStatus::Advisory,
            process_cleanup: PolicyEnforcementStatus::Advisory,
            resource_limits: PolicyEnforcementStatus::Unsupported,
            metadata: RunMetadata::new(),
        }
    }

    /// Capability report for a host executor that enforces sandbox and network
    /// restrictions.
    pub fn enforced(identity: CommandExecutorIdentity) -> Self {
        Self {
            identity,
            one_shot: true,
            pty: false,
            sandbox_enforcement: true,
            network_enforcement: true,
            sandbox: PolicyEnforcementStatus::Enforced,
            network: PolicyEnforcementStatus::Enforced,
            process_cleanup: PolicyEnforcementStatus::Enforced,
            resource_limits: PolicyEnforcementStatus::Unknown,
            metadata: RunMetadata::new(),
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
    /// Executor identity.
    pub executor: CommandExecutorIdentity,
    /// Sandbox policy requested by the harness.
    pub sandbox: SandboxPolicy,
    /// Network policy requested by the harness.
    pub network: NetworkPolicy,
    /// Whether the sandbox policy was technically enforced.
    pub sandbox_enforced: bool,
    /// Whether the network policy was technically enforced.
    pub network_enforced: bool,
    /// Structured sandbox enforcement status.
    pub sandbox_status: PolicyEnforcementStatus,
    /// Structured network enforcement status.
    pub network_status: PolicyEnforcementStatus,
    /// Process tree cleanup status after timeout/cancellation.
    pub process_cleanup_status: PolicyEnforcementStatus,
    /// Resource limit enforcement status.
    pub resource_limit_status: PolicyEnforcementStatus,
    /// Warnings about advisory or unsupported policy.
    pub warnings: Vec<String>,
    /// Host-owned enforcement metadata.
    pub metadata: RunMetadata,
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

/// Validates that executor capabilities can satisfy a command policy before
/// execution starts.
///
/// # Errors
///
/// Returns [`CommandError::UnsupportedPolicy`] when required sandbox or
/// network enforcement is missing and advisory mode is not enabled.
pub fn validate_command_capabilities(
    capabilities: &CommandExecutorCapabilities,
    request: &CommandRequest,
    policy: &ExecutionPolicy,
    mode: PolicyCapabilityMode,
) -> Result<(), CommandError> {
    let sandbox = requested_sandbox(request, policy);
    let network = requested_network(request, policy);
    validate_required_status(
        "sandbox",
        sandbox_requires_enforcement(&sandbox),
        capabilities.sandbox,
        mode,
    )?;
    validate_required_status(
        "network",
        network_requires_enforcement(&network),
        capabilities.network,
        mode,
    )
}

/// Validates that an executor's post-run report matches requested policy and
/// declared capabilities.
///
/// # Errors
///
/// Returns [`CommandError::UnsupportedPolicy`] when the report downgrades or
/// contradicts required enforcement.
pub fn validate_policy_enforcement_report(
    capabilities: &CommandExecutorCapabilities,
    report: &PolicyEnforcementReport,
    mode: PolicyCapabilityMode,
) -> Result<(), CommandError> {
    validate_report_status(
        "sandbox",
        sandbox_requires_enforcement(&report.sandbox),
        capabilities.sandbox,
        report.sandbox_status,
        mode,
    )?;
    validate_report_status(
        "network",
        network_requires_enforcement(&report.network),
        capabilities.network,
        report.network_status,
        mode,
    )
}

/// Local non-PTY, one-shot command executor backed by host process spawning.
///
/// This executor resolves [`CommandRequest::cwd`] through the [`Workspace`],
/// starts `argv` directly without implicit shell parsing, applies
/// [`EnvPolicy`], requires a timeout, captures stdout/stderr separately, and
/// reports output truncation.
///
/// This executor is not an OS sandbox. It does not technically enforce
/// [`SandboxPolicy`], [`NetworkPolicy`], network isolation, process-tree
/// cleanup, or resource limits. Its capability report marks sandbox, network,
/// and process cleanup as advisory, and resource limits as unsupported.
///
/// By default, command execution fails closed when the requested policy requires
/// technical enforcement that this local process backend cannot provide.
/// [`LocalCommandExecutor::with_advisory_policy`] and
/// [`LocalCommandExecutor::with_policy_capability_mode`] can explicitly allow
/// such execution to continue with advisory enforcement reports. That mode is
/// intended for tests, local prototypes, and reference CLI dogfooding; production
/// coding-agent hosts should inject a [`CommandExecutor`] backed by a container,
/// VM, platform sandbox, or remote isolated worker that can enforce the
/// requested policy.
#[derive(Debug, Clone)]
pub struct LocalCommandExecutor<W> {
    workspace: W,
    policy_capability_mode: PolicyCapabilityMode,
}

impl<W> LocalCommandExecutor<W> {
    /// Constructs a local command executor for a workspace.
    pub fn new(workspace: W) -> Self {
        Self {
            workspace,
            policy_capability_mode: PolicyCapabilityMode::RequireEnforced,
        }
    }

    /// Allows policy requirements that cannot be technically enforced locally to
    /// be reported as advisory instead of failing closed.
    ///
    /// Passing `true` does not enable sandboxing, network isolation, process-tree
    /// cleanup, or resource limits. It only permits execution to continue while
    /// recording the downgrade in the returned policy enforcement report.
    pub fn with_advisory_policy(mut self, allow: bool) -> Self {
        self.policy_capability_mode = if allow {
            PolicyCapabilityMode::AllowAdvisory
        } else {
            PolicyCapabilityMode::RequireEnforced
        };
        self
    }

    /// Sets how this executor handles policy/capability mismatches.
    ///
    /// [`PolicyCapabilityMode::RequireEnforced`] is the conservative default.
    /// [`PolicyCapabilityMode::AllowAdvisory`] allows local execution to proceed
    /// without technical enforcement and requires reports to say so explicitly.
    pub fn with_policy_capability_mode(mut self, mode: PolicyCapabilityMode) -> Self {
        self.policy_capability_mode = mode;
        self
    }
}

#[async_trait]
impl<W> CommandExecutor for LocalCommandExecutor<W>
where
    W: Workspace,
{
    fn capabilities(&self) -> CommandExecutorCapabilities {
        CommandExecutorCapabilities::local_process()
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
            .unwrap_or_else(|| policy.sandbox().clone());
        let network = request
            .requested_network
            .clone()
            .unwrap_or_else(|| policy.network().clone());
        let capabilities = self.capabilities();
        validate_command_capabilities(
            &capabilities,
            &request,
            policy,
            self.policy_capability_mode,
        )?;
        let sandbox_status = requested_sandbox_status(&sandbox, capabilities.sandbox);
        let network_status = requested_network_status(&network, capabilities.network);
        let mut warnings = advisory_warnings(sandbox_status, network_status);

        let cwd = self
            .workspace
            .resolve(&request.cwd, super::workspace::WorkspaceAccess::List)
            .await
            .map_err(|error| CommandError::InvalidRequest {
                message: format!("invalid command cwd: {error}"),
            })?;

        let timeout = choose_timeout(request.timeout, policy.timeout(), context.remaining())
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
                warnings.push("process tree cleanup is advisory for LocalCommandExecutor".to_string());
                return Ok(terminal_output(
                    CommandStatus::Cancelled,
                    started.elapsed(),
                    TerminalPolicyReportInput {
                        executor: capabilities.identity.clone(),
                        sandbox,
                        network,
                        sandbox_status,
                        network_status,
                        process_cleanup_status: PolicyEnforcementStatus::Advisory,
                        warnings,
                    },
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
                        warnings.push("process tree cleanup is advisory for LocalCommandExecutor".to_string());
                        return Ok(terminal_output(
                            CommandStatus::TimedOut,
                            started.elapsed(),
                            TerminalPolicyReportInput {
                                executor: capabilities.identity.clone(),
                                sandbox,
                                network,
                                sandbox_status,
                                network_status,
                                process_cleanup_status: PolicyEnforcementStatus::Advisory,
                                warnings,
                            },
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
                executor: capabilities.identity,
                sandbox,
                network,
                sandbox_enforced: sandbox_status.is_enforced(),
                network_enforced: network_status.is_enforced(),
                sandbox_status,
                network_status,
                process_cleanup_status: PolicyEnforcementStatus::NotRequested,
                resource_limit_status: PolicyEnforcementStatus::Unsupported,
                warnings,
                metadata: RunMetadata::new(),
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

fn requested_sandbox(request: &CommandRequest, policy: &ExecutionPolicy) -> SandboxPolicy {
    request
        .requested_sandbox
        .clone()
        .unwrap_or_else(|| policy.sandbox().clone())
}

fn requested_network(request: &CommandRequest, policy: &ExecutionPolicy) -> NetworkPolicy {
    request
        .requested_network
        .clone()
        .unwrap_or_else(|| policy.network().clone())
}

fn sandbox_requires_enforcement(sandbox: &SandboxPolicy) -> bool {
    matches!(
        sandbox,
        SandboxPolicy::ReadOnly | SandboxPolicy::WorkspaceWrite | SandboxPolicy::Custom(_)
    )
}

fn network_requires_enforcement(network: &NetworkPolicy) -> bool {
    matches!(
        network,
        NetworkPolicy::Deny | NetworkPolicy::AllowListed(_) | NetworkPolicy::Custom(_)
    )
}

fn requested_sandbox_status(
    sandbox: &SandboxPolicy,
    capability: PolicyEnforcementStatus,
) -> PolicyEnforcementStatus {
    if sandbox_requires_enforcement(sandbox) {
        capability
    } else {
        PolicyEnforcementStatus::NotRequested
    }
}

fn requested_network_status(
    network: &NetworkPolicy,
    capability: PolicyEnforcementStatus,
) -> PolicyEnforcementStatus {
    if network_requires_enforcement(network) {
        capability
    } else {
        PolicyEnforcementStatus::NotRequested
    }
}

fn validate_required_status(
    dimension: &str,
    requires_enforcement: bool,
    status: PolicyEnforcementStatus,
    mode: PolicyCapabilityMode,
) -> Result<(), CommandError> {
    if !requires_enforcement {
        return Ok(());
    }
    match status {
        PolicyEnforcementStatus::Enforced => Ok(()),
        PolicyEnforcementStatus::Advisory if mode.allows_advisory() => Ok(()),
        PolicyEnforcementStatus::Advisory => Err(CommandError::UnsupportedPolicy {
            message: format!(
                "{dimension} policy requires technical enforcement; executor only supports advisory mode"
            ),
        }),
        PolicyEnforcementStatus::Unsupported => Err(CommandError::UnsupportedPolicy {
            message: format!("{dimension} policy is unsupported by executor"),
        }),
        PolicyEnforcementStatus::NotRequested | PolicyEnforcementStatus::Unknown => {
            Err(CommandError::UnsupportedPolicy {
                message: format!("{dimension} policy enforcement status is {status:?}"),
            })
        }
    }
}

fn validate_report_status(
    dimension: &str,
    requires_enforcement: bool,
    capability: PolicyEnforcementStatus,
    reported: PolicyEnforcementStatus,
    mode: PolicyCapabilityMode,
) -> Result<(), CommandError> {
    if reported == PolicyEnforcementStatus::Enforced
        && capability != PolicyEnforcementStatus::Enforced
    {
        return Err(CommandError::UnsupportedPolicy {
            message: format!(
                "{dimension} report claims enforced but capabilities report {capability:?}"
            ),
        });
    }
    if capability == PolicyEnforcementStatus::Enforced
        && requires_enforcement
        && reported != PolicyEnforcementStatus::Enforced
    {
        return Err(CommandError::UnsupportedPolicy {
            message: format!(
                "{dimension} capabilities require enforced report but executor returned {reported:?}"
            ),
        });
    }
    validate_required_status(dimension, requires_enforcement, reported, mode)
}

fn advisory_warnings(
    sandbox_status: PolicyEnforcementStatus,
    network_status: PolicyEnforcementStatus,
) -> Vec<String> {
    let mut warnings = Vec::new();
    if sandbox_status.is_advisory() {
        warnings.push("sandbox policy is advisory; no OS sandbox was applied".to_string());
    }
    if network_status.is_advisory() {
        warnings.push("network policy is advisory; no network isolation was applied".to_string());
    }
    warnings
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

struct TerminalPolicyReportInput {
    executor: CommandExecutorIdentity,
    sandbox: SandboxPolicy,
    network: NetworkPolicy,
    sandbox_status: PolicyEnforcementStatus,
    network_status: PolicyEnforcementStatus,
    process_cleanup_status: PolicyEnforcementStatus,
    warnings: Vec<String>,
}

fn terminal_output(
    status: CommandStatus,
    duration: Duration,
    policy_report: TerminalPolicyReportInput,
    request: &CommandRequest,
) -> CommandOutput {
    CommandOutput {
        status,
        stdout: truncate_bytes(&[], request.output_limit.stdout_bytes),
        stderr: truncate_bytes(&[], request.output_limit.stderr_bytes),
        duration,
        truncated: false,
        policy_enforcement: PolicyEnforcementReport {
            executor: policy_report.executor,
            sandbox: policy_report.sandbox,
            network: policy_report.network,
            sandbox_enforced: policy_report.sandbox_status.is_enforced(),
            network_enforced: policy_report.network_status.is_enforced(),
            sandbox_status: policy_report.sandbox_status,
            network_status: policy_report.network_status,
            process_cleanup_status: policy_report.process_cleanup_status,
            resource_limit_status: PolicyEnforcementStatus::Unsupported,
            warnings: policy_report.warnings,
            metadata: RunMetadata::new(),
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
        let executor = LocalCommandExecutor::new(workspace).with_advisory_policy(true);
        let output = executor
            .execute(
                CommandRequest::new(["printf", "%s", "a;b"]),
                &ExecutionPolicy::new(SandboxPolicy::ReadOnly, NetworkPolicy::Deny)
                    .with_timeout(Some(Duration::from_secs(5))),
                &RunContext::new("cmd"),
            )
            .await
            .unwrap();
        assert_eq!(output.stdout.text, "a;b");
        assert_eq!(
            output.policy_enforcement.network_status,
            PolicyEnforcementStatus::Advisory
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn local_command_fails_closed_without_advisory_policy() {
        let root = temp_dir("fail-closed");
        let workspace = LocalWorkspace::new(&root).unwrap();
        let executor = LocalCommandExecutor::new(workspace);
        let error = executor
            .execute(
                CommandRequest::new(["printf", "ok"]),
                &ExecutionPolicy::new(SandboxPolicy::ReadOnly, NetworkPolicy::Deny)
                    .with_timeout(Some(Duration::from_secs(5))),
                &RunContext::new("cmd"),
            )
            .await
            .unwrap_err();
        assert!(matches!(error, CommandError::UnsupportedPolicy { .. }));
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn local_command_truncates_stdout_and_stderr_separately() {
        let root = temp_dir("truncate");
        let workspace = LocalWorkspace::new(&root).unwrap();
        let executor = LocalCommandExecutor::new(workspace).with_advisory_policy(true);
        let mut request = CommandRequest::new(["sh", "-c", "printf 12345; printf abcde >&2"]);
        request.output_limit = CommandOutputLimit {
            stdout_bytes: 3,
            stderr_bytes: 2,
        };
        let output = executor
            .execute(
                request,
                &ExecutionPolicy::new(SandboxPolicy::ReadOnly, NetworkPolicy::Deny)
                    .with_timeout(Some(Duration::from_secs(5))),
                &RunContext::new("cmd"),
            )
            .await
            .unwrap();
        assert_eq!(output.stdout.text, "123");
        assert_eq!(output.stderr.text, "ab");
        assert!(output.truncated);
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn local_command_truncation_handles_utf8_boundary() {
        let root = temp_dir("truncate-utf8");
        let workspace = LocalWorkspace::new(&root).unwrap();
        let executor = LocalCommandExecutor::new(workspace).with_advisory_policy(true);
        let mut request = CommandRequest::new(["printf", "éé"]);
        request.output_limit = CommandOutputLimit {
            stdout_bytes: 3,
            stderr_bytes: 3,
        };
        let output = executor
            .execute(
                request,
                &ExecutionPolicy::new(SandboxPolicy::ReadOnly, NetworkPolicy::Deny)
                    .with_timeout(Some(Duration::from_secs(5))),
                &RunContext::new("cmd"),
            )
            .await
            .unwrap();

        assert_eq!(output.stdout.bytes, 4);
        assert!(output.stdout.truncated);
        assert!(!output.stdout.text.is_empty());
        let _ = std::fs::remove_dir_all(root);
    }
}
