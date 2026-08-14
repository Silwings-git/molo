use crate::harness::{EffectExecutor, ExecutionError, ExecutionPolicy, RawEffectOutput};
use crate::{EffectKind, EffectRequest, RunContext};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::command::CommandExecutor;
use super::error::CodingError;
use super::git::{GitInspector, GitOperation};
use super::payload::{
    ApplyPatchPayload, CommandPayload, GitPayload, ListFilesPayload, ReadFilePayload,
    SearchPayload, WriteFilePayload,
};
use super::search::{RepoSearchRequest, RepoSearcher, SearchMode};
use super::workspace::{FileBody, FileReadOptions, PatchRequest, Workspace};

/// Configuration for [`CodingEffectExecutor`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct CodingExecutorConfig {
    /// Whether write and patch effects may execute under
    /// `SandboxPolicy::ReadOnly`. The default is false.
    pub(crate) allow_write_in_read_only_policy: bool,
    /// Default read max bytes when payload omits it.
    pub(crate) default_read_max_bytes: usize,
    /// Default search matches when payload omits it.
    pub(crate) default_search_max_matches: usize,
}

impl Default for CodingExecutorConfig {
    fn default() -> Self {
        Self {
            allow_write_in_read_only_policy: false,
            default_read_max_bytes: 64 * 1024,
            default_search_max_matches: 100,
        }
    }
}

impl CodingExecutorConfig {
    /// Constructs a config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether write and patch effects may execute under read-only sandbox policy.
    pub fn allow_write_in_read_only_policy(&self) -> bool {
        self.allow_write_in_read_only_policy
    }

    /// Returns a config with updated read-only write behavior.
    pub fn with_allow_write_in_read_only_policy(
        mut self,
        allow_write_in_read_only_policy: bool,
    ) -> Self {
        self.allow_write_in_read_only_policy = allow_write_in_read_only_policy;
        self
    }

    /// Default read max bytes when payload omits it.
    pub fn default_read_max_bytes(&self) -> usize {
        self.default_read_max_bytes
    }

    /// Returns a config with an updated default read byte cap.
    pub fn with_default_read_max_bytes(mut self, default_read_max_bytes: usize) -> Self {
        self.default_read_max_bytes = default_read_max_bytes;
        self
    }

    /// Default search matches when payload omits it.
    pub fn default_search_max_matches(&self) -> usize {
        self.default_search_max_matches
    }

    /// Returns a config with an updated default search match cap.
    pub fn with_default_search_max_matches(mut self, default_search_max_matches: usize) -> Self {
        self.default_search_max_matches = default_search_max_matches;
        self
    }
}

/// Effect executor that routes typed coding payloads to coding primitives.
#[derive(Debug, Clone)]
pub struct CodingEffectExecutor<W, C, G, S> {
    workspace: W,
    commands: C,
    git: G,
    searcher: S,
    config: CodingExecutorConfig,
}

impl<W, C, G, S> CodingEffectExecutor<W, C, G, S> {
    /// Constructs a coding effect executor.
    pub fn new(workspace: W, commands: C, git: G, searcher: S) -> Self {
        Self {
            workspace,
            commands,
            git,
            searcher,
            config: CodingExecutorConfig::default(),
        }
    }

    /// Replaces executor configuration.
    pub fn with_config(mut self, config: CodingExecutorConfig) -> Self {
        self.config = config;
        self
    }
}

#[async_trait]
impl<W, C, G, S> EffectExecutor for CodingEffectExecutor<W, C, G, S>
where
    W: Workspace,
    C: CommandExecutor,
    G: GitInspector,
    S: RepoSearcher,
{
    async fn execute(
        &self,
        request: &EffectRequest,
        policy: &ExecutionPolicy,
        context: &RunContext,
    ) -> Result<RawEffectOutput, ExecutionError> {
        match &request.kind {
            EffectKind::ReadFile => self.execute_read(request).await,
            EffectKind::WriteFile => self.execute_write(request, policy).await,
            EffectKind::ApplyPatch => self.execute_patch(request, policy).await,
            EffectKind::Search => self.execute_search(request, context).await,
            EffectKind::ExecuteCommand => self.execute_command(request, policy, context).await,
            EffectKind::Git => self.execute_git(request, context).await,
            other => Err(ExecutionError::Unsupported(format!(
                "coding executor does not support effect kind {other:?}"
            ))),
        }
    }
}

impl<W, C, G, S> CodingEffectExecutor<W, C, G, S>
where
    W: Workspace,
    C: CommandExecutor,
    G: GitInspector,
    S: RepoSearcher,
{
    async fn execute_read(
        &self,
        request: &EffectRequest,
    ) -> Result<RawEffectOutput, ExecutionError> {
        let payload = ReadFilePayload::from_effect(request)?;
        let content = self
            .workspace
            .read_file(
                &payload.path,
                FileReadOptions {
                    max_bytes: Some(
                        payload
                            .max_bytes
                            .unwrap_or(self.config.default_read_max_bytes),
                    ),
                    include_binary: false,
                },
            )
            .await
            .map_err(CodingError::from)?;
        let observation = match &content.body {
            FileBody::Text { text, .. } => {
                format!(
                    "read {} ({} bytes, truncated={}):\n{}",
                    content.path.display(),
                    content.version.len,
                    content.truncated,
                    text
                )
            }
            FileBody::Binary { .. } => format!(
                "read {} as binary metadata ({} bytes, truncated={})",
                content.path.display(),
                content.version.len,
                content.truncated
            ),
        };
        raw_json(observation, &content)
    }

    async fn execute_write(
        &self,
        request: &EffectRequest,
        policy: &ExecutionPolicy,
    ) -> Result<RawEffectOutput, ExecutionError> {
        require_write_policy(policy, self.config.allow_write_in_read_only_policy)?;
        let payload = WriteFilePayload::from_effect(request)?;
        let result = self
            .workspace
            .write_file(payload.into_request())
            .await
            .map_err(CodingError::from)?;
        raw_json(
            format!(
                "wrote {} ({} bytes, created={})",
                result.path.display(),
                result.bytes_written,
                result.created
            ),
            &result,
        )
    }

    async fn execute_patch(
        &self,
        request: &EffectRequest,
        policy: &ExecutionPolicy,
    ) -> Result<RawEffectOutput, ExecutionError> {
        require_write_policy(policy, self.config.allow_write_in_read_only_policy)?;
        let mut payload = ApplyPatchPayload::from_effect(request)?;
        for expected in payload.expected_versions {
            for file in &mut payload.patch.files {
                if file.path == expected.path && file.expected_version.is_none() {
                    file.expected_version = Some(expected.clone());
                }
            }
        }
        let result = self
            .workspace
            .apply_patch(PatchRequest {
                patch: payload.patch,
                dry_run: payload.dry_run,
                allow_partial: false,
            })
            .await
            .map_err(CodingError::from)?;
        raw_json(
            format!(
                "patch applied={} changed={} conflicts={}",
                result.applied,
                result.changed_files.len(),
                result.conflicts.len()
            ),
            &result,
        )
    }

    async fn execute_search(
        &self,
        request: &EffectRequest,
        context: &RunContext,
    ) -> Result<RawEffectOutput, ExecutionError> {
        if let Ok(payload) = ListFilesPayload::from_effect(request) {
            let entries = self
                .workspace
                .list_files(payload.into_query())
                .await
                .map_err(CodingError::from)?;
            return raw_json(format!("listed {} entrie(s)", entries.len()), &entries);
        }
        let payload = SearchPayload::from_effect(request)?;
        let results = self
            .searcher
            .search(
                RepoSearchRequest {
                    query: payload.query,
                    paths: payload.paths,
                    mode: SearchMode::Literal,
                    max_matches: payload
                        .max_matches
                        .unwrap_or(self.config.default_search_max_matches),
                    context_lines: payload.context_lines,
                    include_hidden: false,
                    respect_gitignore: true,
                },
                context,
            )
            .await
            .map_err(CodingError::from)?;
        raw_json(
            format!(
                "search returned {} match(es), truncated={}",
                results.matches.len(),
                results.truncated
            ),
            &results,
        )
    }

    async fn execute_command(
        &self,
        request: &EffectRequest,
        policy: &ExecutionPolicy,
        context: &RunContext,
    ) -> Result<RawEffectOutput, ExecutionError> {
        let payload = CommandPayload::from_effect(request)?;
        let output = self
            .commands
            .execute(payload.request, policy, context)
            .await
            .map_err(CodingError::from)?;
        raw_json(
            format!(
                "command status={:?}, stdout_bytes={}, stderr_bytes={}, truncated={}",
                output.status, output.stdout.bytes, output.stderr.bytes, output.truncated
            ),
            &output,
        )
    }

    async fn execute_git(
        &self,
        request: &EffectRequest,
        context: &RunContext,
    ) -> Result<RawEffectOutput, ExecutionError> {
        let payload = GitPayload::from_effect(request)?;
        match payload.operation {
            GitOperation::Status(request) => {
                let status = self
                    .git
                    .status(request, context)
                    .await
                    .map_err(CodingError::from)?;
                raw_json(
                    format!("git status: {} changed file(s)", status.changed_files.len()),
                    &status,
                )
            }
            GitOperation::Diff(request) => {
                let diff = self
                    .git
                    .diff(request, context)
                    .await
                    .map_err(CodingError::from)?;
                raw_json(
                    format!(
                        "git diff: {} changed path hint(s), truncated={}",
                        diff.changed_files.len(),
                        diff.truncated
                    ),
                    &diff,
                )
            }
            GitOperation::ChangedFiles(request) => {
                let files = self
                    .git
                    .changed_files(request, context)
                    .await
                    .map_err(CodingError::from)?;
                raw_json(format!("git changed files: {}", files.len()), &files)
            }
            GitOperation::Head => {
                let head = self.git.head(context).await.map_err(CodingError::from)?;
                raw_json("git head", &head)
            }
        }
    }
}

fn require_write_policy(
    policy: &ExecutionPolicy,
    allow_read_only: bool,
) -> Result<(), ExecutionError> {
    if allow_read_only {
        return Ok(());
    }
    match policy.sandbox {
        crate::harness::SandboxPolicy::WorkspaceWrite
        | crate::harness::SandboxPolicy::FullAccess => Ok(()),
        _ => Err(ExecutionError::Unsupported(
            "write and patch effects require workspace-write sandbox policy".to_string(),
        )),
    }
}

fn raw_json<T>(summary: impl Into<String>, value: &T) -> Result<RawEffectOutput, ExecutionError>
where
    T: Serialize,
{
    let json = serde_json::to_string_pretty(value).map_err(|error| {
        ExecutionError::Failed(format!("failed to encode coding output: {error}"))
    })?;
    Ok(RawEffectOutput::text(format!("{}\n{}", summary.into(), json)).with_debug(json))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::{
        LocalCommandExecutor, LocalWorkspace, ReadFilePayload, WorkspacePath, WorkspaceSearcher,
        WriteFilePayload,
    };
    use crate::harness::{NetworkPolicy, OutputLimit, SandboxPolicy};
    use std::time::Duration;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("molo-coding-executor-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn policy(sandbox: SandboxPolicy) -> ExecutionPolicy {
        ExecutionPolicy {
            sandbox,
            network: NetworkPolicy::Deny,
            timeout: Some(Duration::from_secs(5)),
            output_limit: OutputLimit::default(),
        }
    }

    #[tokio::test]
    async fn read_file_effect_executes_through_workspace() {
        let root = temp_dir("read");
        std::fs::write(root.join("a.txt"), "alpha").unwrap();
        let workspace = LocalWorkspace::new(&root).unwrap();
        let commands = LocalCommandExecutor::new(workspace.clone());
        let git = crate::coding::CliGitInspector::new(commands.clone());
        let searcher = WorkspaceSearcher::new(workspace.clone());
        let executor = CodingEffectExecutor::new(workspace, commands, git, searcher);
        let effect = ReadFilePayload {
            path: WorkspacePath::parse("a.txt").unwrap(),
            max_bytes: Some(64),
        }
        .into_effect()
        .unwrap();

        let output = executor
            .execute(
                &effect,
                &policy(SandboxPolicy::ReadOnly),
                &crate::RunContext::new("coding-exec"),
            )
            .await
            .unwrap();

        assert!(output.observation_for_model.contains("alpha"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn write_file_requires_workspace_write_policy() {
        let root = temp_dir("write-policy");
        let workspace = LocalWorkspace::new(&root).unwrap();
        let commands = LocalCommandExecutor::new(workspace.clone());
        let git = crate::coding::CliGitInspector::new(commands.clone());
        let searcher = WorkspaceSearcher::new(workspace.clone());
        let executor = CodingEffectExecutor::new(workspace, commands, git, searcher);
        let effect = WriteFilePayload {
            path: WorkspacePath::parse("a.txt").unwrap(),
            content: crate::coding::FileWriteContent::Text("alpha".to_string()),
            expected_version: None,
            create: true,
            overwrite: false,
        }
        .into_effect()
        .unwrap();

        let error = executor
            .execute(
                &effect,
                &policy(SandboxPolicy::ReadOnly),
                &crate::RunContext::new("coding-exec"),
            )
            .await
            .unwrap_err();

        assert!(matches!(error, ExecutionError::Unsupported(_)));
        assert!(!root.join("a.txt").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn config_default_is_conservative() {
        let config = CodingExecutorConfig::default();
        assert!(!config.allow_write_in_read_only_policy);
        assert!(config.default_read_max_bytes > 0);
        assert!(config.default_search_max_matches > 0);
    }
}
