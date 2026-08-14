use crate::args::{ApprovalMode, CliArgs, ProviderKind};
use crate::error::CliError;
use molo::{NetworkPolicy, SandboxPolicy};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Effective CLI configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliConfig {
    /// Workspace root.
    pub workspace_root: PathBuf,
    /// Session store directory.
    pub session_dir: PathBuf,
    /// Provider configuration.
    pub provider: ProviderConfig,
    /// Policy configuration.
    pub policy: PolicyConfig,
    /// Whether the command is non-interactive.
    pub non_interactive: bool,
}

/// Provider configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider kind.
    pub kind: ProviderKind,
    /// Model name.
    pub model: String,
    /// OpenAI-compatible base URL.
    pub base_url: Option<String>,
    /// Environment variable that stores the API key.
    pub api_key_env: String,
}

/// Policy configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyConfig {
    /// Sandbox policy for code mode.
    pub sandbox: SandboxPolicy,
    /// Network policy for command effects.
    pub network: NetworkPolicy,
    /// Approval mode.
    pub approval: ApprovalMode,
    /// Command timeout.
    pub command_timeout: Duration,
}

/// Redacted effective config suitable for terminal display or session files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CliConfigSnapshot {
    /// Workspace root display.
    pub workspace_root: String,
    /// Session directory display.
    pub session_dir: String,
    /// Provider summary.
    pub provider: ProviderConfigSnapshot,
    /// Policy summary.
    pub policy: PolicyConfig,
    /// Whether non-interactive mode is active.
    pub non_interactive: bool,
}

/// Redacted provider summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderConfigSnapshot {
    /// Provider kind.
    pub kind: ProviderKind,
    /// Model name.
    pub model: String,
    /// Base URL, if configured.
    pub base_url: Option<String>,
    /// Environment variable name only. The value is never copied.
    pub api_key_env: String,
}

impl CliConfig {
    /// Constructs the effective config from parsed args and environment.
    pub fn from_args(args: &CliArgs) -> Result<Self, CliError> {
        let workspace_root = match &args.global.workspace {
            Some(root) => root.clone(),
            None => std::env::current_dir()?,
        };
        let session_dir = match &args.global.session_dir {
            Some(path) => path.clone(),
            None => default_session_dir(),
        };
        let model = args
            .global
            .model
            .clone()
            .or_else(|| std::env::var("MOLO_MODEL").ok())
            .unwrap_or_else(|| match args.global.provider {
                ProviderKind::Fake => "fake".to_string(),
                ProviderKind::OpenAi => "gpt-4o-mini".to_string(),
            });
        let base_url = args
            .global
            .base_url
            .clone()
            .or_else(|| std::env::var("MOLO_BASE_URL").ok())
            .or_else(|| {
                (args.global.provider == ProviderKind::OpenAi)
                    .then_some("https://api.openai.com/v1".to_string())
            });

        Ok(Self {
            workspace_root,
            session_dir,
            provider: ProviderConfig {
                kind: args.global.provider,
                model,
                base_url,
                api_key_env: args.global.api_key_env.clone(),
            },
            policy: PolicyConfig {
                sandbox: SandboxPolicy::WorkspaceWrite,
                network: NetworkPolicy::Deny,
                approval: args.global.approval,
                command_timeout: Duration::from_secs(args.global.command_timeout_secs),
            },
            non_interactive: args.global.non_interactive,
        })
    }

    /// Returns a redacted snapshot for persistence or display.
    pub fn snapshot(&self) -> CliConfigSnapshot {
        CliConfigSnapshot {
            workspace_root: self.workspace_root.display().to_string(),
            session_dir: self.session_dir.display().to_string(),
            provider: ProviderConfigSnapshot {
                kind: self.provider.kind,
                model: self.provider.model.clone(),
                base_url: self.provider.base_url.clone(),
                api_key_env: self.provider.api_key_env.clone(),
            },
            policy: self.policy.clone(),
            non_interactive: self.non_interactive,
        }
    }
}

/// Builds a provider from CLI config.
pub fn provider_from_config(config: &CliConfig) -> Result<Box<dyn molo::Provider>, CliError> {
    match config.provider.kind {
        ProviderKind::Fake => Ok(Box::new(molo::FakeProvider::new([molo::FakeReply::Text(
            "fake provider response".to_string(),
        )]))),
        ProviderKind::OpenAi => {
            let base_url =
                config.provider.base_url.clone().ok_or_else(|| {
                    CliError::Config("openai provider requires --base-url".into())
                })?;
            let api_key = std::env::var(&config.provider.api_key_env).map_err(|_| {
                CliError::Config(format!(
                    "environment variable {} is not set",
                    config.provider.api_key_env
                ))
            })?;
            Ok(Box::new(molo::OpenAiProvider::new(
                base_url,
                api_key,
                config.provider.model.clone(),
            )))
        }
    }
}

fn default_session_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("MOLO_SESSION_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".local/share/molo/sessions");
    }
    std::env::temp_dir().join("molo/sessions")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::CliArgs;

    #[test]
    fn snapshot_does_not_read_api_key_value() {
        let args = CliArgs::parse([
            "--provider",
            "openai",
            "--api-key-env",
            "SECRET_ENV",
            "config",
            "check",
        ])
        .unwrap();
        let config = CliConfig::from_args(&args).unwrap();
        let snapshot = config.snapshot();
        assert_eq!(snapshot.provider.api_key_env, "SECRET_ENV");
    }
}
