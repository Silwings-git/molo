use crate::{EffectKind, EffectRequest, RiskLevel, RunMetadata};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use super::command::CommandRequest;
use super::error::CodingError;
use super::git::GitOperation;
use super::workspace::{
    FileVersion, FileWriteContent, ListFilesQuery, Patch, WorkspacePath, WriteFileRequest,
};

/// Typed payload for reading a workspace file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadFilePayload {
    /// Root-relative file path.
    pub path: WorkspacePath,
    /// Maximum bytes returned by the read.
    pub max_bytes: Option<usize>,
}

impl ReadFilePayload {
    /// Converts this payload into an [`EffectRequest`].
    ///
    /// # Errors
    ///
    /// Returns [`CodingError::InvalidPayload`] if JSON serialization fails.
    pub fn into_effect(self) -> Result<EffectRequest, CodingError> {
        payload_into_effect(
            EffectKind::ReadFile,
            format!("Read file {}", self.path.display()),
            RiskLevel::Low,
            self,
        )
    }

    /// Decodes this payload from a matching [`EffectRequest`].
    ///
    /// # Errors
    ///
    /// Returns [`CodingError`] if the kind or payload is invalid.
    pub fn from_effect(request: &EffectRequest) -> Result<Self, CodingError> {
        payload_from_effect(request, &EffectKind::ReadFile)
    }
}

/// Typed payload for listing workspace files.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListFilesPayload {
    /// Directory or file path to list.
    pub path: WorkspacePath,
    /// Whether listing should recurse.
    pub recursive: bool,
    /// Maximum entries returned.
    pub max_entries: Option<usize>,
    /// Whether hidden files are included.
    pub include_hidden: bool,
    /// Whether simple `.gitignore` rules are respected.
    pub respect_gitignore: bool,
}

impl ListFilesPayload {
    /// Converts this payload into a search-kind [`EffectRequest`].
    ///
    /// Listing uses `EffectKind::Search` because Phase 2 does not define a
    /// separate list-files effect kind.
    ///
    /// # Errors
    ///
    /// Returns [`CodingError::InvalidPayload`] if JSON serialization fails.
    pub fn into_effect(self) -> Result<EffectRequest, CodingError> {
        payload_into_effect(
            EffectKind::Search,
            format!("List files under {}", self.path.display()),
            RiskLevel::Low,
            self,
        )
    }

    /// Decodes this payload from a matching [`EffectRequest`].
    ///
    /// # Errors
    ///
    /// Returns [`CodingError`] if the kind or payload is invalid.
    pub fn from_effect(request: &EffectRequest) -> Result<Self, CodingError> {
        payload_from_effect(request, &EffectKind::Search)
    }

    pub(crate) fn into_query(self) -> ListFilesQuery {
        ListFilesQuery {
            path: self.path,
            recursive: self.recursive,
            max_entries: self.max_entries,
            include_hidden: self.include_hidden,
            respect_gitignore: self.respect_gitignore,
        }
    }
}

/// Typed payload for writing a workspace file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteFilePayload {
    /// Root-relative file path.
    pub path: WorkspacePath,
    /// Content to write.
    pub content: FileWriteContent,
    /// Expected existing version.
    pub expected_version: Option<FileVersion>,
    /// Whether a missing file may be created.
    pub create: bool,
    /// Whether an existing file may be overwritten.
    pub overwrite: bool,
}

impl WriteFilePayload {
    /// Converts this payload into an [`EffectRequest`].
    ///
    /// # Errors
    ///
    /// Returns [`CodingError::InvalidPayload`] if JSON serialization fails.
    pub fn into_effect(self) -> Result<EffectRequest, CodingError> {
        payload_into_effect(
            EffectKind::WriteFile,
            format!("Write file {}", self.path.display()),
            RiskLevel::Medium,
            self,
        )
    }

    /// Decodes this payload from a matching [`EffectRequest`].
    ///
    /// # Errors
    ///
    /// Returns [`CodingError`] if the kind or payload is invalid.
    pub fn from_effect(request: &EffectRequest) -> Result<Self, CodingError> {
        payload_from_effect(request, &EffectKind::WriteFile)
    }

    pub(crate) fn into_request(self) -> WriteFileRequest {
        WriteFileRequest {
            path: self.path,
            content: self.content,
            expected_version: self.expected_version,
            create: self.create,
            overwrite: self.overwrite,
        }
    }
}

/// Typed payload for applying a structured patch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplyPatchPayload {
    /// Patch to apply.
    pub patch: Patch,
    /// Expected versions for files referenced by the patch.
    pub expected_versions: Vec<FileVersion>,
    /// Validate without writing.
    pub dry_run: bool,
}

impl ApplyPatchPayload {
    /// Converts this payload into an [`EffectRequest`].
    ///
    /// # Errors
    ///
    /// Returns [`CodingError::InvalidPayload`] if JSON serialization fails.
    pub fn into_effect(self) -> Result<EffectRequest, CodingError> {
        payload_into_effect(
            EffectKind::ApplyPatch,
            format!("Apply patch to {} file(s)", self.patch.files.len()),
            RiskLevel::Medium,
            self,
        )
    }

    /// Decodes this payload from a matching [`EffectRequest`].
    ///
    /// # Errors
    ///
    /// Returns [`CodingError`] if the kind or payload is invalid.
    pub fn from_effect(request: &EffectRequest) -> Result<Self, CodingError> {
        payload_from_effect(request, &EffectKind::ApplyPatch)
    }
}

/// Typed payload for repository search.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchPayload {
    /// Query string or regex pattern.
    pub query: String,
    /// Paths to search. Empty means the workspace root.
    pub paths: Vec<WorkspacePath>,
    /// Maximum matches returned.
    pub max_matches: Option<usize>,
    /// Context lines around matches.
    pub context_lines: usize,
}

impl SearchPayload {
    /// Converts this payload into an [`EffectRequest`].
    ///
    /// # Errors
    ///
    /// Returns [`CodingError::InvalidPayload`] if JSON serialization fails.
    pub fn into_effect(self) -> Result<EffectRequest, CodingError> {
        payload_into_effect(
            EffectKind::Search,
            format!("Search repository for {}", self.query),
            RiskLevel::Low,
            self,
        )
    }

    /// Decodes this payload from a matching [`EffectRequest`].
    ///
    /// # Errors
    ///
    /// Returns [`CodingError`] if the kind or payload is invalid.
    pub fn from_effect(request: &EffectRequest) -> Result<Self, CodingError> {
        payload_from_effect(request, &EffectKind::Search)
    }
}

/// Typed payload for command execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandPayload {
    /// Command request.
    pub request: CommandRequest,
}

impl CommandPayload {
    /// Converts this payload into an [`EffectRequest`].
    ///
    /// # Errors
    ///
    /// Returns [`CodingError::InvalidPayload`] if JSON serialization fails.
    pub fn into_effect(self) -> Result<EffectRequest, CodingError> {
        let description = format!("Run command {}", self.request.argv.join(" "));
        payload_into_effect(
            EffectKind::ExecuteCommand,
            description,
            command_risk(&self.request.argv),
            self,
        )
    }

    /// Decodes this payload from a matching [`EffectRequest`].
    ///
    /// # Errors
    ///
    /// Returns [`CodingError`] if the kind or payload is invalid.
    pub fn from_effect(request: &EffectRequest) -> Result<Self, CodingError> {
        payload_from_effect(request, &EffectKind::ExecuteCommand)
    }
}

/// Typed payload for read-only git inspection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GitPayload {
    /// Git operation.
    pub operation: GitOperation,
}

impl GitPayload {
    /// Converts this payload into an [`EffectRequest`].
    ///
    /// # Errors
    ///
    /// Returns [`CodingError::InvalidPayload`] if JSON serialization fails.
    pub fn into_effect(self) -> Result<EffectRequest, CodingError> {
        payload_into_effect(EffectKind::Git, "Inspect git state", RiskLevel::Low, self)
    }

    /// Decodes this payload from a matching [`EffectRequest`].
    ///
    /// # Errors
    ///
    /// Returns [`CodingError`] if the kind or payload is invalid.
    pub fn from_effect(request: &EffectRequest) -> Result<Self, CodingError> {
        payload_from_effect(request, &EffectKind::Git)
    }
}

fn payload_into_effect<T>(
    kind: EffectKind,
    description: impl Into<String>,
    risk: RiskLevel,
    payload: T,
) -> Result<EffectRequest, CodingError>
where
    T: Serialize,
{
    let payload = serde_json::to_value(payload).map_err(|error| CodingError::InvalidPayload {
        message: format!("failed to encode payload: {error}"),
    })?;
    Ok(EffectRequest::new(kind, description, payload)
        .with_risk(risk)
        .with_metadata(RunMetadata::new()))
}

fn payload_from_effect<T>(request: &EffectRequest, expected: &EffectKind) -> Result<T, CodingError>
where
    T: DeserializeOwned,
{
    if !effect_kind_eq(&request.kind, expected) {
        return Err(CodingError::unexpected_kind(format!(
            "expected {:?}, got {:?}",
            expected, request.kind
        )));
    }
    serde_json::from_value(request.payload.clone()).map_err(|error| CodingError::InvalidPayload {
        message: format!("failed to decode payload: {error}"),
    })
}

fn effect_kind_eq(left: &EffectKind, right: &EffectKind) -> bool {
    match (left, right) {
        (EffectKind::ReadFile, EffectKind::ReadFile)
        | (EffectKind::WriteFile, EffectKind::WriteFile)
        | (EffectKind::ApplyPatch, EffectKind::ApplyPatch)
        | (EffectKind::Search, EffectKind::Search)
        | (EffectKind::ExecuteCommand, EffectKind::ExecuteCommand)
        | (EffectKind::Git, EffectKind::Git)
        | (EffectKind::Network, EffectKind::Network)
        | (EffectKind::Browser, EffectKind::Browser)
        | (EffectKind::Mcp, EffectKind::Mcp) => true,
        (EffectKind::Custom(left), EffectKind::Custom(right)) => left == right,
        _ => false,
    }
}

fn command_risk(argv: &[String]) -> RiskLevel {
    if argv.is_empty() {
        return RiskLevel::Medium;
    }
    let lowered = argv.join(" ").to_ascii_lowercase();
    if argv
        .first()
        .is_some_and(|arg| arg == "sh" || arg == "bash" || arg == "zsh")
        || lowered.contains("rm -rf")
        || lowered.contains("git reset --hard")
        || lowered.contains("push --force")
        || lowered.contains("sudo ")
    {
        RiskLevel::High
    } else {
        RiskLevel::Medium
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::WorkspacePath;

    #[test]
    fn read_payload_round_trips_effect() {
        let payload = ReadFilePayload {
            path: WorkspacePath::parse("src/lib.rs").unwrap(),
            max_bytes: Some(128),
        };
        let effect = payload.clone().into_effect().unwrap();
        assert_eq!(effect.kind, EffectKind::ReadFile);
        assert_eq!(ReadFilePayload::from_effect(&effect).unwrap(), payload);
    }

    #[test]
    fn command_payload_marks_shell_high_risk() {
        let payload = CommandPayload {
            request: CommandRequest::new(["sh", "-c", "echo hi"]),
        };
        let effect = payload.into_effect().unwrap();
        assert_eq!(effect.risk, RiskLevel::High);
    }

    #[test]
    fn command_payload_marks_destructive_patterns_high_risk() {
        let cases = [
            vec!["sudo", "whoami"],
            vec!["git", "push", "--force"],
            vec!["bash", "-lc", "echo hi"],
            vec!["sh", "-c", "rm -rf target"],
        ];
        for argv in cases {
            let payload = CommandPayload {
                request: CommandRequest::new(argv),
            };
            let effect = payload.into_effect().unwrap();
            assert_eq!(effect.risk, RiskLevel::High);
        }
    }

    #[test]
    fn command_payload_keeps_plain_commands_medium_risk() {
        let payload = CommandPayload {
            request: CommandRequest::new(["cargo", "test", "--workspace"]),
        };
        let effect = payload.into_effect().unwrap();
        assert_eq!(effect.risk, RiskLevel::Medium);
    }
}
