use crate::{RunContext, RunMetadata};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::workspace::{FileBody, FileReadOptions, Workspace, WorkspacePath};

/// Instruction file candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstructionFileSpec {
    /// File name to look for in each ancestor directory.
    pub file_name: String,
    /// Maximum bytes read from this file.
    pub max_bytes: Option<usize>,
}

impl Default for InstructionFileSpec {
    fn default() -> Self {
        Self {
            file_name: "AGENTS.md".to_string(),
            max_bytes: None,
        }
    }
}

/// Instruction resolution request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstructionRequest {
    /// Target paths used to determine ancestor search paths.
    pub target_paths: Vec<WorkspacePath>,
    /// Candidate instruction files.
    pub file_specs: Vec<InstructionFileSpec>,
    /// Maximum total bytes returned.
    pub max_bytes: usize,
}

impl Default for InstructionRequest {
    fn default() -> Self {
        Self {
            target_paths: vec![WorkspacePath::root()],
            file_specs: vec![InstructionFileSpec::default()],
            max_bytes: 64 * 1024,
        }
    }
}

/// Resolved project instructions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstructionBundle {
    /// Instruction file contents in application order. More-specific files
    /// appear later.
    pub files: Vec<InstructionFile>,
    /// Resolver warnings.
    pub warnings: Vec<String>,
    /// Whether the bundle was truncated by byte budget.
    pub truncated: bool,
    /// Host-owned metadata.
    pub metadata: RunMetadata,
}

/// One resolved instruction file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstructionFile {
    /// Instruction file path.
    pub path: WorkspacePath,
    /// File content.
    pub content: String,
    /// Whether content was truncated.
    pub truncated: bool,
}

/// Resolves project instruction files.
#[async_trait]
pub trait InstructionResolver: Send + Sync {
    /// Resolves project instructions.
    async fn resolve(
        &self,
        request: InstructionRequest,
        context: &RunContext,
    ) -> Result<InstructionBundle, InstructionError>;
}

/// Default resolver that searches for `AGENTS.md`-style files from root to
/// target parent directories.
#[derive(Debug, Clone)]
pub struct DefaultInstructionResolver<W> {
    workspace: W,
}

impl<W> DefaultInstructionResolver<W> {
    /// Constructs a resolver over a workspace.
    pub fn new(workspace: W) -> Self {
        Self { workspace }
    }
}

#[async_trait]
impl<W> InstructionResolver for DefaultInstructionResolver<W>
where
    W: Workspace,
{
    async fn resolve(
        &self,
        mut request: InstructionRequest,
        _context: &RunContext,
    ) -> Result<InstructionBundle, InstructionError> {
        if request.file_specs.is_empty() {
            request.file_specs.push(InstructionFileSpec::default());
        }
        if request.target_paths.is_empty() {
            request.target_paths.push(WorkspacePath::root());
        }

        let mut candidate_dirs = Vec::new();
        for target in &request.target_paths {
            for dir in ancestors(target) {
                if !candidate_dirs.contains(&dir) {
                    candidate_dirs.push(dir);
                }
            }
        }
        candidate_dirs.sort();
        candidate_dirs.dedup();

        let mut files = Vec::new();
        let mut warnings = Vec::new();
        let mut used = 0usize;
        let mut truncated = false;

        for dir in candidate_dirs {
            for spec in &request.file_specs {
                let candidate = if dir.as_path().as_os_str().is_empty() {
                    WorkspacePath::parse(&spec.file_name)
                } else {
                    dir.join(&spec.file_name)
                }
                .map_err(|error| InstructionError::InvalidRequest {
                    message: error.to_string(),
                })?;
                let remaining = request.max_bytes.saturating_sub(used);
                if remaining == 0 {
                    truncated = true;
                    continue;
                }
                let max_bytes = spec.max_bytes.unwrap_or(remaining).min(remaining);
                let content = match self
                    .workspace
                    .read_file(
                        &candidate,
                        FileReadOptions {
                            max_bytes: Some(max_bytes),
                            include_binary: false,
                        },
                    )
                    .await
                {
                    Ok(content) => content,
                    Err(error) => {
                        let text = error.to_string();
                        if !text.contains("not found") {
                            warnings.push(format!("{}: {text}", candidate.display()));
                        }
                        continue;
                    }
                };
                let FileBody::Text { text, .. } = content.body else {
                    warnings.push(format!(
                        "instruction file is binary: {}",
                        candidate.display()
                    ));
                    continue;
                };
                used += text.len();
                truncated |= content.truncated;
                files.push(InstructionFile {
                    path: candidate,
                    content: text,
                    truncated: content.truncated,
                });
            }
        }

        Ok(InstructionBundle {
            files,
            warnings,
            truncated,
            metadata: RunMetadata::new(),
        })
    }
}

fn ancestors(path: &WorkspacePath) -> Vec<WorkspacePath> {
    let display = path.display();
    if display.is_empty() {
        return vec![WorkspacePath::root()];
    }
    let parts: Vec<_> = display.split('/').collect();
    let parent_parts = if display.ends_with('/') {
        parts
    } else {
        parts[..parts.len().saturating_sub(1)].to_vec()
    };
    let mut dirs = vec![WorkspacePath::root()];
    for index in 0..parent_parts.len() {
        let joined = parent_parts[..=index].join("/");
        if let Ok(path) = WorkspacePath::parse(joined) {
            dirs.push(path);
        }
    }
    dirs
}

/// Instruction resolver errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
#[non_exhaustive]
pub enum InstructionError {
    /// Request is invalid.
    #[error("invalid instruction request: {message}")]
    InvalidRequest {
        /// Model-safe explanation.
        message: String,
    },
    /// Workspace read failed.
    #[error("instruction workspace error: {message}")]
    Workspace {
        /// Model-safe explanation.
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coding::LocalWorkspace;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "molo-instruction-test-{}-{tag}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn resolver_applies_hierarchy() {
        let root = temp_dir("hierarchy");
        std::fs::create_dir_all(root.join("src/nested")).unwrap();
        std::fs::write(root.join("AGENTS.md"), "root").unwrap();
        std::fs::write(root.join("src/AGENTS.md"), "src").unwrap();
        let resolver = DefaultInstructionResolver::new(LocalWorkspace::new(&root).unwrap());
        let bundle = resolver
            .resolve(
                InstructionRequest {
                    target_paths: vec![WorkspacePath::parse("src/nested/lib.rs").unwrap()],
                    ..InstructionRequest::default()
                },
                &RunContext::new("instructions"),
            )
            .await
            .unwrap();
        assert_eq!(bundle.files.len(), 2);
        assert_eq!(bundle.files[0].content, "root");
        assert_eq!(bundle.files[1].content, "src");
        let _ = std::fs::remove_dir_all(root);
    }
}
