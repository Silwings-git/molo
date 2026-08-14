use crate::{RunContext, RunMetadata};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::git::{GitChangedFilesRequest, GitInspector, GitStatus};
use super::instructions::{InstructionBundle, InstructionRequest, InstructionResolver};
use super::search::{RepoSearchRequest, RepoSearchResults, RepoSearcher, SearchMode};
use super::test_runner::VerificationResult;
use super::workspace::{ListFilesQuery, Workspace, WorkspaceEntry, WorkspacePath};

/// Context budget for repository context gathering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextBudget {
    /// Approximate maximum bytes across textual context fields.
    pub max_bytes: usize,
    /// Maximum files in tree summaries.
    pub max_files: usize,
    /// Maximum search matches.
    pub max_search_matches: usize,
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self {
            max_bytes: 64 * 1024,
            max_files: 500,
            max_search_matches: 50,
        }
    }
}

/// Flags controlling which context sources are gathered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodingContextInclude {
    /// Include project instructions.
    pub instructions: bool,
    /// Include repository tree summary.
    pub repo_tree: bool,
    /// Include search matches for goal terms.
    pub search: bool,
    /// Include git status.
    pub git_status: bool,
    /// Include changed files.
    pub changed_files: bool,
}

impl Default for CodingContextInclude {
    fn default() -> Self {
        Self {
            instructions: true,
            repo_tree: true,
            search: true,
            git_status: true,
            changed_files: true,
        }
    }
}

/// Coding context request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodingContextRequest {
    /// User or agent goal.
    pub goal: String,
    /// Paths to prioritize.
    pub focus_paths: Vec<WorkspacePath>,
    /// Included context sources.
    pub include: CodingContextInclude,
    /// Context budget.
    pub budget: ContextBudget,
    /// Recent verification failures.
    pub recent_test_failures: Vec<VerificationResult>,
    /// Recent transcript summary.
    pub transcript_summary: Option<String>,
}

impl CodingContextRequest {
    /// Constructs a context request from a goal.
    pub fn new(goal: impl Into<String>) -> Self {
        Self {
            goal: goal.into(),
            focus_paths: Vec::new(),
            include: CodingContextInclude::default(),
            budget: ContextBudget::default(),
            recent_test_failures: Vec::new(),
            transcript_summary: None,
        }
    }
}

/// Context bundle returned by a [`CodingContextProvider`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodingContextBundle {
    /// Resolved project instructions.
    pub instructions: Option<InstructionBundle>,
    /// Repository tree entries.
    pub repo_tree: Vec<WorkspaceEntry>,
    /// Search results.
    pub search_results: Vec<RepoSearchResults>,
    /// Git status.
    pub git_status: Option<GitStatus>,
    /// Changed files.
    pub changed_files: Vec<WorkspacePath>,
    /// Focus and changed paths considered relevant by deterministic
    /// heuristics.
    pub relevant_files: Vec<WorkspacePath>,
    /// Dependency manifest hints discovered in the repository tree.
    pub dependency_metadata: Vec<DependencyMetadata>,
    /// Recent verification failures.
    pub recent_test_failures: Vec<VerificationResult>,
    /// Recent transcript summary.
    pub transcript_summary: Option<String>,
    /// Warnings about truncation, unsupported tools, or missing context.
    pub warnings: Vec<String>,
    /// Approximate bytes used.
    pub bytes_used: usize,
    /// Whether the bundle hit a configured budget.
    pub truncated: bool,
    /// Host-owned metadata.
    pub metadata: RunMetadata,
}

/// Dependency manifest metadata discovered by context gathering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencyMetadata {
    /// Manifest path.
    pub path: WorkspacePath,
    /// Manifest ecosystem or format.
    pub kind: String,
}

/// Provides repository context outside chat memory.
#[async_trait]
pub trait CodingContextProvider: Send + Sync {
    /// Gathers coding context.
    async fn gather(
        &self,
        request: CodingContextRequest,
        context: &RunContext,
    ) -> Result<CodingContextBundle, CodingContextError>;
}

/// Baseline context provider that combines workspace, search, git, and
/// instruction primitives.
#[derive(Debug, Clone)]
pub struct DefaultCodingContextProvider<W, S, G, I> {
    workspace: W,
    searcher: S,
    git: G,
    instructions: I,
}

impl<W, S, G, I> DefaultCodingContextProvider<W, S, G, I> {
    /// Constructs a default context provider.
    pub fn new(workspace: W, searcher: S, git: G, instructions: I) -> Self {
        Self {
            workspace,
            searcher,
            git,
            instructions,
        }
    }
}

#[async_trait]
impl<W, S, G, I> CodingContextProvider for DefaultCodingContextProvider<W, S, G, I>
where
    W: Workspace,
    S: RepoSearcher,
    G: GitInspector,
    I: InstructionResolver,
{
    async fn gather(
        &self,
        request: CodingContextRequest,
        context: &RunContext,
    ) -> Result<CodingContextBundle, CodingContextError> {
        let mut warnings = Vec::new();
        let mut bytes_used = 0usize;
        let mut truncated = false;

        let instructions = if request.include.instructions {
            match self
                .instructions
                .resolve(
                    InstructionRequest {
                        target_paths: request.focus_paths.clone(),
                        max_bytes: request.budget.max_bytes,
                        ..InstructionRequest::default()
                    },
                    context,
                )
                .await
            {
                Ok(bundle) => {
                    bytes_used += bundle
                        .files
                        .iter()
                        .map(|file| file.content.len())
                        .sum::<usize>();
                    truncated |= bundle.truncated;
                    Some(bundle)
                }
                Err(error) => {
                    warnings.push(error.to_string());
                    None
                }
            }
        } else {
            None
        };

        let repo_tree = if request.include.repo_tree {
            let entries = self
                .workspace
                .list_files(ListFilesQuery {
                    path: WorkspacePath::root(),
                    recursive: true,
                    max_entries: Some(request.budget.max_files),
                    include_hidden: false,
                    respect_gitignore: true,
                })
                .await
                .map_err(|error| CodingContextError::Workspace {
                    message: error.to_string(),
                })?;
            if entries.len() >= request.budget.max_files {
                truncated = true;
            }
            bytes_used += entries
                .iter()
                .map(|entry| entry.path.display().len())
                .sum::<usize>();
            entries
        } else {
            Vec::new()
        };

        let search_results = if request.include.search && !request.goal.trim().is_empty() {
            let query = derive_query(&request.goal);
            match self
                .searcher
                .search(
                    RepoSearchRequest {
                        query,
                        paths: request.focus_paths.clone(),
                        mode: SearchMode::Literal,
                        max_matches: request.budget.max_search_matches,
                        context_lines: 1,
                        include_hidden: false,
                        respect_gitignore: true,
                    },
                    context,
                )
                .await
            {
                Ok(results) => {
                    truncated |= results.truncated;
                    bytes_used += results
                        .matches
                        .iter()
                        .map(|mat| mat.line_text.len())
                        .sum::<usize>();
                    vec![results]
                }
                Err(error) => {
                    warnings.push(error.to_string());
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };

        let git_status = if request.include.git_status {
            match self.git.status(Default::default(), context).await {
                Ok(status) => {
                    bytes_used += status.raw.len();
                    truncated |= status.truncated;
                    Some(status)
                }
                Err(error) => {
                    warnings.push(error.to_string());
                    None
                }
            }
        } else {
            None
        };

        let changed_files = if request.include.changed_files {
            match self
                .git
                .changed_files(GitChangedFilesRequest::default(), context)
                .await
            {
                Ok(files) => files.into_iter().map(|file| file.path).collect(),
                Err(error) => {
                    warnings.push(error.to_string());
                    Vec::new()
                }
            }
        } else {
            Vec::new()
        };
        let mut relevant_files = request.focus_paths.clone();
        for path in &changed_files {
            if !relevant_files.contains(path) {
                relevant_files.push(path.clone());
            }
        }
        let dependency_metadata = dependency_metadata_from_tree(&repo_tree);

        if bytes_used > request.budget.max_bytes {
            truncated = true;
            warnings.push("context byte budget exceeded".to_string());
        }

        Ok(CodingContextBundle {
            instructions,
            repo_tree,
            search_results,
            git_status,
            changed_files,
            relevant_files,
            dependency_metadata,
            recent_test_failures: request.recent_test_failures,
            transcript_summary: request.transcript_summary,
            warnings,
            bytes_used,
            truncated,
            metadata: RunMetadata::new(),
        })
    }
}

fn dependency_metadata_from_tree(entries: &[WorkspaceEntry]) -> Vec<DependencyMetadata> {
    entries
        .iter()
        .filter_map(|entry| {
            let display = entry.path.display();
            let file_name = display.rsplit('/').next().unwrap_or(display.as_str());
            let kind = match file_name {
                "Cargo.toml" => "cargo",
                "package.json" => "npm",
                "pnpm-lock.yaml" => "pnpm",
                "yarn.lock" => "yarn",
                "pyproject.toml" => "python",
                "requirements.txt" => "python",
                "go.mod" => "go",
                "pom.xml" => "maven",
                "build.gradle" | "build.gradle.kts" => "gradle",
                _ => return None,
            };
            Some(DependencyMetadata {
                path: entry.path.clone(),
                kind: kind.to_string(),
            })
        })
        .collect()
}

fn derive_query(goal: &str) -> String {
    goal.split_whitespace()
        .find(|word| word.len() >= 4)
        .unwrap_or(goal)
        .trim_matches(|ch: char| !ch.is_alphanumeric() && ch != '_')
        .to_string()
}

/// Coding context errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CodingContextError {
    /// Workspace context failed.
    #[error("context workspace error: {message}")]
    Workspace {
        /// Model-safe explanation.
        message: String,
    },
    /// Context source failed.
    #[error("context source error: {message}")]
    Source {
        /// Model-safe explanation.
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_query_picks_goal_term() {
        assert_eq!(derive_query("fix parser panic"), "parser");
    }
}
