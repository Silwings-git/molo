use crate::harness::{ExecutionPolicy, NetworkPolicy, OutputLimit, SandboxPolicy};
use crate::{RunContext, RunMetadata};
use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

use super::command::{CommandExecutor, CommandOutputLimit, CommandRequest};
use super::workspace::{FileBody, FileReadOptions, ListFilesQuery, Workspace, WorkspacePath};

/// Repository search mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SearchMode {
    /// Treat the query as a literal substring.
    #[default]
    Literal,
    /// Treat the query as a regular expression.
    Regex,
}

/// Repository search request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepoSearchRequest {
    /// Query string or regex pattern.
    pub query: String,
    /// Paths to search. Empty means workspace root.
    pub paths: Vec<WorkspacePath>,
    /// Search mode.
    pub mode: SearchMode,
    /// Maximum matches returned.
    pub max_matches: usize,
    /// Context lines around each match.
    pub context_lines: usize,
    /// Whether hidden paths are included.
    pub include_hidden: bool,
    /// Whether simple `.gitignore` rules are respected.
    pub respect_gitignore: bool,
}

impl RepoSearchRequest {
    /// Constructs a literal search request.
    pub fn literal(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            paths: Vec::new(),
            mode: SearchMode::Literal,
            max_matches: 100,
            context_lines: 0,
            include_hidden: false,
            respect_gitignore: true,
        }
    }
}

/// One repository search match.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchMatch {
    /// Matched file.
    pub path: WorkspacePath,
    /// 1-based line number.
    pub line: usize,
    /// 1-based column number when known.
    pub column: Option<usize>,
    /// Matched line text.
    pub line_text: String,
    /// Context lines before the match.
    pub before: Vec<String>,
    /// Context lines after the match.
    pub after: Vec<String>,
}

/// Repository search results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepoSearchResults {
    /// Matches in deterministic order.
    pub matches: Vec<SearchMatch>,
    /// Whether matching stopped at the output budget.
    pub truncated: bool,
    /// Warnings such as missing external search binaries.
    pub warnings: Vec<String>,
    /// Host-owned metadata.
    pub metadata: RunMetadata,
}

/// Search implementation for repositories.
#[async_trait]
pub trait RepoSearcher: Send + Sync {
    /// Searches repository content.
    async fn search(
        &self,
        request: RepoSearchRequest,
        context: &RunContext,
    ) -> Result<RepoSearchResults, SearchError>;
}

/// In-process fallback searcher based on [`Workspace`] reads.
#[derive(Debug, Clone)]
pub struct WorkspaceSearcher<W> {
    workspace: W,
    max_file_bytes: usize,
}

impl<W> WorkspaceSearcher<W> {
    /// Constructs a workspace searcher.
    pub fn new(workspace: W) -> Self {
        Self {
            workspace,
            max_file_bytes: 256 * 1024,
        }
    }

    /// Sets the per-file read budget.
    pub fn with_max_file_bytes(mut self, max_file_bytes: usize) -> Self {
        self.max_file_bytes = max_file_bytes;
        self
    }
}

#[async_trait]
impl<W> RepoSearcher for WorkspaceSearcher<W>
where
    W: Workspace,
{
    async fn search(
        &self,
        request: RepoSearchRequest,
        _context: &RunContext,
    ) -> Result<RepoSearchResults, SearchError> {
        if request.query.is_empty() {
            return Err(SearchError::InvalidQuery {
                message: "search query must not be empty".to_string(),
            });
        }
        let matcher = Matcher::new(&request.query, request.mode)?;
        let paths = if request.paths.is_empty() {
            vec![WorkspacePath::root()]
        } else {
            request.paths.clone()
        };
        let mut matches = Vec::new();
        let mut truncated = false;
        for path in paths {
            let entries = self
                .workspace
                .list_files(ListFilesQuery {
                    path,
                    recursive: true,
                    max_entries: Some(20_000),
                    include_hidden: request.include_hidden,
                    respect_gitignore: request.respect_gitignore,
                })
                .await
                .map_err(|error| SearchError::Workspace {
                    message: error.to_string(),
                })?;
            for entry in entries {
                if matches.len() >= request.max_matches {
                    truncated = true;
                    break;
                }
                if entry.kind != super::workspace::ResolvedPathKind::File {
                    continue;
                }
                let content = self
                    .workspace
                    .read_file(
                        &entry.path,
                        FileReadOptions {
                            max_bytes: Some(self.max_file_bytes),
                            include_binary: false,
                        },
                    )
                    .await
                    .map_err(|error| SearchError::Workspace {
                        message: error.to_string(),
                    })?;
                let FileBody::Text { text, .. } = content.body else {
                    continue;
                };
                let file_matches = search_text(&entry.path, &text, &matcher, request.context_lines);
                for mat in file_matches {
                    if matches.len() >= request.max_matches {
                        truncated = true;
                        break;
                    }
                    matches.push(mat);
                }
            }
        }
        Ok(RepoSearchResults {
            matches,
            truncated,
            warnings: Vec::new(),
            metadata: RunMetadata::new(),
        })
    }
}

/// Repository searcher that invokes `rg` through a [`CommandExecutor`].
#[derive(Clone)]
pub struct RipgrepSearcher<C> {
    commands: Arc<C>,
    timeout: Duration,
}

impl<C> std::fmt::Debug for RipgrepSearcher<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RipgrepSearcher")
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

impl<C> RipgrepSearcher<C> {
    /// Constructs a ripgrep searcher from a command executor.
    pub fn new(commands: C) -> Self {
        Self {
            commands: Arc::new(commands),
            timeout: Duration::from_secs(10),
        }
    }

    /// Sets the ripgrep command timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[async_trait]
impl<C> RepoSearcher for RipgrepSearcher<C>
where
    C: CommandExecutor + 'static,
{
    async fn search(
        &self,
        request: RepoSearchRequest,
        context: &RunContext,
    ) -> Result<RepoSearchResults, SearchError> {
        if request.mode == SearchMode::Regex {
            Regex::new(&request.query).map_err(|error| SearchError::InvalidQuery {
                message: format!("invalid regex: {error}"),
            })?;
        }
        let mut argv = vec![
            "rg".to_string(),
            "--line-number".to_string(),
            "--column".to_string(),
            "--no-heading".to_string(),
            "--color".to_string(),
            "never".to_string(),
        ];
        if request.mode == SearchMode::Literal {
            argv.push("--fixed-strings".to_string());
        }
        if request.include_hidden {
            argv.push("--hidden".to_string());
        }
        if !request.respect_gitignore {
            argv.push("--no-ignore".to_string());
        }
        argv.push(request.query.clone());
        for path in &request.paths {
            argv.push(path.display());
        }
        let mut command = CommandRequest::new(argv);
        command.timeout = Some(self.timeout);
        command.output_limit = CommandOutputLimit {
            stdout_bytes: 512 * 1024,
            stderr_bytes: 64 * 1024,
        };
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
            .map_err(|error| SearchError::Command {
                message: error.to_string(),
            })?;
        let mut warnings = Vec::new();
        if !output.stderr.text.trim().is_empty() {
            warnings.push(output.stderr.text);
        }
        let mut matches = Vec::new();
        for line in output.stdout.text.lines() {
            if matches.len() >= request.max_matches {
                break;
            }
            if let Some(mat) = parse_rg_line(line) {
                matches.push(mat);
            }
        }
        let hit_match_limit = matches.len() >= request.max_matches;
        Ok(RepoSearchResults {
            matches,
            truncated: output.truncated || output.stdout.truncated || hit_match_limit,
            warnings,
            metadata: output.metadata,
        })
    }
}

enum Matcher {
    Literal(String),
    Regex(Regex),
}

impl Matcher {
    fn new(query: &str, mode: SearchMode) -> Result<Self, SearchError> {
        match mode {
            SearchMode::Literal => Ok(Self::Literal(query.to_string())),
            SearchMode::Regex => {
                Regex::new(query)
                    .map(Self::Regex)
                    .map_err(|error| SearchError::InvalidQuery {
                        message: format!("invalid regex: {error}"),
                    })
            }
        }
    }

    fn find(&self, line: &str) -> Option<usize> {
        match self {
            Self::Literal(query) => line.find(query),
            Self::Regex(regex) => regex.find(line).map(|mat| mat.start()),
        }
    }
}

fn search_text(
    path: &WorkspacePath,
    text: &str,
    matcher: &Matcher,
    context_lines: usize,
) -> Vec<SearchMatch> {
    let lines: Vec<_> = text.lines().collect();
    let mut matches = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let Some(column) = matcher.find(line) else {
            continue;
        };
        let before_start = index.saturating_sub(context_lines);
        let after_end = (index + context_lines + 1).min(lines.len());
        matches.push(SearchMatch {
            path: path.clone(),
            line: index + 1,
            column: Some(column + 1),
            line_text: (*line).to_string(),
            before: lines[before_start..index]
                .iter()
                .map(|line| (*line).to_string())
                .collect(),
            after: lines[index + 1..after_end]
                .iter()
                .map(|line| (*line).to_string())
                .collect(),
        });
    }
    matches
}

fn parse_rg_line(line: &str) -> Option<SearchMatch> {
    let mut parts = line.splitn(4, ':');
    let path = WorkspacePath::parse(parts.next()?).ok()?;
    let line_number = parts.next()?.parse().ok()?;
    let column = parts.next()?.parse().ok();
    let line_text = parts.next().unwrap_or_default().to_string();
    Some(SearchMatch {
        path,
        line: line_number,
        column,
        line_text,
        before: Vec::new(),
        after: Vec::new(),
    })
}

/// Search errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SearchError {
    /// Query is invalid.
    #[error("invalid search query: {message}")]
    InvalidQuery {
        /// Model-safe explanation.
        message: String,
    },
    /// Workspace operation failed.
    #[error("workspace search error: {message}")]
    Workspace {
        /// Model-safe explanation.
        message: String,
    },
    /// Command search failed.
    #[error("search command error: {message}")]
    Command {
        /// Model-safe explanation.
        message: String,
    },
    /// Search implementation is unsupported.
    #[error("unsupported search operation: {message}")]
    Unsupported {
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
            std::env::temp_dir().join(format!("molo-search-test-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn workspace_search_returns_structured_matches() {
        let root = temp_dir("workspace");
        std::fs::write(root.join("a.txt"), "one\ntwo\nthree").unwrap();
        let searcher = WorkspaceSearcher::new(LocalWorkspace::new(&root).unwrap());
        let results = searcher
            .search(
                RepoSearchRequest {
                    query: "two".to_string(),
                    paths: Vec::new(),
                    mode: SearchMode::Literal,
                    max_matches: 10,
                    context_lines: 1,
                    include_hidden: false,
                    respect_gitignore: true,
                },
                &RunContext::new("search"),
            )
            .await
            .unwrap();
        assert_eq!(results.matches.len(), 1);
        assert_eq!(results.matches[0].line, 2);
        assert_eq!(results.matches[0].before, vec!["one"]);
        assert_eq!(results.matches[0].after, vec!["three"]);
        let _ = std::fs::remove_dir_all(root);
    }
}
