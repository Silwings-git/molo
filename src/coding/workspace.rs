use crate::RunMetadata;
use async_trait::async_trait;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Canonical root directory that bounds workspace filesystem access.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkspaceRoot {
    absolute: PathBuf,
}

impl WorkspaceRoot {
    /// Canonicalizes and validates an existing workspace root directory.
    ///
    /// # Errors
    ///
    /// Returns [`WorkspaceError::NotFound`] when the path does not exist,
    /// [`WorkspaceError::Unsupported`] when it is not a directory, and
    /// [`WorkspaceError::Io`] for canonicalization failures.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, WorkspaceError> {
        let path = path.as_ref();
        let canonical = std::fs::canonicalize(path).map_err(|error| WorkspaceError::Io {
            message: format!("failed to canonicalize workspace root: {error}"),
        })?;
        let metadata = std::fs::metadata(&canonical).map_err(|error| WorkspaceError::Io {
            message: format!("failed to inspect workspace root: {error}"),
        })?;
        if !metadata.is_dir() {
            return Err(WorkspaceError::Unsupported {
                message: format!("workspace root is not a directory: {}", canonical.display()),
            });
        }
        Ok(Self {
            absolute: canonical,
        })
    }

    /// Returns the absolute canonical root path.
    pub fn as_path(&self) -> &Path {
        &self.absolute
    }

    /// Joins a validated workspace path onto this root.
    pub fn join(&self, path: &WorkspacePath) -> PathBuf {
        self.absolute.join(path.as_path())
    }

    fn strip_absolute(&self, path: &Path) -> Result<WorkspacePath, WorkspaceError> {
        let relative =
            path.strip_prefix(&self.absolute)
                .map_err(|_| WorkspaceError::OutsideRoot {
                    path: path.display().to_string(),
                    root: self.absolute.display().to_string(),
                })?;
        WorkspacePath::from_relative_pathbuf(relative.to_path_buf())
    }
}

impl Serialize for WorkspaceRoot {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let Some(path) = self.absolute.to_str() else {
            return Err(serde::ser::Error::custom(
                "workspace root path is not valid UTF-8",
            ));
        };
        serializer.serialize_str(path)
    }
}

impl<'de> Deserialize<'de> for WorkspaceRoot {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let path = String::deserialize(deserializer)?;
        WorkspaceRoot::new(path).map_err(serde::de::Error::custom)
    }
}

/// A root-relative path validated for workspace operations.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkspacePath {
    relative: PathBuf,
}

impl fmt::Debug for WorkspacePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("WorkspacePath")
            .field(&self.display())
            .finish()
    }
}

impl WorkspacePath {
    /// Returns the workspace root path.
    pub fn root() -> Self {
        Self {
            relative: PathBuf::new(),
        }
    }

    /// Parses a model- or user-facing path as a root-relative workspace path.
    ///
    /// Absolute paths, `.` and `..` traversal, empty components, NUL bytes,
    /// Windows-style separators, and platform roots or prefixes are rejected.
    ///
    /// # Errors
    ///
    /// Returns [`WorkspaceError::InvalidPath`] when the path is not a safe
    /// root-relative path.
    pub fn parse(path: impl AsRef<str>) -> Result<Self, WorkspaceError> {
        let path = path.as_ref();
        if path.is_empty() {
            return Ok(Self::root());
        }
        if path.contains('\0') {
            return Err(WorkspaceError::InvalidPath {
                message: "workspace path contains NUL byte".to_string(),
            });
        }
        if path.contains('\\') {
            return Err(WorkspaceError::InvalidPath {
                message: "workspace path must use forward slashes".to_string(),
            });
        }
        if path.split('/').any(str::is_empty) {
            return Err(WorkspaceError::InvalidPath {
                message: "workspace path contains an empty component".to_string(),
            });
        }
        if path.split('/').any(|component| component == ".") {
            return Err(WorkspaceError::InvalidPath {
                message: "workspace path must not contain `.`".to_string(),
            });
        }
        if path.split('/').any(|component| component == "..") {
            return Err(WorkspaceError::InvalidPath {
                message: "workspace path must not contain `..`".to_string(),
            });
        }

        let candidate = Path::new(path);
        if candidate.is_absolute() {
            return Err(WorkspaceError::InvalidPath {
                message: "workspace path must be relative".to_string(),
            });
        }

        let mut relative = PathBuf::new();
        for component in candidate.components() {
            match component {
                Component::Normal(part) => relative.push(part),
                Component::CurDir => {
                    return Err(WorkspaceError::InvalidPath {
                        message: "workspace path must not contain `.`".to_string(),
                    });
                }
                Component::ParentDir => {
                    return Err(WorkspaceError::InvalidPath {
                        message: "workspace path must not contain `..`".to_string(),
                    });
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err(WorkspaceError::InvalidPath {
                        message: "workspace path must not contain a root or prefix".to_string(),
                    });
                }
            }
        }
        Self::from_relative_pathbuf(relative)
    }

    /// Returns the path as a root-relative [`Path`].
    pub fn as_path(&self) -> &Path {
        &self.relative
    }

    /// Returns a UTF-8 display form suitable for JSON payloads.
    pub fn display(&self) -> String {
        if self.relative.as_os_str().is_empty() {
            return String::new();
        }
        self.relative
            .to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "/")
    }

    /// Appends a single relative component path and validates the result.
    ///
    /// # Errors
    ///
    /// Returns [`WorkspaceError::InvalidPath`] if the resulting path would no
    /// longer be a valid workspace path.
    pub fn join(&self, path: impl AsRef<str>) -> Result<Self, WorkspaceError> {
        let suffix = WorkspacePath::parse(path)?;
        if self.relative.as_os_str().is_empty() {
            return Ok(suffix);
        }
        if suffix.relative.as_os_str().is_empty() {
            return Ok(self.clone());
        }
        Self::from_relative_pathbuf(self.relative.join(suffix.relative))
    }

    fn parent(&self) -> Self {
        self.relative
            .parent()
            .map(|path| Self {
                relative: path.to_path_buf(),
            })
            .unwrap_or_else(Self::root)
    }

    fn from_relative_pathbuf(relative: PathBuf) -> Result<Self, WorkspaceError> {
        if relative.is_absolute() {
            return Err(WorkspaceError::InvalidPath {
                message: "workspace path must be relative".to_string(),
            });
        }
        for component in relative.components() {
            match component {
                Component::Normal(_) => {}
                Component::CurDir => {
                    return Err(WorkspaceError::InvalidPath {
                        message: "workspace path must not contain `.`".to_string(),
                    });
                }
                Component::ParentDir => {
                    return Err(WorkspaceError::InvalidPath {
                        message: "workspace path must not contain `..`".to_string(),
                    });
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err(WorkspaceError::InvalidPath {
                        message: "workspace path must not contain a root or prefix".to_string(),
                    });
                }
            }
        }
        if relative.as_os_str().is_empty() {
            return Ok(Self::root());
        }
        if relative.to_str().is_none() {
            return Err(WorkspaceError::NonUtf8Path {
                path: relative.display().to_string(),
            });
        }
        Ok(Self { relative })
    }
}

impl Serialize for WorkspacePath {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.display())
    }
}

impl<'de> Deserialize<'de> for WorkspacePath {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let path = String::deserialize(deserializer)?;
        WorkspacePath::parse(path).map_err(serde::de::Error::custom)
    }
}

impl fmt::Display for WorkspacePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.display())
    }
}

/// Resolved workspace path with canonicalization metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedPath {
    /// Workspace root used for resolution.
    pub root: WorkspaceRoot,
    /// Original validated workspace path.
    pub workspace_path: WorkspacePath,
    /// Absolute path selected for the requested access.
    pub absolute: PathBuf,
    /// Filesystem kind observed at resolution time.
    pub kind: ResolvedPathKind,
    /// Canonical symlink target, when the path itself is a symlink.
    pub symlink_target: Option<PathBuf>,
}

/// Filesystem kind observed when resolving a workspace path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ResolvedPathKind {
    /// Regular file.
    File,
    /// Directory.
    Directory,
    /// Symlink.
    Symlink,
    /// Path does not exist.
    Missing,
}

/// Symlink behavior for local workspace operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SymlinkPolicy {
    /// Read and list may follow symlinks whose canonical target remains
    /// inside the workspace root.
    #[default]
    FollowReadInsideRoot,
    /// Do not follow symlinks.
    NoFollow,
    /// Reject any operation targeting a symlink.
    RejectAll,
}

/// Requested workspace access mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum WorkspaceAccess {
    /// Read file content or metadata.
    Read,
    /// List directory entries.
    List,
    /// Create a new file.
    Create,
    /// Modify an existing file.
    Modify,
    /// Delete an existing file.
    Delete,
}

/// Encoding of text file content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TextEncoding {
    /// UTF-8 text.
    Utf8,
}

/// File content returned by a workspace read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileContent {
    /// File path.
    pub path: WorkspacePath,
    /// Version observed while reading.
    pub version: FileVersion,
    /// File body.
    pub body: FileBody,
    /// Whether the body was truncated by byte budget.
    pub truncated: bool,
    /// Host-owned metadata.
    pub metadata: RunMetadata,
}

/// File body with text and binary separated explicitly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FileBody {
    /// UTF-8 text body.
    Text {
        /// Text content.
        text: String,
        /// Text encoding.
        encoding: TextEncoding,
    },
    /// Binary body. When binary reads are not explicitly enabled, `bytes`
    /// is empty and metadata/version still describe the file.
    Binary {
        /// Binary bytes returned to the caller.
        bytes: Vec<u8>,
        /// Optional media type.
        media_type: Option<String>,
    },
}

/// Stable content digest used in file version preconditions.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ContentDigest {
    /// Digest algorithm name.
    pub algorithm: String,
    /// Hex-encoded digest.
    pub value: String,
}

/// File version used to detect stale writes and patch conflicts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileVersion {
    /// File path.
    pub path: WorkspacePath,
    /// Content digest.
    pub digest: ContentDigest,
    /// File byte length.
    pub len: u64,
    /// Last modification time, when available.
    pub modified: Option<SystemTime>,
}

/// Options for workspace file reads.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileReadOptions {
    /// Maximum bytes to read before truncating.
    pub max_bytes: Option<usize>,
    /// Whether binary bytes may be returned.
    pub include_binary: bool,
}

/// Content to write into a workspace file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FileWriteContent {
    /// UTF-8 text.
    Text(String),
    /// Raw bytes.
    Bytes(Vec<u8>),
}

impl FileWriteContent {
    fn into_bytes(self) -> Vec<u8> {
        match self {
            Self::Text(text) => text.into_bytes(),
            Self::Bytes(bytes) => bytes,
        }
    }
}

/// Request to write a workspace file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WriteFileRequest {
    /// File path.
    pub path: WorkspacePath,
    /// Content to write.
    pub content: FileWriteContent,
    /// Required currently observed version.
    pub expected_version: Option<FileVersion>,
    /// Whether a missing file may be created.
    pub create: bool,
    /// Whether an existing file may be overwritten.
    pub overwrite: bool,
}

/// Result of a successful workspace write.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileWriteResult {
    /// File path.
    pub path: WorkspacePath,
    /// Version before writing, when the file existed.
    pub previous_version: Option<FileVersion>,
    /// Version after writing.
    pub new_version: FileVersion,
    /// Whether the file was newly created.
    pub created: bool,
    /// Number of bytes written.
    pub bytes_written: u64,
    /// Host-owned metadata.
    pub metadata: RunMetadata,
}

/// Query for deterministic workspace listing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListFilesQuery {
    /// Directory or file path to list.
    pub path: WorkspacePath,
    /// Whether to recurse into directories.
    pub recursive: bool,
    /// Maximum number of entries returned.
    pub max_entries: Option<usize>,
    /// Whether hidden path components are included.
    pub include_hidden: bool,
    /// Whether simple `.gitignore` patterns are respected.
    pub respect_gitignore: bool,
}

/// Workspace entry returned by listing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceEntry {
    /// Entry path.
    pub path: WorkspacePath,
    /// Entry kind.
    pub kind: ResolvedPathKind,
    /// File length, when available.
    pub len: Option<u64>,
}

/// Request for a lightweight workspace snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotRequest {
    /// Paths to include. Empty means the workspace root.
    pub paths: Vec<WorkspacePath>,
    /// Whether directory paths are captured recursively.
    pub recursive: bool,
}

/// Lightweight snapshot of file versions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceSnapshot {
    /// Snapshot id.
    pub id: String,
    /// Workspace root.
    pub root: WorkspaceRoot,
    /// File versions.
    pub files: Vec<FileVersion>,
    /// Git head at capture time, when known.
    pub git_head: Option<String>,
    /// Host-owned metadata.
    pub metadata: RunMetadata,
}

/// Request to diff two snapshots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffRequest {
    /// Earlier snapshot.
    pub before: WorkspaceSnapshot,
    /// Later snapshot.
    pub after: WorkspaceSnapshot,
}

/// Workspace diff summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceDiff {
    /// Changed paths.
    pub changed_files: Vec<WorkspacePath>,
    /// Text summary suitable for users and models.
    pub text: String,
    /// Whether the diff text was truncated.
    pub truncated: bool,
    /// Host-owned metadata.
    pub metadata: RunMetadata,
}

/// Structured patch containing one or more file patches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Patch {
    /// File patches.
    pub files: Vec<FilePatch>,
    /// Original patch text, when imported from a textual format.
    pub original_text: Option<String>,
    /// Host-owned metadata.
    pub metadata: RunMetadata,
}

/// Patch for one file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilePatch {
    /// Destination path, or current path for non-rename operations.
    pub path: WorkspacePath,
    /// Patch operation.
    pub operation: PatchOperation,
    /// Required version for the file being modified, deleted, or renamed.
    pub expected_version: Option<FileVersion>,
    /// Text hunks applied in order.
    pub hunks: Vec<PatchHunk>,
}

/// File patch operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PatchOperation {
    /// Create a new file.
    Create,
    /// Modify an existing file.
    Modify,
    /// Delete an existing file.
    Delete,
    /// Rename an existing file to `FilePatch::path`.
    Rename {
        /// Source path.
        from: WorkspacePath,
    },
}

/// Text hunk used by the local workspace patch applier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatchHunk {
    /// Text expected in the current file. For create operations this may be
    /// empty.
    pub old_text: String,
    /// Replacement text.
    pub new_text: String,
}

/// Request to apply or dry-run a patch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatchRequest {
    /// Patch to apply.
    pub patch: Patch,
    /// Validate without writing.
    pub dry_run: bool,
    /// Whether non-conflicting file patches may be applied when another file
    /// conflicts. The local workspace currently reports conflicts without
    /// partial writes.
    pub allow_partial: bool,
}

/// Patch conflict with model-safe details.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatchConflict {
    /// Path that conflicted.
    pub path: WorkspacePath,
    /// Human-readable conflict explanation.
    pub message: String,
    /// Expected version, when supplied.
    pub expected_version: Option<FileVersion>,
    /// Actual version, when observed.
    pub actual_version: Option<FileVersion>,
}

/// Result from applying or dry-running a patch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatchResult {
    /// Whether changes were written.
    pub applied: bool,
    /// Files that would change or did change.
    pub changed_files: Vec<WorkspacePath>,
    /// Patch conflicts.
    pub conflicts: Vec<PatchConflict>,
    /// Diff summary.
    pub diff: WorkspaceDiff,
    /// Snapshot before validation.
    pub snapshot_before: WorkspaceSnapshot,
    /// Snapshot after writing, absent for dry-run or conflict results.
    pub snapshot_after: Option<WorkspaceSnapshot>,
    /// Host-owned metadata.
    pub metadata: RunMetadata,
}

/// Tracks files changed by the agent layer.
#[derive(Debug, Clone, Default)]
pub struct AgentChangeTracker {
    changes: Arc<Mutex<BTreeMap<WorkspacePath, FileChange>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FileChange {
    before: Option<FileVersion>,
    after: Option<FileVersion>,
}

impl AgentChangeTracker {
    /// Constructs an empty change tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a file write, delete, or rename target.
    pub fn record(
        &self,
        path: WorkspacePath,
        before: Option<FileVersion>,
        after: Option<FileVersion>,
    ) {
        self.changes
            .lock()
            .expect("AgentChangeTracker lock poisoned")
            .insert(path, FileChange { before, after });
    }

    /// Returns paths changed by the agent so far.
    pub fn changed_files(&self) -> Vec<WorkspacePath> {
        self.changes
            .lock()
            .expect("AgentChangeTracker lock poisoned")
            .keys()
            .cloned()
            .collect()
    }

    /// Clears recorded changes.
    pub fn clear(&self) {
        self.changes
            .lock()
            .expect("AgentChangeTracker lock poisoned")
            .clear();
    }
}

/// Workspace abstraction for coding workloads.
#[async_trait]
pub trait Workspace: Send + Sync {
    /// Returns this workspace's root.
    async fn root(&self) -> WorkspaceRoot;

    /// Resolves a root-relative path for a specific access mode.
    async fn resolve(
        &self,
        path: &WorkspacePath,
        access: WorkspaceAccess,
    ) -> Result<ResolvedPath, WorkspaceError>;

    /// Reads a file through workspace policy.
    async fn read_file(
        &self,
        path: &WorkspacePath,
        options: FileReadOptions,
    ) -> Result<FileContent, WorkspaceError>;

    /// Writes a file through workspace policy.
    async fn write_file(
        &self,
        request: WriteFileRequest,
    ) -> Result<FileWriteResult, WorkspaceError>;

    /// Lists files through workspace policy.
    async fn list_files(
        &self,
        query: ListFilesQuery,
    ) -> Result<Vec<WorkspaceEntry>, WorkspaceError>;

    /// Captures a lightweight snapshot of file versions.
    async fn snapshot(&self, request: SnapshotRequest)
    -> Result<WorkspaceSnapshot, WorkspaceError>;

    /// Diffs two snapshots.
    async fn diff(&self, request: DiffRequest) -> Result<WorkspaceDiff, WorkspaceError>;

    /// Applies or dry-runs a structured patch.
    async fn apply_patch(&self, request: PatchRequest) -> Result<PatchResult, WorkspaceError>;
}

/// Local filesystem workspace configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct LocalWorkspaceConfig {
    /// Symlink behavior.
    pub(crate) symlink_policy: SymlinkPolicy,
    /// Default maximum bytes returned by read operations.
    pub(crate) max_read_bytes: usize,
    /// Default maximum entries returned by list operations.
    pub(crate) max_list_entries: usize,
    /// Whether hidden files are included when a query does not opt in.
    pub(crate) include_hidden_by_default: bool,
    /// Whether simple `.gitignore` patterns are respected when a query does
    /// not opt out.
    pub(crate) respect_gitignore_by_default: bool,
}

impl Default for LocalWorkspaceConfig {
    fn default() -> Self {
        Self {
            symlink_policy: SymlinkPolicy::FollowReadInsideRoot,
            max_read_bytes: 64 * 1024,
            max_list_entries: 10_000,
            include_hidden_by_default: false,
            respect_gitignore_by_default: true,
        }
    }
}

impl LocalWorkspaceConfig {
    /// Constructs a config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Symlink behavior.
    pub fn symlink_policy(&self) -> SymlinkPolicy {
        self.symlink_policy
    }

    /// Returns a config with updated symlink behavior.
    pub fn with_symlink_policy(mut self, symlink_policy: SymlinkPolicy) -> Self {
        self.symlink_policy = symlink_policy;
        self
    }

    /// Default maximum bytes returned by read operations.
    pub fn max_read_bytes(&self) -> usize {
        self.max_read_bytes
    }

    /// Returns a config with an updated read byte cap.
    pub fn with_max_read_bytes(mut self, max_read_bytes: usize) -> Self {
        self.max_read_bytes = max_read_bytes;
        self
    }

    /// Default maximum entries returned by list operations.
    pub fn max_list_entries(&self) -> usize {
        self.max_list_entries
    }

    /// Returns a config with an updated list entry cap.
    pub fn with_max_list_entries(mut self, max_list_entries: usize) -> Self {
        self.max_list_entries = max_list_entries;
        self
    }

    /// Whether hidden files are included when a query does not opt in.
    pub fn include_hidden_by_default(&self) -> bool {
        self.include_hidden_by_default
    }

    /// Returns a config with updated hidden-file behavior.
    pub fn with_include_hidden_by_default(mut self, include_hidden_by_default: bool) -> Self {
        self.include_hidden_by_default = include_hidden_by_default;
        self
    }

    /// Whether `.gitignore` patterns are respected when a query does not opt out.
    pub fn respect_gitignore_by_default(&self) -> bool {
        self.respect_gitignore_by_default
    }

    /// Returns a config with updated `.gitignore` behavior.
    pub fn with_respect_gitignore_by_default(mut self, respect_gitignore_by_default: bool) -> Self {
        self.respect_gitignore_by_default = respect_gitignore_by_default;
        self
    }
}

/// Local filesystem implementation of [`Workspace`].
#[derive(Debug, Clone)]
pub struct LocalWorkspace {
    root: WorkspaceRoot,
    config: LocalWorkspaceConfig,
    changes: AgentChangeTracker,
}

impl LocalWorkspace {
    /// Constructs a local workspace from a root directory.
    ///
    /// # Errors
    ///
    /// Returns [`WorkspaceError`] if the root cannot be canonicalized or is
    /// not an existing directory.
    pub fn new(root: impl AsRef<Path>) -> Result<Self, WorkspaceError> {
        Self::with_config(root, LocalWorkspaceConfig::default())
    }

    /// Constructs a local workspace with explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns [`WorkspaceError`] if the root cannot be canonicalized or is
    /// not an existing directory.
    pub fn with_config(
        root: impl AsRef<Path>,
        config: LocalWorkspaceConfig,
    ) -> Result<Self, WorkspaceError> {
        Ok(Self {
            root: WorkspaceRoot::new(root)?,
            config,
            changes: AgentChangeTracker::new(),
        })
    }

    /// Returns the local change tracker.
    pub fn change_tracker(&self) -> AgentChangeTracker {
        self.changes.clone()
    }

    fn check_inside_root(&self, absolute: &Path) -> Result<(), WorkspaceError> {
        if absolute.starts_with(self.root.as_path()) {
            Ok(())
        } else {
            Err(WorkspaceError::OutsideRoot {
                path: absolute.display().to_string(),
                root: self.root.as_path().display().to_string(),
            })
        }
    }

    fn resolve_existing_kind(metadata: &std::fs::Metadata) -> ResolvedPathKind {
        if metadata.is_dir() {
            ResolvedPathKind::Directory
        } else {
            ResolvedPathKind::File
        }
    }

    async fn version_for_absolute(
        &self,
        path: &WorkspacePath,
        absolute: &Path,
    ) -> Result<FileVersion, WorkspaceError> {
        let bytes = tokio::fs::read(absolute)
            .await
            .map_err(|error| WorkspaceError::Io {
                message: format!("failed to read file for version: {error}"),
            })?;
        let metadata = tokio::fs::metadata(absolute)
            .await
            .map_err(|error| WorkspaceError::Io {
                message: format!("failed to inspect file for version: {error}"),
            })?;
        Ok(FileVersion {
            path: path.clone(),
            digest: digest_bytes(&bytes),
            len: metadata.len(),
            modified: metadata.modified().ok(),
        })
    }

    async fn read_text_file(
        &self,
        path: &WorkspacePath,
        max_bytes: usize,
    ) -> Result<(String, FileVersion, bool), WorkspaceError> {
        let content = self
            .read_file(
                path,
                FileReadOptions {
                    max_bytes: Some(max_bytes),
                    include_binary: false,
                },
            )
            .await?;
        let text = match content.body {
            FileBody::Text { text, .. } => text,
            FileBody::Binary { .. } => {
                return Err(WorkspaceError::Unsupported {
                    message: format!("file is binary: {}", path.display()),
                });
            }
        };
        Ok((text, content.version, content.truncated))
    }

    async fn snapshot_paths(
        &self,
        paths: Vec<WorkspacePath>,
        recursive: bool,
    ) -> Result<WorkspaceSnapshot, WorkspaceError> {
        let paths = if paths.is_empty() {
            vec![WorkspacePath::root()]
        } else {
            paths
        };
        let mut versions = BTreeMap::new();
        for path in paths {
            let resolved = self.resolve(&path, WorkspaceAccess::Read).await?;
            match resolved.kind {
                ResolvedPathKind::Missing => {}
                ResolvedPathKind::Directory => {
                    let entries = self
                        .list_files(ListFilesQuery {
                            path: path.clone(),
                            recursive,
                            max_entries: Some(self.config.max_list_entries),
                            include_hidden: self.config.include_hidden_by_default,
                            respect_gitignore: self.config.respect_gitignore_by_default,
                        })
                        .await?;
                    for entry in entries {
                        if entry.kind != ResolvedPathKind::File {
                            continue;
                        }
                        let absolute = self.root.join(&entry.path);
                        let version = self.version_for_absolute(&entry.path, &absolute).await?;
                        versions.insert(entry.path, version);
                    }
                }
                ResolvedPathKind::File => {
                    let version = self.version_for_absolute(&path, &resolved.absolute).await?;
                    versions.insert(path, version);
                }
                ResolvedPathKind::Symlink => {}
            }
        }
        Ok(WorkspaceSnapshot {
            id: generated_snapshot_id(),
            root: self.root.clone(),
            files: versions.into_values().collect(),
            git_head: None,
            metadata: RunMetadata::new(),
        })
    }
}

#[async_trait]
impl Workspace for LocalWorkspace {
    async fn root(&self) -> WorkspaceRoot {
        self.root.clone()
    }

    async fn resolve(
        &self,
        path: &WorkspacePath,
        access: WorkspaceAccess,
    ) -> Result<ResolvedPath, WorkspaceError> {
        let joined = self.root.join(path);
        if !joined.starts_with(self.root.as_path()) {
            return Err(WorkspaceError::OutsideRoot {
                path: joined.display().to_string(),
                root: self.root.as_path().display().to_string(),
            });
        }

        match std::fs::symlink_metadata(&joined) {
            Ok(symlink_metadata) => {
                if symlink_metadata.file_type().is_symlink() {
                    let target =
                        std::fs::read_link(&joined).map_err(|error| WorkspaceError::Io {
                            message: format!("failed to read symlink: {error}"),
                        })?;
                    let target_absolute = if target.is_absolute() {
                        target
                    } else {
                        joined
                            .parent()
                            .unwrap_or_else(|| self.root.as_path())
                            .join(target)
                    };
                    let canonical_target =
                        std::fs::canonicalize(&target_absolute).map_err(|error| {
                            WorkspaceError::Io {
                                message: format!("failed to canonicalize symlink target: {error}"),
                            }
                        })?;
                    if !canonical_target.starts_with(self.root.as_path()) {
                        return Err(WorkspaceError::SymlinkEscapesRoot {
                            path: joined.display().to_string(),
                            target: canonical_target.display().to_string(),
                        });
                    }
                    return match self.config.symlink_policy {
                        SymlinkPolicy::RejectAll => Err(WorkspaceError::Unsupported {
                            message: format!("symlink rejected: {}", path.display()),
                        }),
                        SymlinkPolicy::NoFollow => Ok(ResolvedPath {
                            root: self.root.clone(),
                            workspace_path: path.clone(),
                            absolute: joined,
                            kind: ResolvedPathKind::Symlink,
                            symlink_target: Some(canonical_target),
                        }),
                        SymlinkPolicy::FollowReadInsideRoot
                            if matches!(access, WorkspaceAccess::Read | WorkspaceAccess::List) =>
                        {
                            let metadata =
                                std::fs::metadata(&canonical_target).map_err(|error| {
                                    WorkspaceError::Io {
                                        message: format!(
                                            "failed to inspect symlink target: {error}"
                                        ),
                                    }
                                })?;
                            Ok(ResolvedPath {
                                root: self.root.clone(),
                                workspace_path: path.clone(),
                                absolute: canonical_target.clone(),
                                kind: Self::resolve_existing_kind(&metadata),
                                symlink_target: Some(canonical_target),
                            })
                        }
                        SymlinkPolicy::FollowReadInsideRoot => Err(WorkspaceError::Unsupported {
                            message: format!(
                                "write/delete through symlink rejected: {}",
                                path.display()
                            ),
                        }),
                    };
                }

                let canonical =
                    std::fs::canonicalize(&joined).map_err(|error| WorkspaceError::Io {
                        message: format!("failed to canonicalize workspace path: {error}"),
                    })?;
                self.check_inside_root(&canonical)?;
                Ok(ResolvedPath {
                    root: self.root.clone(),
                    workspace_path: path.clone(),
                    absolute: canonical,
                    kind: Self::resolve_existing_kind(&symlink_metadata),
                    symlink_target: None,
                })
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let parent = path.parent();
                let parent_absolute = self.root.join(&parent);
                let canonical_parent =
                    std::fs::canonicalize(&parent_absolute).map_err(|parent_error| {
                        WorkspaceError::NotFound {
                            path: parent.clone(),
                            message: format!("parent does not exist: {parent_error}"),
                        }
                    })?;
                self.check_inside_root(&canonical_parent)?;
                Ok(ResolvedPath {
                    root: self.root.clone(),
                    workspace_path: path.clone(),
                    absolute: joined,
                    kind: ResolvedPathKind::Missing,
                    symlink_target: None,
                })
            }
            Err(error) => Err(WorkspaceError::Io {
                message: format!("failed to inspect workspace path: {error}"),
            }),
        }
    }

    async fn read_file(
        &self,
        path: &WorkspacePath,
        options: FileReadOptions,
    ) -> Result<FileContent, WorkspaceError> {
        let resolved = self.resolve(path, WorkspaceAccess::Read).await?;
        match resolved.kind {
            ResolvedPathKind::Missing => {
                return Err(WorkspaceError::NotFound {
                    path: path.clone(),
                    message: "file does not exist".to_string(),
                });
            }
            ResolvedPathKind::Directory => {
                return Err(WorkspaceError::Unsupported {
                    message: format!("path is a directory: {}", path.display()),
                });
            }
            ResolvedPathKind::Symlink => {
                return Err(WorkspaceError::Unsupported {
                    message: format!("symlink was not followed: {}", path.display()),
                });
            }
            ResolvedPathKind::File => {}
        }

        let max_bytes = options.max_bytes.unwrap_or(self.config.max_read_bytes);
        let mut bytes =
            tokio::fs::read(&resolved.absolute)
                .await
                .map_err(|error| WorkspaceError::Io {
                    message: format!("failed to read file: {error}"),
                })?;
        let total_len = bytes.len();
        let truncated = bytes.len() > max_bytes;
        if truncated {
            bytes.truncate(max_bytes);
        }
        let metadata = tokio::fs::metadata(&resolved.absolute)
            .await
            .map_err(|error| WorkspaceError::Io {
                message: format!("failed to inspect file: {error}"),
            })?;
        let version = FileVersion {
            path: path.clone(),
            digest: digest_bytes(&tokio::fs::read(&resolved.absolute).await.map_err(|error| {
                WorkspaceError::Io {
                    message: format!("failed to read file for digest: {error}"),
                }
            })?),
            len: metadata.len(),
            modified: metadata.modified().ok(),
        };
        let body = match String::from_utf8(bytes) {
            Ok(text) => FileBody::Text {
                text,
                encoding: TextEncoding::Utf8,
            },
            Err(error) if options.include_binary => FileBody::Binary {
                bytes: error.into_bytes(),
                media_type: None,
            },
            Err(_) => FileBody::Binary {
                bytes: Vec::new(),
                media_type: None,
            },
        };
        let mut metadata = RunMetadata::new();
        metadata.insert("bytes_read".to_string(), serde_json::json!(total_len));
        Ok(FileContent {
            path: path.clone(),
            version,
            body,
            truncated,
            metadata,
        })
    }

    async fn write_file(
        &self,
        request: WriteFileRequest,
    ) -> Result<FileWriteResult, WorkspaceError> {
        let joined = self.root.join(&request.path);
        let exists = std::fs::symlink_metadata(&joined).is_ok();
        let access = if exists {
            WorkspaceAccess::Modify
        } else {
            WorkspaceAccess::Create
        };
        let resolved = self.resolve(&request.path, access).await?;
        let previous_version = if exists {
            if resolved.kind != ResolvedPathKind::File {
                return Err(WorkspaceError::Unsupported {
                    message: format!("path is not a regular file: {}", request.path.display()),
                });
            }
            Some(
                self.version_for_absolute(&request.path, &resolved.absolute)
                    .await?,
            )
        } else {
            None
        };

        if previous_version.is_some() && !request.overwrite {
            return Err(WorkspaceError::Conflict {
                conflict: Box::new(PatchConflict {
                    path: request.path.clone(),
                    message: "file exists and overwrite is false".to_string(),
                    expected_version: request.expected_version.clone(),
                    actual_version: previous_version,
                }),
            });
        }
        if previous_version.is_none() && !request.create {
            return Err(WorkspaceError::NotFound {
                path: request.path,
                message: "file does not exist and create is false".to_string(),
            });
        }
        if let Some(expected) = &request.expected_version
            && previous_version.as_ref() != Some(expected)
        {
            return Err(WorkspaceError::Conflict {
                conflict: Box::new(PatchConflict {
                    path: request.path.clone(),
                    message: "stale file version".to_string(),
                    expected_version: Some(expected.clone()),
                    actual_version: previous_version,
                }),
            });
        }

        let bytes = request.content.into_bytes();
        let parent = resolved
            .absolute
            .parent()
            .ok_or_else(|| WorkspaceError::InvalidPath {
                message: "file path has no parent".to_string(),
            })?;
        let tmp = parent.join(format!(
            ".molo-write-{}-{}",
            std::process::id(),
            monotonic_nanos()
        ));
        tokio::fs::write(&tmp, &bytes)
            .await
            .map_err(|error| WorkspaceError::Io {
                message: format!("failed to write temporary file: {error}"),
            })?;
        tokio::fs::rename(&tmp, &resolved.absolute)
            .await
            .map_err(|error| WorkspaceError::Io {
                message: format!("failed to commit file write: {error}"),
            })?;
        let new_version = self
            .version_for_absolute(&request.path, &resolved.absolute)
            .await?;
        self.changes.record(
            request.path.clone(),
            previous_version.clone(),
            Some(new_version.clone()),
        );
        Ok(FileWriteResult {
            path: request.path,
            previous_version,
            new_version,
            created: !exists,
            bytes_written: bytes.len() as u64,
            metadata: RunMetadata::new(),
        })
    }

    async fn list_files(
        &self,
        query: ListFilesQuery,
    ) -> Result<Vec<WorkspaceEntry>, WorkspaceError> {
        let resolved = self.resolve(&query.path, WorkspaceAccess::List).await?;
        if resolved.kind == ResolvedPathKind::Missing {
            return Err(WorkspaceError::NotFound {
                path: query.path,
                message: "path does not exist".to_string(),
            });
        }
        let max_entries = query.max_entries.unwrap_or(self.config.max_list_entries);
        let ignore = if query.respect_gitignore {
            SimpleIgnore::load(self.root.as_path())
        } else {
            SimpleIgnore::default()
        };
        let mut entries = Vec::new();
        collect_entries(
            &self.root,
            &resolved.absolute,
            query.recursive,
            query.include_hidden,
            &ignore,
            max_entries,
            &mut entries,
        )?;
        entries.sort_by(|left, right| left.path.cmp(&right.path));
        if entries.len() > max_entries {
            entries.truncate(max_entries);
        }
        Ok(entries)
    }

    async fn snapshot(
        &self,
        request: SnapshotRequest,
    ) -> Result<WorkspaceSnapshot, WorkspaceError> {
        self.snapshot_paths(request.paths, request.recursive).await
    }

    async fn diff(&self, request: DiffRequest) -> Result<WorkspaceDiff, WorkspaceError> {
        Ok(diff_snapshots(&request.before, &request.after))
    }

    async fn apply_patch(&self, request: PatchRequest) -> Result<PatchResult, WorkspaceError> {
        let before = self
            .snapshot(SnapshotRequest {
                paths: Vec::new(),
                recursive: true,
            })
            .await?;
        let mut conflicts = Vec::new();
        let mut writes: BTreeMap<WorkspacePath, Vec<u8>> = BTreeMap::new();
        let mut deletes: BTreeSet<WorkspacePath> = BTreeSet::new();

        for file_patch in &request.patch.files {
            match validate_file_patch(self, file_patch).await {
                Ok(plan) => match plan {
                    PatchPlan::Write { path, bytes } => {
                        writes.insert(path, bytes);
                    }
                    PatchPlan::Delete { path } => {
                        deletes.insert(path);
                    }
                    PatchPlan::Rename { from, to, bytes } => {
                        deletes.insert(from);
                        writes.insert(to, bytes);
                    }
                },
                Err(conflict) => conflicts.push(conflict),
            }
        }

        let changed_files: Vec<_> = writes
            .keys()
            .chain(deletes.iter())
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if !conflicts.is_empty() || request.dry_run {
            let diff = WorkspaceDiff {
                changed_files: changed_files.clone(),
                text: if conflicts.is_empty() {
                    format!("patch dry-run would change {} file(s)", changed_files.len())
                } else {
                    format!("patch has {} conflict(s)", conflicts.len())
                },
                truncated: false,
                metadata: RunMetadata::new(),
            };
            return Ok(PatchResult {
                applied: false,
                changed_files,
                conflicts,
                diff,
                snapshot_before: before,
                snapshot_after: None,
                metadata: RunMetadata::new(),
            });
        }

        for path in deletes {
            let resolved = self.resolve(&path, WorkspaceAccess::Delete).await?;
            tokio::fs::remove_file(&resolved.absolute)
                .await
                .map_err(|error| WorkspaceError::Io {
                    message: format!("failed to delete file: {error}"),
                })?;
            self.changes.record(path, None, None);
        }
        for (path, bytes) in writes {
            let text = match String::from_utf8(bytes.clone()) {
                Ok(text) => FileWriteContent::Text(text),
                Err(_) => FileWriteContent::Bytes(bytes),
            };
            let exists = std::fs::symlink_metadata(self.root.join(&path)).is_ok();
            self.write_file(WriteFileRequest {
                path,
                content: text,
                expected_version: None,
                create: !exists,
                overwrite: exists,
            })
            .await?;
        }

        let after = self
            .snapshot(SnapshotRequest {
                paths: Vec::new(),
                recursive: true,
            })
            .await?;
        let diff = diff_snapshots(&before, &after);
        Ok(PatchResult {
            applied: true,
            changed_files: diff.changed_files.clone(),
            conflicts: Vec::new(),
            diff,
            snapshot_before: before,
            snapshot_after: Some(after),
            metadata: RunMetadata::new(),
        })
    }
}

#[derive(Debug)]
enum PatchPlan {
    Write {
        path: WorkspacePath,
        bytes: Vec<u8>,
    },
    Delete {
        path: WorkspacePath,
    },
    Rename {
        from: WorkspacePath,
        to: WorkspacePath,
        bytes: Vec<u8>,
    },
}

async fn validate_file_patch(
    workspace: &LocalWorkspace,
    file_patch: &FilePatch,
) -> Result<PatchPlan, PatchConflict> {
    match &file_patch.operation {
        PatchOperation::Create => {
            if std::fs::symlink_metadata(workspace.root.join(&file_patch.path)).is_ok() {
                return Err(PatchConflict {
                    path: file_patch.path.clone(),
                    message: "create target already exists".to_string(),
                    expected_version: file_patch.expected_version.clone(),
                    actual_version: None,
                });
            }
            let bytes = file_patch
                .hunks
                .iter()
                .map(|hunk| hunk.new_text.as_str())
                .collect::<String>()
                .into_bytes();
            Ok(PatchPlan::Write {
                path: file_patch.path.clone(),
                bytes,
            })
        }
        PatchOperation::Modify => {
            let (mut text, version, truncated) = workspace
                .read_text_file(&file_patch.path, workspace.config.max_read_bytes)
                .await
                .map_err(|error| PatchConflict {
                    path: file_patch.path.clone(),
                    message: error.to_string(),
                    expected_version: file_patch.expected_version.clone(),
                    actual_version: None,
                })?;
            if truncated {
                return Err(PatchConflict {
                    path: file_patch.path.clone(),
                    message: "file is too large for local patch applier".to_string(),
                    expected_version: file_patch.expected_version.clone(),
                    actual_version: Some(version),
                });
            }
            if let Some(expected) = &file_patch.expected_version
                && expected != &version
            {
                return Err(PatchConflict {
                    path: file_patch.path.clone(),
                    message: "stale file version".to_string(),
                    expected_version: Some(expected.clone()),
                    actual_version: Some(version),
                });
            }
            for hunk in &file_patch.hunks {
                let Some(index) = text.find(&hunk.old_text) else {
                    return Err(PatchConflict {
                        path: file_patch.path.clone(),
                        message: "patch hunk did not match".to_string(),
                        expected_version: file_patch.expected_version.clone(),
                        actual_version: Some(version),
                    });
                };
                text.replace_range(index..index + hunk.old_text.len(), &hunk.new_text);
            }
            Ok(PatchPlan::Write {
                path: file_patch.path.clone(),
                bytes: text.into_bytes(),
            })
        }
        PatchOperation::Delete => {
            let version = workspace
                .version_for_absolute(&file_patch.path, &workspace.root.join(&file_patch.path))
                .await
                .map_err(|error| PatchConflict {
                    path: file_patch.path.clone(),
                    message: error.to_string(),
                    expected_version: file_patch.expected_version.clone(),
                    actual_version: None,
                })?;
            if let Some(expected) = &file_patch.expected_version
                && expected != &version
            {
                return Err(PatchConflict {
                    path: file_patch.path.clone(),
                    message: "stale file version".to_string(),
                    expected_version: Some(expected.clone()),
                    actual_version: Some(version),
                });
            }
            Ok(PatchPlan::Delete {
                path: file_patch.path.clone(),
            })
        }
        PatchOperation::Rename { from } => {
            let (text, version, truncated) = workspace
                .read_text_file(from, workspace.config.max_read_bytes)
                .await
                .map_err(|error| PatchConflict {
                    path: from.clone(),
                    message: error.to_string(),
                    expected_version: file_patch.expected_version.clone(),
                    actual_version: None,
                })?;
            if truncated {
                return Err(PatchConflict {
                    path: from.clone(),
                    message: "file is too large for local patch applier".to_string(),
                    expected_version: file_patch.expected_version.clone(),
                    actual_version: Some(version),
                });
            }
            if std::fs::symlink_metadata(workspace.root.join(&file_patch.path)).is_ok() {
                return Err(PatchConflict {
                    path: file_patch.path.clone(),
                    message: "rename target already exists".to_string(),
                    expected_version: None,
                    actual_version: None,
                });
            }
            let mut new_text = text;
            for hunk in &file_patch.hunks {
                if hunk.old_text.is_empty() {
                    continue;
                }
                let Some(index) = new_text.find(&hunk.old_text) else {
                    return Err(PatchConflict {
                        path: from.clone(),
                        message: "rename hunk did not match".to_string(),
                        expected_version: file_patch.expected_version.clone(),
                        actual_version: Some(version),
                    });
                };
                new_text.replace_range(index..index + hunk.old_text.len(), &hunk.new_text);
            }
            Ok(PatchPlan::Rename {
                from: from.clone(),
                to: file_patch.path.clone(),
                bytes: new_text.into_bytes(),
            })
        }
    }
}

fn collect_entries(
    root: &WorkspaceRoot,
    absolute: &Path,
    recursive: bool,
    include_hidden: bool,
    ignore: &SimpleIgnore,
    max_entries: usize,
    entries: &mut Vec<WorkspaceEntry>,
) -> Result<(), WorkspaceError> {
    if entries.len() >= max_entries {
        return Ok(());
    }
    let metadata = std::fs::metadata(absolute).map_err(|error| WorkspaceError::Io {
        message: format!("failed to inspect list entry: {error}"),
    })?;
    if metadata.is_file() {
        entries.push(WorkspaceEntry {
            path: root.strip_absolute(absolute)?,
            kind: ResolvedPathKind::File,
            len: Some(metadata.len()),
        });
        return Ok(());
    }

    let mut children = Vec::new();
    for entry in std::fs::read_dir(absolute).map_err(|error| WorkspaceError::Io {
        message: format!("failed to list directory: {error}"),
    })? {
        let entry = entry.map_err(|error| WorkspaceError::Io {
            message: format!("failed to read directory entry: {error}"),
        })?;
        children.push(entry.path());
    }
    children.sort();

    for child in children {
        if entries.len() >= max_entries {
            break;
        }
        let relative = root.strip_absolute(&child)?;
        if !include_hidden && is_hidden(&relative) {
            continue;
        }
        if ignore.is_ignored(&relative) {
            continue;
        }
        let metadata = std::fs::symlink_metadata(&child).map_err(|error| WorkspaceError::Io {
            message: format!("failed to inspect list entry: {error}"),
        })?;
        let kind = if metadata.file_type().is_symlink() {
            ResolvedPathKind::Symlink
        } else if metadata.is_dir() {
            ResolvedPathKind::Directory
        } else {
            ResolvedPathKind::File
        };
        entries.push(WorkspaceEntry {
            path: relative,
            kind,
            len: metadata.is_file().then_some(metadata.len()),
        });
        if recursive && kind == ResolvedPathKind::Directory {
            collect_entries(
                root,
                &child,
                true,
                include_hidden,
                ignore,
                max_entries,
                entries,
            )?;
        }
    }
    Ok(())
}

#[derive(Debug, Default)]
struct SimpleIgnore {
    patterns: Vec<String>,
}

impl SimpleIgnore {
    fn load(root: &Path) -> Self {
        let mut ignore = Self {
            patterns: vec![".git".to_string()],
        };
        let path = root.join(".gitignore");
        let Ok(text) = std::fs::read_to_string(path) else {
            return ignore;
        };
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('!') {
                continue;
            }
            ignore.patterns.push(line.trim_end_matches('/').to_string());
        }
        ignore
    }

    fn is_ignored(&self, path: &WorkspacePath) -> bool {
        let display = path.display();
        self.patterns.iter().any(|pattern| {
            display == *pattern
                || display.starts_with(&format!("{pattern}/"))
                || display
                    .split('/')
                    .any(|component| component == pattern.as_str())
        })
    }
}

fn is_hidden(path: &WorkspacePath) -> bool {
    path.display()
        .split('/')
        .any(|component| component.starts_with('.') && component != "." && !component.is_empty())
}

fn digest_bytes(bytes: &[u8]) -> ContentDigest {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    ContentDigest {
        algorithm: "fnv1a64".to_string(),
        value: format!("{hash:016x}"),
    }
}

fn diff_snapshots(before: &WorkspaceSnapshot, after: &WorkspaceSnapshot) -> WorkspaceDiff {
    let before_map: BTreeMap<_, _> = before
        .files
        .iter()
        .map(|version| (version.path.clone(), version))
        .collect();
    let after_map: BTreeMap<_, _> = after
        .files
        .iter()
        .map(|version| (version.path.clone(), version))
        .collect();
    let paths: BTreeSet<_> = before_map.keys().chain(after_map.keys()).cloned().collect();
    let mut changed = Vec::new();
    let mut lines = Vec::new();
    for path in paths {
        match (before_map.get(&path), after_map.get(&path)) {
            (None, Some(_)) => {
                changed.push(path.clone());
                lines.push(format!("created {}", path.display()));
            }
            (Some(_), None) => {
                changed.push(path.clone());
                lines.push(format!("deleted {}", path.display()));
            }
            (Some(left), Some(right)) if left.digest != right.digest || left.len != right.len => {
                changed.push(path.clone());
                lines.push(format!("modified {}", path.display()));
            }
            _ => {}
        }
    }
    WorkspaceDiff {
        changed_files: changed,
        text: lines.join("\n"),
        truncated: false,
        metadata: RunMetadata::new(),
    }
}

fn monotonic_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn generated_snapshot_id() -> String {
    format!("snapshot-{}-{:#x}", std::process::id(), monotonic_nanos())
}

/// Workspace operation errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
#[non_exhaustive]
pub enum WorkspaceError {
    /// The path syntax is invalid.
    #[error("invalid workspace path: {message}")]
    InvalidPath {
        /// Model-safe explanation.
        message: String,
    },
    /// The resolved path escapes the workspace root.
    #[error("path escapes workspace root: {path} is outside {root}")]
    OutsideRoot {
        /// Requested or resolved path.
        path: String,
        /// Workspace root.
        root: String,
    },
    /// A symlink points outside the workspace root.
    #[error("symlink escapes workspace root: {path} -> {target}")]
    SymlinkEscapesRoot {
        /// Symlink path.
        path: String,
        /// Canonical symlink target.
        target: String,
    },
    /// A required path was not found.
    #[error("workspace path not found: {path}: {message}")]
    NotFound {
        /// Workspace path.
        path: WorkspacePath,
        /// Model-safe explanation.
        message: String,
    },
    /// A write or patch conflict was detected.
    #[error("workspace conflict: {conflict:?}")]
    Conflict {
        /// Conflict detail.
        conflict: Box<PatchConflict>,
    },
    /// A path could not be displayed as UTF-8.
    #[error("workspace path is not UTF-8: {path}")]
    NonUtf8Path {
        /// Lossy path display.
        path: String,
    },
    /// I/O failed.
    #[error("workspace I/O error: {message}")]
    Io {
        /// Model-safe explanation.
        message: String,
    },
    /// Operation is not supported by this workspace.
    #[error("unsupported workspace operation: {message}")]
    Unsupported {
        /// Model-safe explanation.
        message: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "molo-coding-workspace-{}-{tag}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn workspace_path_rejects_escape() {
        assert!(WorkspacePath::parse("../secret").is_err());
        assert!(WorkspacePath::parse("/etc/passwd").is_err());
        assert!(WorkspacePath::parse("a//b").is_err());
        assert!(WorkspacePath::parse("./a").is_err());
        assert!(WorkspacePath::parse("a\\b").is_err());
        assert!(WorkspacePath::parse("a\0b").is_err());
        assert_eq!(
            WorkspacePath::parse("src/lib.rs").unwrap().display(),
            "src/lib.rs"
        );
    }

    #[test]
    fn workspace_path_rejects_escape_corpus() {
        let invalid = [
            "..",
            "../a",
            "a/../b",
            "a/./b",
            "./a",
            "/absolute",
            "a//b",
            "a\\b",
            "a/\0/b",
            "a/",
            "/",
        ];
        for candidate in invalid {
            assert!(
                WorkspacePath::parse(candidate).is_err(),
                "candidate must be rejected: {candidate:?}"
            );
        }

        let valid = ["", "src/lib.rs", "nested/path/file.txt", "unicode/你好.txt"];
        for candidate in valid {
            assert_eq!(
                WorkspacePath::parse(candidate).unwrap().display(),
                candidate
            );
        }
    }

    #[tokio::test]
    async fn local_workspace_rejects_outside_symlink() {
        let root = temp_dir("symlink");
        let outside = root.with_extension("outside");
        let _ = std::fs::remove_dir_all(&outside);
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret"), "secret").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(outside.join("secret"), root.join("link")).unwrap();
        #[cfg(windows)]
        std::os::windows::fs::symlink_file(outside.join("secret"), root.join("link")).unwrap();

        let workspace = LocalWorkspace::new(&root).unwrap();
        let err = workspace
            .read_file(
                &WorkspacePath::parse("link").unwrap(),
                FileReadOptions::default(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, WorkspaceError::SymlinkEscapesRoot { .. }));
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[tokio::test]
    async fn local_workspace_rejects_symlink_chain_escape() {
        let root = temp_dir("symlink-chain");
        let outside = root.with_extension("outside-chain");
        let _ = std::fs::remove_dir_all(&outside);
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret"), "secret").unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(outside.join("secret"), root.join("outside-link")).unwrap();
            std::os::unix::fs::symlink(root.join("outside-link"), root.join("chain-link")).unwrap();
        }
        #[cfg(windows)]
        {
            std::os::windows::fs::symlink_file(outside.join("secret"), root.join("outside-link"))
                .unwrap();
            std::os::windows::fs::symlink_file(root.join("outside-link"), root.join("chain-link"))
                .unwrap();
        }

        let workspace = LocalWorkspace::new(&root).unwrap();
        let err = workspace
            .read_file(
                &WorkspacePath::parse("chain-link").unwrap(),
                FileReadOptions::default(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, WorkspaceError::SymlinkEscapesRoot { .. }));
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(outside);
    }

    #[tokio::test]
    async fn write_stale_version_conflicts() {
        let root = temp_dir("stale");
        std::fs::write(root.join("a.txt"), "one").unwrap();
        let workspace = LocalWorkspace::new(&root).unwrap();
        let path = WorkspacePath::parse("a.txt").unwrap();
        let content = workspace
            .read_file(&path, FileReadOptions::default())
            .await
            .unwrap();
        std::fs::write(root.join("a.txt"), "two").unwrap();
        let err = workspace
            .write_file(WriteFileRequest {
                path,
                content: FileWriteContent::Text("three".to_string()),
                expected_version: Some(content.version),
                create: false,
                overwrite: true,
            })
            .await
            .unwrap_err();
        assert!(matches!(err, WorkspaceError::Conflict { .. }));
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn patch_is_all_or_nothing_on_conflict() {
        let root = temp_dir("patch");
        std::fs::write(root.join("a.txt"), "alpha").unwrap();
        std::fs::write(root.join("b.txt"), "bravo").unwrap();
        let workspace = LocalWorkspace::new(&root).unwrap();
        let result = workspace
            .apply_patch(PatchRequest {
                patch: Patch {
                    files: vec![
                        FilePatch {
                            path: WorkspacePath::parse("a.txt").unwrap(),
                            operation: PatchOperation::Modify,
                            expected_version: None,
                            hunks: vec![PatchHunk {
                                old_text: "alpha".to_string(),
                                new_text: "ALPHA".to_string(),
                            }],
                        },
                        FilePatch {
                            path: WorkspacePath::parse("b.txt").unwrap(),
                            operation: PatchOperation::Modify,
                            expected_version: None,
                            hunks: vec![PatchHunk {
                                old_text: "missing".to_string(),
                                new_text: "BRAVO".to_string(),
                            }],
                        },
                    ],
                    original_text: None,
                    metadata: RunMetadata::new(),
                },
                dry_run: false,
                allow_partial: false,
            })
            .await
            .unwrap();
        assert!(!result.applied);
        assert_eq!(
            std::fs::read_to_string(root.join("a.txt")).unwrap(),
            "alpha"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn patch_stale_precondition_reports_conflict_without_write() {
        let root = temp_dir("patch-stale");
        std::fs::write(root.join("a.txt"), "alpha").unwrap();
        let workspace = LocalWorkspace::new(&root).unwrap();
        let path = WorkspacePath::parse("a.txt").unwrap();
        let content = workspace
            .read_file(&path, FileReadOptions::default())
            .await
            .unwrap();
        std::fs::write(root.join("a.txt"), "changed by user").unwrap();

        let result = workspace
            .apply_patch(PatchRequest {
                patch: Patch {
                    files: vec![FilePatch {
                        path: path.clone(),
                        operation: PatchOperation::Modify,
                        expected_version: Some(content.version),
                        hunks: vec![PatchHunk {
                            old_text: "alpha".to_string(),
                            new_text: "ALPHA".to_string(),
                        }],
                    }],
                    original_text: None,
                    metadata: RunMetadata::new(),
                },
                dry_run: false,
                allow_partial: false,
            })
            .await
            .unwrap();

        assert!(!result.applied);
        assert_eq!(result.conflicts.len(), 1);
        assert_eq!(
            std::fs::read_to_string(root.join("a.txt")).unwrap(),
            "changed by user"
        );
        let _ = std::fs::remove_dir_all(root);
    }
}
