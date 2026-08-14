//! Coding-workload primitives for governed agents.
//!
//! This module is an SDK layer for applications that build coding agents on
//! top of the harness runtime. It provides root-relative workspace paths,
//! local workspace operations, typed effect payloads, command execution,
//! git inspection, repository search, project instruction resolution, and
//! context gathering. It is not a CLI product, and it does not bypass the
//! harness: production file, patch, command, git, or search side effects
//! should be represented as [`EffectRequest`](crate::EffectRequest) values
//! and executed through a [`Harness`](crate::harness::Harness).

mod command;
mod context;
mod effect_executor;
mod error;
mod git;
mod instructions;
mod payload;
mod search;
mod test_runner;
mod tools;
mod workspace;

pub use command::{
    CommandError, CommandExecutor, CommandExecutorCapabilities, CommandOutput, CommandOutputLimit,
    CommandRequest, CommandStatus, EnvPolicy, LocalCommandExecutor, OutputText,
    PolicyEnforcementReport, PtyMode,
};
pub use context::{
    CodingContextBundle, CodingContextError, CodingContextInclude, CodingContextProvider,
    CodingContextRequest, ContextBudget, DefaultCodingContextProvider, DependencyMetadata,
};
pub use effect_executor::{CodingEffectExecutor, CodingExecutorConfig};
pub use error::CodingError;
pub use git::{
    CliGitInspector, GitChangedFile, GitChangedFilesRequest, GitDiffRequest, GitError, GitHead,
    GitInspector, GitOperation, GitStatus, GitStatusRequest,
};
pub use instructions::{
    DefaultInstructionResolver, InstructionBundle, InstructionError, InstructionFile,
    InstructionFileSpec, InstructionRequest, InstructionResolver,
};
pub use payload::{
    ApplyPatchPayload, CommandPayload, GitPayload, ListFilesPayload, ReadFilePayload,
    SearchPayload, WriteFilePayload,
};
pub use search::{
    RepoSearchRequest, RepoSearchResults, RepoSearcher, RipgrepSearcher, SearchError, SearchMatch,
    SearchMode, WorkspaceSearcher,
};
pub use test_runner::{
    CommandTestRunner, TestRunError, TestRunRequest, TestRunner, VerificationResult,
};
pub use tools::{
    ApplyPatchTool, GitStatusTool, ListFilesTool, ReadFileTool, RunCommandTool, SearchRepoTool,
};
pub use workspace::{
    AgentChangeTracker, ContentDigest, DiffRequest, FileBody, FileContent, FilePatch,
    FileReadOptions, FileVersion, FileWriteContent, FileWriteResult, ListFilesQuery,
    LocalWorkspace, LocalWorkspaceConfig, Patch, PatchConflict, PatchHunk, PatchOperation,
    PatchRequest, PatchResult, ResolvedPath, ResolvedPathKind, SnapshotRequest, SymlinkPolicy,
    TextEncoding, Workspace, WorkspaceAccess, WorkspaceDiff, WorkspaceEntry, WorkspaceError,
    WorkspacePath, WorkspaceRoot, WorkspaceSnapshot, WriteFileRequest,
};
