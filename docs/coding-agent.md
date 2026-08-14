# Coding Agents

The `coding` feature provides primitives for building coding-agent products on
top of molo. It is an SDK layer, not a CLI product: applications still own UX,
configuration, approval UI, session storage, final prompts, and model choice.

Enable it explicitly:

```toml
molo = { version = "0.3", features = ["coding"] }
```

`coding` depends on `harness` because production file, patch, command, git, and
search side effects should be governed before execution.

## Workspace

Use `WorkspacePath` for every model- or user-provided path. It accepts only
root-relative paths and rejects absolute paths, traversal, empty components,
NUL bytes, and platform prefixes.

`LocalWorkspace` canonicalizes its root and checks existing paths, new-file
parents, and symlink targets before reading or writing:

```rust
use molo::{
    FileReadOptions, FileWriteContent, LocalWorkspace, Workspace, WorkspacePath,
    WriteFileRequest,
};

# #[tokio::main(flavor = "current_thread")]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
let workspace = LocalWorkspace::new(std::env::current_dir()?)?;
let path = WorkspacePath::parse("target/example.txt")?;

workspace
    .write_file(WriteFileRequest {
        path: path.clone(),
        content: FileWriteContent::Text("hello\n".to_string()),
        expected_version: None,
        create: true,
        overwrite: true,
    })
    .await?;

let content = workspace
    .read_file(&path, FileReadOptions::default())
    .await?;
println!("{} bytes", content.version.len);
# Ok(())
# }
```

Writes and patches support version preconditions through `FileVersion` so a
host can detect stale edits instead of silently overwriting user changes.

## Effects

Model-visible tools should build typed payloads and return `EffectRequest`
instead of touching the filesystem or process table directly:

```rust
use molo::{ReadFilePayload, ToolResult, WorkspacePath};

# fn build() -> Result<ToolResult, Box<dyn std::error::Error>> {
let effect = ReadFilePayload {
    path: WorkspacePath::parse("Cargo.toml")?,
    max_bytes: Some(4096),
}
.into_effect()?;

Ok(ToolResult::Effect(effect))
# }
```

`CodingEffectExecutor` decodes those payloads after `BasicHarness` policy,
approval, audit, transcript, output limiting, and redaction:

```rust
use molo::{
    AlwaysAllowApprovalBroker, BasicHarness, CodingEffectExecutor, LocalCommandExecutor,
    LocalWorkspace, RunContext, WorkspaceSearcher,
};

# #[tokio::main(flavor = "current_thread")]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
let workspace = LocalWorkspace::new(std::env::current_dir()?)?;
let commands = LocalCommandExecutor::new(workspace.clone());
let git = molo::CliGitInspector::new(commands.clone());
let searcher = WorkspaceSearcher::new(workspace.clone());
let executor = CodingEffectExecutor::new(workspace, commands, git, searcher);

let harness = BasicHarness::new(
    executor,
    molo::DefaultPolicyEngine,
    AlwaysAllowApprovalBroker,
    molo::NoopAuditSink,
    molo::NoopTranscriptStore,
);

let _context = RunContext::new("coding-run");
# Ok(())
# }
```

The full runtime shape is:

```text
tool payload -> EffectRequest -> Harness policy/approval/audit -> CodingEffectExecutor
```

## Commands And Git

`CommandRequest` uses `argv: Vec<String>` and does not perform implicit shell
parsing. Shell syntax is an explicit command such as `["sh", "-c", "..."]`,
which is classified as higher risk by the typed payload builder.

`LocalCommandExecutor` is a non-PTY, one-shot local executor. It resolves
`cwd` through the workspace, uses explicit environment policy, requires a
timeout, captures stdout/stderr separately, and reports truncation. OS-level
sandbox and network enforcement are host-dependent; unsupported policy fails
closed unless the host opts into advisory mode.

`LocalCommandExecutor` is not an OS sandbox. It is useful for local prototypes,
tests, and reference CLI dogfooding, but production coding-agent products should
inject a host-provided `CommandExecutor` that can actually enforce the requested
filesystem, network, process, and resource boundaries.

The production boundary is the trait, not the local executor:

```rust
use molo::{
    CommandError, CommandExecutor, CommandExecutorCapabilities, CommandOutput,
    CommandRequest, ExecutionPolicy, RunContext,
};

#[derive(Debug)]
struct IsolatedCommandExecutor {
    // Store handles for a container, VM, remote worker, or platform sandbox.
}

#[molo::async_trait]
impl CommandExecutor for IsolatedCommandExecutor {
    fn capabilities(&self) -> CommandExecutorCapabilities {
        CommandExecutorCapabilities {
            one_shot: true,
            pty: false,
            sandbox_enforcement: true,
            network_enforcement: true,
        }
    }

    async fn execute(
        &self,
        request: CommandRequest,
        policy: &ExecutionPolicy,
        context: &RunContext,
    ) -> Result<CommandOutput, CommandError> {
        // Run `request.argv` in the isolated backend, enforce `policy`, honor
        // `context` cancellation/deadline, then return stdout/stderr and the
        // policy enforcement report.
        todo!("host executor implementation")
    }
}
```

A production executor should fail closed when it cannot enforce the requested
`SandboxPolicy` or `NetworkPolicy`, unless the host explicitly opts into an
advisory mode and records that downgrade in audit/transcript output. It should
also report executor identity/version, cwd and mount/write roots, inherited env
keys, timeout/resource limits, output limits, and whether cancellation/timeout
terminated the full process tree.

`GitInspector` is read-only in the coding SDK. Mutating git operations such as
commit, checkout, reset, push, or force-push should be represented as command
effects and governed by policy/approval.

## Context

Repository context is explicit and separate from chat memory:

- `RepoSearcher` returns structured matches instead of raw stdout.
- `InstructionResolver` reads configured instruction files such as `AGENTS.md`
  from root to target path. Instruction content is context data, not a security
  boundary.
- `CodingContextProvider` can gather project instructions, repo tree entries,
  search matches, git status, changed files, dependency manifest hints, test
  failures, and a transcript summary under a context budget.

Project files and tool output remain untrusted data. They cannot lower harness
policy, approval, sandbox, or network requirements.
