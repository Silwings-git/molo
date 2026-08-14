//! Phase 9 benchmarks for coding-workload primitives.
//!
//! These benches operate on generated temporary fixtures only.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use molo::{
    AlwaysAllowApprovalBroker, BasicHarness, CliGitInspector, CodingEffectExecutor,
    CommandExecutor, CommandOutputLimit, CommandRequest, DefaultPolicyEngine, FilePatch,
    FileWriteContent, Harness, HarnessConfig, LocalCommandExecutor, LocalWorkspace, NetworkPolicy,
    Patch, PatchHunk, PatchOperation, PatchRequest, RepoSearchRequest, RepoSearcher, RunContext,
    RunMetadata, SandboxPolicy, Workspace, WorkspacePath, WorkspaceSearcher, WriteFilePayload,
};
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

type CodingBenchExecutor = CodingEffectExecutor<
    LocalWorkspace,
    LocalCommandExecutor<LocalWorkspace>,
    CliGitInspector<LocalCommandExecutor<LocalWorkspace>>,
    WorkspaceSearcher<LocalWorkspace>,
>;

type CodingBenchHarness = BasicHarness<
    CodingBenchExecutor,
    DefaultPolicyEngine,
    AlwaysAllowApprovalBroker,
    molo::NoopAuditSink,
    molo::NoopTranscriptStore,
>;

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("benchmark runtime")
}

fn temp_root(tag: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let root =
        std::env::temp_dir().join(format!("molo-bench-{tag}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(root.join("src")).expect("create fixture directory");
    root
}

fn patch_fixture() -> (PathBuf, LocalWorkspace, PatchRequest) {
    let root = temp_root("patch");
    fs::write(
        root.join("src/lib.rs"),
        "pub fn answer() -> i32 {\n    41\n}\n",
    )
    .expect("write fixture");
    let workspace = LocalWorkspace::new(&root).expect("workspace");
    let request = PatchRequest {
        patch: Patch {
            files: vec![FilePatch {
                path: WorkspacePath::parse("src/lib.rs").expect("path"),
                operation: PatchOperation::Modify,
                expected_version: None,
                hunks: vec![PatchHunk {
                    old_text: "41".to_string(),
                    new_text: "42".to_string(),
                }],
            }],
            original_text: None,
            metadata: RunMetadata::new(),
        },
        dry_run: false,
        allow_partial: false,
    };
    (root, workspace, request)
}

fn bench_patch_apply(c: &mut Criterion) {
    let runtime = rt();
    let mut group = c.benchmark_group("coding_patch");
    group.bench_function("local_workspace_small_patch", |b| {
        b.iter_batched(
            patch_fixture,
            |(root, workspace, request)| {
                runtime.block_on(async {
                    black_box(workspace.apply_patch(request).await.expect("apply patch"));
                });
                let _ = fs::remove_dir_all(root);
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn command_fixture() -> (PathBuf, LocalWorkspace, CommandRequest) {
    let root = temp_root("command");
    let workspace = LocalWorkspace::new(&root).expect("workspace");
    let mut request = CommandRequest::new(["sh", "-c", "yes x | head -c 200000"]);
    request.output_limit = CommandOutputLimit {
        stdout_bytes: 4096,
        stderr_bytes: 1024,
    };
    request.timeout = Some(Duration::from_secs(5));
    (root, workspace, request)
}

fn bench_command_output_truncation(c: &mut Criterion) {
    let runtime = rt();
    let mut group = c.benchmark_group("coding_command");
    group.bench_function("local_command_large_stdout_truncated", |b| {
        b.iter_batched(
            command_fixture,
            |(root, workspace, request)| {
                runtime.block_on(async {
                    let executor = LocalCommandExecutor::new(workspace);
                    let output = executor
                        .execute(
                            request,
                            &molo::ExecutionPolicy {
                                sandbox: SandboxPolicy::ReadOnly,
                                network: NetworkPolicy::Deny,
                                timeout: Some(Duration::from_secs(5)),
                                output_limit: molo::OutputLimit::default(),
                            },
                            &RunContext::new("bench-command"),
                        )
                        .await
                        .expect("execute command");
                    black_box(output);
                });
                let _ = fs::remove_dir_all(root);
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn search_fixture() -> (PathBuf, LocalWorkspace, RepoSearchRequest) {
    let root = temp_root("search");
    for i in 0..200 {
        fs::write(
            root.join("src").join(format!("file_{i}.rs")),
            format!("pub fn f_{i}() -> &'static str {{ \"needle-{i}\" }}\n"),
        )
        .expect("write fixture");
    }
    let workspace = LocalWorkspace::new(&root).expect("workspace");
    let request = RepoSearchRequest {
        query: "needle-199".to_string(),
        paths: vec![WorkspacePath::parse("src").expect("path")],
        max_matches: 10,
        ..RepoSearchRequest::literal("needle-199")
    };
    (root, workspace, request)
}

fn bench_workspace_search(c: &mut Criterion) {
    let runtime = rt();
    let mut group = c.benchmark_group("coding_search");
    group.bench_function("workspace_search_200_files", |b| {
        b.iter_batched(
            search_fixture,
            |(root, workspace, request)| {
                runtime.block_on(async {
                    let searcher = WorkspaceSearcher::new(workspace);
                    black_box(
                        searcher
                            .search(request, &RunContext::new("bench-search"))
                            .await
                            .expect("search fixture"),
                    );
                });
                let _ = fs::remove_dir_all(root);
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn coding_effect_fixture() -> (PathBuf, CodingBenchHarness, molo::EffectRequest, RunContext) {
    let root = temp_root("effect");
    let workspace = LocalWorkspace::new(&root).expect("workspace");
    let commands = LocalCommandExecutor::new(workspace.clone());
    let git = CliGitInspector::new(commands.clone());
    let searcher = WorkspaceSearcher::new(workspace.clone());
    let executor = CodingEffectExecutor::new(workspace, commands, git, searcher);
    let harness = BasicHarness::new(
        executor,
        DefaultPolicyEngine,
        AlwaysAllowApprovalBroker,
        molo::NoopAuditSink,
        molo::NoopTranscriptStore,
    )
    .with_config(HarnessConfig {
        default_sandbox: SandboxPolicy::WorkspaceWrite,
        default_network: NetworkPolicy::Deny,
        default_timeout: Duration::from_secs(5),
        ..HarnessConfig::default()
    });
    let effect = WriteFilePayload {
        path: WorkspacePath::parse("src/generated.rs").expect("path"),
        content: FileWriteContent::Text("pub const VALUE: i32 = 42;\n".to_string()),
        expected_version: None,
        create: true,
        overwrite: false,
    }
    .into_effect()
    .expect("write effect")
    .with_id("write-effect");
    (
        root,
        harness,
        effect,
        RunContext::new("bench-coding-effect"),
    )
}

fn bench_coding_effect_adapter(c: &mut Criterion) {
    let runtime = rt();
    let mut group = c.benchmark_group("coding_effect_adapter");
    group.bench_function("write_file_through_harness", |b| {
        b.iter_batched(
            coding_effect_fixture,
            |(root, harness, effect, context)| {
                runtime.block_on(async {
                    black_box(
                        harness
                            .execute(effect, &context)
                            .await
                            .expect("execute coding effect"),
                    );
                });
                let _ = fs::remove_dir_all(root);
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_patch_apply,
    bench_command_output_truncation,
    bench_workspace_search,
    bench_coding_effect_adapter
);
criterion_main!(benches);
