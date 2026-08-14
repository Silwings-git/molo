use molo::{
    ApplyPatchPayload, BasicHarness, CliGitInspector, CodingEffectExecutor, CommandPayload,
    CommandRequest, CommandStatus, CommandTestRunner, DefaultPolicyEngine, EffectObservation,
    EffectStatus, ExecutionPolicy, FilePatch, FileWriteContent, GitChangedFilesRequest,
    GitInspector, Harness, HarnessConfig, LocalCommandExecutor, LocalWorkspace, NetworkPolicy,
    OutputLimit, Patch, PatchHunk, PatchOperation, PatternRedactor, ReadFilePayload, RunContext,
    RunMetadata, SandboxPolicy, StaticApprovalBroker, TestRunRequest, TestRunner, VecAuditSink,
    VecTranscriptStore, WorkspacePath, WorkspaceSearcher, WriteFilePayload,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const RESULT_SCHEMA_VERSION: u16 = 1;
const RUNNER_VERSION: &str = env!("CARGO_PKG_VERSION");

#[tokio::main(flavor = "current_thread")]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("molo-eval-runner: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    match Args::parse(env::args().skip(1).collect())? {
        Args::ValidateDir { dir } => {
            let manifests = find_manifests(&dir).map_err(|error| error.to_string())?;
            if manifests.is_empty() {
                return Err(format!("no eval manifests found under {}", dir.display()));
            }
            for manifest in manifests {
                let case = read_case(&manifest)?;
                validate_case(&manifest, &case)?;
                println!("valid {}", case.id);
            }
            Ok(())
        }
        Args::Run {
            manifest,
            output,
            keep_workspace,
        } => {
            let case = read_case(&manifest)?;
            validate_case(&manifest, &case)?;
            let output = match output {
                Some(path) => path,
                None => default_output_path(&case),
            };
            let record = run_case(&manifest, &case, keep_workspace).await?;
            write_result(&output, &record)?;
            println!(
                "{} {:?} {}",
                record.case_id,
                record.verdict,
                output.display()
            );
            Ok(())
        }
    }
}

enum Args {
    ValidateDir {
        dir: PathBuf,
    },
    Run {
        manifest: PathBuf,
        output: Option<PathBuf>,
        keep_workspace: bool,
    },
}

impl Args {
    fn parse(args: Vec<String>) -> Result<Self, String> {
        if args.is_empty() || args.iter().any(|arg| arg == "--help" || arg == "-h") {
            return Err(usage());
        }
        let mut validate_dir = None;
        let mut manifest = None;
        let mut output = None;
        let mut keep_workspace = false;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--validate-dir" => {
                    i += 1;
                    validate_dir =
                        Some(PathBuf::from(args.get(i).ok_or_else(|| {
                            "--validate-dir requires a path".to_string()
                        })?));
                }
                "--manifest" => {
                    i += 1;
                    manifest = Some(PathBuf::from(
                        args.get(i)
                            .ok_or_else(|| "--manifest requires a path".to_string())?,
                    ));
                }
                "--output" => {
                    i += 1;
                    output = Some(PathBuf::from(
                        args.get(i)
                            .ok_or_else(|| "--output requires a path".to_string())?,
                    ));
                }
                "--keep-workspace" => keep_workspace = true,
                other => return Err(format!("unknown argument {other}\n{}", usage())),
            }
            i += 1;
        }
        match (validate_dir, manifest) {
            (Some(dir), None) => Ok(Self::ValidateDir { dir }),
            (None, Some(manifest)) => Ok(Self::Run {
                manifest,
                output,
                keep_workspace,
            }),
            _ => Err(usage()),
        }
    }
}

fn usage() -> String {
    "usage: molo-eval-runner --validate-dir evals/cases/coding\n       molo-eval-runner --manifest evals/cases/coding/edit-function/eval.json [--output evals/results/run.json] [--keep-workspace]".to_string()
}

#[derive(Debug, Clone, Deserialize)]
struct EvalCase {
    schema_version: u16,
    id: String,
    title: String,
    kind: EvalKind,
    fixture_version: String,
    task_file: String,
    fixture_dir: String,
    provider: EvalProviderSummary,
    policy: EvalPolicySummary,
    setup: EvalSetup,
    actions: Vec<EvalAction>,
    expected_changed_files: Vec<String>,
    forbidden_changed_files: Vec<String>,
    verification_commands: Vec<EvalVerificationCommand>,
    success_criteria: Vec<String>,
    failure_categories: Vec<EvalFailureCategory>,
    timeout_ms: u64,
    raw_capture: bool,
    redaction_patterns: Vec<String>,
    #[serde(default)]
    min_redactions_applied: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum EvalKind {
    Deterministic,
    Safety,
    ModelInLoop,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
struct EvalProviderSummary {
    kind: String,
    script_id: Option<String>,
    model_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
struct EvalPolicySummary {
    sandbox: String,
    network: String,
    approval: EvalApprovalMode,
    timeout_ms: u64,
    output_model_bytes: usize,
    output_display_bytes: usize,
    output_debug_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum EvalApprovalMode {
    AllowOnce,
    AllowForSession,
    Deny,
}

impl EvalApprovalMode {
    fn decision(self) -> molo::ApprovalDecision {
        match self {
            Self::AllowOnce => molo::ApprovalDecision::AllowOnce,
            Self::AllowForSession => molo::ApprovalDecision::AllowForSession,
            Self::Deny => molo::ApprovalDecision::Deny {
                reason: "denied by eval approval script".to_string(),
            },
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct EvalSetup {
    #[serde(default)]
    pre_existing_dirty_files: Vec<EvalDirtyFile>,
}

#[derive(Debug, Clone, Deserialize)]
struct EvalDirtyFile {
    path: String,
    content: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum EvalAction {
    ReadFile {
        path: String,
    },
    WriteFile {
        path: String,
        content: String,
        #[serde(default)]
        create: bool,
        #[serde(default = "default_true")]
        overwrite: bool,
    },
    ApplyPatch {
        path: String,
        old_text: String,
        new_text: String,
    },
    RunCommand {
        argv: Vec<String>,
        expect: EvalEffectExpectation,
    },
    ResumeMarker {
        previous_run_id: String,
    },
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum EvalEffectExpectation {
    Succeeded,
    Denied,
    Failed,
    Any,
}

#[derive(Debug, Clone, Deserialize)]
struct EvalVerificationCommand {
    name: String,
    argv: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum EvalVerdict {
    Pass,
    Partial,
    Fail,
    Invalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum EvalFailureCategory {
    TaskMisunderstood,
    ContextMissed,
    WrongEdit,
    VerificationNotRun,
    VerificationFailed,
    CompileFailed,
    TestFailed,
    PolicyDeniedExpected,
    PolicyDeniedUnexpected,
    ApprovalRequired,
    ApprovalBypass,
    WorkspaceEscape,
    DirtyWorktreeOverwrite,
    DestructiveCommandRequested,
    Timeout,
    ProviderError,
    ToolError,
    HarnessError,
    TranscriptError,
    AuditError,
    RedactionFailure,
    NonDeterministic,
    InfraFailure,
    HumanInterventionRequired,
}

#[derive(Debug, Serialize)]
struct EvalRunRecord {
    schema_version: u16,
    case_id: String,
    case_version: String,
    run_id: String,
    session_id: Option<String>,
    molo_git_commit: String,
    runner_version: String,
    started_at: String,
    finished_at: String,
    provider: EvalProviderSummary,
    policy: EvalPolicySummary,
    verdict: EvalVerdict,
    failure_category: Option<EvalFailureCategory>,
    failure_detail: Option<String>,
    model_requests: u32,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    wall_time_ms: u64,
    effects: EvalEffectSummary,
    approvals: EvalApprovalSummary,
    commands: Vec<EvalCommandSummary>,
    changed_files: Vec<String>,
    pre_existing_dirty_files: Vec<String>,
    agent_touched_files: Vec<String>,
    forbidden_files_touched: Vec<String>,
    verification: Vec<EvalVerificationSummary>,
    transcript_digest: Option<String>,
    audit_digest: Option<String>,
    final_summary_digest: Option<String>,
    redactions_applied: u32,
}

#[derive(Debug, Default, Serialize)]
struct EvalEffectSummary {
    total: u32,
    succeeded: u32,
    denied: u32,
    failed: u32,
    timed_out: u32,
    cancelled: u32,
}

#[derive(Debug, Default, Serialize)]
struct EvalApprovalSummary {
    requested: u32,
    allowed: u32,
    denied: u32,
}

#[derive(Debug, Serialize)]
struct EvalCommandSummary {
    name: String,
    argv_digest: String,
    status: String,
    exit_code: Option<i32>,
    stdout_digest: Option<String>,
    stderr_digest: Option<String>,
    stdout_bytes: Option<usize>,
    stderr_bytes: Option<usize>,
}

#[derive(Debug, Serialize)]
struct EvalVerificationSummary {
    name: String,
    passed: bool,
    status: String,
    exit_code: Option<i32>,
    stdout_digest: Option<String>,
    stderr_digest: Option<String>,
}

struct EvalExecution {
    effects: EvalEffectSummary,
    commands: Vec<EvalCommandSummary>,
    redactions_applied: u32,
    failure_category: Option<EvalFailureCategory>,
    failure_detail: Option<String>,
    resume_markers: u32,
}

async fn run_case(
    manifest_path: &Path,
    case: &EvalCase,
    keep_workspace: bool,
) -> Result<EvalRunRecord, String> {
    let started = Instant::now();
    let started_at = timestamp();
    let run_id = format!("eval-{}-{}", case.id, epoch_millis());
    let manifest_dir = manifest_path
        .parent()
        .ok_or_else(|| "manifest path has no parent".to_string())?;
    let fixture_src = manifest_dir.join(&case.fixture_dir);
    let workspace_root = temp_workspace(&case.id)?;
    copy_dir_all(&fixture_src, &workspace_root).map_err(|error| {
        format!(
            "failed to copy fixture {} to {}: {error}",
            fixture_src.display(),
            workspace_root.display()
        )
    })?;
    init_git_baseline(&workspace_root)?;

    let pre_existing_dirty_files = apply_dirty_setup(&workspace_root, &case.setup)?;
    let workspace = LocalWorkspace::new(&workspace_root).map_err(|error| error.to_string())?;
    let commands = LocalCommandExecutor::new(workspace.clone());
    let git = CliGitInspector::new(commands.clone());
    let searcher = WorkspaceSearcher::new(workspace.clone());
    let audit = VecAuditSink::new();
    let transcript = VecTranscriptStore::new();
    let executor =
        CodingEffectExecutor::new(workspace.clone(), commands.clone(), git.clone(), searcher);
    let harness = BasicHarness::new(
        executor,
        DefaultPolicyEngine,
        StaticApprovalBroker::new(case.policy.approval.decision()),
        audit.clone(),
        transcript.clone(),
    )
    .with_config(
        HarnessConfig::default()
            .with_default_sandbox(sandbox_policy(&case.policy.sandbox)?)
            .with_default_network(network_policy(&case.policy.network)?)
            .with_default_timeout(Duration::from_millis(case.policy.timeout_ms))
            .with_output_limit(OutputLimit::new(
                case.policy.output_model_bytes,
                case.policy.output_display_bytes,
                case.policy.output_debug_bytes,
            )),
    )
    .with_redactor(PatternRedactor::new(case.redaction_patterns.clone()));

    let context =
        RunContext::new(run_id.clone()).with_timeout(Duration::from_millis(case.timeout_ms));
    let mut execution = EvalExecution {
        effects: EvalEffectSummary::default(),
        commands: Vec::new(),
        redactions_applied: 0,
        failure_category: None,
        failure_detail: None,
        resume_markers: 0,
    };

    for action in &case.actions {
        if let Err(error) = execute_action(action, &harness, &context, &mut execution).await {
            set_failure(
                &mut execution,
                EvalFailureCategory::HarnessError,
                sanitize(&error, &case.redaction_patterns),
            );
        }
    }

    let verification = run_verification(&case.verification_commands, commands, &context).await?;
    let changed_files = git
        .changed_files(
            GitChangedFilesRequest {
                include_untracked: true,
            },
            &context,
        )
        .await
        .map_err(|error| error.to_string())?
        .into_iter()
        .map(|file| file.path.display())
        .collect::<Vec<_>>();
    let agent_touched_files = workspace
        .change_tracker()
        .changed_files()
        .into_iter()
        .map(|path| path.display())
        .collect::<Vec<_>>();
    let forbidden_files_touched = intersection(&agent_touched_files, &case.forbidden_changed_files);
    if !forbidden_files_touched.is_empty() {
        set_failure(
            &mut execution,
            EvalFailureCategory::DirtyWorktreeOverwrite,
            format!(
                "agent touched forbidden files: {}",
                forbidden_files_touched.join(", ")
            ),
        );
    }
    let missing_expected = missing(&agent_touched_files, &case.expected_changed_files);
    if !missing_expected.is_empty() {
        set_failure(
            &mut execution,
            EvalFailureCategory::WrongEdit,
            format!(
                "missing expected changed files: {}",
                missing_expected.join(", ")
            ),
        );
    }
    if verification.is_empty() && !case.verification_commands.is_empty() {
        set_failure(
            &mut execution,
            EvalFailureCategory::VerificationNotRun,
            "verification commands were configured but not recorded".to_string(),
        );
    }
    if verification.iter().any(|result| !result.passed) {
        set_failure(
            &mut execution,
            EvalFailureCategory::VerificationFailed,
            "one or more verification commands failed".to_string(),
        );
    }
    if execution.redactions_applied < case.min_redactions_applied {
        let detail = format!(
            "expected at least {} redaction(s), observed {}",
            case.min_redactions_applied, execution.redactions_applied
        );
        set_failure(
            &mut execution,
            EvalFailureCategory::RedactionFailure,
            detail,
        );
    }

    let audit_events = audit.events();
    let approvals = approval_summary(&audit_events);
    let transcript_records = transcript.records();
    let audit_digest = digest_json(&audit_events)?;
    let transcript_digest = digest_json(&transcript_records)?;
    let final_summary = serde_json::json!({
        "case_id": &case.id,
        "changed_files": &changed_files,
        "agent_touched_files": &agent_touched_files,
        "verification": &verification,
        "effects": &execution.effects,
        "resume_markers": execution.resume_markers
    });
    let final_summary_digest = digest_value(&final_summary);

    let verdict = if execution.failure_category.is_none() {
        EvalVerdict::Pass
    } else {
        EvalVerdict::Fail
    };
    let mut record = EvalRunRecord {
        schema_version: RESULT_SCHEMA_VERSION,
        case_id: case.id.clone(),
        case_version: case.fixture_version.clone(),
        run_id,
        session_id: None,
        molo_git_commit: git_commit(),
        runner_version: RUNNER_VERSION.to_string(),
        started_at,
        finished_at: timestamp(),
        provider: case.provider.clone(),
        policy: case.policy.clone(),
        verdict,
        failure_category: execution.failure_category,
        failure_detail: execution.failure_detail,
        model_requests: 0,
        input_tokens: None,
        output_tokens: None,
        wall_time_ms: started.elapsed().as_millis() as u64,
        effects: execution.effects,
        approvals,
        commands: execution.commands,
        changed_files: changed_files.clone(),
        pre_existing_dirty_files,
        agent_touched_files: agent_touched_files.clone(),
        forbidden_files_touched,
        verification,
        transcript_digest: Some(transcript_digest),
        audit_digest: Some(audit_digest),
        final_summary_digest: Some(final_summary_digest),
        redactions_applied: execution.redactions_applied,
    };

    let serialized = serde_json::to_string(&record).map_err(|error| error.to_string())?;
    if !case.raw_capture
        && case
            .redaction_patterns
            .iter()
            .any(|pattern| !pattern.is_empty() && serialized.contains(pattern))
    {
        record.verdict = EvalVerdict::Fail;
        record.failure_category = Some(EvalFailureCategory::RedactionFailure);
        record.failure_detail = Some("redaction pattern appeared in result record".to_string());
    }

    if !keep_workspace {
        let _ = fs::remove_dir_all(&workspace_root);
    } else {
        eprintln!("kept eval workspace at {}", workspace_root.display());
    }
    Ok(record)
}

async fn execute_action<H: Harness>(
    action: &EvalAction,
    harness: &H,
    context: &RunContext,
    execution: &mut EvalExecution,
) -> Result<(), String> {
    match action {
        EvalAction::ReadFile { path } => {
            let effect = ReadFilePayload {
                path: WorkspacePath::parse(path).map_err(|error| error.to_string())?,
                max_bytes: Some(64 * 1024),
            }
            .into_effect()
            .map_err(|error| error.to_string())?;
            let observation = harness
                .execute(effect, context)
                .await
                .map_err(|error| error.to_string())?;
            record_effect_observation(execution, &observation, EvalEffectExpectation::Succeeded);
        }
        EvalAction::WriteFile {
            path,
            content,
            create,
            overwrite,
        } => {
            let effect = WriteFilePayload {
                path: WorkspacePath::parse(path).map_err(|error| error.to_string())?,
                content: FileWriteContent::Text(content.clone()),
                expected_version: None,
                create: *create,
                overwrite: *overwrite,
            }
            .into_effect()
            .map_err(|error| error.to_string())?;
            let observation = harness
                .execute(effect, context)
                .await
                .map_err(|error| error.to_string())?;
            record_effect_observation(execution, &observation, EvalEffectExpectation::Succeeded);
        }
        EvalAction::ApplyPatch {
            path,
            old_text,
            new_text,
        } => {
            let effect = ApplyPatchPayload {
                patch: Patch {
                    files: vec![FilePatch {
                        path: WorkspacePath::parse(path).map_err(|error| error.to_string())?,
                        operation: PatchOperation::Modify,
                        expected_version: None,
                        hunks: vec![PatchHunk {
                            old_text: old_text.clone(),
                            new_text: new_text.clone(),
                        }],
                    }],
                    original_text: None,
                    metadata: RunMetadata::new(),
                },
                expected_versions: Vec::new(),
                dry_run: false,
            }
            .into_effect()
            .map_err(|error| error.to_string())?;
            let observation = harness
                .execute(effect, context)
                .await
                .map_err(|error| error.to_string())?;
            record_effect_observation(execution, &observation, EvalEffectExpectation::Succeeded);
        }
        EvalAction::RunCommand { argv, expect } => {
            let mut request = CommandRequest::new(argv.clone());
            request.timeout = Some(Duration::from_secs(5));
            let effect = CommandPayload { request }
                .into_effect()
                .map_err(|error| error.to_string())?;
            let observation = harness
                .execute(effect, context)
                .await
                .map_err(|error| error.to_string())?;
            execution.commands.push(EvalCommandSummary {
                name: "action".to_string(),
                argv_digest: digest_value(&serde_json::json!(argv)),
                status: format!("{:?}", observation.status),
                exit_code: None,
                stdout_digest: None,
                stderr_digest: None,
                stdout_bytes: None,
                stderr_bytes: None,
            });
            record_effect_observation(execution, &observation, *expect);
        }
        EvalAction::ResumeMarker { previous_run_id } => {
            if previous_run_id.is_empty() {
                set_failure(
                    execution,
                    EvalFailureCategory::TranscriptError,
                    "resume marker has an empty previous_run_id".to_string(),
                );
            }
            execution.resume_markers += 1;
        }
    }
    Ok(())
}

fn record_effect_observation(
    execution: &mut EvalExecution,
    observation: &EffectObservation,
    expect: EvalEffectExpectation,
) {
    execution.effects.total += 1;
    match &observation.status {
        EffectStatus::Succeeded => execution.effects.succeeded += 1,
        EffectStatus::Denied => execution.effects.denied += 1,
        EffectStatus::Failed => execution.effects.failed += 1,
        EffectStatus::TimedOut => execution.effects.timed_out += 1,
        EffectStatus::Cancelled => execution.effects.cancelled += 1,
        _ => execution.effects.failed += 1,
    }
    execution.redactions_applied += observation
        .metadata
        .get("redactions_applied")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as u32;
    let matched = match expect {
        EvalEffectExpectation::Succeeded => observation.status == EffectStatus::Succeeded,
        EvalEffectExpectation::Denied => observation.status == EffectStatus::Denied,
        EvalEffectExpectation::Failed => observation.status == EffectStatus::Failed,
        EvalEffectExpectation::Any => true,
    };
    if !matched {
        let category = match expect {
            EvalEffectExpectation::Denied => EvalFailureCategory::ApprovalBypass,
            EvalEffectExpectation::Succeeded => EvalFailureCategory::HarnessError,
            EvalEffectExpectation::Failed | EvalEffectExpectation::Any => {
                EvalFailureCategory::VerificationFailed
            }
        };
        set_failure(
            execution,
            category,
            format!(
                "effect {} had status {:?}, expected {:?}",
                observation.effect_id, observation.status, expect
            ),
        );
    }
}

async fn run_verification(
    commands: &[EvalVerificationCommand],
    executor: LocalCommandExecutor<LocalWorkspace>,
    context: &RunContext,
) -> Result<Vec<EvalVerificationSummary>, String> {
    let runner = CommandTestRunner::new(executor);
    let mut results = Vec::new();
    for command in commands {
        let mut request = CommandRequest::new(command.argv.clone());
        request.timeout = Some(Duration::from_secs(10));
        let result = runner
            .run(
                TestRunRequest {
                    command: request,
                    name: command.name.clone(),
                },
                &ExecutionPolicy::new(SandboxPolicy::ReadOnly, NetworkPolicy::Deny)
                    .with_timeout(Some(Duration::from_secs(10)))
                    .with_output_limit(OutputLimit::default()),
                context,
            )
            .await
            .map_err(|error| error.to_string())?;
        let output = result.output.as_ref();
        results.push(EvalVerificationSummary {
            name: result.name,
            passed: result.passed,
            status: output
                .map(|output| format!("{:?}", output.status))
                .unwrap_or_else(|| "missing_output".to_string()),
            exit_code: output.and_then(|output| exit_code(&output.status)),
            stdout_digest: output.map(|output| digest_bytes(output.stdout.text.as_bytes())),
            stderr_digest: output.map(|output| digest_bytes(output.stderr.text.as_bytes())),
        });
    }
    Ok(results)
}

fn approval_summary(events: &[molo::AuditEvent]) -> EvalApprovalSummary {
    let mut summary = EvalApprovalSummary::default();
    for event in events {
        match event {
            molo::AuditEvent::ApprovalRequested { .. } => summary.requested += 1,
            molo::AuditEvent::ApprovalDecided { decision, .. } => {
                if decision.contains("deny") {
                    summary.denied += 1;
                } else {
                    summary.allowed += 1;
                }
            }
            _ => {}
        }
    }
    summary
}

fn set_failure(execution: &mut EvalExecution, category: EvalFailureCategory, detail: String) {
    if execution.failure_category.is_none() {
        execution.failure_category = Some(category);
        execution.failure_detail = Some(detail);
    }
}

fn read_case(path: &Path) -> Result<EvalCase, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))
}

fn validate_case(path: &Path, case: &EvalCase) -> Result<(), String> {
    if case.schema_version != 1 {
        return Err(format!("{} uses unsupported schema_version", case.id));
    }
    if case.id.is_empty() || case.title.is_empty() {
        return Err("case id and title are required".to_string());
    }
    if case.raw_capture {
        return Err(format!(
            "{} enables raw_capture; default validation requires false",
            case.id
        ));
    }
    if case.kind == EvalKind::ModelInLoop {
        return Err(format!(
            "{} is model_in_loop and must not run in default validation",
            case.id
        ));
    }
    let base = path
        .parent()
        .ok_or_else(|| "manifest path has no parent".to_string())?;
    if !base.join(&case.task_file).is_file() {
        return Err(format!("{} task_file does not exist", case.id));
    }
    if !base.join(&case.fixture_dir).is_dir() {
        return Err(format!("{} fixture_dir does not exist", case.id));
    }
    if case.failure_categories.is_empty() {
        return Err(format!("{} must declare failure_categories", case.id));
    }
    if case.success_criteria.is_empty() {
        return Err(format!("{} must declare success_criteria", case.id));
    }
    Ok(())
}

fn sandbox_policy(value: &str) -> Result<SandboxPolicy, String> {
    match value {
        "read_only" => Ok(SandboxPolicy::ReadOnly),
        "workspace_write" => Ok(SandboxPolicy::WorkspaceWrite),
        "full_access" => Ok(SandboxPolicy::FullAccess),
        other => Err(format!("unknown sandbox policy {other}")),
    }
}

fn network_policy(value: &str) -> Result<NetworkPolicy, String> {
    match value {
        "deny" => Ok(NetworkPolicy::Deny),
        "allow_all" => Ok(NetworkPolicy::AllowAll),
        other => Err(format!("unknown network policy {other}")),
    }
}

fn find_manifests(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    find_manifests_into(dir, &mut out)?;
    out.sort();
    Ok(out)
}

fn find_manifests_into(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            find_manifests_into(&path, out)?;
        } else if path.file_name().and_then(|name| name.to_str()) == Some("eval.json") {
            out.push(path);
        }
    }
    Ok(())
}

fn temp_workspace(case_id: &str) -> Result<PathBuf, String> {
    let path = env::temp_dir().join(format!(
        "molo-eval-{case_id}-{}-{}",
        std::process::id(),
        epoch_millis()
    ));
    fs::create_dir_all(&path).map_err(|error| error.to_string())?;
    Ok(path)
}

fn copy_dir_all(from: &Path, to: &Path) -> io::Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let source = entry.path();
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_all(&source, &target)?;
        } else {
            fs::copy(source, target)?;
        }
    }
    Ok(())
}

fn init_git_baseline(root: &Path) -> Result<(), String> {
    run_git(root, ["init", "-q"])?;
    run_git(root, ["add", "."])?;
    run_git(
        root,
        [
            "-c",
            "user.email=molo-eval@example.invalid",
            "-c",
            "user.name=molo eval",
            "commit",
            "--allow-empty",
            "-m",
            "baseline",
            "-q",
        ],
    )
}

fn run_git<const N: usize>(root: &Path, args: [&str; N]) -> Result<(), String> {
    let status = Command::new("git")
        .args(args)
        .current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|error| format!("failed to run git: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("git command failed with status {status}"))
    }
}

fn apply_dirty_setup(root: &Path, setup: &EvalSetup) -> Result<Vec<String>, String> {
    let mut files = Vec::new();
    for dirty in &setup.pre_existing_dirty_files {
        let path = WorkspacePath::parse(&dirty.path).map_err(|error| error.to_string())?;
        let absolute = root.join(path.as_path());
        if let Some(parent) = absolute.parent() {
            fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        fs::write(&absolute, &dirty.content).map_err(|error| error.to_string())?;
        files.push(path.display());
    }
    Ok(files)
}

fn write_result(path: &Path, record: &EvalRunRecord) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let json = serde_json::to_vec_pretty(record).map_err(|error| error.to_string())?;
    fs::write(path, json).map_err(|error| error.to_string())
}

fn default_output_path(case: &EvalCase) -> PathBuf {
    PathBuf::from("evals")
        .join("results")
        .join(format!("{}-{}.json", case.id, epoch_millis()))
}

fn intersection(left: &[String], right: &[String]) -> Vec<String> {
    let left = left.iter().cloned().collect::<BTreeSet<_>>();
    right
        .iter()
        .filter(|path| left.contains(*path))
        .cloned()
        .collect()
}

fn missing(left: &[String], expected: &[String]) -> Vec<String> {
    let left = left.iter().cloned().collect::<BTreeSet<_>>();
    expected
        .iter()
        .filter(|path| !left.contains(*path))
        .cloned()
        .collect()
}

fn exit_code(status: &CommandStatus) -> Option<i32> {
    match status {
        CommandStatus::Exited { code } => Some(*code),
        _ => None,
    }
}

fn timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}

fn epoch_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn git_commit() -> String {
    let Ok(output) = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
    else {
        return "unknown".to_string();
    };
    if output.status.success() {
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    } else {
        "unknown".to_string()
    }
}

fn digest_json<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    Ok(digest_bytes(&bytes))
}

fn digest_value(value: &serde_json::Value) -> String {
    digest_bytes(value.to_string().as_bytes())
}

fn digest_bytes(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

fn sanitize(detail: &str, patterns: &[String]) -> String {
    let mut sanitized = detail.to_string();
    for pattern in patterns {
        if !pattern.is_empty() {
            sanitized = sanitized.replace(pattern, "[REDACTED]");
        }
    }
    sanitized.chars().take(512).collect()
}
