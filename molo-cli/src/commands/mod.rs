use crate::approval::CliApprovalBroker;
use crate::args::{Command, ProviderKind};
use crate::config::{CliConfig, provider_from_config};
use crate::error::CliError;
use crate::output::{FileChangeSummary, FinalDiffSummary, print_final_summary};
use crate::session::{
    CliAuditLine, CliSessionEnvelope, CliSessionStatus, CliSessionStore, CliTranscriptLine,
    JsonlAuditSink, JsonlTranscriptStore, WorkspaceFingerprint,
};
use futures::StreamExt;
use molo::{
    Agent, ApplyPatchTool, BasicHarness, CliGitInspector, CodingContextProvider,
    CodingContextRequest, CodingEffectExecutor, CommandRequest, DefaultCodingContextProvider,
    DefaultInstructionResolver, DefaultPolicyEngine, DiffRequest, GitChangedFilesRequest,
    GitInspector, GitStatusTool, HarnessConfig, HarnessRuntime, ListFilesTool,
    LocalCommandExecutor, LocalWorkspace, MessageChunk, NetworkPolicy, Provider, ReActAgent,
    ReadFileTool, RunCommandTool, RunContext, RunMetadata, RunRequest, SandboxPolicy,
    SearchRepoTool, SnapshotRequest, ToolCall, ToolRegistry, Workspace, WorkspacePath,
    WorkspaceSearcher,
};
use serde_json::json;
use std::io::Read;
use std::time::Duration;

/// Dispatches one parsed command.
pub async fn dispatch(command: Command, config: CliConfig) -> Result<(), CliError> {
    match command {
        Command::Help => {
            println!("{}", crate::args::help_text());
            Ok(())
        }
        Command::Chat {
            prompt,
            stream,
            project_instructions,
        } => chat(config, prompt, stream, project_instructions).await,
        Command::Code { task, json } => code(config, task, json).await,
        Command::Review {
            paths,
            json,
            allow_readonly_commands,
        } => review(config, paths, json, allow_readonly_commands).await,
        Command::Resume {
            session_id,
            task,
            json,
        } => resume(config, session_id, task, json).await,
        Command::Sessions { json } => sessions(config, json),
        Command::Transcript { session_id } => transcript(config, session_id),
        Command::ConfigCheck { json } => config_check(config, json),
    }
}

async fn chat(
    config: CliConfig,
    prompt: Option<String>,
    stream: bool,
    project_instructions: bool,
) -> Result<(), CliError> {
    let prompt = prompt_or_stdin(prompt)?;
    let store = CliSessionStore::new(config.session_dir.clone());
    store.ensure_root()?;

    let (workspace, _commands, git, searcher, instructions) = workspace_stack(&config)?;
    let fingerprint = workspace_fingerprint(&git, &RunContext::new("fingerprint")).await;
    let mut session = CliSessionEnvelope::new(
        "chat",
        prompt.clone(),
        workspace.root().await.as_path().display().to_string(),
        fingerprint,
        config.snapshot(),
        None,
    );
    store.save(&session)?;
    let run_context = run_context(&session.session_id, config.policy.command_timeout);
    let system_prompt = if project_instructions {
        let provider = DefaultCodingContextProvider::new(
            workspace.clone(),
            searcher.clone(),
            git.clone(),
            instructions,
        );
        context_system_prompt(&provider, &prompt, &run_context).await?
    } else {
        "You are molo's reference CLI chat mode.".to_string()
    };

    append_cli_event(
        &store,
        &session.session_id,
        &run_context,
        "run_started",
        json!({}),
    )?;
    let provider = provider_for_chat(&config)?;
    let mut agent = ReActAgent::new(provider, ToolRegistry::new(), system_prompt);

    let answer = if stream {
        let mut output = String::new();
        let mut stream = agent
            .run_stream_request_with_context(RunRequest::text(prompt), run_context.clone())
            .await?;
        while let Some(chunk) = stream.next().await {
            match chunk? {
                MessageChunk::Delta(delta) => {
                    print!("{delta}");
                    output.push_str(&delta);
                }
                MessageChunk::Done(_) => {
                    if !output.ends_with('\n') {
                        println!();
                    }
                }
                MessageChunk::Cancelled => {
                    session
                        .task_state
                        .interruptions
                        .push(crate::session::CliInterruption {
                            run_id: run_context.run_id.clone(),
                            reason: "cancelled".to_string(),
                        });
                }
                _ => {}
            }
        }
        output
    } else {
        let output = agent
            .run_request_with_context(RunRequest::text(prompt), run_context.clone())
            .await?;
        println!("{}", output.answer);
        output.answer
    };

    append_cli_event(
        &store,
        &session.session_id,
        &run_context,
        "run_completed",
        json!({ "answer_bytes": answer.len() }),
    )?;
    session.finish(CliSessionStatus::Completed);
    store.save(&session)?;
    println!("session: {}", session.session_id);
    Ok(())
}

async fn code(config: CliConfig, task: String, json_output: bool) -> Result<(), CliError> {
    let store = CliSessionStore::new(config.session_dir.clone());
    store.ensure_root()?;

    let (workspace, commands, git, searcher, instructions) = workspace_stack(&config)?;
    let fingerprint_context = RunContext::new("fingerprint");
    let fingerprint = workspace_fingerprint(&git, &fingerprint_context).await;
    let before_snapshot = workspace
        .snapshot(SnapshotRequest {
            paths: Vec::new(),
            recursive: true,
        })
        .await?;
    let mut session = CliSessionEnvelope::new(
        "code",
        task.clone(),
        workspace.root().await.as_path().display().to_string(),
        fingerprint.clone(),
        config.snapshot(),
        None,
    );
    store.save(&session)?;

    let run_context = run_context(&session.session_id, config.policy.command_timeout);
    append_cli_event(
        &store,
        &session.session_id,
        &run_context,
        "run_started",
        json!({}),
    )?;

    let context_provider = DefaultCodingContextProvider::new(
        workspace.clone(),
        searcher.clone(),
        git.clone(),
        instructions,
    );
    let system_prompt = context_system_prompt(&context_provider, &task, &run_context).await?;
    let provider = provider_for_code(&config, &task)?;
    let approval = CliApprovalBroker::new(config.policy.approval, config.non_interactive);
    let executor =
        CodingEffectExecutor::new(workspace.clone(), commands.clone(), git.clone(), searcher);
    let harness = BasicHarness::new(
        executor,
        DefaultPolicyEngine,
        approval.clone(),
        JsonlAuditSink::new(store.clone(), session.session_id.clone()),
        JsonlTranscriptStore::new(store.clone(), session.session_id.clone()),
    )
    .with_config(HarnessConfig {
        default_sandbox: SandboxPolicy::WorkspaceWrite,
        default_network: NetworkPolicy::Deny,
        default_timeout: config.policy.command_timeout,
        ..HarnessConfig::default()
    });
    let runtime = HarnessRuntime::new(provider, harness);
    let mut kernel = ReActAgent::kernel(code_registry(), system_prompt);

    let run_result = runtime
        .run(
            &mut kernel,
            RunRequest::text(task.clone()),
            run_context.clone(),
        )
        .await;
    let (status, model_answer) = match run_result {
        Ok(output) => {
            append_cli_event(
                &store,
                &session.session_id,
                &run_context,
                "run_completed",
                json!({ "answer_bytes": output.answer.len() }),
            )?;
            (CliSessionStatus::Completed, output.answer)
        }
        Err(error) => {
            session.task_state.last_error = Some(error.to_string());
            append_cli_event(
                &store,
                &session.session_id,
                &run_context,
                "run_failed",
                json!({ "error": error.to_string() }),
            )?;
            (CliSessionStatus::Failed, String::new())
        }
    };

    session.task_state.approvals = approval.summaries();
    session.task_state.changed_files = workspace
        .change_tracker()
        .changed_files()
        .into_iter()
        .map(|path| path.display())
        .collect();
    session.finish(status);
    store.save(&session)?;

    let summary = final_summary(FinalSummaryInput {
        store: &store,
        session: &session,
        run_id: &run_context.run_id,
        workspace: &workspace,
        git: &git,
        before_snapshot,
        pre_existing_dirty_files: fingerprint.dirty_files,
        model_answer,
    })
    .await?;
    store.write_final_summary(&session.session_id, &summary)?;
    print_final_summary(&summary, json_output)?;
    Ok(())
}

async fn review(
    config: CliConfig,
    paths: Vec<String>,
    json_output: bool,
    allow_readonly_commands: bool,
) -> Result<(), CliError> {
    let store = CliSessionStore::new(config.session_dir.clone());
    store.ensure_root()?;

    let (workspace, commands, git, searcher, instructions) = workspace_stack(&config)?;
    let focus_paths = paths
        .iter()
        .filter_map(|path| WorkspacePath::parse(path).ok())
        .collect::<Vec<_>>();
    let fingerprint = workspace_fingerprint(&git, &RunContext::new("fingerprint")).await;
    let before_snapshot = workspace
        .snapshot(SnapshotRequest {
            paths: Vec::new(),
            recursive: true,
        })
        .await?;
    let goal = if paths.is_empty() {
        "Review the current workspace diff".to_string()
    } else {
        format!("Review {}", paths.join(" "))
    };
    let mut session = CliSessionEnvelope::new(
        "review",
        goal.clone(),
        workspace.root().await.as_path().display().to_string(),
        fingerprint.clone(),
        config.snapshot(),
        None,
    );
    store.save(&session)?;
    let run_context = run_context(&session.session_id, config.policy.command_timeout);
    append_cli_event(
        &store,
        &session.session_id,
        &run_context,
        "run_started",
        json!({}),
    )?;

    let context_provider = DefaultCodingContextProvider::new(
        workspace.clone(),
        searcher.clone(),
        git.clone(),
        instructions,
    );
    let mut request = CodingContextRequest::new(&goal);
    request.focus_paths = focus_paths;
    let system_prompt =
        context_prompt_from_bundle(context_provider.gather(request, &run_context).await?);
    let provider = provider_for_review(&config)?;
    let approval = CliApprovalBroker::new(config.policy.approval, config.non_interactive);
    let executor =
        CodingEffectExecutor::new(workspace.clone(), commands.clone(), git.clone(), searcher);
    let harness = BasicHarness::new(
        executor,
        DefaultPolicyEngine,
        approval.clone(),
        JsonlAuditSink::new(store.clone(), session.session_id.clone()),
        JsonlTranscriptStore::new(store.clone(), session.session_id.clone()),
    )
    .with_config(HarnessConfig {
        default_sandbox: SandboxPolicy::ReadOnly,
        default_network: NetworkPolicy::Deny,
        default_timeout: config.policy.command_timeout,
        ..HarnessConfig::default()
    });
    let runtime = HarnessRuntime::new(provider, harness);
    let mut kernel = ReActAgent::kernel(review_registry(allow_readonly_commands), system_prompt);
    let output = runtime
        .run(
            &mut kernel,
            RunRequest::text(goal.clone()),
            run_context.clone(),
        )
        .await?;
    append_cli_event(
        &store,
        &session.session_id,
        &run_context,
        "run_completed",
        json!({ "answer_bytes": output.answer.len() }),
    )?;
    session.task_state.approvals = approval.summaries();
    session.finish(CliSessionStatus::Completed);
    store.save(&session)?;

    let summary = final_summary(FinalSummaryInput {
        store: &store,
        session: &session,
        run_id: &run_context.run_id,
        workspace: &workspace,
        git: &git,
        before_snapshot,
        pre_existing_dirty_files: fingerprint.dirty_files,
        model_answer: output.answer,
    })
    .await?;
    store.write_final_summary(&session.session_id, &summary)?;
    print_final_summary(&summary, json_output)?;
    Ok(())
}

async fn resume(
    config: CliConfig,
    session_id: String,
    task: Option<String>,
    json_output: bool,
) -> Result<(), CliError> {
    let store = CliSessionStore::new(config.session_dir.clone());
    let previous = store.load(&session_id)?;
    let task = task.unwrap_or_else(|| format!("Continue previous session {}", previous.session_id));
    let (workspace, commands, git, searcher, instructions) = workspace_stack(&config)?;
    let fingerprint = workspace_fingerprint(&git, &RunContext::new("fingerprint")).await;
    let workspace_root = workspace.root().await.as_path().display().to_string();
    if previous.workspace_root != workspace_root {
        return Err(CliError::Config(format!(
            "resume workspace mismatch: session was {}, current is {}",
            previous.workspace_root, workspace_root
        )));
    }
    if previous.workspace_fingerprint.git_head != fingerprint.git_head {
        return Err(CliError::Config(
            "resume git HEAD mismatch; start a new session instead".to_string(),
        ));
    }
    let dirty_changed = previous.workspace_fingerprint.dirty_files != fingerprint.dirty_files;
    let before_snapshot = workspace
        .snapshot(SnapshotRequest {
            paths: Vec::new(),
            recursive: true,
        })
        .await?;
    let mut session = CliSessionEnvelope::new(
        "resume",
        task.clone(),
        workspace_root,
        fingerprint.clone(),
        config.snapshot(),
        Some(previous.session_id.clone()),
    );
    store.save(&session)?;
    let run_context = run_context(&session.session_id, config.policy.command_timeout);
    append_cli_event(
        &store,
        &session.session_id,
        &run_context,
        "resume_started",
        json!({
            "parent_session_id": previous.session_id,
            "dirty_baseline_changed": dirty_changed,
        }),
    )?;

    let transcript_summary = store
        .transcript_text(&session_id)?
        .lines()
        .take(20)
        .collect::<Vec<_>>()
        .join("\n");
    let mut context_request = CodingContextRequest::new(&task);
    context_request.transcript_summary =
        (!transcript_summary.is_empty()).then_some(transcript_summary);
    let context_provider = DefaultCodingContextProvider::new(
        workspace.clone(),
        searcher.clone(),
        git.clone(),
        instructions,
    );
    let system_prompt = context_prompt_from_bundle(
        context_provider
            .gather(context_request, &run_context)
            .await?,
    );
    let provider = provider_for_code(&config, &task)?;
    let approval = CliApprovalBroker::new(config.policy.approval, config.non_interactive);
    let executor =
        CodingEffectExecutor::new(workspace.clone(), commands.clone(), git.clone(), searcher);
    let harness = BasicHarness::new(
        executor,
        DefaultPolicyEngine,
        approval.clone(),
        JsonlAuditSink::new(store.clone(), session.session_id.clone()),
        JsonlTranscriptStore::new(store.clone(), session.session_id.clone()),
    )
    .with_config(HarnessConfig {
        default_sandbox: SandboxPolicy::WorkspaceWrite,
        default_network: NetworkPolicy::Deny,
        default_timeout: config.policy.command_timeout,
        ..HarnessConfig::default()
    });
    let runtime = HarnessRuntime::new(provider, harness);
    let mut kernel = ReActAgent::kernel(code_registry(), system_prompt);
    let output = runtime
        .run(&mut kernel, RunRequest::text(task), run_context.clone())
        .await?;
    session.task_state.approvals = approval.summaries();
    session.finish(CliSessionStatus::Completed);
    store.save(&session)?;
    let summary = final_summary(FinalSummaryInput {
        store: &store,
        session: &session,
        run_id: &run_context.run_id,
        workspace: &workspace,
        git: &git,
        before_snapshot,
        pre_existing_dirty_files: fingerprint.dirty_files,
        model_answer: output.answer,
    })
    .await?;
    store.write_final_summary(&session.session_id, &summary)?;
    print_final_summary(&summary, json_output)?;
    Ok(())
}

fn sessions(config: CliConfig, json_output: bool) -> Result<(), CliError> {
    let store = CliSessionStore::new(config.session_dir);
    let sessions = store.list()?;
    if json_output {
        println!("{}", serde_json::to_string_pretty(&sessions)?);
        return Ok(());
    }
    for session in sessions {
        println!(
            "{} {:?} {} {}",
            session.session_id, session.status, session.command, session.goal
        );
    }
    Ok(())
}

fn transcript(config: CliConfig, session_id: String) -> Result<(), CliError> {
    let store = CliSessionStore::new(config.session_dir);
    print!("{}", store.transcript_text(&session_id)?);
    Ok(())
}

fn config_check(config: CliConfig, json_output: bool) -> Result<(), CliError> {
    let snapshot = config.snapshot();
    if json_output {
        println!("{}", serde_json::to_string_pretty(&snapshot)?);
    } else {
        println!("workspace: {}", snapshot.workspace_root);
        println!("session_dir: {}", snapshot.session_dir);
        println!(
            "provider: {:?} {}",
            snapshot.provider.kind, snapshot.provider.model
        );
        println!("api_key_env: {}", snapshot.provider.api_key_env);
        println!("sandbox: {:?}", snapshot.policy.sandbox);
        println!("network: {:?}", snapshot.policy.network);
        println!("approval: {:?}", snapshot.policy.approval);
        println!("non_interactive: {}", snapshot.non_interactive);
    }
    Ok(())
}

fn prompt_or_stdin(prompt: Option<String>) -> Result<String, CliError> {
    if let Some(prompt) = prompt {
        return Ok(prompt);
    }
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let prompt = input.trim().to_string();
    if prompt.is_empty() {
        return Err(CliError::Args(
            "prompt is required via argv or stdin".to_string(),
        ));
    }
    Ok(prompt)
}

fn provider_for_chat(config: &CliConfig) -> Result<Box<dyn Provider>, CliError> {
    match config.provider.kind {
        ProviderKind::Fake => Ok(Box::new(molo::FakeProvider::new([molo::FakeReply::Text(
            "fake provider response".to_string(),
        )]))),
        ProviderKind::OpenAi => provider_from_config(config),
    }
}

fn provider_for_code(config: &CliConfig, task: &str) -> Result<Box<dyn Provider>, CliError> {
    match config.provider.kind {
        ProviderKind::OpenAi => provider_from_config(config),
        ProviderKind::Fake if task.contains("fake-patch") => {
            Ok(Box::new(molo::FakeProvider::new([
                molo::FakeReply::ToolCalls {
                    content: String::new(),
                    calls: vec![ToolCall {
                        id: "call-apply-patch".to_string(),
                        name: "apply_patch".to_string(),
                        arguments: serde_json::to_string(&json!({
                            "patch": {
                                "files": [{
                                    "path": "molo-cli-fake-output.txt",
                                    "operation": "Create",
                                    "expected_version": null,
                                    "hunks": [{
                                        "old_text": "",
                                        "new_text": "fake patch from molo-cli\n"
                                    }]
                                }],
                                "original_text": null,
                                "metadata": {}
                            },
                            "expected_versions": [],
                            "dry_run": false
                        }))?,
                    }],
                },
                molo::FakeReply::Text(
                    "Applied the fake patch through the governed harness.".to_string(),
                ),
            ])))
        }
        ProviderKind::Fake => Ok(Box::new(molo::FakeProvider::new([
            molo::FakeReply::ToolCalls {
                content: String::new(),
                calls: vec![ToolCall {
                    id: "call-git-status".to_string(),
                    name: "git_status".to_string(),
                    arguments: "{}".to_string(),
                }],
            },
            molo::FakeReply::Text(
                "Inspected the workspace through the governed harness.".to_string(),
            ),
        ]))),
    }
}

fn provider_for_review(config: &CliConfig) -> Result<Box<dyn Provider>, CliError> {
    match config.provider.kind {
        ProviderKind::OpenAi => provider_from_config(config),
        ProviderKind::Fake => Ok(Box::new(molo::FakeProvider::new([
            molo::FakeReply::ToolCalls {
                content: String::new(),
                calls: vec![ToolCall {
                    id: "call-review-status".to_string(),
                    name: "git_status".to_string(),
                    arguments: "{}".to_string(),
                }],
            },
            molo::FakeReply::Text("Fake review completed with no findings.".to_string()),
        ]))),
    }
}

fn code_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(ReadFileTool);
    registry.register(ListFilesTool);
    registry.register(SearchRepoTool);
    registry.register(ApplyPatchTool);
    registry.register(RunCommandTool);
    registry.register(GitStatusTool);
    registry
}

fn review_registry(allow_readonly_commands: bool) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry.register(ReadFileTool);
    registry.register(ListFilesTool);
    registry.register(SearchRepoTool);
    registry.register(GitStatusTool);
    if allow_readonly_commands {
        registry.register(RunCommandTool);
    }
    registry
}

type WorkspaceStack = (
    LocalWorkspace,
    LocalCommandExecutor<LocalWorkspace>,
    CliGitInspector<LocalCommandExecutor<LocalWorkspace>>,
    WorkspaceSearcher<LocalWorkspace>,
    DefaultInstructionResolver<LocalWorkspace>,
);

fn workspace_stack(config: &CliConfig) -> Result<WorkspaceStack, CliError> {
    let workspace = LocalWorkspace::new(&config.workspace_root)?;
    let commands = LocalCommandExecutor::new(workspace.clone()).with_advisory_policy(false);
    let git = CliGitInspector::new(commands.clone());
    let searcher = WorkspaceSearcher::new(workspace.clone());
    let instructions = DefaultInstructionResolver::new(workspace.clone());
    Ok((workspace, commands, git, searcher, instructions))
}

async fn workspace_fingerprint<G>(git: &G, context: &RunContext) -> WorkspaceFingerprint
where
    G: GitInspector,
{
    let git_head = git.head(context).await.ok().flatten();
    let dirty_files = git
        .changed_files(GitChangedFilesRequest::default(), context)
        .await
        .unwrap_or_default()
        .into_iter()
        .map(|file| file.path.display())
        .collect();
    WorkspaceFingerprint {
        git_head,
        dirty_files,
    }
}

async fn context_system_prompt<C>(
    provider: &C,
    goal: &str,
    context: &RunContext,
) -> Result<String, CliError>
where
    C: CodingContextProvider,
{
    let bundle = provider
        .gather(CodingContextRequest::new(goal), context)
        .await?;
    Ok(context_prompt_from_bundle(bundle))
}

fn context_prompt_from_bundle(bundle: molo::CodingContextBundle) -> String {
    let mut prompt = String::from(
        "You are molo's reference coding CLI. Use governed tools for workspace effects.\n",
    );
    if let Some(instructions) = bundle.instructions {
        for file in instructions.files {
            prompt.push_str("\nProject instructions from ");
            prompt.push_str(&file.path.display());
            prompt.push_str(":\n");
            prompt.push_str(&file.content);
            prompt.push('\n');
        }
    }
    if let Some(status) = bundle.git_status {
        prompt.push_str("\nGit status:\n");
        prompt.push_str(&status.raw);
        prompt.push('\n');
    }
    if !bundle.repo_tree.is_empty() {
        prompt.push_str("\nRepository files:\n");
        for entry in bundle.repo_tree.iter().take(80) {
            prompt.push_str("- ");
            prompt.push_str(&entry.path.display());
            prompt.push('\n');
        }
    }
    if !bundle.warnings.is_empty() {
        prompt.push_str("\nContext warnings:\n");
        for warning in bundle.warnings {
            prompt.push_str("- ");
            prompt.push_str(&warning);
            prompt.push('\n');
        }
    }
    prompt
}

fn run_context(session_id: &str, timeout: Duration) -> RunContext {
    let cancellation = molo::CancellationToken::new();
    let token = cancellation.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            token.cancel();
        }
    });
    let mut metadata = RunMetadata::new();
    metadata.insert("session_id".to_string(), json!(session_id));
    RunContext::generated()
        .with_cancellation(cancellation)
        .with_timeout(timeout)
        .with_metadata(metadata)
}

fn append_cli_event(
    store: &CliSessionStore,
    session_id: &str,
    context: &RunContext,
    name: &str,
    payload: serde_json::Value,
) -> Result<(), CliError> {
    store.append_transcript(
        session_id,
        &CliTranscriptLine::CliEvent {
            run_id: context.run_id.clone(),
            name: name.to_string(),
            payload,
        },
    )
}

struct FinalSummaryInput<'a, G> {
    store: &'a CliSessionStore,
    session: &'a CliSessionEnvelope,
    run_id: &'a str,
    workspace: &'a LocalWorkspace,
    git: &'a G,
    before_snapshot: molo::coding::WorkspaceSnapshot,
    pre_existing_dirty_files: Vec<String>,
    model_answer: String,
}

async fn final_summary<G>(input: FinalSummaryInput<'_, G>) -> Result<FinalDiffSummary, CliError>
where
    G: GitInspector,
{
    let after_snapshot = input
        .workspace
        .snapshot(SnapshotRequest {
            paths: Vec::new(),
            recursive: true,
        })
        .await?;
    let workspace_diff = input
        .workspace
        .diff(DiffRequest {
            before: input.before_snapshot,
            after: after_snapshot,
        })
        .await?;
    let git_changed = input
        .git
        .changed_files(
            GitChangedFilesRequest::default(),
            &RunContext::new("summary"),
        )
        .await
        .unwrap_or_default();
    let mut changed_files = git_changed
        .into_iter()
        .map(|file| FileChangeSummary {
            path: file.path.display(),
            status: file.status,
        })
        .collect::<Vec<_>>();
    for path in workspace_diff.changed_files {
        let display = path.display();
        if !changed_files.iter().any(|file| file.path == display) {
            changed_files.push(FileChangeSummary {
                path: display,
                status: "workspace".to_string(),
            });
        }
    }
    let denied_effects = denied_effects_from_audit(input.store, &input.session.session_id)?;
    Ok(FinalDiffSummary {
        session_id: input.session.session_id.clone(),
        run_ids: vec![input.run_id.to_string()],
        goal: input.session.goal.clone(),
        status: input.session.status,
        changed_files,
        pre_existing_dirty_files: input.pre_existing_dirty_files,
        verification: Vec::new(),
        approvals: input.session.task_state.approvals.clone(),
        denied_effects,
        truncated: workspace_diff.truncated,
        model_answer: input.model_answer,
    })
}

fn denied_effects_from_audit(
    store: &CliSessionStore,
    session_id: &str,
) -> Result<Vec<String>, CliError> {
    let mut denied = Vec::new();
    for line in store.audit_text(session_id)?.lines() {
        let parsed: CliAuditLine = serde_json::from_str(line)?;
        if let molo::AuditEvent::EffectDenied { effect_id, reason } = parsed.event {
            denied.push(format!("{effect_id}: {reason}"));
        }
    }
    Ok(denied)
}

#[allow(dead_code)]
fn read_only_command(argv: &[&str]) -> CommandRequest {
    CommandRequest::new(argv.iter().copied())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::{ApprovalMode, ProviderKind};
    use crate::config::{PolicyConfig, ProviderConfig};

    fn test_config(root: std::path::PathBuf) -> CliConfig {
        CliConfig {
            workspace_root: root.clone(),
            session_dir: root.join("sessions"),
            provider: ProviderConfig {
                kind: ProviderKind::Fake,
                model: "fake".to_string(),
                base_url: None,
                api_key_env: "OPENAI_API_KEY".to_string(),
            },
            policy: PolicyConfig {
                sandbox: SandboxPolicy::WorkspaceWrite,
                network: NetworkPolicy::Deny,
                approval: ApprovalMode::Deny,
                command_timeout: Duration::from_secs(5),
            },
            non_interactive: true,
        }
    }

    #[tokio::test]
    async fn fake_code_patch_creates_file() {
        let root = std::env::temp_dir().join(format!("molo-cli-code-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let config = test_config(root.clone());
        code(config, "fake-patch".to_string(), true).await.unwrap();
        assert!(root.join("molo-cli-fake-output.txt").exists());
        let _ = std::fs::remove_dir_all(root);
    }
}
