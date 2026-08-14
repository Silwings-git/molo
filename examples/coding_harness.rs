use molo::{
    AlwaysAllowApprovalBroker, BasicHarness, CodingEffectExecutor, CodingExecutorConfig,
    CodingPolicyEngine, GitChangedFilesRequest, GitOperation, GitPayload, LocalCommandExecutor,
    LocalWorkspace, PolicyCapabilityMode, ReadFilePayload, RunContext, SearchPayload,
    WorkspacePath, WorkspaceSearcher,
};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let workspace = LocalWorkspace::new(std::env::current_dir()?)?;
    let commands = LocalCommandExecutor::new(workspace.clone()).with_advisory_policy(true);
    let git = molo::CliGitInspector::new(commands.clone());
    let searcher = WorkspaceSearcher::new(workspace.clone());
    let executor = CodingEffectExecutor::new(workspace, commands, git, searcher).with_config(
        CodingExecutorConfig::default()
            .with_command_policy_capability_mode(PolicyCapabilityMode::AllowAdvisory),
    );
    let harness = BasicHarness::new(
        executor,
        CodingPolicyEngine::conservative(),
        AlwaysAllowApprovalBroker,
        molo::NoopAuditSink,
        molo::NoopTranscriptStore,
    );
    let context = RunContext::new("coding-harness-example");

    let read = ReadFilePayload {
        path: WorkspacePath::parse("Cargo.toml")?,
        max_bytes: Some(4096),
    }
    .into_effect()?;
    let observation = molo::Harness::execute(&harness, read, &context).await?;
    println!("{}", observation.output.observation_for_model);

    let search = SearchPayload {
        query: "coding".to_string(),
        paths: vec![WorkspacePath::parse("src")?],
        max_matches: Some(5),
        context_lines: 0,
    }
    .into_effect()?;
    let observation = molo::Harness::execute(&harness, search, &context).await?;
    println!("{}", observation.output.observation_for_model);

    let git = GitPayload {
        operation: GitOperation::ChangedFiles(GitChangedFilesRequest::default()),
    }
    .into_effect()?;
    let observation = molo::Harness::execute(&harness, git, &context).await?;
    println!("{}", observation.output.observation_for_model);

    Ok(())
}
