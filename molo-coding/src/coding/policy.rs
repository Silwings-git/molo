use crate::harness::{
    ClassifiedEffect, DefaultPolicyEngine, HarnessError, PolicyDecision, PolicyEngine,
};
use crate::{EffectKind, RiskLevel, RunContext, RunMetadata};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::command::{CommandRequest, PtyMode};
use super::payload::{CommandPayload, ListFilesPayload};

/// Coding-specific operation class used by conservative policy presets.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CodingPolicyClass {
    /// Read a workspace file.
    ReadWorkspace,
    /// List workspace files.
    ListWorkspace,
    /// Search workspace content.
    SearchWorkspace,
    /// Write a workspace file.
    WriteWorkspace,
    /// Apply a workspace patch.
    ApplyPatch,
    /// Read git state.
    GitRead,
    /// Mutate git state.
    GitMutation,
    /// Destructively mutate git state.
    GitDestructiveMutation,
    /// Host-allowlisted test command.
    TestCommand,
    /// Host-allowlisted build command.
    BuildCommand,
    /// Host-allowlisted lint command.
    LintCommand,
    /// Package manager install/update command.
    PackageInstall,
    /// Command that can perform network I/O.
    NetworkCommand,
    /// Shell command such as `sh -c` or `bash -lc`.
    ShellCommand,
    /// Command requesting a PTY.
    PtyCommand,
    /// Destructive command.
    DestructiveCommand,
    /// Command that does not match a safer known class.
    UnknownCommand,
    /// Trusted MCP operation.
    McpTrusted,
    /// Untrusted MCP operation.
    McpUntrusted,
}

/// Prefix pattern used by [`CommandTaxonomy`] allowlists.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandPattern {
    /// Argument prefix that must match from `argv[0]`.
    pub argv_prefix: Vec<String>,
}

impl CommandPattern {
    /// Constructs an argument-prefix pattern.
    pub fn new<I, S>(argv_prefix: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            argv_prefix: argv_prefix.into_iter().map(Into::into).collect(),
        }
    }

    fn matches(&self, argv: &[String]) -> bool {
        !self.argv_prefix.is_empty()
            && argv.len() >= self.argv_prefix.len()
            && self
                .argv_prefix
                .iter()
                .zip(argv)
                .all(|(expected, actual)| expected == actual)
    }
}

/// Command taxonomy and host-provided allowlists for coding policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct CommandTaxonomy {
    /// Test command prefixes allowed without approval.
    pub allowed_test_commands: Vec<CommandPattern>,
    /// Build command prefixes allowed without approval.
    pub allowed_build_commands: Vec<CommandPattern>,
    /// Lint command prefixes allowed without approval.
    pub allowed_lint_commands: Vec<CommandPattern>,
    /// Program names treated as network-capable.
    pub network_programs: Vec<String>,
    /// Program names treated as shells.
    pub shell_programs: Vec<String>,
    /// Program names treated as package managers.
    pub package_managers: Vec<String>,
    /// Lowercase command fragments treated as destructive.
    pub destructive_fragments: Vec<String>,
}

impl Default for CommandTaxonomy {
    fn default() -> Self {
        Self {
            allowed_test_commands: Vec::new(),
            allowed_build_commands: Vec::new(),
            allowed_lint_commands: Vec::new(),
            network_programs: [
                "curl", "wget", "ssh", "scp", "sftp", "rsync", "nc", "netcat",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            shell_programs: ["sh", "bash", "zsh", "fish", "cmd", "powershell", "pwsh"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            package_managers: [
                "npm", "pnpm", "yarn", "cargo", "pip", "pip3", "uv", "poetry", "brew", "apt",
                "apt-get",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
            destructive_fragments: [
                "rm -rf",
                "git reset --hard",
                "git clean -fd",
                "push --force",
                "push -f",
                "sudo ",
                "mkfs",
                "dd if=",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
        }
    }
}

impl CommandTaxonomy {
    /// Constructs the conservative taxonomy with no test/build/lint allowlist.
    pub fn conservative() -> Self {
        Self::default()
    }

    /// Adds a test command allowlist prefix.
    pub fn with_allowed_test_command<I, S>(mut self, argv_prefix: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_test_commands
            .push(CommandPattern::new(argv_prefix));
        self
    }

    /// Adds a build command allowlist prefix.
    pub fn with_allowed_build_command<I, S>(mut self, argv_prefix: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_build_commands
            .push(CommandPattern::new(argv_prefix));
        self
    }

    /// Adds a lint command allowlist prefix.
    pub fn with_allowed_lint_command<I, S>(mut self, argv_prefix: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_lint_commands
            .push(CommandPattern::new(argv_prefix));
        self
    }

    /// Classifies one command request.
    pub fn classify_command(&self, request: &CommandRequest) -> Vec<CodingPolicyClass> {
        let mut classes = Vec::new();
        if request.pty != PtyMode::Disabled {
            classes.push(CodingPolicyClass::PtyCommand);
        }
        let Some(program) = request.argv.first() else {
            classes.push(CodingPolicyClass::UnknownCommand);
            return classes;
        };
        let program = program.to_ascii_lowercase();
        let lowered = request.argv.join(" ").to_ascii_lowercase();

        if self
            .destructive_fragments
            .iter()
            .any(|fragment| lowered.contains(fragment))
        {
            if program == "git" {
                classes.push(CodingPolicyClass::GitDestructiveMutation);
            }
            classes.push(CodingPolicyClass::DestructiveCommand);
        }
        if self.shell_programs.iter().any(|shell| shell == &program) {
            classes.push(CodingPolicyClass::ShellCommand);
        }
        if self
            .network_programs
            .iter()
            .any(|network_program| network_program == &program)
        {
            classes.push(CodingPolicyClass::NetworkCommand);
        }
        if self.is_package_install(&program, &request.argv) {
            classes.push(CodingPolicyClass::PackageInstall);
        }
        if program == "git" {
            if is_git_read_only(&request.argv) {
                classes.push(CodingPolicyClass::GitRead);
            } else if !classes.contains(&CodingPolicyClass::GitDestructiveMutation) {
                classes.push(CodingPolicyClass::GitMutation);
            }
        }
        if matches_any(&self.allowed_test_commands, &request.argv) {
            classes.push(CodingPolicyClass::TestCommand);
        }
        if matches_any(&self.allowed_build_commands, &request.argv) {
            classes.push(CodingPolicyClass::BuildCommand);
        }
        if matches_any(&self.allowed_lint_commands, &request.argv) {
            classes.push(CodingPolicyClass::LintCommand);
        }
        if classes.is_empty() {
            classes.push(CodingPolicyClass::UnknownCommand);
        }
        classes.sort();
        classes.dedup();
        classes
    }

    fn is_package_install(&self, program: &str, argv: &[String]) -> bool {
        if !self
            .package_managers
            .iter()
            .any(|manager| manager == program)
        {
            return false;
        }
        argv.iter().skip(1).any(|arg| {
            matches!(
                arg.as_str(),
                "install" | "add" | "update" | "upgrade" | "sync"
            )
        })
    }
}

/// Typed input produced for coding policy evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CodingPolicyInput {
    /// Effect id.
    pub effect_id: String,
    /// Effect kind.
    pub kind: EffectKind,
    /// Coding policy classes.
    pub classes: Vec<CodingPolicyClass>,
    /// Command request, for command effects.
    pub command: Option<CommandRequest>,
    /// Host-owned metadata.
    pub metadata: RunMetadata,
}

impl CodingPolicyInput {
    /// Builds policy input from a classified effect.
    pub fn from_classified_effect(effect: &ClassifiedEffect, taxonomy: &CommandTaxonomy) -> Self {
        let mut command = None;
        let classes = match &effect.request.kind {
            EffectKind::ReadFile => vec![CodingPolicyClass::ReadWorkspace],
            EffectKind::WriteFile => vec![CodingPolicyClass::WriteWorkspace],
            EffectKind::ApplyPatch => vec![CodingPolicyClass::ApplyPatch],
            EffectKind::Search => {
                if serde_json::from_value::<ListFilesPayload>(effect.request.payload.clone())
                    .is_ok()
                {
                    vec![CodingPolicyClass::ListWorkspace]
                } else {
                    vec![CodingPolicyClass::SearchWorkspace]
                }
            }
            EffectKind::Git => vec![CodingPolicyClass::GitRead],
            EffectKind::ExecuteCommand => {
                match serde_json::from_value::<CommandPayload>(effect.request.payload.clone()) {
                    Ok(payload) => {
                        let classes = taxonomy.classify_command(&payload.request);
                        command = Some(payload.request);
                        classes
                    }
                    Err(_) => vec![CodingPolicyClass::UnknownCommand],
                }
            }
            EffectKind::Mcp => vec![CodingPolicyClass::McpUntrusted],
            _ => Vec::new(),
        };
        Self {
            effect_id: effect.request.id.clone(),
            kind: effect.request.kind.clone(),
            classes,
            command,
            metadata: effect.request.metadata.clone(),
        }
    }
}

/// Conservative coding policy wrapper around a host policy engine.
#[derive(Debug, Clone)]
pub struct CodingPolicyEngine<P = DefaultPolicyEngine> {
    inner: P,
    taxonomy: CommandTaxonomy,
}

impl CodingPolicyEngine<DefaultPolicyEngine> {
    /// Constructs the conservative preset with the default fallback policy.
    pub fn conservative() -> Self {
        Self::new(DefaultPolicyEngine)
    }
}

impl<P> CodingPolicyEngine<P> {
    /// Constructs a coding policy wrapper around a fallback policy engine.
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            taxonomy: CommandTaxonomy::conservative(),
        }
    }

    /// Replaces the command taxonomy and host allowlists.
    pub fn with_taxonomy(mut self, taxonomy: CommandTaxonomy) -> Self {
        self.taxonomy = taxonomy;
        self
    }

    /// Returns the configured taxonomy.
    pub fn taxonomy(&self) -> &CommandTaxonomy {
        &self.taxonomy
    }
}

#[async_trait]
impl<P> PolicyEngine for CodingPolicyEngine<P>
where
    P: PolicyEngine,
{
    async fn evaluate(
        &self,
        effect: &ClassifiedEffect,
        context: &RunContext,
    ) -> Result<PolicyDecision, HarnessError> {
        let input = CodingPolicyInput::from_classified_effect(effect, &self.taxonomy);
        if let Some(decision) = conservative_decision(&input, effect.effective_risk) {
            return Ok(decision);
        }
        self.inner.evaluate(effect, context).await
    }
}

fn conservative_decision(input: &CodingPolicyInput, risk: RiskLevel) -> Option<PolicyDecision> {
    if risk == RiskLevel::Critical
        || has_any(
            input,
            &[
                CodingPolicyClass::DestructiveCommand,
                CodingPolicyClass::GitDestructiveMutation,
                CodingPolicyClass::NetworkCommand,
            ],
        )
    {
        return Some(PolicyDecision::Deny {
            reason: "coding policy denied destructive or network command".to_string(),
        });
    }

    if risk == RiskLevel::High
        || has_any(
            input,
            &[
                CodingPolicyClass::PackageInstall,
                CodingPolicyClass::ShellCommand,
                CodingPolicyClass::PtyCommand,
                CodingPolicyClass::GitMutation,
                CodingPolicyClass::UnknownCommand,
            ],
        )
    {
        return Some(PolicyDecision::RequireApproval {
            reason: "coding policy requires approval".to_string(),
        });
    }

    if has_any(
        input,
        &[
            CodingPolicyClass::ReadWorkspace,
            CodingPolicyClass::ListWorkspace,
            CodingPolicyClass::SearchWorkspace,
            CodingPolicyClass::WriteWorkspace,
            CodingPolicyClass::ApplyPatch,
            CodingPolicyClass::GitRead,
            CodingPolicyClass::TestCommand,
            CodingPolicyClass::BuildCommand,
            CodingPolicyClass::LintCommand,
        ],
    ) {
        return Some(PolicyDecision::Allow);
    }

    None
}

fn has_any(input: &CodingPolicyInput, classes: &[CodingPolicyClass]) -> bool {
    classes.iter().any(|class| input.classes.contains(class))
}

fn matches_any(patterns: &[CommandPattern], argv: &[String]) -> bool {
    patterns.iter().any(|pattern| pattern.matches(argv))
}

fn is_git_read_only(argv: &[String]) -> bool {
    let Some(subcommand) = argv.get(1).map(String::as_str) else {
        return false;
    };
    matches!(
        subcommand,
        "status" | "diff" | "show" | "rev-parse" | "branch" | "log" | "ls-files"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{DefaultRiskClassifier, RiskClassifier};
    use serde_json::json;

    async fn classify(effect: crate::EffectRequest) -> ClassifiedEffect {
        DefaultRiskClassifier
            .classify(effect, &RunContext::new("policy"))
            .await
            .unwrap()
    }

    #[test]
    fn taxonomy_classifies_network_shell_package_and_allowlisted_test() {
        let taxonomy = CommandTaxonomy::conservative().with_allowed_test_command(["cargo", "test"]);
        assert_eq!(
            taxonomy.classify_command(&CommandRequest::new(["curl", "https://example.com"])),
            vec![CodingPolicyClass::NetworkCommand]
        );
        assert!(
            taxonomy
                .classify_command(&CommandRequest::new(["sh", "-c", "echo hi"]))
                .contains(&CodingPolicyClass::ShellCommand)
        );
        assert!(
            taxonomy
                .classify_command(&CommandRequest::new(["npm", "install"]))
                .contains(&CodingPolicyClass::PackageInstall)
        );
        assert_eq!(
            taxonomy.classify_command(&CommandRequest::new(["cargo", "test", "-q"])),
            vec![CodingPolicyClass::TestCommand]
        );
    }

    #[tokio::test]
    async fn conservative_policy_denies_network_command() {
        let payload = CommandPayload {
            request: CommandRequest::new(["curl", "https://example.com"]),
        };
        let effect = classify(payload.into_effect().unwrap()).await;
        let decision = CodingPolicyEngine::conservative()
            .evaluate(&effect, &RunContext::new("policy"))
            .await
            .unwrap();
        assert!(matches!(decision, PolicyDecision::Deny { .. }));
    }

    #[tokio::test]
    async fn conservative_policy_requires_approval_for_unknown_command() {
        let payload = CommandPayload {
            request: CommandRequest::new(["custom-tool", "--flag"]),
        };
        let effect = classify(payload.into_effect().unwrap()).await;
        let decision = CodingPolicyEngine::conservative()
            .evaluate(&effect, &RunContext::new("policy"))
            .await
            .unwrap();
        assert!(matches!(decision, PolicyDecision::RequireApproval { .. }));
    }

    #[tokio::test]
    async fn conservative_policy_covers_command_hardening_cases() {
        for argv in [
            vec!["wget", "https://example.com"],
            vec!["sudo", "whoami"],
            vec!["rm", "-rf", "target"],
            vec!["git", "reset", "--hard"],
            vec!["git", "push", "--force"],
        ] {
            let payload = CommandPayload {
                request: CommandRequest::new(argv),
            };
            let effect = classify(payload.into_effect().unwrap()).await;
            let decision = CodingPolicyEngine::conservative()
                .evaluate(&effect, &RunContext::new("policy"))
                .await
                .unwrap();
            assert!(matches!(decision, PolicyDecision::Deny { .. }));
        }

        let package_install = classify(
            CommandPayload {
                request: CommandRequest::new(["npm", "install"]),
            }
            .into_effect()
            .unwrap(),
        )
        .await;
        assert!(matches!(
            CodingPolicyEngine::conservative()
                .evaluate(&package_install, &RunContext::new("policy"))
                .await
                .unwrap(),
            PolicyDecision::RequireApproval { .. }
        ));

        let shell = classify(
            CommandPayload {
                request: CommandRequest::new(["sh", "-c", "echo hi"]),
            }
            .into_effect()
            .unwrap(),
        )
        .await;
        assert!(matches!(
            CodingPolicyEngine::conservative()
                .evaluate(&shell, &RunContext::new("policy"))
                .await
                .unwrap(),
            PolicyDecision::RequireApproval { .. }
        ));
    }

    #[tokio::test]
    async fn conservative_policy_allows_read_effect() {
        let effect = classify(crate::EffectRequest::new(
            EffectKind::ReadFile,
            "read",
            json!({"path": "src/lib.rs"}),
        ))
        .await;
        let decision = CodingPolicyEngine::conservative()
            .evaluate(&effect, &RunContext::new("policy"))
            .await
            .unwrap();
        assert_eq!(decision, PolicyDecision::Allow);
    }
}
