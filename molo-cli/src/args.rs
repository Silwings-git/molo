use crate::error::CliError;
use std::path::PathBuf;

/// Parsed top-level CLI arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliArgs {
    /// Global options.
    pub global: GlobalArgs,
    /// Command to run.
    pub command: Command,
}

/// Global options shared by all commands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalArgs {
    /// Workspace root. Defaults to the current directory.
    pub workspace: Option<PathBuf>,
    /// Session directory. Defaults to a user data directory.
    pub session_dir: Option<PathBuf>,
    /// Provider kind.
    pub provider: ProviderKind,
    /// Model name.
    pub model: Option<String>,
    /// OpenAI-compatible base URL.
    pub base_url: Option<String>,
    /// Environment variable that stores the provider API key.
    pub api_key_env: String,
    /// Approval mode.
    pub approval: ApprovalMode,
    /// Whether the command is non-interactive.
    pub non_interactive: bool,
    /// Default command/effect timeout in seconds.
    pub command_timeout_secs: u64,
}

impl Default for GlobalArgs {
    fn default() -> Self {
        Self {
            workspace: None,
            session_dir: None,
            provider: ProviderKind::Fake,
            model: None,
            base_url: None,
            api_key_env: "OPENAI_API_KEY".to_string(),
            approval: ApprovalMode::Ask,
            non_interactive: false,
            command_timeout_secs: 30,
        }
    }
}

/// Provider selected by the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ProviderKind {
    /// Deterministic fake provider for tests and rehearsals.
    Fake,
    /// OpenAI-compatible HTTP provider.
    OpenAi,
}

/// Approval behavior selected by the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub enum ApprovalMode {
    /// Prompt on the terminal when possible.
    Ask,
    /// Allow approvable effects.
    Allow,
    /// Deny approval requests.
    Deny,
}

/// Parsed command.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Command {
    /// Print help.
    Help,
    /// Chat without coding tools.
    Chat {
        /// Optional prompt; stdin is used when absent.
        prompt: Option<String>,
        /// Stream output when supported.
        stream: bool,
        /// Load project instructions as protected context.
        project_instructions: bool,
    },
    /// Run a coding task.
    Code {
        /// Task text.
        task: String,
        /// Emit machine-readable JSON summary.
        json: bool,
    },
    /// Review files or the current diff.
    Review {
        /// Optional path/range arguments.
        paths: Vec<String>,
        /// Emit machine-readable JSON summary.
        json: bool,
        /// Allow read-only commands if configured.
        allow_readonly_commands: bool,
    },
    /// Resume a previous session.
    Resume {
        /// Session id.
        session_id: String,
        /// Optional continuation task.
        task: Option<String>,
        /// Emit machine-readable JSON summary.
        json: bool,
    },
    /// List sessions.
    Sessions {
        /// Emit machine-readable JSON.
        json: bool,
    },
    /// Print a transcript.
    Transcript {
        /// Session id.
        session_id: String,
    },
    /// Configuration helper.
    ConfigCheck {
        /// Emit machine-readable JSON.
        json: bool,
    },
}

impl Command {
    /// Whether the command is help.
    pub fn is_help(&self) -> bool {
        matches!(self, Self::Help)
    }
}

impl CliArgs {
    /// Parses CLI arguments.
    pub fn parse<I, S>(args: I) -> Result<Self, CliError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut tokens: Vec<String> = args.into_iter().map(Into::into).collect();
        if tokens.is_empty() {
            return Ok(Self {
                global: GlobalArgs::default(),
                command: Command::Help,
            });
        }
        if tokens
            .iter()
            .any(|token| token == "-h" || token == "--help")
        {
            return Ok(Self {
                global: GlobalArgs::default(),
                command: Command::Help,
            });
        }

        let mut global = GlobalArgs::default();
        let mut index = 0;
        while index < tokens.len() {
            if !tokens[index].starts_with('-') {
                break;
            }
            let token = tokens[index].clone();
            match token.as_str() {
                "--workspace" => {
                    global.workspace =
                        Some(PathBuf::from(take_value(&tokens, &mut index, &token)?));
                }
                "--session-dir" => {
                    global.session_dir =
                        Some(PathBuf::from(take_value(&tokens, &mut index, &token)?));
                }
                "--provider" => {
                    global.provider = parse_provider(&take_value(&tokens, &mut index, &token)?)?;
                }
                "--model" => {
                    global.model = Some(take_value(&tokens, &mut index, &token)?);
                }
                "--base-url" => {
                    global.base_url = Some(take_value(&tokens, &mut index, &token)?);
                }
                "--api-key-env" => {
                    global.api_key_env = take_value(&tokens, &mut index, &token)?;
                }
                "--approval" => {
                    global.approval = parse_approval(&take_value(&tokens, &mut index, &token)?)?;
                }
                "--non-interactive" => {
                    global.non_interactive = true;
                    index += 1;
                }
                "--command-timeout" => {
                    let value = take_value(&tokens, &mut index, &token)?;
                    global.command_timeout_secs = value.parse().map_err(|_| {
                        CliError::Args("--command-timeout expects integer seconds".to_string())
                    })?;
                }
                _ => return Err(CliError::Args(format!("unknown global option: {token}"))),
            }
        }

        let command_tokens = tokens.split_off(index);
        let command = parse_command(command_tokens)?;
        Ok(Self { global, command })
    }
}

fn parse_command(tokens: Vec<String>) -> Result<Command, CliError> {
    let Some((name, rest)) = tokens.split_first() else {
        return Ok(Command::Help);
    };
    match name.as_str() {
        "chat" => parse_chat(rest),
        "code" => parse_code(rest),
        "review" => parse_review(rest),
        "resume" => parse_resume(rest),
        "sessions" => parse_sessions(rest),
        "transcript" => parse_transcript(rest),
        "config" => parse_config(rest),
        other => Err(CliError::Args(format!("unknown command: {other}"))),
    }
}

fn parse_chat(tokens: &[String]) -> Result<Command, CliError> {
    let mut stream = true;
    let mut project_instructions = false;
    let mut prompt = Vec::new();
    let mut index = 0;
    while index < tokens.len() {
        match tokens[index].as_str() {
            "--stream" => stream = true,
            "--no-stream" => stream = false,
            "--project-instructions" => project_instructions = true,
            token if token.starts_with('-') => {
                return Err(CliError::Args(format!("unknown chat option: {token}")));
            }
            token => prompt.push(token.to_string()),
        }
        index += 1;
    }
    Ok(Command::Chat {
        prompt: joined(prompt),
        stream,
        project_instructions,
    })
}

fn parse_code(tokens: &[String]) -> Result<Command, CliError> {
    let mut json = false;
    let mut task = Vec::new();
    for token in tokens {
        match token.as_str() {
            "--json" => json = true,
            value if value.starts_with('-') => {
                return Err(CliError::Args(format!("unknown code option: {value}")));
            }
            value => task.push(value.to_string()),
        }
    }
    let Some(task) = joined(task) else {
        return Err(CliError::Args("molo code requires a task".to_string()));
    };
    Ok(Command::Code { task, json })
}

fn parse_review(tokens: &[String]) -> Result<Command, CliError> {
    let mut json = false;
    let mut allow_readonly_commands = false;
    let mut paths = Vec::new();
    for token in tokens {
        match token.as_str() {
            "--json" => json = true,
            "--allow-readonly-commands" => allow_readonly_commands = true,
            value if value.starts_with('-') => {
                return Err(CliError::Args(format!("unknown review option: {value}")));
            }
            value => paths.push(value.to_string()),
        }
    }
    Ok(Command::Review {
        paths,
        json,
        allow_readonly_commands,
    })
}

fn parse_resume(tokens: &[String]) -> Result<Command, CliError> {
    let mut json = false;
    let mut positionals = Vec::new();
    for token in tokens {
        match token.as_str() {
            "--json" => json = true,
            value if value.starts_with('-') => {
                return Err(CliError::Args(format!("unknown resume option: {value}")));
            }
            value => positionals.push(value.to_string()),
        }
    }
    let Some(session_id) = positionals.first().cloned() else {
        return Err(CliError::Args(
            "molo resume requires a session id".to_string(),
        ));
    };
    Ok(Command::Resume {
        session_id,
        task: joined(positionals.into_iter().skip(1).collect()),
        json,
    })
}

fn parse_sessions(tokens: &[String]) -> Result<Command, CliError> {
    let mut json = false;
    for token in tokens {
        match token.as_str() {
            "--json" => json = true,
            value => return Err(CliError::Args(format!("unknown sessions option: {value}"))),
        }
    }
    Ok(Command::Sessions { json })
}

fn parse_transcript(tokens: &[String]) -> Result<Command, CliError> {
    if tokens.len() != 1 {
        return Err(CliError::Args(
            "molo transcript requires exactly one session id".to_string(),
        ));
    }
    Ok(Command::Transcript {
        session_id: tokens[0].clone(),
    })
}

fn parse_config(tokens: &[String]) -> Result<Command, CliError> {
    let Some((subcommand, rest)) = tokens.split_first() else {
        return Err(CliError::Args(
            "molo config requires a subcommand".to_string(),
        ));
    };
    if subcommand != "check" {
        return Err(CliError::Args(format!(
            "unknown config subcommand: {subcommand}"
        )));
    }
    let mut json = false;
    for token in rest {
        match token.as_str() {
            "--json" => json = true,
            value => {
                return Err(CliError::Args(format!(
                    "unknown config check option: {value}"
                )));
            }
        }
    }
    Ok(Command::ConfigCheck { json })
}

fn take_value(tokens: &[String], index: &mut usize, option: &str) -> Result<String, CliError> {
    let value_index = *index + 1;
    let Some(value) = tokens.get(value_index) else {
        return Err(CliError::Args(format!("{option} expects a value")));
    };
    *index += 2;
    Ok(value.clone())
}

fn parse_provider(value: &str) -> Result<ProviderKind, CliError> {
    match value {
        "fake" => Ok(ProviderKind::Fake),
        "openai" | "openai-compatible" => Ok(ProviderKind::OpenAi),
        _ => Err(CliError::Args(format!("unknown provider: {value}"))),
    }
}

fn parse_approval(value: &str) -> Result<ApprovalMode, CliError> {
    match value {
        "ask" => Ok(ApprovalMode::Ask),
        "allow" => Ok(ApprovalMode::Allow),
        "deny" => Ok(ApprovalMode::Deny),
        _ => Err(CliError::Args(format!("unknown approval mode: {value}"))),
    }
}

fn joined(values: Vec<String>) -> Option<String> {
    let text = values.join(" ");
    (!text.trim().is_empty()).then_some(text)
}

/// Returns CLI help text.
pub fn help_text() -> &'static str {
    r#"molo reference CLI

Usage:
  molo [GLOBAL_OPTIONS] chat [--stream|--no-stream] [PROMPT]
  molo [GLOBAL_OPTIONS] code [--json] TASK
  molo [GLOBAL_OPTIONS] review [--json] [PATH_OR_RANGE...]
  molo [GLOBAL_OPTIONS] resume [--json] SESSION_ID [TASK]
  molo [GLOBAL_OPTIONS] sessions [--json]
  molo [GLOBAL_OPTIONS] transcript SESSION_ID
  molo [GLOBAL_OPTIONS] config check [--json]

Global options:
  --workspace DIR          Workspace root (default: current directory)
  --session-dir DIR        Session store directory
  --provider fake|openai   Provider (default: fake)
  --model MODEL            Provider model
  --base-url URL           OpenAI-compatible base URL
  --api-key-env NAME       API key environment variable (default: OPENAI_API_KEY)
  --approval ask|allow|deny
  --non-interactive        Fail closed for required approvals
  --command-timeout SECS   Default command/effect timeout (default: 30)
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_code_command() {
        let args = CliArgs::parse([
            "--workspace",
            ".",
            "--approval",
            "deny",
            "code",
            "--json",
            "fix",
            "bug",
        ])
        .unwrap();
        assert_eq!(args.global.approval, ApprovalMode::Deny);
        assert!(matches!(args.command, Command::Code { json: true, .. }));
    }

    #[test]
    fn rejects_unknown_command() {
        let error = CliArgs::parse(["wat"]).unwrap_err();
        assert!(error.to_string().contains("unknown command"));
    }
}
