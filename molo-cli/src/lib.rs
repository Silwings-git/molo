#![warn(missing_docs)]
//! Reference CLI for molo.
//!
//! This crate is a `publish = false` binary crate used to validate how the
//! molo provider, harness, and coding layers compose in a real command-line
//! application. CLI config, sessions, approval prompts, and final summaries
//! are intentionally private to this crate.

mod approval;
mod args;
mod commands;
mod config;
mod error;
mod output;
mod session;

use crate::args::CliArgs;
use crate::config::CliConfig;

pub use crate::error::CliError;

/// Runs the CLI from an iterator of argument strings on a current-thread
/// tokio runtime.
pub fn run_blocking<I, S>(args: I) -> Result<(), CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| CliError::Runtime(error.to_string()))?;
    runtime.block_on(run(args))
}

/// Runs the CLI from an iterator of argument strings.
pub async fn run<I, S>(args: I) -> Result<(), CliError>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let parsed = CliArgs::parse(args)?;
    if parsed.command.is_help() {
        println!("{}", args::help_text());
        return Ok(());
    }

    let config = CliConfig::from_args(&parsed)?;
    commands::dispatch(parsed.command, config).await
}
