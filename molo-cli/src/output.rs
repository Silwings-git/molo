use crate::session::{CliApprovalSummary, CliSessionStatus};
use molo::VerificationResult;
use serde::{Deserialize, Serialize};

/// Final diff summary emitted by `molo code` and `molo review`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FinalDiffSummary {
    /// Session id.
    pub session_id: String,
    /// Run ids.
    pub run_ids: Vec<String>,
    /// Goal.
    pub goal: String,
    /// Session status.
    pub status: CliSessionStatus,
    /// Changed files.
    pub changed_files: Vec<FileChangeSummary>,
    /// Dirty files present before the run.
    pub pre_existing_dirty_files: Vec<String>,
    /// Verification results.
    pub verification: Vec<VerificationResult>,
    /// Non-fatal warnings about reference CLI execution.
    pub warnings: Vec<String>,
    /// Approval summaries.
    pub approvals: Vec<CliApprovalSummary>,
    /// Denied effects.
    pub denied_effects: Vec<String>,
    /// Whether output was truncated.
    pub truncated: bool,
    /// Model answer.
    pub model_answer: String,
}

/// Changed file summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileChangeSummary {
    /// File path.
    pub path: String,
    /// Change status.
    pub status: String,
}

/// Prints a final summary.
pub fn print_final_summary(
    summary: &FinalDiffSummary,
    json: bool,
) -> Result<(), serde_json::Error> {
    if json {
        println!("{}", serde_json::to_string_pretty(summary)?);
        return Ok(());
    }
    println!("session: {}", summary.session_id);
    println!("status: {:?}", summary.status);
    println!("goal: {}", summary.goal);
    if !summary.changed_files.is_empty() {
        println!("changed files:");
        for file in &summary.changed_files {
            println!("  {} {}", file.status, file.path);
        }
    } else {
        println!("changed files: none");
    }
    if !summary.pre_existing_dirty_files.is_empty() {
        println!("pre-existing dirty files:");
        for path in &summary.pre_existing_dirty_files {
            println!("  {path}");
        }
    }
    if !summary.denied_effects.is_empty() {
        println!("denied effects:");
        for effect in &summary.denied_effects {
            println!("  {effect}");
        }
    }
    if !summary.warnings.is_empty() {
        println!("warnings:");
        for warning in &summary.warnings {
            println!("  {warning}");
        }
    }
    if !summary.model_answer.trim().is_empty() {
        println!();
        println!("{}", summary.model_answer.trim());
    }
    Ok(())
}
