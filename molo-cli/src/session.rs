use crate::config::CliConfigSnapshot;
use crate::error::CliError;
use async_trait::async_trait;
use molo::{
    AuditError, AuditEvent, AuditSink, GitHead, RunContext, TranscriptError, TranscriptRecord,
    TranscriptStore,
};
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);

/// CLI session status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CliSessionStatus {
    /// Run is active.
    Running,
    /// Run completed.
    Completed,
    /// Run was interrupted.
    Interrupted,
    /// Run failed.
    Failed,
}

/// Workspace fingerprint stored in the session envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceFingerprint {
    /// Current git head, when available.
    pub git_head: Option<GitHead>,
    /// Dirty files at session start.
    pub dirty_files: Vec<String>,
}

/// CLI-owned task state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CliTaskState {
    /// Task goal.
    pub goal: String,
    /// Plan items.
    pub plan: Vec<CliPlanItem>,
    /// Files changed by the agent.
    pub changed_files: Vec<String>,
    /// Verification summaries.
    pub verification: Vec<String>,
    /// Approval summaries.
    pub approvals: Vec<CliApprovalSummary>,
    /// Interruptions.
    pub interruptions: Vec<CliInterruption>,
    /// Last error.
    pub last_error: Option<String>,
}

impl CliTaskState {
    /// Constructs task state for a goal.
    pub fn new(goal: impl Into<String>) -> Self {
        Self {
            goal: goal.into(),
            plan: Vec::new(),
            changed_files: Vec::new(),
            verification: Vec::new(),
            approvals: Vec::new(),
            interruptions: Vec::new(),
            last_error: None,
        }
    }
}

/// CLI plan item.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CliPlanItem {
    /// Step text.
    pub step: String,
    /// Step status.
    pub status: String,
}

/// Approval summary stored in CLI session state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CliApprovalSummary {
    /// Effect id.
    pub effect_id: String,
    /// Effect kind.
    pub kind: String,
    /// Risk level.
    pub risk: String,
    /// Decision.
    pub decision: String,
    /// Reason.
    pub reason: String,
}

/// Interruption summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CliInterruption {
    /// Run id.
    pub run_id: String,
    /// Reason.
    pub reason: String,
}

/// CLI session envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CliSessionEnvelope {
    /// Schema version.
    pub schema_version: u16,
    /// Session id.
    pub session_id: String,
    /// Parent session id for resume sessions.
    pub parent_session_id: Option<String>,
    /// Creation timestamp.
    pub created_at: String,
    /// Update timestamp.
    pub updated_at: String,
    /// Workspace root.
    pub workspace_root: String,
    /// Workspace fingerprint.
    pub workspace_fingerprint: WorkspaceFingerprint,
    /// Command name.
    pub command: String,
    /// Goal text.
    pub goal: String,
    /// Status.
    pub status: CliSessionStatus,
    /// Redacted config snapshot.
    pub config_snapshot: CliConfigSnapshot,
    /// Task state.
    pub task_state: CliTaskState,
}

impl CliSessionEnvelope {
    /// Constructs a new session envelope.
    pub fn new(
        command: impl Into<String>,
        goal: impl Into<String>,
        workspace_root: impl Into<String>,
        fingerprint: WorkspaceFingerprint,
        config_snapshot: CliConfigSnapshot,
        parent_session_id: Option<String>,
    ) -> Self {
        let goal = goal.into();
        let now = timestamp();
        Self {
            schema_version: 1,
            session_id: generated_session_id(),
            parent_session_id,
            created_at: now.clone(),
            updated_at: now,
            workspace_root: workspace_root.into(),
            workspace_fingerprint: fingerprint,
            command: command.into(),
            goal: goal.clone(),
            status: CliSessionStatus::Running,
            config_snapshot,
            task_state: CliTaskState::new(goal),
        }
    }

    /// Marks the session with a terminal status.
    pub fn finish(&mut self, status: CliSessionStatus) {
        self.status = status;
        self.updated_at = timestamp();
    }
}

/// CLI transcript record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CliTranscriptLine {
    /// Harness transcript record.
    Harness {
        /// Run id.
        run_id: String,
        /// Record payload.
        record: Box<TranscriptRecord>,
    },
    /// CLI event.
    CliEvent {
        /// Run id.
        run_id: String,
        /// Event name.
        name: String,
        /// Event payload.
        payload: serde_json::Value,
    },
}

/// CLI audit record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CliAuditLine {
    /// Run id.
    pub run_id: String,
    /// Audit event.
    pub event: AuditEvent,
}

/// Filesystem session store.
#[derive(Debug, Clone)]
pub struct CliSessionStore {
    root: PathBuf,
}

impl CliSessionStore {
    /// Constructs a session store.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Creates the store root.
    pub fn ensure_root(&self) -> Result<(), CliError> {
        fs::create_dir_all(&self.root)?;
        Ok(())
    }

    /// Returns one session directory.
    pub fn session_dir(&self, session_id: &str) -> PathBuf {
        self.root.join(session_id)
    }

    /// Saves a session envelope atomically.
    pub fn save(&self, session: &CliSessionEnvelope) -> Result<(), CliError> {
        let dir = self.session_dir(&session.session_id);
        fs::create_dir_all(&dir)?;
        let path = dir.join("session.json");
        let tmp = dir.join("session.json.tmp");
        let json = serde_json::to_vec_pretty(session)?;
        fs::write(&tmp, json)?;
        fs::rename(tmp, path)?;
        Ok(())
    }

    /// Loads a session envelope.
    pub fn load(&self, session_id: &str) -> Result<CliSessionEnvelope, CliError> {
        let path = self.session_dir(session_id).join("session.json");
        let bytes = fs::read(path)?;
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Lists known sessions.
    pub fn list(&self) -> Result<Vec<CliSessionEnvelope>, CliError> {
        self.ensure_root()?;
        let mut sessions: Vec<CliSessionEnvelope> = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let session_path = entry.path().join("session.json");
            if !session_path.exists() {
                continue;
            }
            let bytes = fs::read(session_path)?;
            sessions.push(serde_json::from_slice(&bytes)?);
        }
        sessions.sort_by(|left, right| left.created_at.cmp(&right.created_at));
        Ok(sessions)
    }

    /// Appends a transcript line.
    pub fn append_transcript(
        &self,
        session_id: &str,
        line: &CliTranscriptLine,
    ) -> Result<(), CliError> {
        self.append_jsonl(session_id, "transcript.jsonl", line)
    }

    /// Appends an audit line.
    pub fn append_audit(&self, session_id: &str, line: &CliAuditLine) -> Result<(), CliError> {
        self.append_jsonl(session_id, "audit.jsonl", line)
    }

    /// Writes a final summary artifact.
    pub fn write_final_summary<T: Serialize>(
        &self,
        session_id: &str,
        summary: &T,
    ) -> Result<(), CliError> {
        let dir = self.session_dir(session_id);
        fs::create_dir_all(&dir)?;
        fs::write(
            dir.join("final-summary.json"),
            serde_json::to_vec_pretty(summary)?,
        )?;
        Ok(())
    }

    /// Reads transcript lines as raw JSONL text.
    pub fn transcript_text(&self, session_id: &str) -> Result<String, CliError> {
        let path = self.session_dir(session_id).join("transcript.jsonl");
        if !path.exists() {
            return Ok(String::new());
        }
        Ok(fs::read_to_string(path)?)
    }

    /// Reads audit lines as raw JSONL text.
    pub fn audit_text(&self, session_id: &str) -> Result<String, CliError> {
        let path = self.session_dir(session_id).join("audit.jsonl");
        if !path.exists() {
            return Ok(String::new());
        }
        Ok(fs::read_to_string(path)?)
    }

    fn append_jsonl<T: Serialize>(
        &self,
        session_id: &str,
        file_name: &str,
        line: &T,
    ) -> Result<(), CliError> {
        let dir = self.session_dir(session_id);
        fs::create_dir_all(&dir)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join(file_name))?;
        serde_json::to_writer(&mut file, line)?;
        file.write_all(b"\n")?;
        Ok(())
    }
}

/// JSONL audit sink backed by a CLI session.
#[derive(Debug, Clone)]
pub struct JsonlAuditSink {
    store: CliSessionStore,
    session_id: String,
}

impl JsonlAuditSink {
    /// Constructs an audit sink.
    pub fn new(store: CliSessionStore, session_id: impl Into<String>) -> Self {
        Self {
            store,
            session_id: session_id.into(),
        }
    }
}

#[async_trait]
impl AuditSink for JsonlAuditSink {
    async fn record(&self, event: AuditEvent, context: &RunContext) -> Result<(), AuditError> {
        self.store
            .append_audit(
                &self.session_id,
                &CliAuditLine {
                    run_id: context.run_id.clone(),
                    event,
                },
            )
            .map_err(|error| AuditError::Sink(error.to_string()))
    }
}

/// JSONL transcript store backed by a CLI session.
#[derive(Debug, Clone)]
pub struct JsonlTranscriptStore {
    store: CliSessionStore,
    session_id: String,
}

impl JsonlTranscriptStore {
    /// Constructs a transcript store.
    pub fn new(store: CliSessionStore, session_id: impl Into<String>) -> Self {
        Self {
            store,
            session_id: session_id.into(),
        }
    }
}

#[async_trait]
impl TranscriptStore for JsonlTranscriptStore {
    async fn append(
        &self,
        record: TranscriptRecord,
        context: &RunContext,
    ) -> Result<(), TranscriptError> {
        self.store
            .append_transcript(
                &self.session_id,
                &CliTranscriptLine::Harness {
                    run_id: context.run_id.clone(),
                    record: Box::new(record),
                },
            )
            .map_err(|error| TranscriptError::Store(error.to_string()))
    }
}

fn generated_session_id() -> String {
    let count = SESSION_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("s-{}-{count}", timestamp_compact())
}

fn timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}.{:03}Z", now.as_secs(), now.subsec_millis())
}

fn timestamp_compact() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saves_and_loads_session() {
        let root =
            std::env::temp_dir().join(format!("molo-cli-session-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let store = CliSessionStore::new(&root);
        let session = CliSessionEnvelope::new(
            "chat",
            "hello",
            ".",
            WorkspaceFingerprint {
                git_head: None,
                dirty_files: Vec::new(),
            },
            crate::config::CliConfigSnapshot {
                workspace_root: ".".to_string(),
                session_dir: root.display().to_string(),
                provider: crate::config::ProviderConfigSnapshot {
                    kind: crate::args::ProviderKind::Fake,
                    model: "fake".to_string(),
                    base_url: None,
                    api_key_env: "OPENAI_API_KEY".to_string(),
                },
                policy: crate::config::PolicyConfig {
                    sandbox: molo::SandboxPolicy::WorkspaceWrite,
                    network: molo::NetworkPolicy::Deny,
                    approval: crate::args::ApprovalMode::Ask,
                    command_timeout: std::time::Duration::from_secs(30),
                },
                non_interactive: false,
            },
            None,
        );
        store.save(&session).unwrap();
        let loaded = store.load(&session.session_id).unwrap();
        assert_eq!(loaded.goal, "hello");
        let _ = fs::remove_dir_all(root);
    }
}
