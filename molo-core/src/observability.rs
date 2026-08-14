//! Lightweight observability record types.
//!
//! These types are data shapes, not a telemetry backend. They let framework
//! events expose redacted, serializable summaries for logs, devtools, tests,
//! and metrics adapters while keeping raw prompt/model/tool content out of
//! default records.

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Current schema version for [`AgentEventRecord`].
pub const AGENT_EVENT_RECORD_SCHEMA_VERSION: u16 = 1;

/// A redaction that was applied to an exported record or text field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RedactionRecord {
    /// Field where redaction occurred.
    pub field: String,
    /// Human-readable redaction reason.
    pub reason: String,
}

/// Severity for a serializable agent event record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EventSeverity {
    /// Very detailed diagnostic event.
    Trace,
    /// Debug diagnostic event.
    Debug,
    /// Informational lifecycle event.
    Info,
    /// Warning event.
    Warn,
    /// Error event.
    Error,
}

/// Serializable, redacted event record for out-of-process observers.
///
/// `AgentEventRecord` complements the low-cost `Arc<dyn AgentEvent>` channel:
/// framework-owned events can expose sanitized JSON summaries, while custom
/// events may keep returning `None` from `AgentEvent::to_record`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentEventRecord {
    /// Record schema version. Starts at 1; payload additions should be
    /// backward-compatible.
    pub schema_version: u16,
    /// Run id when the event has one.
    pub run_id: Option<String>,
    /// Optional producer-local sequence number.
    pub sequence: Option<u64>,
    /// Record creation timestamp.
    pub timestamp: SystemTime,
    /// Event name.
    pub name: String,
    /// Event severity.
    pub severity: EventSeverity,
    /// Redacted, event-specific summary payload.
    pub payload: serde_json::Value,
    /// Redactions or omitted raw-content fields.
    pub redactions: Vec<RedactionRecord>,
}

impl AgentEventRecord {
    /// Constructs a record with the current schema version and timestamp.
    pub fn new(
        name: impl Into<String>,
        severity: EventSeverity,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            schema_version: AGENT_EVENT_RECORD_SCHEMA_VERSION,
            run_id: None,
            sequence: None,
            timestamp: SystemTime::now(),
            name: name.into(),
            severity,
            payload,
            redactions: Vec::new(),
        }
    }

    /// Sets the run id.
    pub fn with_run_id(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    /// Sets the optional sequence number.
    pub fn with_sequence(mut self, sequence: u64) -> Self {
        self.sequence = Some(sequence);
        self
    }

    /// Sets redaction records.
    pub fn with_redactions(mut self, redactions: Vec<RedactionRecord>) -> Self {
        self.redactions = redactions;
        self
    }
}
