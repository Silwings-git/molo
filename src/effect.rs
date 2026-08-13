//! Effect protocol: side-effect requests and observations.
//!
//! Effects are the boundary between an agent kernel and an outer harness.
//! Tools may parse model arguments into an [`EffectRequest`], but the
//! request is not executed by the tool itself. A harness or application
//! runtime classifies, approves, sandboxes, executes, audits, and returns an
//! [`EffectObservation`] for the agent to consume.

use crate::run::{Artifact, RunMetadata};
use crate::tool::ToolMemoryPolicy;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Output intended for host/UI display.
///
/// Display output is not automatically inserted into model context. The
/// model-visible text is carried separately by
/// [`ToolOutput::content`](crate::tool::ToolOutput::content) or
/// [`EffectOutput::observation_for_model`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DisplayOutput {
    /// Display format.
    pub format: DisplayFormat,
    /// Host/UI-facing content.
    pub content: String,
    /// Host-owned display metadata.
    pub metadata: RunMetadata,
}

impl DisplayOutput {
    /// Constructs display output.
    pub fn new(format: DisplayFormat, content: impl Into<String>) -> Self {
        Self {
            format,
            content: content.into(),
            metadata: RunMetadata::new(),
        }
    }

    /// Sets host-owned metadata.
    pub fn with_metadata(mut self, metadata: RunMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Display output format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DisplayFormat {
    /// Plain text.
    PlainText,
    /// Markdown.
    Markdown,
    /// JSON text.
    Json,
    /// Application-specific format, preferably namespaced.
    Custom(String),
}

/// Kind of side effect requested by an agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EffectKind {
    /// Read a file or file-like workspace resource.
    ReadFile,
    /// Write a file or file-like workspace resource.
    WriteFile,
    /// Apply a patch.
    ApplyPatch,
    /// Search data, files, or indexes.
    Search,
    /// Execute a command.
    ExecuteCommand,
    /// Inspect or mutate git state.
    Git,
    /// Perform network I/O.
    Network,
    /// Drive a browser.
    Browser,
    /// Call an MCP server or MCP-like adapter.
    Mcp,
    /// Application-specific effect kind, preferably namespaced.
    Custom(String),
}

/// Request-declared risk level.
///
/// A harness may reclassify or override this value. The requester-provided
/// risk is a signal, not an authorization decision.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RiskLevel {
    /// Low-risk operation.
    #[default]
    Low,
    /// Medium-risk operation.
    Medium,
    /// High-risk operation.
    High,
    /// Critical-risk operation.
    Critical,
}

/// Source tool call that produced an effect request.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectSource {
    /// Source model tool-call id, when produced by a tool.
    pub tool_call_id: Option<String>,
    /// Source tool name, when produced by a tool.
    pub tool_name: Option<String>,
}

/// Request for an outer harness to govern and execute a side effect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectRequest {
    /// Effect id, unique within the process or harness session.
    pub id: String,
    /// Effect kind.
    pub kind: EffectKind,
    /// User-visible approval/audit description.
    pub description: String,
    /// Kind-specific JSON payload.
    pub payload: serde_json::Value,
    /// Source tool-call metadata, when produced by a tool.
    pub source: EffectSource,
    /// Request-declared risk.
    pub risk: RiskLevel,
    /// Request-level timeout suggestion.
    pub timeout: Option<Duration>,
    /// Host/application metadata.
    pub metadata: RunMetadata,
}

impl EffectRequest {
    /// Constructs an effect request with a generated id.
    pub fn new(
        kind: EffectKind,
        description: impl Into<String>,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            id: generated_effect_id(),
            kind,
            description: description.into(),
            payload,
            source: EffectSource::default(),
            risk: RiskLevel::Low,
            timeout: None,
            metadata: RunMetadata::new(),
        }
    }

    /// Overrides the generated effect id.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Sets the source tool-call metadata.
    pub fn with_source(
        mut self,
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
    ) -> Self {
        self.source.tool_call_id = Some(tool_call_id.into());
        self.source.tool_name = Some(tool_name.into());
        self
    }

    /// Fills missing source fields without overwriting fields already set by
    /// the tool.
    pub(crate) fn with_source_if_missing(
        mut self,
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
    ) -> Self {
        if self.source.tool_call_id.is_none() {
            self.source.tool_call_id = Some(tool_call_id.into());
        }
        if self.source.tool_name.is_none() {
            self.source.tool_name = Some(tool_name.into());
        }
        self
    }

    /// Sets the request-declared risk.
    pub fn with_risk(mut self, risk: RiskLevel) -> Self {
        self.risk = risk;
        self
    }

    /// Sets the request-level timeout suggestion.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Sets host/application metadata.
    pub fn with_metadata(mut self, metadata: RunMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Terminal status of an executed effect.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EffectStatus {
    /// Effect succeeded.
    Succeeded,
    /// Effect was denied by policy or approval.
    Denied,
    /// Effect execution failed.
    Failed,
    /// Effect execution was cancelled.
    Cancelled,
    /// Effect execution timed out.
    TimedOut,
}

/// Output produced by an executed effect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectOutput {
    /// Model-visible observation text.
    pub observation_for_model: String,
    /// Optional host/UI display output.
    pub display: Option<DisplayOutput>,
    /// Artifact handles produced by the effect.
    pub artifacts: Vec<Artifact>,
    /// Memory policy for the model-visible observation.
    pub memory_policy: ToolMemoryPolicy,
    /// Host/application metadata.
    pub metadata: RunMetadata,
}

impl EffectOutput {
    /// Constructs a text observation for the model.
    pub fn text(observation_for_model: impl Into<String>) -> Self {
        Self {
            observation_for_model: observation_for_model.into(),
            display: None,
            artifacts: Vec::new(),
            memory_policy: ToolMemoryPolicy::Normal,
            metadata: RunMetadata::new(),
        }
    }

    /// Sets host/UI display output.
    pub fn with_display(mut self, display: DisplayOutput) -> Self {
        self.display = Some(display);
        self
    }

    /// Sets artifact handles.
    pub fn with_artifacts(mut self, artifacts: Vec<Artifact>) -> Self {
        self.artifacts = artifacts;
        self
    }

    /// Sets the memory policy.
    pub fn with_memory_policy(mut self, policy: ToolMemoryPolicy) -> Self {
        self.memory_policy = policy;
        self
    }

    /// Sets host/application metadata.
    pub fn with_metadata(mut self, metadata: RunMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Observation returned to an agent after an effect request is governed and
/// executed by an outer harness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectObservation {
    /// Effect id matching [`EffectRequest::id`].
    pub effect_id: String,
    /// Execution status.
    pub status: EffectStatus,
    /// Effect output.
    pub output: EffectOutput,
    /// Host/application metadata.
    pub metadata: RunMetadata,
}

impl EffectObservation {
    /// Constructs a successful text observation.
    pub fn succeeded(effect_id: impl Into<String>, observation: impl Into<String>) -> Self {
        Self {
            effect_id: effect_id.into(),
            status: EffectStatus::Succeeded,
            output: EffectOutput::text(observation),
            metadata: RunMetadata::new(),
        }
    }

    /// Sets host/application metadata.
    pub fn with_metadata(mut self, metadata: RunMetadata) -> Self {
        self.metadata = metadata;
        self
    }
}

fn generated_effect_id() -> String {
    static START_NANOS: OnceLock<u128> = OnceLock::new();
    static EFFECT_COUNTER: AtomicU64 = AtomicU64::new(0);

    let start_nanos = *START_NANOS.get_or_init(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    });
    let n = EFFECT_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("effect-{start_nanos}-{n}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn effect_request_builder_sets_fields() {
        let request = EffectRequest::new(EffectKind::ReadFile, "read", json!({"path": "a.rs"}))
            .with_id("effect-1")
            .with_source("call-1", "read_file")
            .with_risk(RiskLevel::Medium)
            .with_timeout(Duration::from_secs(5));

        assert_eq!(request.id, "effect-1");
        assert_eq!(request.source.tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(request.source.tool_name.as_deref(), Some("read_file"));
        assert_eq!(request.risk, RiskLevel::Medium);
        assert_eq!(request.timeout, Some(Duration::from_secs(5)));
    }

    #[test]
    fn effect_output_text_is_model_visible_only() {
        let output = EffectOutput::text("observed")
            .with_display(DisplayOutput::new(DisplayFormat::Markdown, "**observed**"));
        assert_eq!(output.observation_for_model, "observed");
        assert_eq!(output.display.unwrap().content, "**observed**");
    }
}
