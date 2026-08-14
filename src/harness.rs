//! Harness runtime: governed execution for effect-producing agents.
//!
//! A harness sits outside an [`AgentKernel`]. The kernel
//! decides the next action; the harness runtime executes provider requests
//! and routes side-effect requests through policy, approval, executor,
//! output limiting, audit, and transcript recording before feeding
//! observations back to the kernel.
//!
//! This module deliberately does not include production filesystem, shell,
//! git, browser, or MCP executors. Those belong in higher layers such as a
//! coding-workload SDK. The built-in executors here are for tests,
//! applications that provide their own effect semantics, and validating the
//! runtime boundary.

use crate::agent::{
    AgentAction, AgentError, AgentKernel, ModelObservation, ModelRequest, Observation,
};
use crate::effect::{
    DisplayOutput, EffectKind, EffectObservation, EffectOutput, EffectRequest, EffectStatus,
    RiskLevel,
};
use crate::provider::{Provider, ProviderError, ProviderRequestContext};
use crate::run::{Artifact, RunContext, RunMetadata, RunOutput, RunRequest};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub use crate::observability::RedactionRecord;

/// Outer runtime that drives an [`AgentKernel`] with a [`Provider`] and
/// governed [`Harness`].
///
/// The runtime owns model and effect execution. The agent kernel only
/// maintains reasoning state and requests the next action.
#[derive(Debug)]
pub struct HarnessRuntime<P, H> {
    provider: P,
    harness: H,
    config: HarnessRuntimeConfig,
}

/// Runtime loop configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessRuntimeConfig {
    /// Maximum number of agent actions before the runtime fails the run.
    pub max_agent_steps: usize,
    /// Intended upper bound for batch effect concurrency.
    ///
    /// The default [`Harness::execute_batch`] is sequential. Custom harness
    /// implementations can read their own configuration to use this value's
    /// semantic equivalent.
    pub max_effect_batch_concurrency: usize,
    /// Whether a batch effect runtime should fail the whole run on the first
    /// harness error.
    ///
    /// Effect-level denied/failed/timed-out results should normally be
    /// represented as [`EffectObservation`] values, not as runtime errors.
    pub fail_fast_effect_batches: bool,
}

impl Default for HarnessRuntimeConfig {
    fn default() -> Self {
        Self {
            max_agent_steps: 256,
            max_effect_batch_concurrency: 1,
            fail_fast_effect_batches: false,
        }
    }
}

impl<P, H> HarnessRuntime<P, H>
where
    P: Provider,
    H: Harness,
{
    /// Constructs a runtime from a provider and harness.
    pub fn new(provider: P, harness: H) -> Self {
        Self {
            provider,
            harness,
            config: HarnessRuntimeConfig::default(),
        }
    }

    /// Replaces runtime configuration.
    pub fn with_config(mut self, config: HarnessRuntimeConfig) -> Self {
        self.config = config;
        self
    }

    /// Runs a kernel to completion.
    ///
    /// # Errors
    ///
    /// Returns [`HarnessRuntimeError::Provider`] when model communication
    /// fails, [`HarnessRuntimeError::Harness`] when governance infrastructure
    /// fails, [`HarnessRuntimeError::Agent`] when the kernel rejects a step,
    /// and [`HarnessRuntimeError::TooManyAgentSteps`] when the configured
    /// step limit is exceeded.
    pub async fn run<K>(
        &self,
        kernel: &mut K,
        request: RunRequest,
        context: RunContext,
    ) -> Result<RunOutput, HarnessRuntimeError>
    where
        K: AgentKernel,
    {
        check_run_context(&context)?;
        let mut action = kernel.start(request, &context).await?;
        for _ in 0..self.config.max_agent_steps {
            check_run_context(&context)?;
            let observation = match action {
                AgentAction::Respond { output } => return Ok(output),
                AgentAction::RequestModel { request } => {
                    Observation::Model(self.execute_model_request(request, &context).await?)
                }
                AgentAction::RequestEffect { request } => {
                    let observation = self.harness.execute(request, &context).await?;
                    Observation::Effect(observation)
                }
                AgentAction::RequestEffects { requests } => {
                    let observations = self.harness.execute_batch(requests, &context).await?;
                    Observation::Effects(observations)
                }
            };
            action = kernel.observe(observation, &context).await?;
        }
        Err(HarnessRuntimeError::TooManyAgentSteps(
            self.config.max_agent_steps,
        ))
    }

    async fn execute_model_request(
        &self,
        request: ModelRequest,
        context: &RunContext,
    ) -> Result<ModelObservation, ProviderError> {
        let request_id = request.id;
        let provider_context = ProviderRequestContext::from_run_context(&request_id, context);
        let response = self
            .provider
            .chat_with_context(request.chat, &provider_context)
            .await?;
        Ok(ModelObservation::new(request_id, response))
    }
}

/// Governs and executes one or more effect requests.
#[async_trait]
pub trait Harness: Send + Sync {
    /// Executes one effect through the harness lifecycle.
    async fn execute(
        &self,
        request: EffectRequest,
        context: &RunContext,
    ) -> Result<EffectObservation, HarnessError>;

    /// Executes a batch of effects.
    ///
    /// The default implementation executes sequentially and returns
    /// observations in request order.
    async fn execute_batch(
        &self,
        requests: Vec<EffectRequest>,
        context: &RunContext,
    ) -> Result<Vec<EffectObservation>, HarnessError> {
        let mut observations = Vec::with_capacity(requests.len());
        for request in requests {
            observations.push(self.execute(request, context).await?);
        }
        Ok(observations)
    }
}

/// Classified effect request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassifiedEffect {
    /// Original request.
    pub request: EffectRequest,
    /// Risk declared by the requester.
    pub requested_risk: RiskLevel,
    /// Harness-classified effective risk.
    pub effective_risk: RiskLevel,
    /// Human-readable classification reasons.
    pub reasons: Vec<String>,
    /// Classification metadata.
    pub metadata: RunMetadata,
}

/// Classifies effect risk before policy evaluation.
#[async_trait]
pub trait RiskClassifier: Send + Sync {
    /// Classifies a request.
    async fn classify(
        &self,
        request: EffectRequest,
        context: &RunContext,
    ) -> Result<ClassifiedEffect, HarnessError>;
}

/// Conservative default risk classifier.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultRiskClassifier;

#[async_trait]
impl RiskClassifier for DefaultRiskClassifier {
    async fn classify(
        &self,
        request: EffectRequest,
        _context: &RunContext,
    ) -> Result<ClassifiedEffect, HarnessError> {
        validate_effect_request(&request)?;
        let requested_risk = request.risk;
        let kind_floor = risk_floor_for_kind(&request.kind);
        let mut effective_risk = max_risk(requested_risk, kind_floor);
        let mut reasons = vec![format!("kind floor: {:?}", kind_floor)];

        let payload_text = request.payload.to_string().to_ascii_lowercase();
        let description = request.description.to_ascii_lowercase();
        let combined = format!("{description} {payload_text}");
        if contains_critical_pattern(&combined) {
            effective_risk = max_risk(effective_risk, RiskLevel::Critical);
            reasons.push("critical payload pattern".to_string());
        } else if contains_high_pattern(&combined) {
            effective_risk = max_risk(effective_risk, RiskLevel::High);
            reasons.push("high-risk payload pattern".to_string());
        }

        Ok(ClassifiedEffect {
            request,
            requested_risk,
            effective_risk,
            reasons,
            metadata: RunMetadata::new(),
        })
    }
}

/// Policy decision for a classified effect.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PolicyDecision {
    /// Allow execution without approval.
    Allow,
    /// Deny execution.
    Deny {
        /// Denial reason.
        reason: String,
    },
    /// Ask an approval broker before execution.
    RequireApproval {
        /// Approval reason.
        reason: String,
    },
}

/// Evaluates host policy for a classified effect.
#[async_trait]
pub trait PolicyEngine: Send + Sync {
    /// Evaluates policy.
    async fn evaluate(
        &self,
        effect: &ClassifiedEffect,
        context: &RunContext,
    ) -> Result<PolicyDecision, HarnessError>;
}

/// Risk-based default policy.
///
/// Low and medium risk are allowed by default, high risk requires approval,
/// and critical risk is denied.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultPolicyEngine;

#[async_trait]
impl PolicyEngine for DefaultPolicyEngine {
    async fn evaluate(
        &self,
        effect: &ClassifiedEffect,
        _context: &RunContext,
    ) -> Result<PolicyDecision, HarnessError> {
        Ok(match effect.effective_risk {
            RiskLevel::Low | RiskLevel::Medium => PolicyDecision::Allow,
            RiskLevel::High => PolicyDecision::RequireApproval {
                reason: "high-risk effect requires approval".to_string(),
            },
            RiskLevel::Critical => PolicyDecision::Deny {
                reason: "critical-risk effect denied by default policy".to_string(),
            },
        })
    }
}

/// Approval request passed to an [`ApprovalBroker`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalRequest {
    /// Run id.
    pub run_id: String,
    /// Effect id.
    pub effect_id: String,
    /// Effect kind.
    pub kind: EffectKind,
    /// Request description.
    pub description: String,
    /// Effective risk.
    pub risk: RiskLevel,
    /// Approval reason.
    pub reason: String,
    /// Short payload summary.
    pub payload_summary: String,
    /// Sandbox policy that would be used for execution.
    pub sandbox: SandboxPolicy,
    /// Network policy that would be used for execution.
    pub network: NetworkPolicy,
    /// Approval metadata.
    pub metadata: RunMetadata,
}

/// Approval decision.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ApprovalDecision {
    /// Allow only this request.
    AllowOnce,
    /// Allow matching requests for this session.
    AllowForSession,
    /// Deny execution.
    Deny {
        /// Denial reason.
        reason: String,
    },
}

/// Broker that obtains approval from an application-specific authority.
#[async_trait]
pub trait ApprovalBroker: Send + Sync {
    /// Approves or denies a request.
    async fn approve(
        &self,
        request: ApprovalRequest,
        context: &RunContext,
    ) -> Result<ApprovalDecision, ApprovalError>;
}

/// Approval broker that always allows requests.
#[derive(Debug, Default, Clone, Copy)]
pub struct AlwaysAllowApprovalBroker;

#[async_trait]
impl ApprovalBroker for AlwaysAllowApprovalBroker {
    async fn approve(
        &self,
        _request: ApprovalRequest,
        _context: &RunContext,
    ) -> Result<ApprovalDecision, ApprovalError> {
        Ok(ApprovalDecision::AllowOnce)
    }
}

/// Approval broker that always denies requests.
#[derive(Debug, Default, Clone, Copy)]
pub struct AlwaysDenyApprovalBroker;

#[async_trait]
impl ApprovalBroker for AlwaysDenyApprovalBroker {
    async fn approve(
        &self,
        _request: ApprovalRequest,
        _context: &RunContext,
    ) -> Result<ApprovalDecision, ApprovalError> {
        Ok(ApprovalDecision::Deny {
            reason: "denied by approval broker".to_string(),
        })
    }
}

/// Static approval broker configured with a single decision.
#[derive(Debug, Clone)]
pub struct StaticApprovalBroker {
    decision: ApprovalDecision,
}

impl StaticApprovalBroker {
    /// Constructs a static broker.
    pub fn new(decision: ApprovalDecision) -> Self {
        Self { decision }
    }
}

#[async_trait]
impl ApprovalBroker for StaticApprovalBroker {
    async fn approve(
        &self,
        _request: ApprovalRequest,
        _context: &RunContext,
    ) -> Result<ApprovalDecision, ApprovalError> {
        Ok(self.decision.clone())
    }
}

/// Filesystem/process sandbox policy requested of an executor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SandboxPolicy {
    /// Read-only access.
    ReadOnly,
    /// Writes only inside the workspace.
    WorkspaceWrite,
    /// Full host access.
    FullAccess,
    /// Application-specific sandbox.
    Custom(String),
}

/// Network access policy requested of an executor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum NetworkPolicy {
    /// No network access.
    Deny,
    /// Access only to listed hosts or patterns.
    AllowListed(Vec<String>),
    /// Unrestricted network access.
    AllowAll,
    /// Application-specific network policy.
    Custom(String),
}

/// Output size limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputLimit {
    /// Maximum model-visible output bytes.
    pub model_bytes: usize,
    /// Maximum display output bytes.
    pub display_bytes: usize,
    /// Maximum debug output bytes.
    pub debug_bytes: usize,
}

impl Default for OutputLimit {
    fn default() -> Self {
        Self {
            model_bytes: 64 * 1024,
            display_bytes: 256 * 1024,
            debug_bytes: 16 * 1024,
        }
    }
}

/// Execution policy passed to an [`EffectExecutor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionPolicy {
    /// Sandbox policy.
    pub sandbox: SandboxPolicy,
    /// Network policy.
    pub network: NetworkPolicy,
    /// Execution timeout.
    pub timeout: Option<Duration>,
    /// Output limits.
    pub output_limit: OutputLimit,
}

/// Executes an already approved effect.
#[async_trait]
pub trait EffectExecutor: Send + Sync {
    /// Executes an effect under the provided policy.
    async fn execute(
        &self,
        request: &EffectRequest,
        policy: &ExecutionPolicy,
        context: &RunContext,
    ) -> Result<RawEffectOutput, ExecutionError>;
}

/// Raw executor output before limiter/redactor processing.
#[derive(Debug, Clone, PartialEq)]
pub struct RawEffectOutput {
    /// Model-visible observation text.
    pub observation_for_model: String,
    /// Optional host/UI display output.
    pub display: Option<DisplayOutput>,
    /// Artifact handles produced by execution.
    pub artifacts: Vec<Artifact>,
    /// Executor metadata.
    pub metadata: RunMetadata,
    /// Debug text that is never fed to the model.
    pub debug: Option<String>,
}

impl RawEffectOutput {
    /// Constructs raw text output.
    pub fn text(observation_for_model: impl Into<String>) -> Self {
        Self {
            observation_for_model: observation_for_model.into(),
            display: None,
            artifacts: Vec::new(),
            metadata: RunMetadata::new(),
            debug: None,
        }
    }

    /// Sets display output.
    pub fn with_display(mut self, display: DisplayOutput) -> Self {
        self.display = Some(display);
        self
    }

    /// Sets artifact handles.
    pub fn with_artifacts(mut self, artifacts: Vec<Artifact>) -> Self {
        self.artifacts = artifacts;
        self
    }

    /// Sets metadata.
    pub fn with_metadata(mut self, metadata: RunMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Sets debug text.
    pub fn with_debug(mut self, debug: impl Into<String>) -> Self {
        self.debug = Some(debug.into());
        self
    }
}

/// Executor that refuses every effect without performing side effects.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopEffectExecutor;

#[async_trait]
impl EffectExecutor for NoopEffectExecutor {
    async fn execute(
        &self,
        request: &EffectRequest,
        _policy: &ExecutionPolicy,
        _context: &RunContext,
    ) -> Result<RawEffectOutput, ExecutionError> {
        Err(ExecutionError::Unsupported(format!(
            "effect kind {:?} is not supported by NoopEffectExecutor",
            request.kind
        )))
    }
}

/// Test executor that returns preconfigured outputs by effect id.
#[derive(Debug, Clone, Default)]
pub struct StaticEffectExecutor {
    outputs: BTreeMap<String, Result<RawEffectOutput, ExecutionError>>,
}

impl StaticEffectExecutor {
    /// Constructs an empty static executor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a successful output for an effect id.
    pub fn with_output(mut self, effect_id: impl Into<String>, output: RawEffectOutput) -> Self {
        self.outputs.insert(effect_id.into(), Ok(output));
        self
    }

    /// Adds a failure for an effect id.
    pub fn with_error(mut self, effect_id: impl Into<String>, error: ExecutionError) -> Self {
        self.outputs.insert(effect_id.into(), Err(error));
        self
    }
}

#[async_trait]
impl EffectExecutor for StaticEffectExecutor {
    async fn execute(
        &self,
        request: &EffectRequest,
        _policy: &ExecutionPolicy,
        _context: &RunContext,
    ) -> Result<RawEffectOutput, ExecutionError> {
        match self.outputs.get(&request.id) {
            Some(Ok(output)) => Ok(output.clone()),
            Some(Err(error)) => Err(error.clone()),
            None => Err(ExecutionError::Unsupported(format!(
                "no static output for effect {}",
                request.id
            ))),
        }
    }
}

/// Executor that dispatches by [`EffectKind`].
#[derive(Default, Clone)]
pub struct RouterEffectExecutor {
    routes: HashMap<EffectKindKey, Arc<dyn EffectExecutor>>,
}

impl fmt::Debug for RouterEffectExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RouterEffectExecutor")
            .field("routes", &self.routes.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl RouterEffectExecutor {
    /// Constructs an empty router.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers an executor for an effect kind.
    pub fn route(mut self, kind: EffectKind, executor: impl EffectExecutor + 'static) -> Self {
        self.routes
            .insert(EffectKindKey::from(kind), Arc::new(executor));
        self
    }
}

#[async_trait]
impl EffectExecutor for RouterEffectExecutor {
    async fn execute(
        &self,
        request: &EffectRequest,
        policy: &ExecutionPolicy,
        context: &RunContext,
    ) -> Result<RawEffectOutput, ExecutionError> {
        let key = EffectKindKey::from(request.kind.clone());
        let Some(executor) = self.routes.get(&key) else {
            return Err(ExecutionError::Unsupported(format!(
                "no executor registered for effect kind {:?}",
                request.kind
            )));
        };
        executor.execute(request, policy, context).await
    }
}

/// Output after limiting and redaction.
#[derive(Debug, Clone, PartialEq)]
pub struct LimitedOutput {
    /// Effect output.
    pub output: EffectOutput,
    /// Whether any field was truncated.
    pub truncated: bool,
    /// Redaction records.
    pub redactions: Vec<RedactionRecord>,
    /// Redacted debug text.
    pub debug: Option<String>,
}

/// Redacted text and metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedactedText {
    /// Redacted text.
    pub text: String,
    /// Redaction records.
    pub redactions: Vec<RedactionRecord>,
}

/// Redacts executor output before model/audit/transcript use.
pub trait Redactor: Send + Sync {
    /// Redacts model-visible text.
    fn redact_model_text(&self, text: &str) -> RedactedText;

    /// Redacts display text.
    fn redact_display_text(&self, text: &str) -> RedactedText;

    /// Redacts debug text.
    fn redact_debug_text(&self, text: &str) -> RedactedText;

    /// Redacts metadata.
    fn redact_metadata(&self, metadata: RunMetadata) -> RunMetadata;
}

/// Redactor that leaves output unchanged.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopRedactor;

impl Redactor for NoopRedactor {
    fn redact_model_text(&self, text: &str) -> RedactedText {
        RedactedText {
            text: text.to_string(),
            redactions: Vec::new(),
        }
    }

    fn redact_display_text(&self, text: &str) -> RedactedText {
        RedactedText {
            text: text.to_string(),
            redactions: Vec::new(),
        }
    }

    fn redact_debug_text(&self, text: &str) -> RedactedText {
        RedactedText {
            text: text.to_string(),
            redactions: Vec::new(),
        }
    }

    fn redact_metadata(&self, metadata: RunMetadata) -> RunMetadata {
        metadata
    }
}

/// Secret-pattern redactor for examples and tests.
///
/// This is intentionally simple and deterministic; production users should
/// provide a redactor that matches their secret taxonomy.
#[derive(Debug, Clone)]
pub struct PatternRedactor {
    patterns: Vec<String>,
    replacement: String,
}

impl PatternRedactor {
    /// Constructs a pattern redactor.
    pub fn new(patterns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            patterns: patterns.into_iter().map(Into::into).collect(),
            replacement: "[REDACTED]".to_string(),
        }
    }

    /// Sets the replacement text.
    pub fn with_replacement(mut self, replacement: impl Into<String>) -> Self {
        self.replacement = replacement.into();
        self
    }

    fn redact_field(&self, field: &str, text: &str) -> RedactedText {
        let mut redacted = text.to_string();
        let mut records = Vec::new();
        for pattern in &self.patterns {
            if pattern.is_empty() || !redacted.contains(pattern) {
                continue;
            }
            redacted = redacted.replace(pattern, &self.replacement);
            records.push(RedactionRecord {
                field: field.to_string(),
                reason: "pattern match".to_string(),
            });
        }
        RedactedText {
            text: redacted,
            redactions: records,
        }
    }
}

impl Redactor for PatternRedactor {
    fn redact_model_text(&self, text: &str) -> RedactedText {
        self.redact_field("model", text)
    }

    fn redact_display_text(&self, text: &str) -> RedactedText {
        self.redact_field("display", text)
    }

    fn redact_debug_text(&self, text: &str) -> RedactedText {
        self.redact_field("debug", text)
    }

    fn redact_metadata(&self, mut metadata: RunMetadata) -> RunMetadata {
        for value in metadata.values_mut() {
            redact_json_value(value, &self.patterns, &self.replacement);
        }
        metadata
    }
}

/// Reliable effect-governance audit event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AuditEvent {
    /// Effect request received.
    EffectRequested {
        /// Effect id.
        effect_id: String,
        /// Effect kind.
        kind: EffectKind,
        /// Description.
        description: String,
        /// Requested risk.
        risk: RiskLevel,
    },
    /// Effect risk was classified.
    EffectClassified {
        /// Effect id.
        effect_id: String,
        /// Requested risk.
        requested_risk: RiskLevel,
        /// Effective risk.
        effective_risk: RiskLevel,
        /// Classification reasons.
        reasons: Vec<String>,
    },
    /// Policy decision was made.
    PolicyDecided {
        /// Effect id.
        effect_id: String,
        /// Decision summary.
        decision: String,
    },
    /// Approval was requested.
    ApprovalRequested {
        /// Effect id.
        effect_id: String,
        /// Reason.
        reason: String,
    },
    /// Approval decision was made.
    ApprovalDecided {
        /// Effect id.
        effect_id: String,
        /// Decision summary.
        decision: String,
    },
    /// Effect execution started.
    EffectStarted {
        /// Effect id.
        effect_id: String,
        /// Execution policy.
        policy: ExecutionPolicySummary,
    },
    /// Effect completed successfully.
    EffectCompleted {
        /// Effect id.
        effect_id: String,
        /// Whether output was truncated.
        truncated: bool,
    },
    /// Effect was denied.
    EffectDenied {
        /// Effect id.
        effect_id: String,
        /// Denial reason.
        reason: String,
    },
    /// Effect failed.
    EffectFailed {
        /// Effect id.
        effect_id: String,
        /// Failure reason.
        reason: String,
    },
    /// Effect timed out.
    EffectTimedOut {
        /// Effect id.
        effect_id: String,
        /// Timeout reason.
        reason: String,
    },
    /// Effect was cancelled.
    EffectCancelled {
        /// Effect id.
        effect_id: String,
        /// Cancellation reason.
        reason: String,
    },
}

/// Serializable summary of an execution policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionPolicySummary {
    /// Sandbox policy.
    pub sandbox: String,
    /// Network policy.
    pub network: String,
    /// Timeout in milliseconds.
    pub timeout_ms: Option<u64>,
}

/// Reliable audit sink.
#[async_trait]
pub trait AuditSink: Send + Sync {
    /// Records an audit event.
    async fn record(&self, event: AuditEvent, context: &RunContext) -> Result<(), AuditError>;
}

/// Explicit opt-out audit sink.
///
/// This sink performs no reliable recording and should only be used in
/// tests, examples, or applications that have an equivalent audit path.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopAuditSink;

#[async_trait]
impl AuditSink for NoopAuditSink {
    async fn record(&self, _event: AuditEvent, _context: &RunContext) -> Result<(), AuditError> {
        Ok(())
    }
}

/// In-memory audit sink useful for tests.
#[derive(Debug, Default, Clone)]
pub struct VecAuditSink {
    events: Arc<Mutex<Vec<AuditEvent>>>,
}

impl VecAuditSink {
    /// Constructs an empty sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns recorded events.
    pub fn events(&self) -> Vec<AuditEvent> {
        self.events
            .lock()
            .expect("VecAuditSink lock poisoned")
            .clone()
    }
}

#[async_trait]
impl AuditSink for VecAuditSink {
    async fn record(&self, event: AuditEvent, _context: &RunContext) -> Result<(), AuditError> {
        self.events
            .lock()
            .expect("VecAuditSink lock poisoned")
            .push(event);
        Ok(())
    }
}

/// Transcript record for run replay and debugging.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TranscriptRecord {
    /// Run started.
    RunStarted {
        /// Run id.
        run_id: String,
        /// Run request.
        request: RunRequest,
    },
    /// Agent action summary.
    AgentAction {
        /// Run id.
        run_id: String,
        /// Action summary.
        action: AgentActionSummary,
    },
    /// Model observation summary.
    ModelObservation {
        /// Run id.
        run_id: String,
        /// Model request id.
        request_id: String,
        /// Model summary.
        summary: ModelSummary,
    },
    /// Effect observation summary.
    EffectObservation {
        /// Run id.
        run_id: String,
        /// Effect id.
        effect_id: String,
        /// Effect status.
        status: EffectStatus,
    },
    /// Run completed.
    RunCompleted {
        /// Run id.
        run_id: String,
        /// Run output.
        output: RunOutput,
    },
    /// Run failed.
    RunFailed {
        /// Run id.
        run_id: String,
        /// Error summary.
        error: String,
    },
}

/// Summary of an agent action for transcript records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AgentActionSummary {
    /// Final response.
    Respond,
    /// Model request.
    RequestModel {
        /// Model request id.
        request_id: String,
    },
    /// Effect request.
    RequestEffect {
        /// Effect id.
        effect_id: String,
    },
    /// Batch effect request.
    RequestEffects {
        /// Effect ids.
        effect_ids: Vec<String>,
    },
}

/// Summary of a model observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSummary {
    /// Assistant content byte length.
    pub content_bytes: usize,
    /// Tool calls in the response.
    pub tool_calls: usize,
}

/// Transcript store for resumable run traces.
#[async_trait]
pub trait TranscriptStore: Send + Sync {
    /// Appends one transcript record.
    async fn append(
        &self,
        record: TranscriptRecord,
        context: &RunContext,
    ) -> Result<(), TranscriptError>;
}

/// Transcript store that drops all records.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopTranscriptStore;

#[async_trait]
impl TranscriptStore for NoopTranscriptStore {
    async fn append(
        &self,
        _record: TranscriptRecord,
        _context: &RunContext,
    ) -> Result<(), TranscriptError> {
        Ok(())
    }
}

/// In-memory transcript store useful for tests.
#[derive(Debug, Default, Clone)]
pub struct VecTranscriptStore {
    records: Arc<Mutex<Vec<TranscriptRecord>>>,
}

impl VecTranscriptStore {
    /// Constructs an empty transcript store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns recorded transcript entries.
    pub fn records(&self) -> Vec<TranscriptRecord> {
        self.records
            .lock()
            .expect("VecTranscriptStore lock poisoned")
            .clone()
    }
}

#[async_trait]
impl TranscriptStore for VecTranscriptStore {
    async fn append(
        &self,
        record: TranscriptRecord,
        _context: &RunContext,
    ) -> Result<(), TranscriptError> {
        self.records
            .lock()
            .expect("VecTranscriptStore lock poisoned")
            .push(record);
        Ok(())
    }
}

/// Minimal in-process harness implementation.
pub struct BasicHarness<E, P, A, S, T> {
    executor: E,
    policy: P,
    approval: A,
    audit: S,
    transcript: T,
    classifier: DefaultRiskClassifier,
    redactor: Arc<dyn Redactor>,
    session_approvals: Mutex<Vec<SessionApproval>>,
    config: HarnessConfig,
}

impl<E, P, A, S, T> fmt::Debug for BasicHarness<E, P, A, S, T>
where
    E: fmt::Debug,
    P: fmt::Debug,
    A: fmt::Debug,
    S: fmt::Debug,
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BasicHarness")
            .field("executor", &self.executor)
            .field("policy", &self.policy)
            .field("approval", &self.approval)
            .field("audit", &self.audit)
            .field("transcript", &self.transcript)
            .field("classifier", &self.classifier)
            .field("redactor", &"dyn Redactor")
            .field("session_approvals", &self.session_approvals)
            .field("config", &self.config)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SessionApproval {
    run_id: String,
    kind: EffectKindKey,
    risk_ceiling: RiskLevel,
}

/// Harness configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessConfig {
    /// Default sandbox policy.
    pub default_sandbox: SandboxPolicy,
    /// Default network policy.
    pub default_network: NetworkPolicy,
    /// Default timeout.
    pub default_timeout: Duration,
    /// Output limits.
    pub output_limit: OutputLimit,
    /// Whether audit failures stop execution.
    pub fail_closed_on_audit_error: bool,
    /// Whether transcript failures stop execution.
    pub fail_closed_on_transcript_error: bool,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            default_sandbox: SandboxPolicy::ReadOnly,
            default_network: NetworkPolicy::Deny,
            default_timeout: Duration::from_secs(30),
            output_limit: OutputLimit::default(),
            fail_closed_on_audit_error: true,
            fail_closed_on_transcript_error: false,
        }
    }
}

impl
    BasicHarness<
        NoopEffectExecutor,
        DefaultPolicyEngine,
        AlwaysDenyApprovalBroker,
        NoopAuditSink,
        NoopTranscriptStore,
    >
{
    /// Constructs a harness that performs no side effects.
    pub fn noop() -> Self {
        Self::new(
            NoopEffectExecutor,
            DefaultPolicyEngine,
            AlwaysDenyApprovalBroker,
            NoopAuditSink,
            NoopTranscriptStore,
        )
    }
}

impl<E, P, A, S, T> BasicHarness<E, P, A, S, T>
where
    E: EffectExecutor,
    P: PolicyEngine,
    A: ApprovalBroker,
    S: AuditSink,
    T: TranscriptStore,
{
    /// Constructs a harness from its lifecycle components.
    pub fn new(executor: E, policy: P, approval: A, audit: S, transcript: T) -> Self {
        Self {
            executor,
            policy,
            approval,
            audit,
            transcript,
            classifier: DefaultRiskClassifier,
            redactor: Arc::new(NoopRedactor),
            session_approvals: Mutex::new(Vec::new()),
            config: HarnessConfig::default(),
        }
    }

    /// Replaces harness configuration.
    pub fn with_config(mut self, config: HarnessConfig) -> Self {
        self.config = config;
        self
    }

    /// Uses the default no-op redactor.
    pub fn with_noop_redactor(mut self) -> Self {
        self.redactor = Arc::new(NoopRedactor);
        self
    }

    /// Replaces the output redactor used before observations are returned,
    /// audited, or stored in transcripts.
    pub fn with_redactor(mut self, redactor: impl Redactor + 'static) -> Self {
        self.redactor = Arc::new(redactor);
        self
    }

    async fn audit(&self, event: AuditEvent, context: &RunContext) -> Result<(), HarnessError> {
        match self.audit.record(event, context).await {
            Ok(()) => Ok(()),
            Err(error) if self.config.fail_closed_on_audit_error => Err(error.into()),
            Err(_) => Ok(()),
        }
    }

    async fn transcript(
        &self,
        record: TranscriptRecord,
        context: &RunContext,
    ) -> Result<(), HarnessError> {
        match self.transcript.append(record, context).await {
            Ok(()) => Ok(()),
            Err(error) if self.config.fail_closed_on_transcript_error => Err(error.into()),
            Err(_) => Ok(()),
        }
    }
}

#[async_trait]
impl<E, P, A, S, T> Harness for BasicHarness<E, P, A, S, T>
where
    E: EffectExecutor,
    P: PolicyEngine,
    A: ApprovalBroker,
    S: AuditSink,
    T: TranscriptStore,
{
    async fn execute(
        &self,
        request: EffectRequest,
        context: &RunContext,
    ) -> Result<EffectObservation, HarnessError> {
        check_run_context(context)?;
        validate_effect_request(&request)?;
        self.audit(
            AuditEvent::EffectRequested {
                effect_id: request.id.clone(),
                kind: request.kind.clone(),
                description: request.description.clone(),
                risk: request.risk,
            },
            context,
        )
        .await?;

        let classified = self.classifier.classify(request, context).await?;
        self.audit(
            AuditEvent::EffectClassified {
                effect_id: classified.request.id.clone(),
                requested_risk: classified.requested_risk,
                effective_risk: classified.effective_risk,
                reasons: classified.reasons.clone(),
            },
            context,
        )
        .await?;

        let execution_policy = self.execution_policy(&classified, context);
        let decision = self.policy.evaluate(&classified, context).await?;
        self.audit(
            AuditEvent::PolicyDecided {
                effect_id: classified.request.id.clone(),
                decision: policy_decision_summary(&decision),
            },
            context,
        )
        .await?;

        match decision {
            PolicyDecision::Allow => {}
            PolicyDecision::Deny { reason } => {
                return self
                    .denied_observation(classified.request, reason, context)
                    .await;
            }
            PolicyDecision::RequireApproval { reason } => {
                if self.is_session_approved(&classified, context) {
                    self.audit(
                        AuditEvent::ApprovalDecided {
                            effect_id: classified.request.id.clone(),
                            decision: "allow for session".to_string(),
                        },
                        context,
                    )
                    .await?;
                } else {
                    let approval_request = ApprovalRequest {
                        run_id: context.run_id.clone(),
                        effect_id: classified.request.id.clone(),
                        kind: classified.request.kind.clone(),
                        description: classified.request.description.clone(),
                        risk: classified.effective_risk,
                        reason: reason.clone(),
                        payload_summary: payload_summary(&classified.request.payload),
                        sandbox: execution_policy.sandbox.clone(),
                        network: execution_policy.network.clone(),
                        metadata: classified.request.metadata.clone(),
                    };
                    self.audit(
                        AuditEvent::ApprovalRequested {
                            effect_id: classified.request.id.clone(),
                            reason,
                        },
                        context,
                    )
                    .await?;
                    let approval = self.approval.approve(approval_request, context).await?;
                    self.audit(
                        AuditEvent::ApprovalDecided {
                            effect_id: classified.request.id.clone(),
                            decision: approval_decision_summary(&approval),
                        },
                        context,
                    )
                    .await?;
                    if let ApprovalDecision::Deny { reason } = approval {
                        return self
                            .denied_observation(classified.request, reason, context)
                            .await;
                    }
                    if matches!(approval, ApprovalDecision::AllowForSession) {
                        self.remember_session_approval(&classified, context);
                    }
                }
            }
        }

        self.audit(
            AuditEvent::EffectStarted {
                effect_id: classified.request.id.clone(),
                policy: ExecutionPolicySummary::from_policy(&execution_policy),
            },
            context,
        )
        .await?;
        let effect_id = classified.request.id.clone();
        let execution = run_executor_with_context(
            &self.executor,
            &classified.request,
            &execution_policy,
            context,
        )
        .await;
        match execution {
            Ok(raw) => {
                let limited =
                    limit_and_redact(raw, &self.config.output_limit, self.redactor.as_ref());
                self.audit(
                    AuditEvent::EffectCompleted {
                        effect_id: effect_id.clone(),
                        truncated: limited.truncated,
                    },
                    context,
                )
                .await?;
                let mut observation_metadata = RunMetadata::new();
                observation_metadata.insert(
                    "truncated".to_string(),
                    serde_json::json!(limited.truncated),
                );
                observation_metadata.insert(
                    "redactions_applied".to_string(),
                    serde_json::json!(limited.redactions.len()),
                );
                let observation = EffectObservation {
                    effect_id: effect_id.clone(),
                    status: EffectStatus::Succeeded,
                    output: limited.output,
                    metadata: observation_metadata,
                };
                self.transcript(
                    TranscriptRecord::EffectObservation {
                        run_id: context.run_id.clone(),
                        effect_id,
                        status: observation.status.clone(),
                    },
                    context,
                )
                .await?;
                Ok(observation)
            }
            Err(ExecutionError::TimedOut(reason)) => {
                self.audit(
                    AuditEvent::EffectTimedOut {
                        effect_id: effect_id.clone(),
                        reason: reason.clone(),
                    },
                    context,
                )
                .await?;
                self.terminal_observation(
                    effect_id,
                    EffectStatus::TimedOut,
                    format!("effect timed out: {reason}"),
                    context,
                )
                .await
            }
            Err(ExecutionError::Cancelled(reason)) => {
                self.audit(
                    AuditEvent::EffectCancelled {
                        effect_id: effect_id.clone(),
                        reason: reason.clone(),
                    },
                    context,
                )
                .await?;
                self.terminal_observation(
                    effect_id,
                    EffectStatus::Cancelled,
                    format!("effect cancelled: {reason}"),
                    context,
                )
                .await
            }
            Err(error) => {
                let reason = error.to_string();
                self.audit(
                    AuditEvent::EffectFailed {
                        effect_id: effect_id.clone(),
                        reason: reason.clone(),
                    },
                    context,
                )
                .await?;
                self.terminal_observation(
                    effect_id,
                    EffectStatus::Failed,
                    format!("effect failed: {reason}"),
                    context,
                )
                .await
            }
        }
    }
}

impl<E, P, A, S, T> BasicHarness<E, P, A, S, T>
where
    E: EffectExecutor,
    P: PolicyEngine,
    A: ApprovalBroker,
    S: AuditSink,
    T: TranscriptStore,
{
    fn execution_policy(
        &self,
        classified: &ClassifiedEffect,
        context: &RunContext,
    ) -> ExecutionPolicy {
        let mut timeout = Some(self.config.default_timeout);
        if let Some(request_timeout) = classified.request.timeout {
            timeout = Some(timeout.map_or(request_timeout, |default| default.min(request_timeout)));
        }
        if let Some(remaining) = context.remaining() {
            timeout = Some(timeout.map_or(remaining, |current| current.min(remaining)));
        }
        ExecutionPolicy {
            sandbox: self.config.default_sandbox.clone(),
            network: self.config.default_network.clone(),
            timeout,
            output_limit: self.config.output_limit.clone(),
        }
    }

    fn is_session_approved(&self, classified: &ClassifiedEffect, context: &RunContext) -> bool {
        let kind = EffectKindKey::from(classified.request.kind.clone());
        let approvals = self
            .session_approvals
            .lock()
            .expect("BasicHarness session approval lock poisoned");
        approvals.iter().any(|approval| {
            approval.run_id == context.run_id
                && approval.kind == kind
                && risk_rank(classified.effective_risk) <= risk_rank(approval.risk_ceiling)
        })
    }

    fn remember_session_approval(&self, classified: &ClassifiedEffect, context: &RunContext) {
        let approval = SessionApproval {
            run_id: context.run_id.clone(),
            kind: EffectKindKey::from(classified.request.kind.clone()),
            risk_ceiling: classified.effective_risk,
        };
        let mut approvals = self
            .session_approvals
            .lock()
            .expect("BasicHarness session approval lock poisoned");
        if !approvals.contains(&approval) {
            approvals.push(approval);
        }
    }

    async fn denied_observation(
        &self,
        request: EffectRequest,
        reason: String,
        context: &RunContext,
    ) -> Result<EffectObservation, HarnessError>
    where
        S: AuditSink,
        T: TranscriptStore,
    {
        self.audit(
            AuditEvent::EffectDenied {
                effect_id: request.id.clone(),
                reason: reason.clone(),
            },
            context,
        )
        .await?;
        self.terminal_observation(
            request.id,
            EffectStatus::Denied,
            format!("effect denied: {reason}"),
            context,
        )
        .await
    }

    async fn terminal_observation(
        &self,
        effect_id: String,
        status: EffectStatus,
        observation_for_model: String,
        context: &RunContext,
    ) -> Result<EffectObservation, HarnessError>
    where
        T: TranscriptStore,
    {
        let observation = EffectObservation {
            effect_id: effect_id.clone(),
            status,
            output: EffectOutput::text(observation_for_model),
            metadata: RunMetadata::new(),
        };
        self.transcript(
            TranscriptRecord::EffectObservation {
                run_id: context.run_id.clone(),
                effect_id,
                status: observation.status.clone(),
            },
            context,
        )
        .await?;
        Ok(observation)
    }
}

/// Errors returned by a harness.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HarnessError {
    /// Effect envelope is invalid.
    #[error("invalid effect request: {0}")]
    InvalidRequest(String),
    /// Policy evaluation failed.
    #[error("policy error: {0}")]
    Policy(String),
    /// Approval failed.
    #[error("approval error: {0}")]
    Approval(#[from] ApprovalError),
    /// Execution failed at infrastructure level.
    #[error("execution error: {0}")]
    Execution(#[from] ExecutionError),
    /// Audit failed.
    #[error("audit error: {0}")]
    Audit(#[from] AuditError),
    /// Transcript failed.
    #[error("transcript error: {0}")]
    Transcript(#[from] TranscriptError),
    /// Run was cancelled.
    #[error("run cancelled")]
    Cancelled,
    /// Run deadline was exceeded.
    #[error("run deadline exceeded")]
    DeadlineExceeded,
}

/// Approval errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ApprovalError {
    /// Broker failed.
    #[error("{0}")]
    Broker(String),
}

/// Executor errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ExecutionError {
    /// Effect kind or request is unsupported by this executor.
    #[error("unsupported effect: {0}")]
    Unsupported(String),
    /// Execution failed.
    #[error("failed: {0}")]
    Failed(String),
    /// Execution timed out.
    #[error("timed out: {0}")]
    TimedOut(String),
    /// Execution was cancelled.
    #[error("cancelled: {0}")]
    Cancelled(String),
}

/// Audit errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum AuditError {
    /// Audit sink failed.
    #[error("{0}")]
    Sink(String),
}

/// Transcript errors.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum TranscriptError {
    /// Transcript store failed.
    #[error("{0}")]
    Store(String),
}

/// Errors returned by [`HarnessRuntime`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HarnessRuntimeError {
    /// Agent kernel failed.
    #[error("agent error: {0}")]
    Agent(#[from] AgentError),
    /// Provider failed.
    #[error("provider error: {0}")]
    Provider(#[from] ProviderError),
    /// Harness failed.
    #[error("harness error: {0}")]
    Harness(#[from] HarnessError),
    /// Agent requested too many steps.
    #[error("too many agent steps: {0}")]
    TooManyAgentSteps(usize),
}

impl ExecutionPolicySummary {
    fn from_policy(policy: &ExecutionPolicy) -> Self {
        Self {
            sandbox: format!("{:?}", policy.sandbox),
            network: format!("{:?}", policy.network),
            timeout_ms: policy
                .timeout
                .map(|timeout| timeout.as_millis().min(u128::from(u64::MAX)) as u64),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum EffectKindKey {
    ReadFile,
    WriteFile,
    ApplyPatch,
    Search,
    ExecuteCommand,
    Git,
    Network,
    Browser,
    Mcp,
    Custom(String),
}

impl From<EffectKind> for EffectKindKey {
    fn from(kind: EffectKind) -> Self {
        match kind {
            EffectKind::ReadFile => Self::ReadFile,
            EffectKind::WriteFile => Self::WriteFile,
            EffectKind::ApplyPatch => Self::ApplyPatch,
            EffectKind::Search => Self::Search,
            EffectKind::ExecuteCommand => Self::ExecuteCommand,
            EffectKind::Git => Self::Git,
            EffectKind::Network => Self::Network,
            EffectKind::Browser => Self::Browser,
            EffectKind::Mcp => Self::Mcp,
            EffectKind::Custom(kind) => Self::Custom(kind),
        }
    }
}

fn validate_effect_request(request: &EffectRequest) -> Result<(), HarnessError> {
    if request.id.trim().is_empty() {
        return Err(HarnessError::InvalidRequest(
            "effect id must not be empty".to_string(),
        ));
    }
    if request.description.trim().is_empty() {
        return Err(HarnessError::InvalidRequest(
            "effect description must not be empty".to_string(),
        ));
    }
    Ok(())
}

fn check_run_context(context: &RunContext) -> Result<(), HarnessError> {
    if context.is_cancelled() {
        Err(HarnessError::Cancelled)
    } else if context.is_expired() {
        Err(HarnessError::DeadlineExceeded)
    } else {
        Ok(())
    }
}

async fn run_executor_with_context<E>(
    executor: &E,
    request: &EffectRequest,
    policy: &ExecutionPolicy,
    context: &RunContext,
) -> Result<RawEffectOutput, ExecutionError>
where
    E: EffectExecutor,
{
    if context.is_cancelled() {
        return Err(ExecutionError::Cancelled("run cancelled".to_string()));
    }
    if context.is_expired() {
        return Err(ExecutionError::TimedOut(
            "run deadline exceeded".to_string(),
        ));
    }
    match policy.timeout {
        Some(timeout) if timeout.is_zero() => {
            Err(ExecutionError::TimedOut("timeout elapsed".to_string()))
        }
        Some(timeout) => {
            tokio::select! {
                _ = context.cancellation.cancelled() => {
                    Err(ExecutionError::Cancelled("run cancelled".to_string()))
                }
                _ = tokio::time::sleep(timeout) => {
                    Err(ExecutionError::TimedOut(format!("exceeded {timeout:?}")))
                }
                output = executor.execute(request, policy, context) => output,
            }
        }
        None => {
            tokio::select! {
                _ = context.cancellation.cancelled() => {
                    Err(ExecutionError::Cancelled("run cancelled".to_string()))
                }
                output = executor.execute(request, policy, context) => output,
            }
        }
    }
}

fn limit_and_redact(
    raw: RawEffectOutput,
    limit: &OutputLimit,
    redactor: &(impl Redactor + ?Sized),
) -> LimitedOutput {
    let model_redacted = redactor.redact_model_text(&raw.observation_for_model);
    let (mut model_text, model_truncated) = truncate_with_marker(
        model_redacted.text,
        limit.model_bytes,
        "\n[output truncated]",
    );

    let mut redactions = model_redacted.redactions;
    let mut truncated = model_truncated;
    if model_truncated && !model_text.contains("[output truncated]") {
        model_text.push_str("\n[output truncated]");
    }

    let display = raw.display.map(|display| {
        let display_redacted = redactor.redact_display_text(&display.content);
        redactions.extend(display_redacted.redactions);
        let (content, was_truncated) =
            truncate_with_marker(display_redacted.text, limit.display_bytes, "\n[truncated]");
        truncated |= was_truncated;
        DisplayOutput {
            content,
            metadata: redactor.redact_metadata(display.metadata),
            ..display
        }
    });

    let debug = raw.debug.map(|debug| {
        let debug_redacted = redactor.redact_debug_text(&debug);
        redactions.extend(debug_redacted.redactions);
        let (debug, was_truncated) =
            truncate_with_marker(debug_redacted.text, limit.debug_bytes, "\n[truncated]");
        truncated |= was_truncated;
        debug
    });

    LimitedOutput {
        output: EffectOutput::text(model_text)
            .with_artifacts(raw.artifacts)
            .with_metadata(redactor.redact_metadata(raw.metadata)),
        truncated,
        redactions,
        debug,
    }
    .with_display(display)
}

impl LimitedOutput {
    fn with_display(mut self, display: Option<DisplayOutput>) -> Self {
        self.output.display = display;
        self
    }
}

fn truncate_with_marker(mut text: String, limit: usize, marker: &str) -> (String, bool) {
    if text.len() <= limit {
        return (text, false);
    }
    if limit == 0 {
        return (String::new(), true);
    }

    let marker_len = marker.len().min(limit);
    let keep = limit.saturating_sub(marker_len);
    let mut end = keep.min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    if marker_len == marker.len() {
        text.push_str(marker);
    }
    (text, true)
}

fn risk_floor_for_kind(kind: &EffectKind) -> RiskLevel {
    match kind {
        EffectKind::ReadFile | EffectKind::Search => RiskLevel::Low,
        EffectKind::WriteFile
        | EffectKind::ApplyPatch
        | EffectKind::ExecuteCommand
        | EffectKind::Git
        | EffectKind::Network
        | EffectKind::Browser
        | EffectKind::Mcp
        | EffectKind::Custom(_) => RiskLevel::Medium,
    }
}

fn max_risk(left: RiskLevel, right: RiskLevel) -> RiskLevel {
    if risk_rank(left) >= risk_rank(right) {
        left
    } else {
        right
    }
}

fn risk_rank(risk: RiskLevel) -> u8 {
    match risk {
        RiskLevel::Low => 0,
        RiskLevel::Medium => 1,
        RiskLevel::High => 2,
        RiskLevel::Critical => 3,
    }
}

fn contains_high_pattern(text: &str) -> bool {
    ["sudo", "force push", "--force", " outside workspace"]
        .iter()
        .any(|pattern| text.contains(pattern))
}

fn contains_critical_pattern(text: &str) -> bool {
    [
        "rm -rf /",
        "mkfs",
        "dd if=",
        "shutdown",
        "reboot",
        "chmod -r 777 /",
    ]
    .iter()
    .any(|pattern| text.contains(pattern))
}

fn payload_summary(payload: &serde_json::Value) -> String {
    payload.to_string().chars().take(512).collect()
}

fn policy_decision_summary(decision: &PolicyDecision) -> String {
    match decision {
        PolicyDecision::Allow => "allow".to_string(),
        PolicyDecision::Deny { reason } => format!("deny: {reason}"),
        PolicyDecision::RequireApproval { reason } => format!("require approval: {reason}"),
    }
}

fn approval_decision_summary(decision: &ApprovalDecision) -> String {
    match decision {
        ApprovalDecision::AllowOnce => "allow once".to_string(),
        ApprovalDecision::AllowForSession => "allow for session".to_string(),
        ApprovalDecision::Deny { reason } => format!("deny: {reason}"),
    }
}

fn redact_json_value(value: &mut serde_json::Value, patterns: &[String], replacement: &str) {
    match value {
        serde_json::Value::String(text) => {
            for pattern in patterns {
                if !pattern.is_empty() && text.contains(pattern) {
                    *text = text.replace(pattern, replacement);
                }
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                redact_json_value(value, patterns, replacement);
            }
        }
        serde_json::Value::Object(map) => {
            for value in map.values_mut() {
                redact_json_value(value, patterns, replacement);
            }
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentKernel, ModelRequest};
    use crate::message::Message;
    use crate::provider::{FakeProvider, FakeReply, ProviderError};
    use crate::tool::{Tool, ToolContext, ToolError, ToolRegistry, ToolResult, ToolSchema};
    use serde_json::json;

    struct SingleModelKernel;

    #[async_trait]
    impl AgentKernel for SingleModelKernel {
        async fn start(
            &mut self,
            _request: RunRequest,
            _context: &RunContext,
        ) -> Result<AgentAction, AgentError> {
            Ok(AgentAction::RequestModel {
                request: ModelRequest::new("model-1", Default::default()),
            })
        }

        async fn observe(
            &mut self,
            observation: Observation,
            context: &RunContext,
        ) -> Result<AgentAction, AgentError> {
            let Observation::Model(observation) = observation else {
                return Err(AgentError::InvalidStep(
                    "expected model observation".to_string(),
                ));
            };
            let answer = match &observation.response.message {
                Message::Assistant { content, .. } => content.clone(),
                _ => String::new(),
            };
            Ok(AgentAction::Respond {
                output: RunOutput {
                    run_id: context.run_id.clone(),
                    answer,
                    summary: Default::default(),
                    final_message: observation.response.message,
                    artifacts: Vec::new(),
                    metadata: RunMetadata::new(),
                },
            })
        }
    }

    struct EffectKernel {
        observed: bool,
    }

    #[async_trait]
    impl AgentKernel for EffectKernel {
        async fn start(
            &mut self,
            _request: RunRequest,
            _context: &RunContext,
        ) -> Result<AgentAction, AgentError> {
            Ok(AgentAction::RequestEffect {
                request: EffectRequest::new(EffectKind::ReadFile, "read config", json!({}))
                    .with_id("effect-1"),
            })
        }

        async fn observe(
            &mut self,
            observation: Observation,
            context: &RunContext,
        ) -> Result<AgentAction, AgentError> {
            let Observation::Effect(observation) = observation else {
                return Err(AgentError::InvalidStep(
                    "expected effect observation".to_string(),
                ));
            };
            self.observed = true;
            Ok(AgentAction::Respond {
                output: RunOutput {
                    run_id: context.run_id.clone(),
                    answer: observation.output.observation_for_model,
                    summary: Default::default(),
                    final_message: Message::assistant("done"),
                    artifacts: Vec::new(),
                    metadata: RunMetadata::new(),
                },
            })
        }
    }

    #[tokio::test]
    async fn runtime_drives_model_request() {
        let provider = FakeProvider::new([FakeReply::Text("hello".to_string())]);
        let harness = BasicHarness::noop();
        let runtime = HarnessRuntime::new(provider, harness);
        let output = runtime
            .run(
                &mut SingleModelKernel,
                RunRequest::text("hi"),
                RunContext::new("run-model"),
            )
            .await
            .unwrap();
        assert_eq!(output.answer, "hello");
    }

    #[tokio::test]
    async fn runtime_drives_effect_request() {
        let provider = FakeProvider::new([]);
        let executor = StaticEffectExecutor::new()
            .with_output("effect-1", RawEffectOutput::text("file contents"));
        let audit = VecAuditSink::new();
        let harness = BasicHarness::new(
            executor,
            DefaultPolicyEngine,
            AlwaysAllowApprovalBroker,
            audit.clone(),
            NoopTranscriptStore,
        );
        let runtime = HarnessRuntime::new(provider, harness);
        let output = runtime
            .run(
                &mut EffectKernel { observed: false },
                RunRequest::text("read"),
                RunContext::new("run-effect"),
            )
            .await
            .unwrap();
        assert_eq!(output.answer, "file contents");
        assert!(audit
            .events()
            .iter()
            .any(|event| matches!(event, AuditEvent::EffectCompleted { effect_id, .. } if effect_id == "effect-1")));
    }

    #[tokio::test]
    async fn runtime_returns_provider_error() {
        let provider = FakeProvider::new([FakeReply::Error(ProviderError::Api {
            status: 500,
            code: None,
            message: "provider down".to_string(),
        })]);
        let harness = BasicHarness::noop();
        let runtime = HarnessRuntime::new(provider, harness);
        let err = runtime
            .run(
                &mut SingleModelKernel,
                RunRequest::text("hi"),
                RunContext::new("run-provider-error"),
            )
            .await
            .unwrap_err();

        assert!(matches!(err, HarnessRuntimeError::Provider(_)));
    }

    struct LoopKernel;

    #[async_trait]
    impl AgentKernel for LoopKernel {
        async fn start(
            &mut self,
            _request: RunRequest,
            _context: &RunContext,
        ) -> Result<AgentAction, AgentError> {
            Ok(AgentAction::RequestModel {
                request: ModelRequest::new("model-1", Default::default()),
            })
        }

        async fn observe(
            &mut self,
            _observation: Observation,
            _context: &RunContext,
        ) -> Result<AgentAction, AgentError> {
            Ok(AgentAction::RequestModel {
                request: ModelRequest::new("model-loop", Default::default()),
            })
        }
    }

    #[tokio::test]
    async fn runtime_limits_agent_steps() {
        let provider = FakeProvider::new([
            FakeReply::Text("one".to_string()),
            FakeReply::Text("two".to_string()),
            FakeReply::Text("three".to_string()),
        ]);
        let harness = BasicHarness::noop();
        let runtime = HarnessRuntime::new(provider, harness).with_config(HarnessRuntimeConfig {
            max_agent_steps: 1,
            ..Default::default()
        });
        let err = runtime
            .run(
                &mut LoopKernel,
                RunRequest::text("hi"),
                RunContext::new("run-step-limit"),
            )
            .await
            .unwrap_err();

        assert!(matches!(err, HarnessRuntimeError::TooManyAgentSteps(1)));
    }

    #[tokio::test]
    async fn react_kernel_runs_effect_through_runtime() {
        let provider = FakeProvider::new([
            FakeReply::ToolCalls {
                content: String::new(),
                calls: vec![crate::ToolCall {
                    id: "call-1".to_string(),
                    name: "read_file".to_string(),
                    arguments: "{}".to_string(),
                }],
            },
            FakeReply::Text("done after observation".to_string()),
        ]);
        let executor =
            StaticEffectExecutor::new().with_output("effect-1", RawEffectOutput::text("observed"));
        let harness = BasicHarness::new(
            executor,
            DefaultPolicyEngine,
            AlwaysAllowApprovalBroker,
            NoopAuditSink,
            NoopTranscriptStore,
        );
        let runtime = HarnessRuntime::new(provider, harness);
        let mut registry = ToolRegistry::new();
        registry.register(EffectTool);
        let mut kernel = crate::ReActAgent::kernel(registry, "");

        let output = runtime
            .run(
                &mut kernel,
                RunRequest::text("read"),
                RunContext::new("run-react-runtime"),
            )
            .await
            .unwrap();

        assert_eq!(output.answer, "done after observation");
    }

    #[tokio::test]
    async fn basic_harness_denies_critical_effect() {
        let harness = BasicHarness::new(
            StaticEffectExecutor::new(),
            DefaultPolicyEngine,
            AlwaysAllowApprovalBroker,
            NoopAuditSink,
            NoopTranscriptStore,
        );
        let observation = harness
            .execute(
                EffectRequest::new(
                    EffectKind::ExecuteCommand,
                    "run rm -rf /",
                    json!({"cmd": "rm -rf /"}),
                )
                .with_id("critical"),
                &RunContext::new("run-deny"),
            )
            .await
            .unwrap();
        assert_eq!(observation.status, EffectStatus::Denied);
        assert!(
            observation
                .output
                .observation_for_model
                .contains("effect denied")
        );
    }

    #[tokio::test]
    async fn basic_harness_batch_can_mix_denied_and_succeeded() {
        let harness = BasicHarness::new(
            StaticEffectExecutor::new()
                .with_output("read-ok", RawEffectOutput::text("read succeeded")),
            DefaultPolicyEngine,
            AlwaysAllowApprovalBroker,
            NoopAuditSink,
            NoopTranscriptStore,
        );
        let observations = harness
            .execute_batch(
                vec![
                    EffectRequest::new(EffectKind::ReadFile, "read", json!({})).with_id("read-ok"),
                    EffectRequest::new(
                        EffectKind::ExecuteCommand,
                        "run rm -rf /",
                        json!({"cmd": "rm -rf /"}),
                    )
                    .with_id("deny-critical"),
                ],
                &RunContext::new("run-batch"),
            )
            .await
            .unwrap();

        assert_eq!(observations[0].status, EffectStatus::Succeeded);
        assert_eq!(observations[1].status, EffectStatus::Denied);
    }

    #[tokio::test]
    async fn approval_deny_becomes_denied_observation() {
        let harness = BasicHarness::new(
            StaticEffectExecutor::new(),
            DefaultPolicyEngine,
            AlwaysDenyApprovalBroker,
            NoopAuditSink,
            NoopTranscriptStore,
        );
        let observation = harness
            .execute(
                EffectRequest::new(
                    EffectKind::ExecuteCommand,
                    "run high-risk command",
                    json!({}),
                )
                .with_id("approval-deny")
                .with_risk(RiskLevel::High),
                &RunContext::new("run-approval-deny"),
            )
            .await
            .unwrap();

        assert_eq!(observation.status, EffectStatus::Denied);
        assert!(
            observation
                .output
                .observation_for_model
                .contains("approval broker")
        );
    }

    #[derive(Debug, Clone)]
    struct CountingApprovalBroker {
        calls: Arc<Mutex<usize>>,
        decision: ApprovalDecision,
    }

    #[async_trait]
    impl ApprovalBroker for CountingApprovalBroker {
        async fn approve(
            &self,
            _request: ApprovalRequest,
            _context: &RunContext,
        ) -> Result<ApprovalDecision, ApprovalError> {
            *self.calls.lock().expect("approval counter lock poisoned") += 1;
            Ok(self.decision.clone())
        }
    }

    #[tokio::test]
    async fn allow_for_session_skips_repeated_approval_for_same_run_and_kind() {
        let calls = Arc::new(Mutex::new(0));
        let broker = CountingApprovalBroker {
            calls: calls.clone(),
            decision: ApprovalDecision::AllowForSession,
        };
        let harness = BasicHarness::new(
            StaticEffectExecutor::new()
                .with_output("effect-1", RawEffectOutput::text("first"))
                .with_output("effect-2", RawEffectOutput::text("second")),
            DefaultPolicyEngine,
            broker,
            NoopAuditSink,
            NoopTranscriptStore,
        );
        let context = RunContext::new("run-session-approval");

        let first = harness
            .execute(
                EffectRequest::new(EffectKind::ExecuteCommand, "run command", json!({}))
                    .with_id("effect-1")
                    .with_risk(RiskLevel::High),
                &context,
            )
            .await
            .unwrap();
        let second = harness
            .execute(
                EffectRequest::new(EffectKind::ExecuteCommand, "run another command", json!({}))
                    .with_id("effect-2")
                    .with_risk(RiskLevel::High),
                &context,
            )
            .await
            .unwrap();

        assert_eq!(first.status, EffectStatus::Succeeded);
        assert_eq!(second.status, EffectStatus::Succeeded);
        assert_eq!(*calls.lock().expect("approval counter lock poisoned"), 1);
    }

    #[tokio::test]
    async fn executor_failure_is_effect_observation() {
        let harness = BasicHarness::new(
            StaticEffectExecutor::new()
                .with_error("effect-1", ExecutionError::Failed("boom".to_string())),
            DefaultPolicyEngine,
            AlwaysAllowApprovalBroker,
            NoopAuditSink,
            NoopTranscriptStore,
        );
        let observation = harness
            .execute(
                EffectRequest::new(EffectKind::ReadFile, "read", json!({})).with_id("effect-1"),
                &RunContext::new("run-fail"),
            )
            .await
            .unwrap();
        assert_eq!(observation.status, EffectStatus::Failed);
        assert!(observation.output.observation_for_model.contains("boom"));
    }

    #[derive(Debug, Default, Clone, Copy)]
    struct SlowExecutor;

    #[async_trait]
    impl EffectExecutor for SlowExecutor {
        async fn execute(
            &self,
            _request: &EffectRequest,
            _policy: &ExecutionPolicy,
            _context: &RunContext,
        ) -> Result<RawEffectOutput, ExecutionError> {
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(RawEffectOutput::text("late"))
        }
    }

    #[tokio::test]
    async fn executor_timeout_becomes_timed_out_observation() {
        let harness = BasicHarness::new(
            SlowExecutor,
            DefaultPolicyEngine,
            AlwaysAllowApprovalBroker,
            NoopAuditSink,
            NoopTranscriptStore,
        )
        .with_config(HarnessConfig {
            default_timeout: Duration::from_millis(1),
            ..Default::default()
        });
        let observation = harness
            .execute(
                EffectRequest::new(EffectKind::ReadFile, "read slowly", json!({}))
                    .with_id("slow-effect"),
                &RunContext::new("run-timeout"),
            )
            .await
            .unwrap();

        assert_eq!(observation.status, EffectStatus::TimedOut);
    }

    #[tokio::test]
    async fn output_is_limited_and_redacted() {
        let raw =
            RawEffectOutput::text("secret-token-1234567890 plus a long tail that must be cut")
                .with_debug("debug secret-token");
        let limit = OutputLimit {
            model_bytes: 32,
            display_bytes: 20,
            debug_bytes: 20,
        };
        let redactor = PatternRedactor::new(["secret-token"]);
        let output = limit_and_redact(raw, &limit, &redactor);
        assert!(output.truncated);
        assert!(!output.output.observation_for_model.contains("secret-token"));
        assert!(output.output.observation_for_model.contains("[REDACTED]"));
    }

    #[tokio::test]
    async fn basic_harness_uses_configured_redactor() {
        let harness = BasicHarness::new(
            StaticEffectExecutor::new()
                .with_output("effect-1", RawEffectOutput::text("secret-token in output")),
            DefaultPolicyEngine,
            AlwaysAllowApprovalBroker,
            NoopAuditSink,
            NoopTranscriptStore,
        )
        .with_redactor(PatternRedactor::new(["secret-token"]));

        let observation = harness
            .execute(
                EffectRequest::new(EffectKind::ReadFile, "read", json!({})).with_id("effect-1"),
                &RunContext::new("run-redactor"),
            )
            .await
            .unwrap();

        assert_eq!(observation.status, EffectStatus::Succeeded);
        assert!(
            !observation
                .output
                .observation_for_model
                .contains("secret-token")
        );
        assert!(
            observation
                .output
                .observation_for_model
                .contains("[REDACTED]")
        );
        assert_eq!(
            observation
                .metadata
                .get("redactions_applied")
                .and_then(serde_json::Value::as_u64),
            Some(1)
        );
    }

    #[derive(Debug, Default, Clone, Copy)]
    struct FailingAuditSink;

    #[async_trait]
    impl AuditSink for FailingAuditSink {
        async fn record(
            &self,
            _event: AuditEvent,
            _context: &RunContext,
        ) -> Result<(), AuditError> {
            Err(AuditError::Sink("audit offline".to_string()))
        }
    }

    #[tokio::test]
    async fn audit_failure_fails_closed_before_execution() {
        let harness = BasicHarness::new(
            StaticEffectExecutor::new()
                .with_output("effect-1", RawEffectOutput::text("should not execute")),
            DefaultPolicyEngine,
            AlwaysAllowApprovalBroker,
            FailingAuditSink,
            NoopTranscriptStore,
        );
        let err = harness
            .execute(
                EffectRequest::new(EffectKind::ReadFile, "read", json!({})).with_id("effect-1"),
                &RunContext::new("run-audit-fail"),
            )
            .await
            .unwrap_err();

        assert!(matches!(err, HarnessError::Audit(_)));
    }

    #[derive(Debug, Default, Clone, Copy)]
    struct FailingTranscriptStore;

    #[async_trait]
    impl TranscriptStore for FailingTranscriptStore {
        async fn append(
            &self,
            _record: TranscriptRecord,
            _context: &RunContext,
        ) -> Result<(), TranscriptError> {
            Err(TranscriptError::Store("transcript offline".to_string()))
        }
    }

    #[tokio::test]
    async fn transcript_failure_defaults_to_fail_open() {
        let harness = BasicHarness::new(
            StaticEffectExecutor::new().with_output("effect-1", RawEffectOutput::text("ok")),
            DefaultPolicyEngine,
            AlwaysAllowApprovalBroker,
            NoopAuditSink,
            FailingTranscriptStore,
        );
        let observation = harness
            .execute(
                EffectRequest::new(EffectKind::ReadFile, "read", json!({})).with_id("effect-1"),
                &RunContext::new("run-transcript-open"),
            )
            .await
            .unwrap();

        assert_eq!(observation.status, EffectStatus::Succeeded);
    }

    #[tokio::test]
    async fn transcript_failure_can_fail_closed() {
        let harness = BasicHarness::new(
            StaticEffectExecutor::new().with_output("effect-1", RawEffectOutput::text("ok")),
            DefaultPolicyEngine,
            AlwaysAllowApprovalBroker,
            NoopAuditSink,
            FailingTranscriptStore,
        )
        .with_config(HarnessConfig {
            fail_closed_on_transcript_error: true,
            ..Default::default()
        });
        let err = harness
            .execute(
                EffectRequest::new(EffectKind::ReadFile, "read", json!({})).with_id("effect-1"),
                &RunContext::new("run-transcript-closed"),
            )
            .await
            .unwrap_err();

        assert!(matches!(err, HarnessError::Transcript(_)));
    }

    #[tokio::test]
    async fn invalid_effect_request_returns_harness_error() {
        let harness = BasicHarness::noop();
        let err = harness
            .execute(
                EffectRequest::new(EffectKind::ReadFile, "read", json!({})).with_id(""),
                &RunContext::new("run-invalid"),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, HarnessError::InvalidRequest(_)));
    }

    struct EffectTool;

    #[async_trait]
    impl Tool for EffectTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema::new("read_file", "Read a file", json!({}))
        }

        async fn call(
            &self,
            _arguments: serde_json::Value,
            _context: ToolContext<'_>,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolResult::Effect(
                EffectRequest::new(EffectKind::ReadFile, "read", json!({})).with_id("effect-1"),
            ))
        }
    }

    #[tokio::test]
    async fn registry_effect_tool_stays_external_to_harness() {
        let mut registry = ToolRegistry::new();
        registry.register(EffectTool);
        assert_eq!(registry.names(), ["read_file"]);
    }
}
