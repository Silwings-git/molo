//! The ReAct reasoning loop: the classic Agent assembly shipped with the
//! framework.
//!
//! The generic loop shape: record the user input → start a conversation →
//! execute tool requests from the model and feed the results back → until
//! the model answers directly; tool execution failures are fed back to the
//! model as text, and the model decides what to do next.
//!
//! Getting started needs just three required parameters: provider
//! ([`Provider`]) + tools ([`ToolRegistry`]) + system_prompt; Memory
//! defaults to a bounded window ([`WindowMemory`], 128k token budget, long
//! conversations auto-trim the oldest rounds); replace it with
//! [`with_memory`](ReActAgent::with_memory) when you need something custom.
//!
//! Pre-call approval at the loop level is implemented by the application
//! layer; this module doesn't build it in.

use super::config::AgentConfig;
use super::structured::{StructuredOutcome, StructuredValidator};
use crate::CancellationToken;
use crate::agent::events::ReActEvent;
use crate::agent::{
    Agent, AgentError, AgentEvent, CancellableAgent, MessageChunk, RunSummary, TypedAgent,
};
use crate::event_channel::EventChannel;
use crate::memory::{Memory, WindowMemory};
use crate::message::{Message, ToolCall};
use crate::provider::{ChatRequest, Provider, ProviderError, StreamEvent, Usage};
use crate::skill::{LoadSkillTool, SkillRegistry};
use crate::tool::{SharedState, ToolRegistry};
use futures::StreamExt;
use futures::stream::BoxStream;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use std::collections::HashSet;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::Instrument;

/// Convenience assembly macro: registers a list of tools (possibly
/// heterogeneous) with automatic boxing, creating a
/// [`ToolRegistry`](crate::tool::ToolRegistry) internally. The system prompt
/// is **optional** (omitted = no system prompt). Six arms:
///
/// - `react_agent!(provider)` — no tools, no system prompt;
/// - `react_agent!(provider, "system prompt")` — no tools, with a system
///   prompt (the string must be a **literal**, distinguishing it from the
///   "existing registry" arm; to use a variable, write
///   `react_agent!(provider, [], system_var)` or the three-arg arm);
/// - `react_agent!(provider, [tool_1, tool_2, ...])` — a heterogeneous tool
///   list (types need not match), auto-registered, no system prompt;
/// - `react_agent!(provider, [tool_1, ...], "system prompt")` — list +
///   system prompt;
/// - `react_agent!(provider, registry)` — an existing registry (e.g. a
///   sub-agent trimmed via `subset`), no system prompt;
/// - `react_agent!(provider, registry, "system prompt")` — registry +
///   system prompt.
///
/// Returns [`ReActAgent`](crate::agent::ReActAgent); chained
/// [`with_memory`](ReActAgent::with_memory) / [`with_config`](ReActAgent::with_config) /
/// [`with_state`](ReActAgent::with_state) / [`with_event_channel`](ReActAgent::with_event_channel) /
/// [`with_skills`](ReActAgent::with_skills) work as usual.
/// `ReActAgent::new` keeps a single signature; the macro handles the
/// multiple shapes.
///
/// ```
/// # #[tokio::main]
/// # async fn main() -> Result<(), molo::AgentError> {
/// use molo::{react_agent, Agent, FakeProvider, FakeReply};
///
/// let mut agent = react_agent!(
///     FakeProvider::new([FakeReply::Text("Hello".into())]),
///     "You are a helpful assistant",
/// );
/// assert_eq!(agent.run("Are you there").await?, "Hello");
/// # Ok(())
/// # }
/// ```
#[macro_export]
macro_rules! react_agent {
    // No tools: bare / with a system prompt (a string literal).
    ($provider:expr $(,)?) => {
        $crate::agent::ReActAgent::new(
            $provider,
            $crate::tool::ToolRegistry::new(),
            "",
        )
    };
    ($provider:expr, $system:literal $(,)?) => {
        $crate::agent::ReActAgent::new(
            $provider,
            $crate::tool::ToolRegistry::new(),
            $system,
        )
    };
    // Tool list: auto-registered, without / with a system prompt.
    ($provider:expr, [$($tool:expr),* $(,)?] $(,)?) => {{
        let mut __molo_registry = $crate::tool::ToolRegistry::new();
        $(__molo_registry.register($tool);)*
        $crate::agent::ReActAgent::new($provider, __molo_registry, "")
    }};
    ($provider:expr, [$($tool:expr),* $(,)?], $system:expr $(,)?) => {{
        let mut __molo_registry = $crate::tool::ToolRegistry::new();
        $(__molo_registry.register($tool);)*
        $crate::agent::ReActAgent::new($provider, __molo_registry, $system)
    }};
    // Existing registry: without / with a system prompt.
    ($provider:expr, $registry:expr $(,)?) => {
        $crate::agent::ReActAgent::new($provider, $registry, "")
    };
    ($provider:expr, $registry:expr, $system:expr $(,)?) => {
        $crate::agent::ReActAgent::new($provider, $registry, $system)
    };
}

/// The classic ReAct reasoning loop: conversation → tool execution fed
/// back → until the model answers directly.
///
/// Assembly: three required parameters (Provider / ToolRegistry /
/// system_prompt) + chained optional
/// ([`with_memory`](ReActAgent::with_memory) / [`with_config`](ReActAgent::with_config) /
/// [`with_state`](ReActAgent::with_state) / [`with_event_channel`](ReActAgent::with_event_channel));
/// for simpler assembly use the macro [`react_agent!`](crate::react_agent).
///
/// # Examples
///
/// ```rust
/// # #[tokio::main]
/// # async fn main() -> Result<(), molo::AgentError> {
/// use molo::agent::{Agent, ReActAgent};
/// use molo::provider::{FakeProvider, FakeReply};
/// use molo::tool::ToolRegistry;
///
/// let mut agent = ReActAgent::new(
///     FakeProvider::new([FakeReply::Text("Hello".into())]),
///     ToolRegistry::new(),
///     "You are a helpful assistant",
/// );
/// let answer = agent.run("Are you there").await?;
/// assert_eq!(answer, "Hello");
/// # Ok(())
/// # }
/// ```
///
/// # Cancellation semantics
///
/// ReActAgent implements [`CancellableAgent`] — every run carries a
/// [`CancellationToken`]; cancellation is cooperative and only checked
/// at safe points:
///
/// - Cancelled mid-conversation: the in-flight request is dropped, and
///   nothing is recorded for this round;
/// - Cancelled during a tool round: the running tool call is not
///   interrupted (tools are user code; a partial execution risks side
///   effects); the Assistant message already recorded this round and the
///   ToolResult fed back right after remain complete;
/// - Streaming path: already-dispatched `Delta` chunks are kept, then
///   the stream concludes with a [`MessageChunk::Cancelled`] terminal
///   chunk — no `Done`;
/// - After cancellation, already-recorded messages are kept, not rolled
///   back; subsequent runs continue from the retained memory.
///
/// Callers that don't need cancellation can just use
/// [`run`](Agent::run) / [`run_stream`](Agent::run_stream) — they
/// internally use a token that is never cancelled.
///
/// # Errors
///
/// Tool execution failures are **not** [`AgentError`] — the error text
/// is fed back to the model via `ToolResult`, the model decides what to
/// do next, and the loop continues. [`AgentError`] only represents
/// run-level failures (Memory / Provider failures, exceeding the
/// tool-round limit, cancellation).
///
/// # Structured output
///
/// The ability to require the model's answer to conform to a given JSON
/// Schema; both entry points share the same validation-and-retry loop
/// (see [`run_typed`](ReActAgent::run_typed) and
/// [`with_structured_output`](ReActAgent::with_structured_output)):
///
/// - [`run_typed`](ReActAgent::run_typed): **typed output** —
///   `run_typed`'s return type directly declares the target type; this
///   run auto-generates a JSON Schema from the type
///   ([`schemars`](https://docs.rs/schemars)-derived, the same pipeline
///   as tool parameter schemas), and deserializes after validation;
/// - [`with_structured_output`](ReActAgent::with_structured_output): a
///   hand-written Schema (or the serialized result of
///   `schemars::schema_for!(T)`), paired with [`Agent::run`](Agent::run)
///   returning JSON text that you parse yourself.
///
/// Both entry points validate the final answer with framework-side
/// jsonschema; on failure the error is fed back to the model for retry
/// (budget [`AgentConfig::max_structured_retries`]); compatible
/// endpoints are additionally constrained as best effort via
/// `response_format` (see
/// [`ModelOptions::structured`](crate::provider::ModelOptions::structured)).
pub struct ReActAgent {
    provider: Box<dyn Provider>,
    memory: Box<dyn Memory>,
    registry: ToolRegistry,
    system_prompt: String,
    config: AgentConfig,
    /// Shared state: the application reads and writes this field directly
    /// across runs; it is injected via [`Tool::call`](crate::Tool::call) on
    /// every tool call, so multiple tools / Agents can share one instance.
    pub state: SharedState,
    /// Observation channel (optional, not attached by default): the loop
    /// pushes process events here, and the host side subscribes.
    events: Option<Arc<dyn EventChannel>>,
    /// Skill registry (optional, empty by default): held when skills are
    /// assembled; the application reads and writes this field directly
    /// across runs (hot-swappable — additions/removals take effect on the
    /// next request).
    pub skills: Arc<SkillRegistry>,
    /// Session-visible allowlist (`None` = all skills visible; only
    /// effective in dynamic mode).
    enabled_skills: Option<Arc<HashSet<String>>>,
    /// Pre-activated skill names, in activation order (bodies join the
    /// system prompt without the model's involvement).
    activated_skills: Vec<String>,
    /// Skill assembly mode (none / dynamic progressive disclosure / static
    /// inlining).
    skill_mode: SkillMode,
    /// Source of run ids: the instance's construction time (nanoseconds)
    /// combined with a process-wide global sequence number, producing
    /// `run-{ts}-{n}` — no collisions across runs / instances in the same
    /// process.
    created_nanos: u128,
}

/// Skill assembly mode: how skills enter the system prompt.
///
/// - [`Dynamic`](SkillMode::Dynamic): progressive disclosure — the menu
///   stays in the system prompt, and the
///   [`load_skill`](crate::skill::LoadSkillTool) tool reads bodies on
///   demand;
/// - [`Inline`](SkillMode::Inline): static inlining — all bodies stay in
///   the system prompt, and load_skill is not registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SkillMode {
    /// No skills assembled.
    None,
    /// Dynamic progressive disclosure (the protocol's primary form).
    Dynamic,
    /// Static inlining (a quick form for small skill sets /
    /// deterministic scenarios).
    Inline,
}

/// Process-wide global run sequence number: the source of run-id
/// uniqueness. Back-to-back instances may share the same construction
/// timestamp (coarse clock granularity), and per-instance counters would
/// collide; the global sequence naturally distinguishes any two instances.
static PROCESS_RUN_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Per-round reply text accumulation limit (4 MiB, shared by streaming and
/// non-streaming): exceeding it is treated as a malicious / abnormal
/// endpoint and terminates the round with `AgentError::Provider`.
const MAX_ROUND_TEXT: usize = 4 << 20;

/// Token budget of the default Memory (128k): `ReActAgent::new` defaults
/// to [`WindowMemory`](crate::memory::WindowMemory), which auto-trims the
/// oldest rounds in long conversations — an unbounded default would let
/// memory grow indefinitely with the conversation; callers needing full
/// history explicitly replace it with
/// [`with_memory`](ReActAgent::with_memory).
const DEFAULT_MEMORY_TOKENS: usize = 128_000;

impl ReActAgent {
    /// Simple construction: three required parameters — provider
    /// ([`Provider`]) + tools ([`ToolRegistry`]) + system_prompt (empty
    /// string = no system prompt); `run` returns the model's answer text
    /// (consistent with [`Agent::run`]).
    ///
    /// Memory defaults to a bounded window
    /// ([`WindowMemory`](crate::memory::WindowMemory), 128k token budget,
    /// long conversations auto-trim the oldest rounds); when you need
    /// something custom (full history / persistence, etc.), chain
    /// [`with_memory`](ReActAgent::with_memory) to replace it; other
    /// optional behaviors (such as the max tool rounds) go through
    /// [`with_config`](ReActAgent::with_config).
    ///
    /// Structured output: typed output goes straight to
    /// [`run_typed`](ReActAgent::run_typed) (this run generates the Schema
    /// from the target type, no constructor configuration needed); for
    /// hand-written Schemas use
    /// [`with_structured_output`](ReActAgent::with_structured_output).
    ///
    /// For simpler assembly see the macro
    /// [`react_agent!`](crate::react_agent).
    pub fn new(
        provider: impl Provider + 'static,
        tools: ToolRegistry,
        system_prompt: impl Into<String>,
    ) -> Self {
        Self {
            provider: Box::new(provider),
            memory: Box::new(WindowMemory::new(DEFAULT_MEMORY_TOKENS)),
            registry: tools,
            system_prompt: system_prompt.into(),
            config: AgentConfig::default(),
            state: SharedState::default(),
            events: None,
            skills: Arc::new(SkillRegistry::new()),
            enabled_skills: None,
            activated_skills: Vec::new(),
            skill_mode: SkillMode::None,
            created_nanos: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
        }
    }

    /// Observability identifier for this run: construction time + a
    /// process-wide global sequence number, unique across runs / instances
    /// in the same process; trace spans and the event stream's `RunStarted`
    /// carry the same id — the correlation key between the two channels.
    fn next_run_id(&self) -> String {
        let n = PROCESS_RUN_COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("run-{}-{n}", self.created_nanos)
    }

    /// Replace the default Memory (default
    /// [`WindowMemory`](crate::memory::WindowMemory), 128k token budget,
    /// long conversations auto-trim the oldest rounds).
    ///
    /// The Agent owns the Memory: `Memory::record` needs `&mut self`, which
    /// a shared (Arc) form can't write through; sharing the same
    /// conversation history across Agents is a Workflow-orchestration
    /// concern, out of this method's scope.
    ///
    /// # Examples
    ///
    /// Replace it with [`InMemoryMemory`](crate::memory::InMemoryMemory),
    /// which keeps all history verbatim:
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() {
    /// use molo::agent::{Agent, ReActAgent};
    /// use molo::memory::InMemoryMemory;
    /// use molo::provider::{FakeProvider, FakeReply};
    /// use molo::tool::ToolRegistry;
    ///
    /// let mut agent = ReActAgent::new(
    ///     FakeProvider::new([FakeReply::Text("Hello".into())]),
    ///     ToolRegistry::new(),
    ///     "",
    /// )
    /// .with_memory(InMemoryMemory::default());
    ///
    /// assert_eq!(agent.run("Are you there").await.unwrap(), "Hello");
    /// # }
    /// ```
    pub fn with_memory(mut self, memory: impl Memory + 'static) -> Self {
        self.memory = Box::new(memory);
        self
    }

    /// Replace the default config (default [`AgentConfig::default`]): the
    /// `max_tool_rounds` round limit and the `options` model parameters
    /// (temperature / max_tokens / extra parameters).
    ///
    /// See [`AgentConfig`](crate::agent::AgentConfig) for how to write a
    /// config.
    pub fn with_config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    /// Enable structured output: the final answer must be JSON conforming
    /// to this **JSON Schema document**.
    ///
    /// `schema` is usually the serialized `RootSchema` produced by
    /// `schemars::schema_for!(T)` (the same pipeline as tool parameter
    /// schemas); it can also be hand-written. Semantics:
    ///
    /// - Compatible endpoints constrain the model as best effort via
    ///   `response_format` once they receive it;
    /// - **Framework-side fallback validation**: when the final answer
    ///   doesn't conform, the validation error is fed back to the model for
    ///   retry (independent budget
    ///   [`AgentConfig::max_structured_retries`]; exceeding it fails);
    /// - Once validation passes, the JSON text is returned as-is as the
    ///   answer.
    ///
    /// Typed output goes through [`run_typed`](ReActAgent::run_typed): the
    /// schema is auto-generated from the target type and deserialized,
    /// ignoring the hand-written schema set here — the two paths don't
    /// combine (the hand-written schema serves `Agent::run`'s text form;
    /// `run_typed` is always "the type is the schema").
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), molo::AgentError> {
    /// use molo::{react_agent, Agent, FakeProvider, FakeReply};
    /// use serde_json::json;
    ///
    /// let mut agent = react_agent!(
    ///     FakeProvider::new([FakeReply::Text(r#"{"city":"Beijing"}"#.into())]),
    ///     "You are a structured-output assistant",
    /// )
    /// .with_structured_output(json!({
    ///     "type": "object",
    ///     "properties": { "city": { "type": "string" } },
    ///     "required": ["city"],
    /// }));
    ///
    /// let answer = agent.run("How's the weather in Beijing").await?;
    /// assert_eq!(answer, r#"{"city":"Beijing"}"#);
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_structured_output(mut self, schema: serde_json::Value) -> Self {
        self.config.options.structured = Some(schema);
        self
    }

    /// Attach shared state (empty by default). The application reads and
    /// writes via [`state`](ReActAgent::state) across runs, and tools
    /// access the same instance through the `state` parameter at call time.
    pub fn with_state(mut self, state: SharedState) -> Self {
        self.state = state;
        self
    }

    /// Attach an observation channel (not attached by default, zero cost):
    /// the loop pushes process events into the pipeline, and the host side
    /// `subscribe`s.
    ///
    /// The pipeline is **long-lived**: attached once, events from many runs
    /// flow into the same pipeline (the Agent is the event source); each
    /// run is one segment of the pipeline, delimited by
    /// [`ReActEvent::RunStarted`] / [`ReActEvent::RunEnded`]; both the
    /// streaming and non-streaming paths publish. See
    /// [`ReActEvent`](crate::agent::ReActEvent) for the event set.
    pub fn with_event_channel(mut self, channel: impl EventChannel + 'static) -> Self {
        self.events = Some(Arc::new(channel));
        self
    }

    /// Attach a skill registry (dynamic progressive disclosure): the skill
    /// menu joins the system prompt, the
    /// [`load_skill`](crate::skill::LoadSkillTool) tool is registered into
    /// the ToolRegistry, and the model reads skill bodies on demand.
    ///
    /// Skills are **hot-swappable**: the agent exposes a
    /// [`skills`](ReActAgent::skills) handle; the application side can add
    /// or remove anytime, taking effect on the next request's menu and
    /// loading; assembling an empty registry = unchanged system prompt
    /// (zero cost).
    ///
    /// Mutually exclusive with
    /// [`with_skills_inline`](ReActAgent::with_skills_inline); the last
    /// call wins.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), molo::AgentError> {
    /// use molo::agent::{Agent, ReActAgent};
    /// use molo::provider::{FakeProvider, FakeReply};
    /// use molo::skill::{Skill, SkillRegistry};
    /// use molo::tool::ToolRegistry;
    ///
    /// let skills = SkillRegistry::new();
    /// // The sample input is known-valid: a parse failure means the sample
    /// // text is wrong, and the assertion surfaces it directly.
    /// skills.add(
    ///     Skill::parse("---\nname: greet\ndescription: Greet the user\n---\nRule: start with 'Hello'.")
    ///         .unwrap(),
    /// );
    ///
    /// let mut agent = ReActAgent::new(
    ///     FakeProvider::new([FakeReply::Text("Hello".into())]),
    ///     ToolRegistry::new(),
    ///     "You are an assistant",
    /// )
    /// .with_skills(skills);
    ///
    /// assert_eq!(agent.run("Are you there").await?, "Hello");
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_skills(mut self, registry: SkillRegistry) -> Self {
        self.skills = Arc::new(registry);
        self.skill_mode = SkillMode::Dynamic;
        let enabled = self.enabled_skills.clone();
        self.registry
            .register(LoadSkillTool::new(self.skills.clone(), enabled));
        self
    }

    /// Attach a skill registry (static inlining): all bodies stay in the
    /// system prompt, and no load_skill tool is registered.
    ///
    /// The counterpart of [`with_skills`](ReActAgent::with_skills)'s
    /// "menu + load on demand": with small skill sets and deterministic
    /// flows, resident bodies save the tool round-trip; with many skills,
    /// the system-prompt cost grows with the bodies, and the dynamic mode
    /// should be used instead.
    ///
    /// Mutually exclusive with [`with_skills`](ReActAgent::with_skills);
    /// the last call wins.
    pub fn with_skills_inline(mut self, registry: SkillRegistry) -> Self {
        self.skills = Arc::new(registry);
        // If switching from dynamic mode: remove load_skill, to avoid
        // leaving a loader tool with no menu to guide it.
        self.registry.remove("load_skill");
        self.skill_mode = SkillMode::Inline;
        self
    }

    /// Restrict the allowlist of skills visible in this session (all
    /// visible by default).
    ///
    /// The allowlist is set at construction time and immutable; skills
    /// outside the allowlist:
    /// - don't appear in the system prompt's menu, and load_skill returns
    ///   "skill not enabled" when loading them;
    /// - can't be pre-activated via
    ///   [`activate_skill`](ReActAgent::activate_skill).
    ///
    /// Disabling ≠ deleting: metadata always stays in the registry
    /// ([`skills`](ReActAgent::skills)), and a new session (a newly
    /// constructed agent) restores full visibility by default. The
    /// allowlist only applies to dynamic mode
    /// ([`with_skills`](ReActAgent::with_skills)); in inline mode
    /// ([`with_skills_inline`](ReActAgent::with_skills_inline)) all bodies
    /// are already resident, so the allowlist has no effect.
    pub fn with_enabled_skills(mut self, names: &[&str]) -> Self {
        let set: HashSet<String> = names.iter().map(|n| n.to_string()).collect();
        self.enabled_skills = Some(Arc::new(set));
        // Dynamic mode: re-register load_skill (same-name replacement) so
        // the allowlist takes effect immediately.
        if self.skill_mode == SkillMode::Dynamic {
            let enabled = self.enabled_skills.clone();
            self.registry
                .register(LoadSkillTool::new(self.skills.clone(), enabled));
        }
        self
    }

    /// Pre-activate a skill: the body joins the system prompt without the
    /// model's involvement.
    ///
    /// The counterpart of load_skill's "model loads on its own", this is
    /// the second path for **explicit user activation** (e.g. interactive
    /// UI selection, application-level parsing of slash commands): the
    /// skill enters the system prompt immediately and deterministically,
    /// consuming no tool rounds. Repeated activation is idempotent;
    /// activated skills are excluded from the menu (no duplicate
    /// disclosure), and the model can still load_skill other skills.
    ///
    /// Pre-activated bodies stay resident in the system prompt and are not
    /// trimmed by window Memory — the activation count is the user's token
    /// trade-off; session cleanup = constructing a new agent resets it.
    ///
    /// # Returns
    ///
    /// Activation succeeded (the skill exists and is visible) or it was
    /// already activated → `true`; the skill doesn't exist, is outside the
    /// allowlist, or the mode is inline → `false`.
    pub fn activate_skill(&mut self, name: &str) -> bool {
        if self.skill_mode != SkillMode::Dynamic {
            return false;
        }
        if !self.skill_visible(name) || self.skills.get(name).is_none() {
            return false;
        }
        if self.is_activated(name) {
            return true;
        }
        self.activated_skills.push(name.to_string());
        true
    }

    /// Deactivate a pre-activated skill: the body leaves the system prompt
    /// and returns to the menu (the model can reload it via load_skill).
    ///
    /// # Returns
    ///
    /// Successfully deactivated (previously activated) → `true`; the skill
    /// wasn't activated or the mode is inline → `false`.
    ///
    /// # Examples
    ///
    /// With no skills assembled (not dynamic mode), deactivation is always
    /// `false`:
    ///
    /// ```
    /// use molo::{FakeProvider, FakeReply, react_agent};
    ///
    /// let mut agent = react_agent!(FakeProvider::new([FakeReply::Text("Hello".into())]));
    /// assert!(!agent.deactivate_skill("greet"));
    /// ```
    pub fn deactivate_skill(&mut self, name: &str) -> bool {
        if self.skill_mode != SkillMode::Dynamic {
            return false;
        }
        if let Some(pos) = self.activated_skills.iter().position(|n| n == name) {
            self.activated_skills.remove(pos);
            true
        } else {
            false
        }
    }

    /// Whether the skill is in the session allowlist (no allowlist set =
    /// everything visible).
    fn skill_visible(&self, name: &str) -> bool {
        match &self.enabled_skills {
            None => true,
            Some(enabled) => enabled.contains(name),
        }
    }

    /// Whether the skill is pre-activated.
    fn is_activated(&self, name: &str) -> bool {
        self.activated_skills.iter().any(|n| n == name)
    }

    /// Publish an event (no-op when no channel is attached). Takes a
    /// **construction closure** rather than the event itself: with no
    /// observation channel attached (the default), construction cost is
    /// zero — the hot path (every Delta chunk) skips an unconditional heap
    /// allocation and copy.
    fn publish<E: AgentEvent + 'static>(&self, make_event: impl FnOnce() -> Arc<E>) {
        if let Some(pipe) = &self.events {
            pipe.publish(make_event());
        }
    }

    /// Non-streaming round loop (same semantics as the streaming path);
    /// rounds / tools / structured retries / usage are accumulated in
    /// [`RunCounters`] for the `RunEnded` summary at run wrap-up.
    async fn run_rounds_cancellable(
        &mut self,
        token: &CancellationToken,
        run_id: &str,
        counters: &mut RunCounters,
        schema: Option<&serde_json::Value>,
    ) -> Result<String, AgentError> {
        let schemas = self.registry.schemas();
        // Structured validator: built when this run has a schema; the retry
        // budget lives in the component (the count accumulates across
        // rounds and doesn't consume the tool-round budget).
        let mut validator = schema.map(|schema| {
            StructuredValidator::new(schema.clone(), self.config.max_structured_retries)
        });
        // Tool-round counter: the limit check only counts rounds where the
        // model requested tools — conversation rounds and structured
        // validation-retry rounds don't consume the tool-round budget
        // (structured retries have their own `max_structured_retries`
        // budget); errors no longer report "tool round limit exceeded"
        // because of structured retries.
        let mut tool_rounds = 0usize;
        loop {
            if tool_rounds >= self.config.max_tool_rounds {
                return Err(AgentError::TooManyToolRounds(self.config.max_tool_rounds));
            }
            counters.rounds += 1;
            // Between-round check: cancellation takes effect immediately —
            // this round hasn't started a conversation, so the memory is
            // clean (nothing from this round beyond the user message).
            if token.is_cancelled() {
                return Err(AgentError::Cancelled);
            }

            // Provider-call span for this round: duration comes from span
            // timing automatically; usage goes out through both channels
            // (the event stream's RunSummary as usual, and observability
            // records it on the span fields at wrap-up); the span is created
            // while the run is on the stack (the outer run block is
            // instrumented), so the hierarchy is established by the ambient
            // parent automatically. The round body runs directly inside the
            // run scope.
            // The block returns Option<String> = whether this round already
            // answered directly; internal errors propagate via `?` (the two
            // Ok arms pin the error type with turbofish — inference would
            // otherwise fail under multiple From impls).
            let answer: Option<String> = async {
                let llm_span = span_llm(run_id, counters.rounds);
                // Cancelled mid-conversation: run_until_cancelled drops the
                // in-flight request and returns immediately (nothing is
                // recorded for this round). When cancellation races with
                // completion, completion wins (primitive semantics).
                // Provider errors are recorded on the llm span here
                // (observability pinpoints the failed call); cancellations
                // are not (cancellation is a run-level outcome, covered by
                // the run span and the RunEnded event).
                // Model parameters on the request: the schema passed for
                // this run (typed path) takes precedence over the
                // hand-written schema in the config — endpoint-side
                // constraint and framework-side validation use the same
                // one.
                let mut options = self.config.options.clone();
                if options.structured.is_none() {
                    options.structured = schema.cloned();
                }
                let response = match token
                    .run_until_cancelled(
                        self.provider
                            .chat(ChatRequest {
                                messages: self.assemble_messages(self.memory.context().await?),
                                tools: schemas.clone(),
                                options,
                            })
                            .instrument(llm_span.clone()),
                    )
                    .await
                {
                    Some(Ok(response)) => response,
                    Some(Err(e)) => {
                        llm_span.record("error", e.to_string());
                        return Err(AgentError::Provider(e));
                    }
                    None => return Err(AgentError::Cancelled),
                };
                llm_span.record("usage.prompt_tokens", response.usage.prompt_tokens);
                llm_span.record("usage.completion_tokens", response.usage.completion_tokens);
                counters.usage_total += response.usage;

                // Per the Provider contract, this round's reply is exactly
                // one Assistant message; text, reasoning, and tool requests
                // stay in the same message (wire constraint: multiple tool
                // requests in the same round are not split apart).
                let Message::Assistant {
                    content,
                    reasoning,
                    tool_calls,
                } = response.message
                else {
                    // Defensive handling: a custom Provider may return a
                    // non-Assistant message; the library boundary responds
                    // to expected inputs with an error (same rigor as tool
                    // panic catching, see the registry).
                    return Err(AgentError::Provider(ProviderError::Api {
                        status: 0,
                        message: "provider returned a non-assistant message".into(),
                    }));
                };
                // Round text limit: a malicious endpoint emitting unbounded
                // text in one round could blow up memory (the Provider layer
                // only limits single lines; this is the per-round backstop,
                // sharing the same constant as the streaming path; reasoning
                // counts toward it too).
                if content.len() + reasoning.as_deref().map_or(0, str::len) > MAX_ROUND_TEXT {
                    return Err(AgentError::Provider(ProviderError::Api {
                        status: 0,
                        message: format!("round text exceeds size limit ({MAX_ROUND_TEXT} bytes)"),
                    }));
                }

                // Empty replies are not recorded: Assistant messages with no
                // content, no reasoning, and no tool requests don't go into
                // Memory; reasoning and tool requests are saved along with
                // the content.
                if !content.is_empty() || reasoning.is_some() || !tool_calls.is_empty() {
                    self.memory
                        .record(Message::Assistant {
                            content: content.clone(),
                            reasoning,
                            tool_calls: tool_calls.clone(),
                        })
                        .await?;
                }

                if tool_calls.is_empty() {
                    // The model answered directly. Structured output: the
                    // component validates the final answer — on failure,
                    // feed it back to the model for retry (budget is built
                    // into the component, decoupled from the tool-round
                    // limit); exceeding it fails the run.
                    if let Some(validator) = &mut validator {
                        match validator.validate(&content) {
                            StructuredOutcome::Passed => {}
                            StructuredOutcome::Retry { message } => {
                                self.memory.record(message).await?;
                                return Ok::<Option<String>, AgentError>(None);
                            }
                            StructuredOutcome::Exhausted { max_retries } => {
                                return Err(AgentError::StructuredRetriesExhausted(max_retries));
                            }
                        }
                    }
                    return Ok::<Option<String>, AgentError>(Some(content));
                }

                counters.tool_calls_total += tool_calls.len();
                tool_rounds += 1;
                // Tool round: executed atomically and never interrupted —
                // tools are user code with no cancellation interface, and a
                // partial execution risks side effects; interrupting
                // mid-way would also leave the Assistant recorded without
                // its ToolResult, breaking the next round's message
                // sequence. Cancellation takes effect naturally after the
                // tool round ends, before the next round's conversation.
                // When multiple tools in the same round have all run, their
                // results are fed back one by one right after.
                for call in tool_calls {
                    // Execution + events + span live in run_tool_call
                    // (shared by both paths); the text is recorded and fed
                    // back to the model.
                    let outcome = self.run_tool_call(call, run_id, counters.rounds).await;
                    // Protected results (skill bodies, etc.) are recorded
                    // via record_protected, exempt from window trimming;
                    // when recording fails, the error text is recorded as a
                    // fallback (memory integrity, see record_tool_result).
                    self.record_tool_result(&outcome).await?;
                }
                Ok::<Option<String>, AgentError>(None)
            }
            .await?;
            if let Some(answer) = answer {
                return Ok(answer);
            }
        }
    }

    /// Assemble the full prompt for each request: System first (base prompt
    /// + skill disclosure, per the assembly mode), followed by Memory's
    /// conversation history.
    ///
    /// Assembly happens when the conversation starts and is not written to
    /// Memory — the system prompt is static configuration while Memory is
    /// the dynamic conversation record; the two are managed separately. The
    /// skill part reads the registry fresh on every request: hot-swapped
    /// additions/removals take effect on the next request.
    fn assemble_messages(&self, context: Vec<Message>) -> Vec<Message> {
        let mut messages = Vec::with_capacity(context.len() + 1);
        let system = self.assemble_system_prompt();
        if !system.is_empty() {
            messages.push(Message::system(&system));
        }
        messages.extend(context);
        messages
    }

    /// Assemble the system prompt: base prompt + skill disclosure (per the
    /// assembly mode); an empty result = no system prompt.
    ///
    /// - No skills assembled: the base prompt is returned as-is;
    /// - Dynamic mode: base prompt + menu (skills in the allowlist that are
    ///   not activated, `- name: desc` in registration order) +
    ///   pre-activated bodies ([Skill name] sections, in activation order);
    /// - Inline mode: base prompt + all skill bodies (the allowlist has no
    ///   effect).
    fn assemble_system_prompt(&self) -> String {
        let base = self.system_prompt.as_str();
        let skills = self.skills.skills();
        if skills.is_empty() {
            return base.to_string();
        }
        let mut out = String::new();
        out.push_str(base);
        match self.skill_mode {
            SkillMode::None => {}
            SkillMode::Dynamic => {
                // Menu: skills in the allowlist that are not activated, in
                // registration order.
                let menu: Vec<String> = skills
                    .iter()
                    .filter(|s| self.skill_visible(s.name()) && !self.is_activated(s.name()))
                    .map(|s| format!("- {}: {}", s.name(), s.description()))
                    .collect();
                append_sections(&mut out, &menu);
                // Pre-activated bodies: in activation order; skills already
                // removed are skipped when get misses.
                let activated: Vec<String> = self
                    .activated_skills
                    .iter()
                    .filter_map(|n| self.skills.get(n))
                    .map(|s| format!("[Skill {}]\n{}", s.name(), s.body()))
                    .collect();
                append_sections(&mut out, &activated);
            }
            SkillMode::Inline => {
                let bodies: Vec<String> = skills
                    .iter()
                    .map(|s| format!("[Skill {}]\n{}", s.name(), s.body()))
                    .collect();
                append_sections(&mut out, &bodies);
            }
        }
        out
    }

    /// Execute a single tool call (shared by the run / run_stream paths):
    /// tool span + ToolStarted / ToolCompleted events + registry execution,
    /// returning [`ToolCallOutcome`] — where recording and streaming
    /// dispatch happen is up to the caller (streaming dispatches first,
    /// then records).
    async fn run_tool_call(
        &mut self,
        call: ToolCall,
        run_id: &str,
        round: usize,
    ) -> ToolCallOutcome {
        // Tool-call span: carries duration; records error on failure. The
        // span is created while the run is on the stack (ambient parent on
        // both paths, see the span-construction comments), so the hierarchy
        // is correct automatically.
        let tool_span = span_tool(run_id, round, &call.name);
        // Tool-started event (carries id / arguments; subscribers pair it
        // with ToolCompleted by id).
        self.publish(|| {
            Arc::new(ReActEvent::ToolStarted {
                id: call.id.clone(),
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            })
        });
        // Execute; the Result status rides along with the event (Ok/Err is
        // classified by the registry), and the text (Ok result / Err's
        // Display) is recorded and fed back to the model.
        let result = self
            .registry
            .call(&call.name, &call.arguments, &self.state)
            .instrument(tool_span.clone())
            .await;
        if let Err(e) = &result {
            tool_span.record("error", e.to_string());
        }
        let content = match &result {
            Ok(text) => text.clone(),
            Err(e) => e.to_string(),
        };
        // The protected declaration is owned by the tool (skill bodies,
        // etc.); the recorder uses it to choose between
        // record / record_protected.
        let protected = self
            .registry
            .get(&call.name)
            .map(|tool| tool.protected_output())
            .unwrap_or(false);
        let publish_tool_completed = {
            let id = call.id.clone();
            let name = call.name.clone();
            move || Arc::new(ReActEvent::ToolCompleted { id, name, result })
        };
        self.publish(publish_tool_completed);
        ToolCallOutcome {
            call,
            content,
            protected,
        }
    }

    /// Record a tool result, falling back to recording the error text
    /// before re-raising on failure.
    ///
    /// The tool has already run (side effects happened), so its result text
    /// must not be lost: an Assistant message recorded without its
    /// ToolResult leaves the memory incomplete, the next request would
    /// carry an unpaired assistant/tool sequence, real endpoints reject it
    /// (400), and the reported cause would be disconnected from the real
    /// failure. If the fallback recording itself fails, the error is
    /// swallowed — the original error takes priority, and there's no point
    /// retrying when the same Memory keeps failing.
    async fn record_tool_result(&mut self, outcome: &ToolCallOutcome) -> Result<(), AgentError> {
        let message = Message::tool_result(outcome.call.id.clone(), outcome.content.clone());
        let record_result = if outcome.protected {
            self.memory.record_protected(message).await
        } else {
            self.memory.record(message).await
        };
        match record_result {
            Ok(()) => Ok(()),
            Err(e) => {
                let fallback = Message::tool_result(
                    outcome.call.id.clone(),
                    format!("memory record failed: {e}"),
                );
                let _ = if outcome.protected {
                    self.memory.record_protected(fallback).await
                } else {
                    self.memory.record(fallback).await
                };
                Err(e.into())
            }
        }
    }
}

/// The execution outcome of one tool call: the call info is returned as-is,
/// and the caller decides recording and dispatch.
struct ToolCallOutcome {
    /// The call as-is (with id / name / arguments, used for event pairing
    /// and locating the record).
    call: ToolCall,
    /// The text fed back (Ok result / Err's Display, visible to the model).
    content: String,
    /// Whether the result is protected (declared by the tool via
    /// [`Tool::protected_output`](crate::Tool::protected_output); protected
    /// results are exempt from window trimming when recorded).
    protected: bool,
}

impl ReActAgent {
    /// Typed run: same semantics as [`Agent::run`](Agent::run) (records
    /// input, drives the loop), but the final answer is deserialized into
    /// the type parameter `U` after validation — this run auto-generates a
    /// JSON Schema from `U`
    /// ([`schemars`](https://docs.rs/schemars)-derived, the same pipeline
    /// as tool parameter schemas), with no constructor configuration
    /// needed.
    ///
    /// Relation to [`Agent::run`](Agent::run): run returns the model's
    /// answer text (free text, or the JSON text of a hand-written Schema
    /// from
    /// [`with_structured_output`](ReActAgent::with_structured_output));
    /// run_typed generates the Schema from the type and returns the type.
    /// The return type is declared in the let annotation (see `# Examples`);
    /// no turbofish needed.
    ///
    /// # Errors
    ///
    /// - [`AgentError::StructuredRetriesExhausted`][]: validation failures
    ///   are fed back to the model for retry, but the
    ///   [`AgentConfig::max_structured_retries`] budget is exhausted
    ///   without success;
    /// - [`AgentError::StructuredParse`][]: validation passed but
    ///   deserialization failed (when a derived schema customized with
    ///   `#[schemars(...)]` disagrees with the serde representation);
    /// - otherwise the same as [`Agent::run`](Agent::run) (Memory /
    ///   Provider / round limit / cancellation).
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), molo::AgentError> {
    /// use molo::{FakeProvider, FakeReply, ReActAgent};
    /// use schemars::JsonSchema;
    /// use serde::Deserialize;
    ///
    /// #[derive(Deserialize, JsonSchema)]
    /// struct Weather {
    ///     city: String,
    /// }
    ///
    /// let mut agent = ReActAgent::new(
    ///     FakeProvider::new([FakeReply::Text(r#"{"city":"Beijing"}"#.into())]),
    ///     molo::tool::ToolRegistry::new(),
    ///     "You are a weather assistant",
    /// );
    /// let weather: Weather = agent.run_typed("How's the weather in Beijing").await?;
    /// assert_eq!(weather.city, "Beijing");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn run_typed<U>(&mut self, input: &str) -> Result<U, AgentError>
    where
        U: DeserializeOwned + JsonSchema + Send + Sync,
    {
        // Implementation lives in [`TypedAgent`]; the inherent delegate
        // lets call sites avoid importing the trait.
        TypedAgent::run_typed(self, input).await
    }
}

#[async_trait::async_trait]
impl TypedAgent for ReActAgent {
    async fn run_typed<U>(&mut self, input: &str) -> Result<U, AgentError>
    where
        U: DeserializeOwned + JsonSchema + Send + Sync,
    {
        let schema = serde_json::to_value(schemars::schema_for!(U))
            .expect("schemars-generated schema always serializes (pure JSON value structure)");
        let text = self
            .run_cancellable_inner(input, &CancellationToken::new(), Some(&schema))
            .await?;
        serde_json::from_str(&text).map_err(|e| AgentError::StructuredParse(e.to_string()))
    }
}

/// Cumulative counters for a single run (rounds / tools / usage), for the
/// `RunEnded` summary at wrap-up; the non-streaming path accumulates
/// through a mutable reference (the streaming path accumulates in local
/// variables inside the generator, same semantics).
/// Structured validation-retry counts don't live here — they're held by the
/// [`StructuredValidator`] component.
#[derive(Default)]
struct RunCounters {
    /// Number of conversation rounds (same semantics as
    /// `RunSummary.rounds`).
    rounds: usize,
    /// Total number of tool executions.
    tool_calls_total: usize,
    /// Sum of token usage across rounds.
    usage_total: Usage,
}

impl fmt::Debug for ReActAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Provider / Memory are trait objects and can't be Debug'd; print
        // the composition and behavior config.
        f.debug_struct("ReActAgent")
            .field("provider", &"Box<dyn Provider>")
            .field("memory", &"Box<dyn Memory>")
            .field("tools", &self.registry)
            .field("system_prompt", &self.system_prompt)
            .field("config", &self.config)
            .field("state", &self.state)
            .field(
                "events",
                &match &self.events {
                    Some(_) => "Some<dyn EventChannel>",
                    None => "None",
                },
            )
            .field("skills", &self.skills)
            .finish()
    }
}

/// Append a set of sections to the system prompt: sections are separated
/// by blank lines, with a blank line inserted before existing content. An
/// empty section list is a no-op.
fn append_sections(out: &mut String, sections: &[String]) {
    if sections.is_empty() {
        return;
    }
    if !out.is_empty() {
        out.push_str("\n\n");
    }
    for (i, section) in sections.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(section);
    }
}

#[async_trait::async_trait]
impl Agent for ReActAgent {
    async fn run(&mut self, input: &str) -> Result<String, AgentError> {
        // Convenience no-cancellation form: create a token that is never
        // cancelled and delegate to the main implementation — the loop
        // logic exists in only one place (see CancellableAgent's
        // run_cancellable).
        self.run_cancellable(input, &CancellationToken::new()).await
    }

    async fn run_stream<'a>(
        &'a mut self,
        input: &'a str,
    ) -> Result<BoxStream<'a, Result<MessageChunk, AgentError>>, AgentError> {
        let token = CancellationToken::new();
        self.run_stream_cancellable(input, &token).await
    }
}

// Span construction is unified — both run / run_stream paths share the
// same field set. round is passed in by the caller (already incremented on
// the non-streaming path; current round + 1 on the streaming path). A
// single hierarchy mechanism: the ambient parent at creation time — llm/tool
// spans are created inside the round body, and while the round body runs,
// `agent.run` is always on the stack (non-streaming: the run block
// instruments the whole span; streaming: SpanStream enters on every poll),
// so no explicit parent and no double instrumenting are needed.

/// The `agent.run` root span: the observability identifier of a whole run
/// (including the error recorded on failure).
fn span_run(run_id: &str) -> tracing::Span {
    tracing::info_span!("agent.run", "run.id" = %run_id, error = tracing::field::Empty)
}

/// Provider-call span: duration comes from span timing automatically; usage
/// goes out through both channels (the business-side RunSummary as usual,
/// observability records it on the span fields at wrap-up); errors are
/// recorded when the Provider fails (both paths' error branches record).
fn span_llm(run_id: &str, round: usize) -> tracing::Span {
    tracing::debug_span!(
        "llm_request",
        "run.id" = %run_id,
        round = round,
        usage.prompt_tokens = tracing::field::Empty,
        usage.completion_tokens = tracing::field::Empty,
        error = tracing::field::Empty,
    )
}

/// Tool-call span: carries duration; records error on failure (see
/// run_tool_call).
fn span_tool(run_id: &str, round: usize, name: &str) -> tracing::Span {
    tracing::debug_span!(
        "tool",
        "run.id" = %run_id,
        round = round,
        name = %name,
        error = tracing::field::Empty,
    )
}

/// Generator terminal wrap-up: publish `RunEnded` (with the accumulated
/// summary and error state; error=None on normal completion, error set on
/// cancellation / failure; no-op when no channel is attached).
///
/// A free function rather than a method: the generator borrows both
/// `self.events` (immutably) and other `&mut self` calls (record, etc.)
/// simultaneously; passing references pointwise avoids closure-capture
/// borrow tangles.
fn publish_ended(
    events: &Option<Arc<dyn EventChannel>>,
    rounds: usize,
    tool_calls_total: usize,
    usage_total: Usage,
    error: Option<AgentError>,
) {
    if let Some(pipe) = events {
        pipe.publish(Arc::new(ReActEvent::RunEnded {
            summary: RunSummary {
                rounds,
                tool_calls: tool_calls_total,
                usage: usage_total,
            },
            error,
        }));
    }
}

/// Streaming termination helper: observability wrap-up (span error +
/// RunEnded event) plus the terminal chunk — cancellation →
/// `Ok(Cancelled)`, other errors → `Err(e)`; the call site `yield`s and
/// `break`s. Consolidates the fixed four-step sequence of every failure
/// branch in the generator (record → publish_ended → yield → break).
///
/// A free function rather than a method: the generator borrows both
/// `self.events` and `&mut self` calls (record, etc.) simultaneously;
/// passing references pointwise avoids closure-capture borrow tangles (same
/// as publish_ended).
fn stream_end(
    events: &Option<Arc<dyn EventChannel>>,
    run_span: &tracing::Span,
    rounds: usize,
    tool_calls_total: usize,
    usage_total: Usage,
    error: AgentError,
) -> Result<MessageChunk, AgentError> {
    // Cancellation is a normal outcome (terminating with the Cancelled
    // terminal chunk), so no span error is recorded — otherwise the
    // observability dashboard would mark a user-initiated stop as a
    // failure.
    if !matches!(error, AgentError::Cancelled) {
        run_span.record("error", error.to_string());
    }
    publish_ended(
        events,
        rounds,
        tool_calls_total,
        usage_total,
        Some(error.clone()),
    );
    match error {
        AgentError::Cancelled => Ok(MessageChunk::Cancelled),
        e => Err(e),
    }
}

impl ReActAgent {
    /// The non-streaming main implementation for cooperative cancellation
    /// (returns the model's answer text); public entry points:
    /// `Agent::run` / `CancellableAgent::run_cancellable` (text, carrying
    /// the hand-written Schema) and [`run_typed`](ReActAgent::run_typed)
    /// (typed, carrying this run's generated Schema).
    ///
    /// `schema` = the JSON Schema used for this run's structured
    /// validation; `None` = free text (no structured constraint).
    async fn run_cancellable_inner(
        &mut self,
        input: &str,
        token: &CancellationToken,
        schema: Option<&serde_json::Value>,
    ) -> Result<String, AgentError> {
        // Run id and root span — one agent.run span for the whole run (from
        // recording the input to the wrap-up event), sharing the same
        // run.id as the event stream's RunStarted (the correlation key
        // between the two channels); on Err, the error is recorded at
        // wrap-up (observability spots the failed run at a glance).
        let run_id = self.next_run_id();
        let run_span = span_run(&run_id);
        let result = async {
            // Record first, publish after (consistent with the streaming
            // path): RunStarted's claim that "the user input is recorded"
            // holds; when recording fails, neither RunStarted nor RunEnded
            // is published, leaving no dangling segment in the event
            // stream.
            self.memory.record(Message::user(input)).await?;
            self.publish(|| {
                Arc::new(ReActEvent::RunStarted {
                    run_id: run_id.clone(),
                    input: input.to_string(),
                })
            });

            // Round / tool / structured-retry / usage counters: for the
            // RunEnded summary at wrap-up (also accumulated on the
            // non-streaming path).
            let mut counters = RunCounters::default();
            let result = self
                .run_rounds_cancellable(token, &run_id, &mut counters, schema)
                .await;
            // Wrap-up event: error=None on success; error set on
            // cancellation / failure (bystanders can observe the outcome of
            // a non-streaming run); built with publish_ended, shared with
            // the streaming path.
            publish_ended(
                &self.events,
                counters.rounds,
                counters.tool_calls_total,
                counters.usage_total,
                result.as_ref().err().cloned(),
            );
            result
        }
        .instrument(run_span.clone())
        .await;
        if let Err(e) = &result {
            run_span.record("error", e.to_string());
        }
        result
    }

    /// The streaming main implementation for cooperative cancellation
    /// (returns text-delta chunks; structured output is assembled and
    /// parsed by the caller); public entry points: `Agent::run_stream` /
    /// `CancellableAgent::run_stream_cancellable`.
    async fn run_stream_cancellable_inner<'a>(
        &'a mut self,
        input: &'a str,
        token: &CancellationToken,
    ) -> Result<BoxStream<'a, Result<MessageChunk, AgentError>>, AgentError> {
        // Run id and root span — the run span covers the consumption period
        // from "stream returned" to "stream dropped" (SpanStream enters on
        // every poll); the error field is recorded by the generator's error
        // branches; the same run.id is shared with the event stream's
        // RunStarted (the correlation key between the two channels).
        //
        // Recording and RunStarted are published before the stream is
        // returned (failing fast, so callers see the error before holding
        // the stream): if a caller drops the returned stream without
        // polling, the event stream is left with a dangling segment without
        // RunEnded — event consumers pair by run.id and wrap up such cases
        // themselves.
        let run_id = self.next_run_id();
        let run_span = span_run(&run_id);
        let stream_span = run_span.clone();
        self.memory.record(Message::user(input)).await?;
        self.publish(|| {
            Arc::new(ReActEvent::RunStarted {
                run_id: run_id.clone(),
                input: input.to_string(),
            })
        });
        let schemas = self.registry.schemas();
        let max_rounds = self.config.max_tool_rounds;

        // Streaming state machine: an async_stream generator — sequential
        // awaits in the loop body, syntax isomorphic to a synchronous loop
        // (replacing a hand-written unfold state machine).
        // The token is cloned into the generator (move-captured; the stream
        // holds a copy — sharing the same cancellation state with the
        // caller; the returned stream doesn't borrow the caller's token, so
        // lifetimes are decoupled).
        let token = token.clone();
        let stream = async_stream::stream! {
            let mut rounds = 0usize;
            // Execution summary (carried with Done): rounds is the `rounds`
            // above; tool counts and usage accumulate per round.
            let mut tool_calls_total = 0usize;
            let mut usage_total = Usage::default();
            // Structured validator: built when the config has a hand-written
            // schema; the retry budget lives in the component (the streaming
            // path only reads the config, so it doesn't interfere with
            // run_typed's per-run schema).
            let mut validator = self.config.options.structured.as_ref().map(|schema| {
                StructuredValidator::new(schema.clone(), self.config.max_structured_retries)
            });
            // Tool-round counter (same semantics as the non-streaming
            // path): the limit only counts rounds where the model requested
            // tools; structured validation-retry rounds don't consume it
            // (independent max_structured_retries budget).
            let mut tool_rounds = 0usize;
            // Labeled loop: in-stream errors must terminate the whole
            // generator — a plain break only exits the current loop;
            // break 'rounds terminates the entire generator.
            'rounds: loop {
                // Same semantics as run: the model still requests tools
                // past the limit → error event.
                if tool_rounds >= max_rounds {
                    yield stream_end(&self.events, &run_span, rounds, tool_calls_total,
                        usage_total, AgentError::TooManyToolRounds(max_rounds));
                    break;
                }

                // Increment at the start of the round (same semantics as
                // the non-streaming path): RunEnded reports "rounds
                // started" — both paths report 1 for a pre-cancelled token,
                // no drift.
                rounds += 1;

                // Between-round check: cancellation takes effect
                // immediately — this round hasn't started a conversation,
                // so the memory is clean.
                if token.is_cancelled() {
                    yield stream_end(&self.events, &run_span, rounds, tool_calls_total,
                        usage_total, AgentError::Cancelled);
                    break;
                }

                // Start a streaming conversation round (cancelled during
                // establishment → nothing recorded for this round, memory
                // clean).
                let context = match self.memory.context().await {
                    Ok(messages) => messages,
                    Err(e) => {
                        yield stream_end(&self.events, &run_span, rounds, tool_calls_total,
                            usage_total, AgentError::Memory(e));
                        break;
                    }
                };
                // Provider-call span (streaming: establishment and
                // per-event consumption both belong to llm_request, the span
                // is shared across the round's consumption loop; usage is
                // recorded at wrap-up when the Done event arrives — the
                // business-side RunSummary as usual); the span is created
                // while the run is on the stack (SpanStream enters on every
                // poll), so the hierarchy is correct automatically.
                let llm_span = span_llm(&run_id, rounds);
                let mut provider_stream = match token
                    .run_until_cancelled(
                        self.provider
                            .stream_chat(ChatRequest {
                                messages: self.assemble_messages(context),
                                tools: schemas.clone(),
                                options: self.config.options.clone(),
                            })
                            .instrument(llm_span.clone()),
                    )
                    .await
                {
                    Some(Ok(stream)) => stream,
                    Some(Err(e)) => {
                        // Provider errors are recorded on the llm span
                        // (observability pinpoints the failed call).
                        llm_span.record("error", e.to_string());
                        yield stream_end(&self.events, &run_span, rounds, tool_calls_total,
                            usage_total, AgentError::Provider(e));
                        break;
                    }
                    None => {
                        // Cancelled before the conversation was
                        // established.
                        yield stream_end(&self.events, &run_span, rounds, tool_calls_total,
                            usage_total, AgentError::Cancelled);
                        break;
                    }
                };

                // Consume this round's full event stream: text dispatches
                // character by character in real time; tool requests are
                // collected whole at the end of the round.
                // Every next is wrapped in run_until_cancelled —
                // cancellation stops per-character dispatch immediately;
                // when cancellation races with event arrival, completion
                // wins (same trade-off as the non-streaming path): keep
                // consuming as long as events keep flowing, and only void
                // the round when next itself is preempted by cancellation
                // (None) (record happens at the end of the round, so the
                // memory is clean when voided).
                let mut text = String::new();
                let mut reasoning = String::new();
                let mut calls = Vec::new();
                loop {
                    // next()'s Output is itself an Option, and
                    // run_until_cancelled wraps it in another (returns None
                    // on cancellation): Some(Some(event)) / Some(None) =
                    // stream ended naturally / None = cancelled during next
                    // (this round is voided).
                    let next = token
                        .run_until_cancelled(provider_stream.next().instrument(llm_span.clone()))
                        .await;
                    let Some(Some(event)) = next else {
                        if next.is_none() {
                            // Cancelled during next: this round wasn't fully
                            // consumed; void it with Cancelled.
                            yield stream_end(&self.events, &run_span, rounds, tool_calls_total,
                                usage_total, AgentError::Cancelled);
                            break 'rounds;
                        }
                        // The event stream ended naturally; move to
                        // end-of-round wrap-up.
                        break;
                    };
                    match event {
                        Ok(StreamEvent::Delta(delta)) => {
                            // Round text accumulation limit: a malicious
                            // endpoint can keep sending Deltas without
                            // bound (the Provider layer only limits single
                            // lines; this is the per-round backstop).
                            if text.len() + delta.len() > MAX_ROUND_TEXT {
                                yield stream_end(&self.events, &run_span, rounds,
                                    tool_calls_total, usage_total,
                                    AgentError::Provider(ProviderError::Api {
                                        status: 0,
                                        message: format!(
                                            "round text exceeds size limit ({MAX_ROUND_TEXT} bytes)"
                                        ),
                                    }));
                                break 'rounds;
                            }
                            text.push_str(&delta);
                            self.publish(|| Arc::new(ReActEvent::Delta { text: delta.clone() }));
                            yield Ok(MessageChunk::Delta(delta));
                        }
                        Ok(StreamEvent::Reasoning(chunk)) => {
                            // Same limit as text: a custom Provider may
                            // send reasoning deltas without bound.
                            if reasoning.len() + chunk.len() > MAX_ROUND_TEXT {
                                yield stream_end(&self.events, &run_span, rounds,
                                    tool_calls_total, usage_total,
                                    AgentError::Provider(ProviderError::Api {
                                        status: 0,
                                        message: format!(
                                            "round reasoning exceeds size limit ({MAX_ROUND_TEXT} bytes)"
                                        ),
                                    }));
                                break 'rounds;
                            }
                            reasoning.push_str(&chunk);
                            self.publish(move || Arc::new(ReActEvent::Reasoning { text: chunk }));
                        }
                        Ok(StreamEvent::ToolCall { id, name, arguments }) => {
                            calls.push(ToolCall {
                                id: id.clone(),
                                name: name.clone(),
                                arguments: arguments.clone(),
                            });
                            yield Ok(MessageChunk::ToolCall { id, name, arguments });
                        }
                        Ok(StreamEvent::Done { usage, .. }) => {
                            // Streaming usage may be absent (the endpoint
                            // didn't return it); missing rounds count as
                            // zero.
                            // usage goes through both channels:
                            // observability records it on the llm_request
                            // span at wrap-up, and the business-side
                            // RunSummary accumulates as usual.
                            if let Some(usage) = usage {
                                llm_span.record("usage.prompt_tokens", usage.prompt_tokens);
                                llm_span.record("usage.completion_tokens", usage.completion_tokens);
                                usage_total += usage;
                            }
                            // Done = this round is complete: move to
                            // end-of-round wrap-up immediately, consuming no
                            // further events — the Provider's Done is always
                            // the stream-final event, and anything after it
                            // was already discarded by the Provider; this
                            // break is belt and braces.
                            break;
                        }
                        Err(e) => {
                            // Errors are produced as Err events and
                            // terminate the stream (no Done afterwards);
                            // recorded on the llm span (observability
                            // pinpoints the failed call).
                            llm_span.record("error", e.to_string());
                            yield stream_end(&self.events, &run_span, rounds, tool_calls_total,
                                usage_total, AgentError::Provider(e));
                            break 'rounds;
                        }
                    }
                }

                // Empty Assistant messages are not recorded (consistent
                // with run).
                if !text.is_empty() || !reasoning.is_empty() || !calls.is_empty() {
                    let message = Message::Assistant {
                        content: text.clone(),
                        reasoning: (!reasoning.is_empty()).then_some(reasoning),
                        tool_calls: calls.clone(),
                    };
                    match self.memory.record(message).await {
                        Ok(()) => {}
                        Err(e) => {
                            yield stream_end(&self.events, &run_span, rounds, tool_calls_total,
                                usage_total, AgentError::Memory(e));
                            break;
                        }
                    }
                }

                if calls.is_empty() {
                    // The model answered directly. Structured output: the
                    // component validates the final answer — on failure,
                    // feed it back to the model for retry (no Done
                    // dispatched; the loop continues; the budget is built
                    // into the component, decoupled from the tool-round
                    // limit), and exceeding it terminates the stream with
                    // an error; only a pass wraps up.
                    if let Some(validator) = &mut validator {
                        match validator.validate(&text) {
                            StructuredOutcome::Passed => {}
                            StructuredOutcome::Retry { message } => {
                                // Record the feedback and continue to the
                                // next round (no Done dispatched).
                                match self.memory.record(message).await {
                                    Ok(()) => continue 'rounds,
                                    Err(e) => {
                                        yield stream_end(&self.events, &run_span, rounds,
                                            tool_calls_total, usage_total,
                                            AgentError::Memory(e));
                                        break;
                                    }
                                }
                            }
                            StructuredOutcome::Exhausted { max_retries } => {
                                yield stream_end(&self.events, &run_span, rounds,
                                    tool_calls_total, usage_total,
                                    AgentError::StructuredRetriesExhausted(max_retries));
                                break 'rounds;
                            }
                        }
                    }
                    // Validation passed (or no structured constraint): this
                    // round ran to completion; wrap up directly.
                    publish_ended(
                        &self.events,
                        rounds,
                        tool_calls_total,
                        usage_total,
                        None,
                    );
                    yield Ok(MessageChunk::Done(RunSummary {
                        rounds,
                        tool_calls: tool_calls_total,
                        usage: usage_total,
                    }));
                    break;
                }

                // Tool round: execute one by one, feeding results back
                // right after each.
                tool_calls_total += calls.len();
                tool_rounds += 1;
                for call in calls {
                    // Execution + events + span live in run_tool_call
                    // (shared by both paths); the result text is fed back
                    // right after, then recorded.
                    let outcome = self.run_tool_call(call, &run_id, rounds).await;
                    yield Ok(MessageChunk::ToolResult {
                        id: outcome.call.id.clone(),
                        name: outcome.call.name.clone(),
                        content: outcome.content.clone(),
                    });
                    // Protected results (skill bodies, etc.) are recorded
                    // via record_protected, exempt from window trimming;
                    // when recording fails, the error text is recorded as a
                    // fallback (memory integrity, see record_tool_result).
                    if let Err(e) = self.record_tool_result(&outcome).await {
                        yield stream_end(&self.events, &run_span, rounds, tool_calls_total,
                            usage_total, e);
                        // Errors are produced as Err items and terminate
                        // the stream — the break must exit the whole round
                        // loop, not just the tool for (otherwise the loop
                        // would continue to the next round after an Err).
                        break 'rounds;
                    }
                }
            }
        };
        Ok(Box::pin(SpanStream {
            stream: Box::pin(stream),
            span: stream_span,
        }))
    }
}

#[async_trait::async_trait]
impl CancellableAgent for ReActAgent {
    async fn run_cancellable(
        &mut self,
        input: &str,
        token: &CancellationToken,
    ) -> Result<String, AgentError> {
        // The text form carries the hand-written Schema (if any); the typed
        // path passes this run's generated Schema directly via
        // [`run_typed`](ReActAgent::run_typed).
        // Clone first: avoids conflicting mutable borrows of self with
        // immutable borrows of config.
        let schema = self.config.options.structured.clone();
        self.run_cancellable_inner(input, token, schema.as_ref())
            .await
    }

    async fn run_stream_cancellable<'a>(
        &'a mut self,
        input: &'a str,
        token: &CancellationToken,
    ) -> Result<BoxStream<'a, Result<MessageChunk, AgentError>>, AgentError> {
        self.run_stream_cancellable_inner(input, token).await
    }
}

/// Wrap a span around a stream: enters on each poll and exits on return;
/// the span's lifetime = the stream's consumption period (creation to drop,
/// including time spent waiting while the consumer hasn't polled yet).
///
/// tracing's `Instrument` only applies to Futures; Streams need this thin
/// wrapper (per-await instrumenting inside the generator is already done
/// explicitly at each await point; this covers the `agent.run` root span
/// across the whole consumption). The inner stream is boxed and pinned at
/// construction (the generator is not Unpin — async_stream uses internal
/// pin_project; once pinned it never moves), and both `Pin<Box<S>>` and
/// `Span` are Unpin ⇒ the wrapper is always Unpin, so projection is safe,
/// no unsafe.
struct SpanStream<S> {
    stream: Pin<Box<S>>,
    span: tracing::Span,
}

impl<S: futures::Stream> futures::Stream for SpanStream<S> {
    type Item = S::Item;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        // Clone the span handle before entering: the guard borrows the
        // local handle, not self (we need &mut self below; Span is a shared
        // handle, so the clone refers to the same span).
        let span = self.span.clone();
        let _enter = span.enter();
        // SpanStream is Unpin (both Pin<Box<S>> and Span are Unpin) ⇒
        // DerefMut is safe; poll the pinned inner stream directly, no
        // unsafe.
        self.stream.as_mut().poll_next(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CancellationToken;
    use crate::memory::MemoryError;
    use crate::message::ContentBlock;
    use crate::provider::{
        ChatResponse, FakeProvider, FakeReply, FinishReason, ProviderError, StreamEvent,
        TimeoutStage,
    };
    use crate::tool::{Tool, ToolError, ToolSchema};
    use futures::StreamExt;
    use serde::Deserialize;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Shared wrapper: a shared reference to FakeProvider + Provider
    /// delegation, letting tests inspect the request history
    /// ([`SharedFake::requests`]) after the Agent has run to assert
    /// behavioral expectations.
    #[derive(Clone)]
    struct SharedFake(Arc<FakeProvider>);

    impl SharedFake {
        fn new(replies: impl IntoIterator<Item = FakeReply>) -> Self {
            Self(Arc::new(FakeProvider::new(replies)))
        }

        fn requests(&self) -> Vec<ChatRequest> {
            self.0.requests()
        }
    }

    #[async_trait::async_trait]
    impl Provider for SharedFake {
        async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ProviderError> {
            self.0.chat(request).await
        }

        async fn stream_chat(
            &self,
            request: ChatRequest,
        ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError> {
            self.0.stream_chat(request).await
        }
    }

    /// Fake tool for tests: returns fixed text and counts calls.
    #[derive(Debug, Clone)]
    struct FakeTool {
        name: &'static str,
        result: &'static str,
        calls: Arc<AtomicUsize>,
    }

    impl FakeTool {
        fn new(name: &'static str, result: &'static str) -> (Self, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    name,
                    result,
                    calls: calls.clone(),
                },
                calls,
            )
        }
    }

    #[async_trait::async_trait]
    impl Tool for FakeTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: self.name.into(),
                description: "Test tool".into(),
                parameters: serde_json::json!({}),
            }
        }

        async fn call(
            &self,
            _arguments: serde_json::Value,
            _state: &SharedState,
        ) -> Result<String, ToolError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(self.result.to_string())
        }
    }

    fn call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    /// Simple assembly: default Memory + optional system prompt (empty
    /// string = none).
    fn agent(fake: SharedFake, system_prompt: &str) -> ReActAgent {
        ReActAgent::new(fake, ToolRegistry::new(), system_prompt)
    }

    /// Assembly: custom registry + config (chained).
    fn agent_with_registry(
        fake: SharedFake,
        registry: ToolRegistry,
        config: AgentConfig,
    ) -> ReActAgent {
        ReActAgent::new(fake, registry, "").with_config(config)
    }

    #[tokio::test]
    async fn direct_answer() {
        let fake = SharedFake::new([FakeReply::Text("Hello".into())]);
        let mut agent = agent(fake.clone(), "");

        let answer = agent.run("Are you there").await.unwrap();
        assert_eq!(answer, "Hello");

        let requests = fake.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].messages.len(), 1);
        assert_eq!(requests[0].messages[0], Message::user("Are you there"));
    }

    #[tokio::test]
    async fn single_tool_round() {
        let (calc, calls) = FakeTool::new("calc", "42");
        let mut registry = ToolRegistry::new();
        registry.register(calc);
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "calc", r#"{"a":1}"#)],
            },
            FakeReply::Text("The answer is 42".into()),
        ]);
        let mut agent = agent_with_registry(fake.clone(), registry, AgentConfig::default());

        let answer = agent.run("Compute 1+1").await.unwrap();
        assert_eq!(answer, "The answer is 42");
        assert_eq!(calls.load(Ordering::Relaxed), 1);

        // Second-round request: the tool result was fed back (the history
        // contains the ToolResult).
        let requests = fake.requests();
        assert_eq!(requests.len(), 2);
        assert!(requests[1].messages.iter().any(|m| matches!(
            m,
            Message::ToolResult { id, content } if id == "c1" && content == "42"
        )));
    }

    #[tokio::test]
    async fn multiple_tools_same_round() {
        let (t1, calls1) = FakeTool::new("t1", "one");
        let (t2, calls2) = FakeTool::new("t2", "two");
        let mut registry = ToolRegistry::new();
        registry.register(t1).register(t2);
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "t1", "{}"), call("c2", "t2", "{}")],
            },
            FakeReply::Text("done".into()),
        ]);
        let mut agent = agent_with_registry(fake.clone(), registry, AgentConfig::default());

        let answer = agent.run("Run them all").await.unwrap();
        assert_eq!(answer, "done");
        assert_eq!(calls1.load(Ordering::Relaxed), 1);
        assert_eq!(calls2.load(Ordering::Relaxed), 1);

        // Two calls in the same round live in the same Assistant message;
        // results follow in order.
        let requests = fake.requests();
        let assistant = requests[1]
            .messages
            .iter()
            .find_map(|m| match m {
                Message::Assistant { tool_calls, .. } => Some(tool_calls),
                _ => None,
            })
            .expect("second round should contain Assistant");
        assert_eq!(assistant.len(), 2);

        let results: Vec<&str> = requests[1]
            .messages
            .iter()
            .filter_map(|m| match m {
                Message::ToolResult { content, .. } => Some(content.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(results, vec!["one", "two"]);
    }

    /// Empty Assistant messages are not recorded: a purely empty reply is
    /// not recorded, and the request history only contains the user.
    #[tokio::test]
    async fn empty_assistant_not_recorded() {
        let fake = SharedFake::new([FakeReply::Text("".into())]);
        let mut agent = agent(fake.clone(), "");

        let answer = agent.run("hi").await.unwrap();
        assert_eq!(answer, "");

        let requests = fake.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].messages.len(), 1); // user only
    }

    /// Default Memory is a bounded window: over-budget conversations
    /// auto-trim the oldest rounds and stop growing unboundedly; short
    /// conversations don't trigger trimming.
    #[tokio::test]
    async fn default_memory_is_bounded_window() {
        let fake = SharedFake::new([FakeReply::Text("hi".into())]);
        let mut agent = agent(fake, "");

        // First round: 600k ASCII chars ≈ 150k tokens (CharTokenCounter,
        // 4 chars / 1 token) > the default 128k budget — triggers trimming;
        // the second round is small and should be fully kept.
        agent
            .memory
            .record(Message::user("x".repeat(600_000)))
            .await
            .unwrap();
        agent
            .memory
            .record(Message::assistant("first-round reply"))
            .await
            .unwrap();
        agent
            .memory
            .record(Message::user("second round"))
            .await
            .unwrap();
        agent
            .memory
            .record(Message::assistant("second-round reply"))
            .await
            .unwrap();

        let ctx = agent.memory.context().await.unwrap();
        assert_eq!(
            ctx,
            vec![
                Message::user("second round"),
                Message::assistant("second-round reply")
            ]
        );
    }

    /// Round limit: the model keeps requesting tools → Err(TooManyToolRounds);
    /// no further conversations are started.
    #[tokio::test]
    async fn too_many_tool_rounds() {
        let (calc, _calls) = FakeTool::new("calc", "42");
        let mut registry = ToolRegistry::new();
        registry.register(calc);
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "calc", "{}")],
            },
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c2", "calc", "{}")],
            },
        ]);
        let mut agent = agent_with_registry(
            fake.clone(),
            registry,
            AgentConfig {
                max_tool_rounds: 2,
                ..Default::default()
            },
        );

        let err = agent.run("Keep computing").await.unwrap_err();
        assert!(matches!(err, AgentError::TooManyToolRounds(2)));
        assert_eq!(fake.requests().len(), 2); // only two conversation rounds were sent
    }

    /// system_prompt: assembled on every request, exactly one System at the
    /// front; not written to Memory.
    #[tokio::test]
    async fn system_prompt_assembled_every_request() {
        let fake = SharedFake::new([
            FakeReply::Text("Hello".into()),
            FakeReply::Text("Goodbye".into()),
        ]);
        let mut agent = agent(fake.clone(), "You are an assistant");

        agent.run("Are you there").await.unwrap();
        agent.run("Any more?").await.unwrap();

        let requests = fake.requests();
        assert_eq!(requests.len(), 2);
        // Each request: exactly one System and it's first; the second
        // request also carries the previous round's conversation history.
        for request in &requests {
            let systems = request
                .messages
                .iter()
                .filter(|m| matches!(m, Message::System(_)))
                .count();
            assert_eq!(systems, 1);
            assert_eq!(request.messages[0], Message::system("You are an assistant"));
        }
        assert_eq!(requests[1].messages.len(), 4); // System + previous round's user/assistant + this round's user
        assert_eq!(requests[1].messages[3], Message::user("Any more?"));
    }

    /// Macro: arms without a system prompt — no System message is
    /// assembled when system_prompt is omitted.
    #[tokio::test]
    async fn macro_arms_without_system_prompt() {
        // Bare arm: no tools, no system prompt.
        let fake = SharedFake::new([FakeReply::Text("hi".into())]);
        let mut agent = crate::react_agent!(fake.clone());
        assert_eq!(agent.run("hi").await.unwrap(), "hi");
        assert_eq!(fake.requests()[0].messages.len(), 1); // user only, no System

        // Tool-list arm: no system prompt.
        let (t1, calls1) = FakeTool::new("t1", "one");
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "t1", "{}")],
            },
            FakeReply::Text("done".into()),
        ]);
        let mut agent = crate::react_agent!(fake.clone(), [t1]);
        assert_eq!(agent.run("x").await.unwrap(), "done");
        assert_eq!(calls1.load(Ordering::Relaxed), 1);
        assert_eq!(fake.requests()[1].messages.len(), 3); // user + assistant + tool, no System

        // Registry arm: no system prompt.
        let (t2, _calls2) = FakeTool::new("t2", "two");
        let mut registry = ToolRegistry::new();
        registry.register(t2);
        let fake = SharedFake::new([FakeReply::Text("hi".into())]);
        let mut agent = crate::react_agent!(fake.clone(), registry);
        assert_eq!(agent.run("hi").await.unwrap(), "hi");
        assert_eq!(fake.requests()[0].messages.len(), 1); // no System
    }

    /// Macro, three arms: no tools / heterogeneous tool list
    /// (auto-registered) / existing registry.
    #[tokio::test]
    async fn macro_three_arms() {
        // No-tool arm: empty registry, answers directly.
        let fake = SharedFake::new([FakeReply::Text("hi".into())]);
        let mut agent = crate::react_agent!(fake.clone(), "");
        assert_eq!(agent.run("hi").await.unwrap(), "hi");
        assert_eq!(fake.requests()[0].messages.len(), 1);

        // Heterogeneous tool-list arm: two tools of different types are
        // auto-registered and actually executed.
        let (t1, calls1) = FakeTool::new("t1", "one");
        let (t2, calls2) = FakeTool::new("t2", "two");
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "t1", "{}")],
            },
            FakeReply::Text("done".into()),
        ]);
        let mut agent = crate::react_agent!(fake.clone(), [t1, t2], "");
        assert_eq!(agent.run("x").await.unwrap(), "done");
        assert_eq!(calls1.load(Ordering::Relaxed), 1);
        assert_eq!(calls2.load(Ordering::Relaxed), 0);

        // Registry arm: the existing registry is passed through as-is.
        let (t3, calls3) = FakeTool::new("t3", "three");
        let mut registry = ToolRegistry::new();
        registry.register(t3);
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "t3", "{}")],
            },
            FakeReply::Text("done".into()),
        ]);
        let mut agent = crate::react_agent!(fake, registry, "");
        assert_eq!(agent.run("x").await.unwrap(), "done");
        assert_eq!(calls3.load(Ordering::Relaxed), 1);
    }

    /// All six macro arms construct (isomorphic to the examples/
    /// react_agent.rs doc demo, ensuring the example doc code compiles).
    #[test]
    fn macro_all_arms_compile() {
        struct Echo;
        #[async_trait::async_trait]
        impl Tool for Echo {
            fn schema(&self) -> ToolSchema {
                ToolSchema {
                    name: "echo".into(),
                    description: "Echo".into(),
                    parameters: serde_json::json!({}),
                }
            }
            async fn call(
                &self,
                _arguments: serde_json::Value,
                _state: &SharedState,
            ) -> Result<String, ToolError> {
                Ok("echo".into())
            }
        }

        fn fake() -> SharedFake {
            SharedFake::new([FakeReply::Text("hi".into())])
        }

        let a1 = crate::react_agent!(fake()); // no tools, no system prompt
        let a2 = crate::react_agent!(fake(), "You are an assistant"); // no tools, with system prompt
        let a3 = crate::react_agent!(fake(), [Echo]); // tool list, no system prompt
        let a4 = crate::react_agent!(fake(), [Echo], "You are an assistant"); // tool list, with system prompt
        let mut registry = ToolRegistry::new();
        registry.register(Echo);
        let a5 = crate::react_agent!(fake(), registry.clone()); // existing registry, no system prompt
        let a6 = crate::react_agent!(fake(), registry, "You are an assistant"); // existing registry, with system prompt
        let _ = (a1, a2, a3, a4, a5, a6);
    }

    /// Empty system prompt: no System message is assembled.
    #[tokio::test]
    async fn empty_system_prompt_skips_system_message() {
        let fake = SharedFake::new([FakeReply::Text("hi".into())]);
        let mut agent = agent(fake.clone(), "");

        agent.run("hi").await.unwrap();

        let requests = fake.requests();
        assert_eq!(requests[0].messages.len(), 1);
        assert_eq!(requests[0].messages[0], Message::user("hi"));
    }

    /// Tool failures are not AgentError: the error text is fed back to the
    /// model and the loop continues.
    #[tokio::test]
    async fn tool_failure_returns_text_and_continues() {
        // Failing tool: always Err on execution.
        struct FailingTool;
        #[async_trait::async_trait]
        impl Tool for FailingTool {
            fn schema(&self) -> ToolSchema {
                ToolSchema {
                    name: "boom".into(),
                    description: "Tool that always fails".into(),
                    parameters: serde_json::json!({}),
                }
            }
            async fn call(
                &self,
                _arguments: serde_json::Value,
                _state: &SharedState,
            ) -> Result<String, ToolError> {
                Err(ToolError::Execution("internal error".into()))
            }
        }

        let mut registry = ToolRegistry::new();
        registry.register(FailingTool);
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "boom", "{}")],
            },
            FakeReply::Text("Got it".into()),
        ]);
        let mut agent = agent_with_registry(fake.clone(), registry, AgentConfig::default());

        let answer = agent.run("Trigger failure").await.unwrap();
        assert_eq!(answer, "Got it");

        // The error was turned into text and fed back (registry
        // semantics); the loop didn't stop.
        let requests = fake.requests();
        assert!(requests[1].messages.iter().any(|m| matches!(
            m,
            Message::ToolResult { content, .. } if content.contains("internal error")
        )));
    }

    // ---- Streaming-path error semantics ----

    /// Streaming: exceeding the round limit terminates with
    /// Err(TooManyToolRounds); no Done / third request.
    #[tokio::test]
    async fn stream_too_many_tool_rounds_terminates() {
        let (calc, _calls) = FakeTool::new("calc", "42");
        let mut registry = ToolRegistry::new();
        registry.register(calc);
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "calc", "{}")],
            },
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c2", "calc", "{}")],
            },
        ]);
        let mut agent = agent_with_registry(
            fake.clone(),
            registry,
            AgentConfig {
                max_tool_rounds: 2,
                ..Default::default()
            },
        );

        let mut stream = agent.run_stream("Keep computing").await.unwrap();
        let chunks: Vec<Result<MessageChunk, AgentError>> = stream.by_ref().collect().await;
        // Both rounds fully execute (tool rounds run before the limit
        // check: ToolCall+ToolResult ×2), and the top-of-round check of the
        // third round hits the limit: Err terminates, no Done, only two
        // conversation rounds sent.
        assert_eq!(chunks.len(), 5);
        assert!(matches!(
            chunks.last(),
            Some(Err(AgentError::TooManyToolRounds(2)))
        ));
        assert_eq!(fake.requests().len(), 2);
    }

    /// Streaming: entry record(user) failure → direct Err, no stream
    /// produced.
    #[tokio::test]
    async fn stream_entry_record_user_failure_returns_error() {
        struct FailingUserMemory;
        #[async_trait::async_trait]
        impl Memory for FailingUserMemory {
            async fn record(&mut self, _message: Message) -> Result<(), MemoryError> {
                Err(MemoryError::Storage("disk full".into()))
            }
            async fn context(&self) -> Result<Vec<Message>, MemoryError> {
                Ok(Vec::new())
            }
        }

        let fake = SharedFake::new([FakeReply::Text("hi".into())]);
        let mut agent = agent(fake.clone(), "").with_memory(FailingUserMemory);
        let err = match agent.run_stream("hi").await {
            Err(e) => e,
            Ok(_) => panic!("expected input recording failure to return Err directly"),
        };
        assert!(matches!(err, AgentError::Memory(MemoryError::Storage(_))));
    }

    /// Streaming: in-round context() failure → Err terminates the stream
    /// (no further chunks).
    #[tokio::test]
    async fn stream_context_failure_terminates_with_err() {
        struct FailingContextMemory;
        #[async_trait::async_trait]
        impl Memory for FailingContextMemory {
            async fn record(&mut self, _message: Message) -> Result<(), MemoryError> {
                Ok(())
            }
            async fn context(&self) -> Result<Vec<Message>, MemoryError> {
                Err(MemoryError::Storage("disk full".into()))
            }
        }

        let fake = SharedFake::new([FakeReply::Text("hi".into())]);
        let mut agent = agent(fake.clone(), "").with_memory(FailingContextMemory);
        let mut stream = agent.run_stream("hi").await.unwrap();
        let chunks: Vec<Result<MessageChunk, AgentError>> = stream.by_ref().collect().await;
        assert_eq!(chunks.len(), 1);
        assert!(matches!(
            chunks[0],
            Err(AgentError::Memory(MemoryError::Storage(_)))
        ));
    }

    /// Streaming: Assistant record failure → Err terminates the stream
    /// (Delta already produced, no Done).
    #[tokio::test]
    async fn stream_assistant_record_failure_terminates_with_err() {
        struct FailingAssistantMemory;
        #[async_trait::async_trait]
        impl Memory for FailingAssistantMemory {
            async fn record(&mut self, message: Message) -> Result<(), MemoryError> {
                if matches!(message, Message::Assistant { .. }) {
                    Err(MemoryError::Storage("disk full".into()))
                } else {
                    Ok(())
                }
            }
            async fn context(&self) -> Result<Vec<Message>, MemoryError> {
                Ok(Vec::new())
            }
        }

        let fake = SharedFake::new([FakeReply::Text("hi".into())]);
        let mut agent = agent(fake.clone(), "").with_memory(FailingAssistantMemory);
        let mut stream = agent.run_stream("hi").await.unwrap();
        let chunks: Vec<Result<MessageChunk, AgentError>> = stream.by_ref().collect().await;
        assert_eq!(chunks.len(), 2); // Delta + Err
        assert!(matches!(
            chunks.last(),
            Some(Err(AgentError::Memory(MemoryError::Storage(_))))
        ));
    }

    /// Streaming: tool failures turn into text fed back; the stream
    /// continues to Done (same semantics as run).
    #[tokio::test]
    async fn stream_tool_failure_returns_text_and_continues() {
        struct FailingTool;
        #[async_trait::async_trait]
        impl Tool for FailingTool {
            fn schema(&self) -> ToolSchema {
                ToolSchema {
                    name: "boom".into(),
                    description: "Tool that always fails".into(),
                    parameters: serde_json::json!({}),
                }
            }
            async fn call(
                &self,
                _arguments: serde_json::Value,
                _state: &SharedState,
            ) -> Result<String, ToolError> {
                Err(ToolError::Execution("internal error".into()))
            }
        }

        let mut registry = ToolRegistry::new();
        registry.register(FailingTool);
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "boom", "{}")],
            },
            FakeReply::Text("Got it".into()),
        ]);
        let mut agent = agent_with_registry(fake.clone(), registry, AgentConfig::default());

        let mut stream = agent.run_stream("Trigger failure").await.unwrap();
        let chunks: Vec<MessageChunk> = stream.by_ref().map(|e| e.unwrap()).collect().await;
        assert!(matches!(
            &chunks[1],
            MessageChunk::ToolResult { content, .. } if content.contains("internal error")
        ));
        // The stream continues: the model answers directly in the second
        // round and Done follows.
        assert_eq!(
            chunks.last(),
            Some(&MessageChunk::Done(RunSummary {
                rounds: 2,
                tool_calls: 1,
                usage: Usage::default(),
            }))
        );
    }

    /// Streaming: multiple tools in one round, chunks produced strictly in
    /// request order → result order, with correct summary counts.
    #[tokio::test]
    async fn stream_multiple_tools_same_round() {
        let (calc_a, _calls) = FakeTool::new("calc_a", "A");
        let (calc_b, _calls) = FakeTool::new("calc_b", "B");
        let mut registry = ToolRegistry::new();
        registry.register(calc_a).register(calc_b);
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "calc_a", "{}"), call("c2", "calc_b", "{}")],
            },
            FakeReply::Text("Done".into()),
        ]);
        let mut agent = agent_with_registry(fake.clone(), registry, AgentConfig::default());

        let mut stream = agent.run_stream("Compute").await.unwrap();
        let chunks: Vec<MessageChunk> = stream.by_ref().map(|e| e.unwrap()).collect().await;
        assert_eq!(
            chunks,
            vec![
                MessageChunk::ToolCall {
                    id: "c1".into(),
                    name: "calc_a".into(),
                    arguments: "{}".into()
                },
                MessageChunk::ToolCall {
                    id: "c2".into(),
                    name: "calc_b".into(),
                    arguments: "{}".into()
                },
                MessageChunk::ToolResult {
                    id: "c1".into(),
                    name: "calc_a".into(),
                    content: "A".into()
                },
                MessageChunk::ToolResult {
                    id: "c2".into(),
                    name: "calc_b".into(),
                    content: "B".into()
                },
                MessageChunk::Delta("Done".into()),
                MessageChunk::Done(RunSummary {
                    rounds: 2,
                    tool_calls: 2,
                    usage: Usage::default(),
                }),
            ]
        );
    }

    /// Error semantics are equivalent across paths: with the same failure
    /// script (round limit), run and run_stream agree on error type and
    /// request history.
    #[tokio::test]
    async fn run_and_stream_error_semantics_equivalent() {
        let script = |fake: &SharedFake| {
            let (calc, _calls) = FakeTool::new("calc", "42");
            let mut registry = ToolRegistry::new();
            registry.register(calc);
            agent_with_registry(
                fake.clone(),
                registry,
                AgentConfig {
                    max_tool_rounds: 1,
                    ..Default::default()
                },
            )
        };

        // Non-streaming: round-limit Err.
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "calc", "{}")],
            },
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c2", "calc", "{}")],
            },
        ]);
        let mut agent = script(&fake);
        let run_err = agent.run("Compute").await.unwrap_err();

        // Streaming: same script, terminates with Err.
        let fake2 = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "calc", "{}")],
            },
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c2", "calc", "{}")],
            },
        ]);
        let mut agent = script(&fake2);
        let mut stream = agent.run_stream("Compute").await.unwrap();
        let chunks: Vec<Result<MessageChunk, AgentError>> = stream.by_ref().collect().await;
        let stream_err = chunks.into_iter().find_map(|e| e.err());

        // Same error type, same request count.
        assert_eq!(run_err, AgentError::TooManyToolRounds(1));
        assert_eq!(stream_err, Some(AgentError::TooManyToolRounds(1)));
        assert_eq!(fake.requests().len(), fake2.requests().len());
    }

    /// run_id is unique across instances: two back-to-back instances have
    /// different ids on their first run.
    #[tokio::test]
    async fn run_id_differs_across_instances() {
        let fake_a = SharedFake::new([FakeReply::Text("hi".into())]);
        let fake_b = SharedFake::new([FakeReply::Text("hi".into())]);
        let agent_a = agent(fake_a.clone(), "");
        let agent_b = agent(fake_b.clone(), "");

        let id_a = agent_a.next_run_id();
        let id_b = agent_b.next_run_id();
        assert_ne!(
            id_a, id_b,
            "back-to-back instances must not collide on run_id"
        );
        // Same format as the trace tests: run-{ts}-{n}.
        assert!(id_a.starts_with("run-") && id_b.starts_with("run-"));
    }

    /// Non-streaming: in-round context() failure → Err(Memory) passes
    /// through (symmetric with streaming).
    #[tokio::test]
    async fn run_context_failure_returns_memory_error() {
        struct FailingContextMemory;
        #[async_trait::async_trait]
        impl Memory for FailingContextMemory {
            async fn record(&mut self, _message: Message) -> Result<(), MemoryError> {
                Ok(())
            }
            async fn context(&self) -> Result<Vec<Message>, MemoryError> {
                Err(MemoryError::Storage("disk full".into()))
            }
        }

        let fake = SharedFake::new([FakeReply::Text("hi".into())]);
        let mut agent = agent(fake.clone(), "").with_memory(FailingContextMemory);
        let err = agent.run("hi").await.unwrap_err();
        assert!(matches!(err, AgentError::Memory(MemoryError::Storage(_))));
        // No conversation was started.
        assert!(fake.requests().is_empty());
    }

    /// Empty-stream semantics: end-of-round wrap-up with no answer
    /// recorded, and a normal Done.
    ///
    /// Note: reporting `Err` for truncated / empty streams is the Provider
    /// implementation's job; this test covers the Agent's behavior when the
    /// Provider delivers a legitimately empty stream.
    #[tokio::test]
    async fn stream_empty_provider_stream_yields_empty_answer() {
        struct EmptyStreamProvider;
        #[async_trait::async_trait]
        impl Provider for EmptyStreamProvider {
            async fn chat(&self, _r: ChatRequest) -> Result<ChatResponse, ProviderError> {
                unreachable!("this test uses streaming only")
            }
            async fn stream_chat(
                &self,
                _r: ChatRequest,
            ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError>
            {
                Ok(Box::pin(futures::stream::empty()))
            }
        }

        let mut agent = ReActAgent::new(EmptyStreamProvider, ToolRegistry::new(), "");
        let chunks: Vec<Result<MessageChunk, AgentError>> = {
            let mut stream = agent.run_stream("hi").await.unwrap();
            stream.by_ref().collect().await
        };
        // Empty answer: not recorded + normal Done (rounds counts 1).
        assert_eq!(
            chunks,
            vec![Ok(MessageChunk::Done(RunSummary {
                rounds: 1,
                tool_calls: 0,
                usage: Usage::default(),
            }))]
        );
        // The memory only holds the user input; no empty Assistant was
        // recorded.
        assert_eq!(
            agent.memory.context().await.unwrap(),
            vec![Message::user("hi")]
        );
    }

    /// Streaming: event order Delta → ToolCall → ToolResult → Done;
    /// reasoning never enters MessageChunk (only recorded with the
    /// message).
    #[tokio::test]
    async fn stream_event_order() {
        let (calc, _calls) = FakeTool::new("calc", "42");
        let mut registry = ToolRegistry::new();
        registry.register(calc);
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "Thinking: ".into(),
                calls: vec![call("c1", "calc", "{}")],
            },
            FakeReply::TextWithReasoning {
                content: "The answer is 42".into(),
                reasoning: "Reasoning steps".into(),
            },
        ]);
        let mut agent = agent_with_registry(fake.clone(), registry, AgentConfig::default());

        let mut stream = agent.run_stream("Compute").await.unwrap();
        let events: Vec<MessageChunk> = stream.by_ref().map(|e| e.unwrap()).collect().await;
        assert_eq!(
            events,
            vec![
                MessageChunk::Delta("Thinking: ".into()),
                MessageChunk::ToolCall {
                    id: "c1".into(),
                    name: "calc".into(),
                    arguments: "{}".into()
                },
                MessageChunk::ToolResult {
                    id: "c1".into(),
                    name: "calc".into(),
                    content: "42".into()
                },
                MessageChunk::Delta("The answer is 42".into()),
                // Two conversation rounds (tool round + direct-answer
                // round), one tool execution; the fake injects no usage, so
                // it's always zero.
                MessageChunk::Done(RunSummary {
                    rounds: 2,
                    tool_calls: 1,
                    usage: Usage::default(),
                }),
            ]
        );
    }

    /// Pure tool round: no text, no Delta dispatched; the tool call still
    /// executes and feeds back.
    #[tokio::test]
    async fn stream_pure_tool_round_no_delta() {
        let (calc, _calls) = FakeTool::new("calc", "42");
        let mut registry = ToolRegistry::new();
        registry.register(calc);
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "calc", "{}")],
            },
            FakeReply::Text("42".into()),
        ]);
        let mut agent = agent_with_registry(fake.clone(), registry, AgentConfig::default());

        let mut stream = agent.run_stream("Compute").await.unwrap();
        let events: Vec<MessageChunk> = stream.by_ref().map(|e| e.unwrap()).collect().await;
        assert_eq!(
            events,
            vec![
                MessageChunk::ToolCall {
                    id: "c1".into(),
                    name: "calc".into(),
                    arguments: "{}".into()
                },
                MessageChunk::ToolResult {
                    id: "c1".into(),
                    name: "calc".into(),
                    content: "42".into()
                },
                MessageChunk::Delta("42".into()),
                MessageChunk::Done(RunSummary {
                    rounds: 2,
                    tool_calls: 1,
                    usage: Usage::default(),
                }),
            ]
        );
    }

    /// The Done summary carries real token usage (injected via
    /// FakeReply::WithUsage, summed across rounds), with rounds and tool
    /// counts tallied by execution.
    #[tokio::test]
    async fn stream_done_summary_accumulates_usage() {
        let (calc, _calls) = FakeTool::new("calc", "42");
        let mut registry = ToolRegistry::new();
        registry.register(calc);
        let fake = SharedFake::new([
            FakeReply::WithUsage {
                reply: Box::new(FakeReply::ToolCalls {
                    content: "".into(),
                    calls: vec![call("c1", "calc", "{}")],
                }),
                usage: Usage::new(10, 2),
            },
            FakeReply::text_with_usage("42", Usage::new(20, 5)),
        ]);
        let mut agent = agent_with_registry(fake.clone(), registry, AgentConfig::default());

        let mut stream = agent.run_stream("Compute").await.unwrap();
        let events: Vec<MessageChunk> = stream.by_ref().map(|e| e.unwrap()).collect().await;

        // Two conversation rounds (tool round + direct-answer round), one
        // tool execution; usage accumulates per round.
        assert_eq!(
            events.last(),
            Some(&MessageChunk::Done(RunSummary {
                rounds: 2,
                tool_calls: 1,
                usage: Usage::new(30, 7), // prompt 10+20,completion 2+5
            }))
        );
    }

    /// run and run_stream with the same script share semantics: identical
    /// final text, identical request history (identical records).
    #[tokio::test]
    async fn run_and_stream_same_semantics() {
        let script = [
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "calc", "{}")],
            },
            FakeReply::Text("42".into()),
        ];
        let (calc, _calls) = FakeTool::new("calc", "42");

        // Non-streaming path.
        let mut registry = ToolRegistry::new();
        registry.register(calc.clone());
        let fake1 = SharedFake::new(script.clone());
        let mut agent1 = agent_with_registry(fake1.clone(), registry, AgentConfig::default());
        let answer1 = agent1.run("Compute").await.unwrap();

        // Streaming path.
        let mut registry = ToolRegistry::new();
        registry.register(calc);
        let fake2 = SharedFake::new(script);
        let mut agent2 = agent_with_registry(fake2.clone(), registry, AgentConfig::default());
        let events: Vec<MessageChunk> = agent2
            .run_stream("Compute")
            .await
            .unwrap()
            .map(|e| e.unwrap())
            .collect()
            .await;
        // Final answer = concatenation of the Deltas, verbatim.
        let answer2: String = events
            .iter()
            .filter_map(|e| match e {
                MessageChunk::Delta(d) => Some(d.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(answer1, answer2);
        assert_eq!(answer1, "42");

        // Both paths send the same message sequence to the model (same
        // recorded history).
        assert_eq!(fake1.requests(), fake2.requests());
    }

    /// In-stream error: after an Err event the stream terminates; no Done.
    ///
    /// Note: this path (an error mid-event-stream after establishment)
    /// can't be built with FakeReply::Error — its semantics are "this round
    /// failed; the method returns Err directly"; here a custom Provider
    /// produces a stream that errors midway.
    #[tokio::test]
    async fn stream_error_terminates_without_done() {
        struct FailInStream;
        #[async_trait::async_trait]
        impl Provider for FailInStream {
            async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, ProviderError> {
                unreachable!("this test uses streaming path only")
            }
            async fn stream_chat(
                &self,
                _request: ChatRequest,
            ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError>
            {
                Ok(Box::pin(futures::stream::iter(vec![
                    Ok(StreamEvent::Delta("hi".into())),
                    Err(ProviderError::Api {
                        status: 0,
                        message: "boom".into(),
                    }),
                ])))
            }
        }

        let mut agent = ReActAgent::new(FailInStream, ToolRegistry::new(), "");

        let mut stream = agent.run_stream("two").await.unwrap();
        assert_eq!(
            stream.next().await.unwrap().unwrap(),
            MessageChunk::Delta("hi".into())
        );
        assert!(matches!(
            stream.next().await.unwrap(),
            Err(AgentError::Provider(ProviderError::Api { message: m, .. })) if m == "boom"
        ));
        assert!(stream.next().await.is_none()); // terminated, no Done
    }

    /// Script exhausted (one extra round): Err(ProviderError::Api("script
    /// exhausted")).
    #[tokio::test]
    async fn script_exhausted_fails_explicitly() {
        let fake = SharedFake::new([FakeReply::Text("hi".into())]);
        let mut agent = agent(fake, "");
        agent.run("one").await.unwrap();

        let err = agent.run("two").await.unwrap_err();
        assert!(
            matches!(err, AgentError::Provider(ProviderError::Api { message: m, .. }) if m.contains("exhausted"))
        );
    }

    /// Shared state flows: the loop injects the agent-held state into every
    /// tool call, and tools read/write the same instance (the application
    /// side can also read/write across runs).
    #[tokio::test]
    async fn shared_state_flows_to_tools() {
        struct CounterTool;
        #[async_trait::async_trait]
        impl Tool for CounterTool {
            fn schema(&self) -> ToolSchema {
                ToolSchema {
                    name: "counter".into(),
                    description: "Count".into(),
                    parameters: serde_json::json!({}),
                }
            }
            async fn call(
                &self,
                _arguments: serde_json::Value,
                state: &SharedState,
            ) -> Result<String, ToolError> {
                state.with_mut::<usize>(|n| *n += 1);
                Ok(format!("count={}", state.get::<usize>().unwrap_or(0)))
            }
        }

        let mut registry = ToolRegistry::new();
        registry.register(CounterTool);
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "counter", "{}"), call("c2", "counter", "{}")],
            },
            FakeReply::Text("done".into()),
        ]);
        let state = SharedState::new();
        state.insert(0usize);
        let mut agent = ReActAgent::new(fake, registry, "").with_state(state.clone());

        agent.run("Count").await.unwrap();

        // Two calls in the same round share one instance: the count
        // accumulates, and the application side reads the final cross-run
        // value.
        assert_eq!(state.get::<usize>(), Some(2));
    }

    /// Memory errors pass through (custom Memory injected via
    /// with_memory).
    #[tokio::test]
    async fn memory_error_passthrough() {
        struct FailingMemory;
        #[async_trait::async_trait]
        impl Memory for FailingMemory {
            async fn record(&mut self, _message: Message) -> Result<(), MemoryError> {
                Err(MemoryError::Storage("disk full".into()))
            }
            async fn context(&self) -> Result<Vec<Message>, MemoryError> {
                Ok(Vec::new())
            }
        }

        let mut agent = ReActAgent::new(
            FakeProvider::new([FakeReply::Text("hi".into())]),
            ToolRegistry::new(),
            "",
        )
        .with_memory(FailingMemory);
        let err = agent.run("hi").await.unwrap_err();
        assert!(matches!(err, AgentError::Memory(MemoryError::Storage(_))));
    }

    /// Streaming: tool-result record failure → an Err item terminates the
    /// stream; no further chunks.
    /// Errors are produced as Err items and terminate the stream — the
    /// break must exit the whole round loop, otherwise the stream would
    /// keep producing the next round after the Err (the memory would lack
    /// the ToolResult, leaving the next round's message sequence
    /// incomplete).
    #[tokio::test]
    async fn stream_tool_result_record_failure_terminates_stream() {
        struct FailingToolResultMemory;
        #[async_trait::async_trait]
        impl Memory for FailingToolResultMemory {
            async fn record(&mut self, message: Message) -> Result<(), MemoryError> {
                if matches!(message, Message::ToolResult { .. }) {
                    Err(MemoryError::Storage("disk full".into()))
                } else {
                    Ok(())
                }
            }
            async fn context(&self) -> Result<Vec<Message>, MemoryError> {
                Ok(Vec::new())
            }
        }

        let (calc, _calls) = FakeTool::new("calc", "42");
        let mut registry = ToolRegistry::new();
        registry.register(calc);
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "calc", "{}")],
            },
            FakeReply::Text("42".into()),
        ]);
        let mut agent = agent_with_registry(fake.clone(), registry, AgentConfig::default())
            .with_memory(FailingToolResultMemory);

        let mut stream = agent.run_stream("Compute").await.unwrap();
        let chunks: Vec<Result<MessageChunk, AgentError>> = stream.by_ref().collect().await;
        // ToolCall → ToolResult → Err (terminates): nothing after the Err.
        assert_eq!(chunks.len(), 3);
        assert!(matches!(
            chunks[2],
            Err(AgentError::Memory(MemoryError::Storage(_)))
        ));
    }

    /// AgentConfig.options passes through: every round's ChatRequest
    /// carries the configured model parameters.
    #[tokio::test]
    async fn config_options_forwarded_to_chat_request() {
        use crate::provider::ModelOptions;

        let fake = SharedFake::new([FakeReply::Text("hi".into())]);
        let mut agent = agent(fake.clone(), "").with_config(AgentConfig {
            max_tool_rounds: 10,
            options: ModelOptions {
                temperature: Some(0.2),
                max_tokens: Some(128),
                extra: Default::default(),
                structured: None,
            },
            ..Default::default()
        });
        agent.run("hi").await.unwrap();

        let req = &fake.requests()[0];
        assert_eq!(req.options.temperature, Some(0.2));
        assert_eq!(req.options.max_tokens, Some(128));
    }

    // ---- Cancellation: run_cancellable / run_stream_cancellable ----

    /// Pre-round cancellation: with an already-cancelled token,
    /// run_cancellable returns Cancelled immediately, having started no
    /// conversation; the user message is already recorded (kept, consistent
    /// with the main path).
    #[tokio::test]
    async fn cancelled_before_run_returns_cancelled() {
        let fake = SharedFake::new([FakeReply::Text("hi".into())]);
        let mut agent = agent(fake.clone(), "");
        let token = CancellationToken::new();
        token.cancel();

        let err = agent.run_cancellable("hi", &token).await.unwrap_err();
        assert!(matches!(err, AgentError::Cancelled));
        assert_eq!(fake.requests().len(), 0); // no conversation started
        assert_eq!(agent.memory.context().await.unwrap().len(), 1); // user only
    }

    /// Pre-cancelled token: both paths report the same rounds in RunEnded
    /// (incremented at the start of the round, both report 1).
    #[tokio::test]
    async fn pre_cancelled_token_rounds_consistent_across_paths() {
        let fake = SharedFake::new([FakeReply::Text("hi".into())]);
        let token = CancellationToken::new();
        token.cancel();

        // Non-streaming path.
        let (mut run_agent, mut rx) = attach_channel(agent(fake.clone(), ""));
        let err = run_agent.run_cancellable("hi", &token).await.unwrap_err();
        assert!(matches!(err, AgentError::Cancelled));
        drop(run_agent);
        let events = drain(&mut rx).await;
        let rounds_run = match react_event(&**events.last().unwrap()) {
            ReActEvent::RunEnded { summary, error, .. } => {
                assert_eq!(error, &Some(AgentError::Cancelled));
                summary.rounds
            }
            _ => panic!("expected RunEnded"),
        };

        // Streaming path.
        let (mut stream_agent, mut rx) = attach_channel(agent(fake, ""));
        let mut stream = stream_agent
            .run_stream_cancellable("hi", &token)
            .await
            .unwrap();
        let chunks: Vec<MessageChunk> = stream.by_ref().map(|e| e.unwrap()).collect().await;
        assert!(chunks.contains(&MessageChunk::Cancelled));
        drop(stream);
        drop(stream_agent);
        let events = drain(&mut rx).await;
        let rounds_stream = match react_event(&**events.last().unwrap()) {
            ReActEvent::RunEnded { summary, error, .. } => {
                assert_eq!(error, &Some(AgentError::Cancelled));
                summary.rounds
            }
            _ => panic!("expected RunEnded"),
        };

        assert_eq!(rounds_run, rounds_stream);
        assert_eq!(rounds_run, 1);
    }

    /// Immediate cancellation mid-conversation: cancelling while chat is
    /// pending makes run return Cancelled immediately (dropping the
    /// in-flight request); nothing is recorded for this round, leaving no
    /// orphan Assistant in the memory.
    #[tokio::test]
    async fn cancel_during_chat_drops_inflight() {
        struct PendingProvider;
        #[async_trait::async_trait]
        impl Provider for PendingProvider {
            async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, ProviderError> {
                std::future::pending().await // never completes: simulates a slow LLM
            }
            async fn stream_chat(
                &self,
                _request: ChatRequest,
            ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError>
            {
                unreachable!("this test uses non-streaming path only")
            }
        }

        let mut agent = ReActAgent::new(PendingProvider, ToolRegistry::new(), "");
        let token = CancellationToken::new();
        let result = tokio::select! {
            r = agent.run_cancellable("hi", &token) => r,
            _ = async {
                tokio::time::sleep(Duration::from_millis(20)).await;
                token.cancel();
                std::future::pending::<()>().await; // stay pending so the run branch wins
            } => unreachable!("cancellation branch only sends a signal"),
        };
        assert!(matches!(result, Err(AgentError::Cancelled)));
        assert_eq!(agent.memory.context().await.unwrap().len(), 1); // user only
    }

    /// Non-cancellation error propagation: when the Provider returns a
    /// Timeout, both run / run_stream return AgentError::Provider(Timeout),
    /// and the memory holds no orphan messages (user only, no half-recorded
    /// residue).
    #[tokio::test]
    async fn provider_error_propagates_from_both_paths() {
        struct FailProvider(ProviderError);
        #[async_trait::async_trait]
        impl Provider for FailProvider {
            async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, ProviderError> {
                Err(self.0.clone())
            }
            async fn stream_chat(
                &self,
                _request: ChatRequest,
            ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError>
            {
                Err(self.0.clone())
            }
        }

        let err = ProviderError::Timeout(TimeoutStage::Request);
        // Non-streaming: the error is returned directly.
        let mut agent = ReActAgent::new(FailProvider(err.clone()), ToolRegistry::new(), "");
        let got = agent.run("hi").await.unwrap_err();
        assert!(matches!(
            got,
            AgentError::Provider(ProviderError::Timeout(_))
        ));
        assert_eq!(agent.memory.context().await.unwrap().len(), 1); // user only

        // Streaming: the error terminates the stream as an in-stream Err
        // item; nothing follows (no Done).
        let mut agent = ReActAgent::new(FailProvider(err.clone()), ToolRegistry::new(), "");
        let mut stream = agent.run_stream("hi").await.unwrap();
        let item = stream.next().await.unwrap().unwrap_err();
        assert!(matches!(
            item,
            AgentError::Provider(ProviderError::Timeout(_))
        ));
        assert!(stream.next().await.is_none());
        drop(stream); // release the borrow of agent before inspecting the memory
        assert_eq!(agent.memory.context().await.unwrap().len(), 1);
    }

    /// Structured output: an answer conforming to the schema is returned
    /// directly (validation passes in a single round).
    #[tokio::test]
    async fn structured_output_valid_answer_passes() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        });
        let fake = SharedFake::new([FakeReply::Text(r#"{"city":"Beijing"}"#.into())]);
        let mut agent = agent(fake.clone(), "").with_structured_output(schema);
        let answer = agent.run("Beijing weather").await.unwrap();
        assert_eq!(answer, r#"{"city":"Beijing"}"#);
        assert_eq!(fake.requests().len(), 1); // passed in one round, no retry
    }

    /// Structured output: invalid JSON is first fed back to the model for
    /// retry; the corrected second round passes.
    #[tokio::test]
    async fn structured_output_retries_after_invalid_answer() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        });
        let fake = SharedFake::new([
            FakeReply::Text("Not JSON".into()),
            FakeReply::Text(r#"{"city":"Beijing"}"#.into()),
        ]);
        let mut agent = agent(fake.clone(), "").with_structured_output(schema);
        let answer = agent.run("Beijing weather").await.unwrap();
        assert_eq!(answer, r#"{"city":"Beijing"}"#);
        assert_eq!(fake.requests().len(), 2); // round one fails, round two passes after feedback
        // The validation-failure feedback was recorded (a User message);
        // the memory contains the retry round.
        let context = agent.memory.context().await.unwrap();
        assert!(context.iter().any(|m| matches!(
            m,
            Message::User(blocks) if blocks.iter().any(|b| matches!(b, ContentBlock::Text(t) if t.contains("JSON schema validation")))
        )));
    }

    /// Structured output: validation retries have an independent budget
    /// (`max_structured_retries`); exhausting it without success fails the
    /// run — the tool-round limit is not consumed, and no "tool round limit
    /// exceeded" is reported.
    #[tokio::test]
    async fn structured_output_exhausts_retry_budget() {
        let schema = serde_json::json!({ "type": "object" });
        let fake = SharedFake::new([
            FakeReply::Text("bad1".into()),
            FakeReply::Text("bad2".into()),
            FakeReply::Text("bad3".into()),
            FakeReply::Text("bad4".into()),
        ]);
        // Even a tool-round limit of 1 doesn't affect structured retries
        // (independent budget).
        let mut agent = agent(fake.clone(), "").with_config(AgentConfig {
            max_tool_rounds: 1,
            max_structured_retries: 3,
            ..Default::default()
        });
        agent = agent.with_structured_output(schema);
        let err = agent.run("hi").await.unwrap_err();
        assert!(matches!(err, AgentError::StructuredRetriesExhausted(3)));
        assert_eq!(fake.requests().len(), 4); // 4 attempts: the 4th failure hits the limit
    }

    /// Typed run: the schema is auto-generated from the type and injected
    /// into the request; after validation passes, the answer deserializes
    /// into the target type.
    #[tokio::test]
    async fn typed_output_parses_valid_answer() {
        #[derive(Debug, Deserialize, JsonSchema)]
        struct Weather {
            city: String,
        }
        let fake = SharedFake::new([FakeReply::Text(r#"{"city":"Beijing"}"#.into())]);
        let mut agent = agent(fake.clone(), "");
        let weather: Weather = agent.run_typed("Beijing weather").await.unwrap();
        assert_eq!(weather.city, "Beijing");
        assert_eq!(fake.requests().len(), 1); // passed in one round, no retry
        // This run's generated schema was injected into the request (the
        // config is untouched — no pollution of later text runs).
        assert!(fake.requests()[0].options.structured.is_some());
        assert!(agent.config.options.structured.is_none());
    }

    /// Typed run: an invalid answer is fed back for retry first; the
    /// corrected second round deserializes successfully.
    #[tokio::test]
    async fn typed_output_retries_then_parses() {
        #[derive(Debug, Deserialize, JsonSchema)]
        struct Weather {
            city: String,
        }
        let fake = SharedFake::new([
            FakeReply::Text("Not JSON".into()),
            FakeReply::Text(r#"{"city":"Beijing"}"#.into()),
        ]);
        let mut agent = agent(fake.clone(), "");
        let weather: Weather = agent.run_typed("Beijing weather").await.unwrap();
        assert_eq!(weather.city, "Beijing");
        assert_eq!(fake.requests().len(), 2);
    }

    /// Typed run: schema disagrees with the serde type (the custom schema
    /// says string while the type expects number) → validation passes but
    /// deserialization fails → StructuredParse (the schemars-generated
    /// path agrees with serde by default; conflicts only come from
    /// user-custom schemas).
    #[tokio::test]
    async fn typed_output_parse_failure_on_schema_mismatch() {
        fn string_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
            serde_json::from_value(serde_json::json!({ "type": "string" })).unwrap()
        }
        #[derive(Debug, Deserialize, JsonSchema)]
        #[allow(dead_code)] // error-path test: the field serves deserialization validation, never read
        struct Weather {
            #[schemars(schema_with = "string_schema")]
            temperature: i32,
        }
        let fake = SharedFake::new([FakeReply::Text(r#"{"temperature":"30"}"#.into())]);
        let mut agent = agent(fake.clone(), "");
        let err: AgentError = agent
            .run_typed::<Weather>("Beijing weather")
            .await
            .unwrap_err();
        assert!(matches!(err, AgentError::StructuredParse(_)));
    }

    /// TypedAgent interface: code with the generic bound `A: TypedAgent`
    /// can call run_typed on any implementation (independent of the
    /// concrete type; Box<dyn Agent> is unaffected).
    #[tokio::test]
    async fn typed_agent_trait_generic_call() {
        #[derive(Debug, Deserialize, JsonSchema)]
        struct Weather {
            city: String,
        }
        async fn typed_run<A: TypedAgent>(
            agent: &mut A,
            input: &str,
        ) -> Result<Weather, AgentError> {
            agent.run_typed(input).await
        }

        let fake = SharedFake::new([FakeReply::Text(r#"{"city":"Beijing"}"#.into())]);
        let mut agent = agent(fake.clone(), "");
        let weather = typed_run(&mut agent, "Beijing weather").await.unwrap();
        assert_eq!(weather.city, "Beijing");
    }

    /// Typed and text runs can be mixed: run_typed yields typed results,
    /// Agent::run yields text (no structured constraint; free text is
    /// unaffected by this run's schema).
    #[tokio::test]
    async fn typed_output_agent_trait_run_returns_text() {
        #[derive(Debug, Deserialize, JsonSchema)]
        struct Weather {
            city: String,
        }
        let fake = SharedFake::new([
            FakeReply::Text(r#"{"city":"Beijing"}"#.into()),
            FakeReply::Text("Hello".into()),
        ]);
        let mut agent = agent(fake.clone(), "");
        let weather: Weather = agent.run_typed("Beijing weather").await.unwrap();
        assert_eq!(weather.city, "Beijing");
        // Later text run: unaffected by run_typed's schema (no structured
        // constraint).
        let text = Agent::run(&mut agent, "Say hi").await.unwrap();
        assert_eq!(text, "Hello");
    }

    /// Streaming: structured validation retries have an independent budget;
    /// exceeding it terminates with an in-stream error (same semantics as
    /// run).
    #[tokio::test]
    async fn structured_output_stream_exhausts_retry_budget() {
        let schema = serde_json::json!({ "type": "object" });
        let fake = SharedFake::new([
            FakeReply::Text("bad1".into()),
            FakeReply::Text("bad2".into()),
        ]);
        let mut agent = agent(fake.clone(), "").with_config(AgentConfig {
            max_structured_retries: 1,
            ..Default::default()
        });
        agent = agent.with_structured_output(schema);
        let mut stream = agent.run_stream("hi").await.unwrap();
        let mut saw_err = false;
        while let Some(item) = stream.next().await {
            if let Err(e) = item {
                assert!(matches!(e, AgentError::StructuredRetriesExhausted(1)));
                saw_err = true;
                break;
            }
        }
        assert!(saw_err);
    }

    /// Tool-round atomicity: cancellation during a tool round doesn't stop
    /// the tool from finishing and recording its result (no interruption,
    /// no half-recorded residue); cancellation takes effect before the next
    /// round's conversation.
    #[tokio::test]
    async fn tool_round_atomic_under_cancel() {
        // Slow tool: 100ms execution, counts calls.
        struct SlowTool {
            calls: Arc<AtomicUsize>,
        }
        #[async_trait::async_trait]
        impl Tool for SlowTool {
            fn schema(&self) -> ToolSchema {
                ToolSchema {
                    name: "slow".into(),
                    description: "Slow tool".into(),
                    parameters: serde_json::json!({}),
                }
            }
            async fn call(
                &self,
                _arguments: serde_json::Value,
                _state: &SharedState,
            ) -> Result<String, ToolError> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok("42".into())
            }
        }

        let calls = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        registry.register(SlowTool {
            calls: calls.clone(),
        });
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "slow", "{}")],
            },
            FakeReply::Text("42".into()),
        ]);
        let mut agent = agent_with_registry(fake.clone(), registry, AgentConfig::default());
        let token = CancellationToken::new();

        // Cancel at 50ms: right in the middle of the tool round (tool takes
        // 100ms).
        let result = tokio::select! {
            r = agent.run_cancellable("Compute", &token) => r,
            _ = async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                token.cancel();
                std::future::pending::<()>().await; // stay pending so the run branch wins
            } => unreachable!("cancellation branch only sends a signal"),
        };
        assert!(matches!(result, Err(AgentError::Cancelled)));
        assert_eq!(calls.load(Ordering::Relaxed), 1); // tool finished (atomic)
        assert_eq!(fake.requests().len(), 1); // only one conversation round sent
        // Memory is complete: user + assistant (with tool requests) +
        // tool_result paired.
        let ctx = agent.memory.context().await.unwrap();
        assert_eq!(ctx.len(), 3);
        assert!(matches!(
            &ctx[1],
            Message::Assistant { tool_calls, .. } if tool_calls.len() == 1
        ));
        assert!(matches!(&ctx[2], Message::ToolResult { content, .. } if content == "42"));
    }

    /// Mid-stream cancellation: already-dispatched Deltas are kept, a
    /// Cancelled terminal event wraps up, no Done; nothing is recorded for
    /// this round (recording happens at the end of the round).
    #[tokio::test]
    async fn stream_cancel_mid_generation() {
        // Slow-stream Provider: 100ms between Deltas (wide gaps, so the
        // cancellation landing point is stable).
        struct SlowStreamProvider;
        #[async_trait::async_trait]
        impl Provider for SlowStreamProvider {
            async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, ProviderError> {
                unreachable!("this test uses streaming path only")
            }
            async fn stream_chat(
                &self,
                _request: ChatRequest,
            ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError>
            {
                Ok(Box::pin(async_stream::stream! {
                    yield Ok(StreamEvent::Delta("d0".into()));
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    yield Ok(StreamEvent::Delta("d1".into()));
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    yield Ok(StreamEvent::Done {
                        reason: FinishReason::Stop,
                        usage: None,
                    });
                }))
            }
        }

        let mut agent = ReActAgent::new(SlowStreamProvider, ToolRegistry::new(), "");
        let token = CancellationToken::new();
        let mut stream = agent.run_stream_cancellable("hi", &token).await.unwrap();

        // Cancel at 50ms: d0 already received (0ms), d1 still pending
        // (100ms) — cancellation lands in the gap.
        let mut first = Vec::new();
        tokio::select! {
            _ = async {
                while let Some(ev) = stream.next().await {
                    let ev = ev.unwrap();
                    if ev == MessageChunk::Cancelled {
                        break;
                    }
                    first.push(ev);
                }
            } => {}
            _ = async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                token.cancel();
            } => {}
        }
        assert_eq!(first, vec![MessageChunk::Delta("d0".into())]);

        // Keep consuming: after cancellation the stream terminates with
        // Cancelled, no Done.
        let rest: Vec<MessageChunk> = stream.by_ref().map(|e| e.unwrap()).collect().await;
        assert_eq!(rest, vec![MessageChunk::Cancelled]);
        drop(stream); // release the borrow of agent
        // Nothing recorded for this round: the memory only holds the user.
        assert_eq!(agent.memory.context().await.unwrap().len(), 1);
    }

    /// No cancellation residue: after a cancellation, a fresh token runs
    /// normally (each run has an independent cancellation source — the
    /// basis of multi-round conversations).
    #[tokio::test]
    async fn cancelled_then_fresh_token_run_works() {
        let fake = SharedFake::new([FakeReply::Text("hi".into())]);
        let mut agent = agent(fake, "");

        let t1 = CancellationToken::new();
        t1.cancel();
        assert!(matches!(
            agent.run_cancellable("one", &t1).await,
            Err(AgentError::Cancelled)
        ));

        let t2 = CancellationToken::new();
        assert_eq!(agent.run_cancellable("two", &t2).await.unwrap(), "hi");
    }

    // ---- Observation channel: the loop pushes process events once an
    // EventChannel is attached ----

    use crate::event_channel::{BroadcastEventChannel, EventReceiver};
    use crate::tool::RegistryError;

    /// Attach a broadcast channel and return a receiver; after the run,
    /// dropping the agent (the channel's last holder) closes the channel so
    /// the receiver can drain the final events.
    fn attach_channel(agent: ReActAgent) -> (ReActAgent, Box<dyn EventReceiver>) {
        let channel = BroadcastEventChannel::new(64);
        let rx = channel.subscribe();
        (agent.with_event_channel(channel), rx)
    }

    /// Drain: all events until the channel closes (None).
    async fn drain(rx: &mut Box<dyn EventReceiver>) -> Vec<Arc<dyn AgentEvent>> {
        let mut out = Vec::new();
        while let Some(ev) = rx.recv().await {
            out.push(ev);
        }
        out
    }

    fn names(events: &[Arc<dyn AgentEvent>]) -> Vec<&'static str> {
        events.iter().map(|e| e.name()).collect()
    }

    /// Downcast an event to ReActEvent (enum event set: one downcast, then
    /// an exhaustive match).
    fn react_event(ev: &dyn AgentEvent) -> &ReActEvent {
        ev.as_any()
            .downcast_ref::<ReActEvent>()
            .expect("test event should be ReActEvent")
    }

    /// Non-streaming run event sequence: RunStarted → ToolStarted →
    /// ToolCompleted(Ok) → RunEnded (summary, error=None); bystanders can
    /// observe a non-streaming run.
    #[tokio::test]
    async fn events_published_on_run() {
        let (calc, _calls) = FakeTool::new("calc", "42");
        let mut registry = ToolRegistry::new();
        registry.register(calc);
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "calc", r#"{"a":1}"#)],
            },
            FakeReply::Text("The answer is 42".into()),
        ]);
        let (mut agent, mut rx) =
            attach_channel(agent_with_registry(fake, registry, AgentConfig::default()));
        let answer = agent.run("Compute").await.unwrap();
        assert_eq!(answer, "The answer is 42");
        drop(agent);
        let events = drain(&mut rx).await;

        assert_eq!(
            names(&events),
            ["run.started", "tool.started", "tool.completed", "run.ended"]
        );
        // One downcast + exhaustive match (the consumption style of an
        // enum event set).
        match react_event(&*events[1]) {
            // ToolStarted: carries id / arguments, paired with
            // ToolCompleted.
            ReActEvent::ToolStarted {
                id,
                name,
                arguments,
            } => {
                assert_eq!(id, "c1");
                assert_eq!(name, "calc");
                assert_eq!(arguments, r#"{"a":1}"#);
            }
            _ => panic!("expected ToolStarted"),
        }
        match react_event(&*events[2]) {
            // ToolCompleted: Ok carries the result text.
            ReActEvent::ToolCompleted { result, .. } => {
                assert_eq!(result, &Ok("42".to_string()));
            }
            _ => panic!("expected ToolCompleted"),
        }
        match react_event(&*events[3]) {
            // RunEnded: normal end with error=None; the non-streaming
            // summary is real (2 rounds + 1 tool).
            ReActEvent::RunEnded { summary, error } => {
                assert_eq!(error, &None);
                assert_eq!(summary.rounds, 2);
                assert_eq!(summary.tool_calls, 1);
            }
            _ => panic!("expected RunEnded"),
        }
    }

    /// Tool failure: ToolCompleted's Err carries the registry
    /// classification (NotFound), whose Display is the error text fed back
    /// to the model.
    #[tokio::test]
    async fn events_carry_tool_failure() {
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "nope", "{}")],
            },
            FakeReply::Text("Got it".into()),
        ]);
        let (mut agent, mut rx) = attach_channel(agent(fake, ""));
        agent.run("Trigger").await.unwrap();
        drop(agent);

        let events = drain(&mut rx).await;
        let completed = events
            .iter()
            .find_map(|e| match react_event(&**e) {
                ReActEvent::ToolCompleted { result, .. } => Some(result),
                _ => None,
            })
            .expect("expected ToolCompleted event");
        assert!(matches!(completed, Err(RegistryError::NotFound(n)) if n == "nope"));
        assert_eq!(
            completed.as_ref().unwrap_err().to_string(),
            "tool not found: nope"
        );
    }

    /// Streaming path: Delta / Reasoning events push along with the stream;
    /// the RunEnded summary matches the pulling path's semantics.
    #[tokio::test]
    async fn stream_events_include_delta_and_reasoning() {
        let (calc, _calls) = FakeTool::new("calc", "42");
        let mut registry = ToolRegistry::new();
        registry.register(calc);
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "Thinking: ".into(),
                calls: vec![call("c1", "calc", "{}")],
            },
            FakeReply::TextWithReasoning {
                content: "The answer is 42".into(),
                reasoning: "Reasoning steps".into(),
            },
        ]);
        let (mut agent, mut rx) =
            attach_channel(agent_with_registry(fake, registry, AgentConfig::default()));

        // Pulling path as usual: text = concatenated Deltas.
        let answer: String = agent
            .run_stream("Compute")
            .await
            .unwrap()
            .map(|e| e.unwrap())
            .filter_map(|e| async move {
                match e {
                    MessageChunk::Delta(d) => Some(d),
                    _ => None,
                }
            })
            .collect()
            .await;
        assert_eq!(answer, "Thinking: The answer is 42");
        drop(agent);

        let events = drain(&mut rx).await;
        // Order matches the StreamEvent docs: content Deltas first,
        // Reasoning at the end of the round (OpenAiProvider and
        // FakeProvider share the order).
        assert_eq!(
            names(&events),
            [
                "run.started",
                "delta",
                "tool.started",
                "tool.completed",
                "delta",
                "reasoning",
                "run.ended",
            ]
        );
        let reasoning = events
            .iter()
            .find_map(|e| match react_event(&**e) {
                ReActEvent::Reasoning { text } => Some(text.as_str()),
                _ => None,
            })
            .expect("expected Reasoning event");
        assert_eq!(reasoning, "Reasoning steps");
        match react_event(&**events.last().unwrap()) {
            ReActEvent::RunEnded { summary, error } => {
                assert_eq!(error, &None);
                assert_eq!(summary.rounds, 2);
                assert_eq!(summary.tool_calls, 1);
            }
            _ => panic!("expected RunEnded"),
        }
    }

    /// Cancellation: RunEnded carries error=Some(Cancelled), so bystanders
    /// know the run's outcome.
    #[tokio::test]
    async fn cancelled_run_publishes_run_ended_with_error() {
        // Slow-stream Provider: 100ms between Deltas (stable cancellation landing point).
        struct SlowStreamProvider;
        #[async_trait::async_trait]
        impl Provider for SlowStreamProvider {
            async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, ProviderError> {
                unreachable!("this test uses streaming path only")
            }
            async fn stream_chat(
                &self,
                _request: ChatRequest,
            ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError>
            {
                Ok(Box::pin(async_stream::stream! {
                    yield Ok(StreamEvent::Delta("d0".into()));
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    yield Ok(StreamEvent::Delta("d1".into()));
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    yield Ok(StreamEvent::Done {
                        reason: FinishReason::Stop,
                        usage: None,
                    });
                }))
            }
        }

        let (mut agent, mut rx) =
            attach_channel(ReActAgent::new(SlowStreamProvider, ToolRegistry::new(), ""));
        let token = CancellationToken::new();
        let mut stream = agent.run_stream_cancellable("hi", &token).await.unwrap();

        // Cancel at 50ms: the select cancellation branch wins and the
        // consumption branch is dropped; after cancellation, keep consuming
        // the stream and wrap up with Cancelled (isomorphic to
        // stream_cancel_mid_generation).
        tokio::select! {
            _ = async {
                while let Some(ev) = stream.next().await {
                    let ev = ev.unwrap();
                    if ev == MessageChunk::Cancelled {
                        break;
                    }
                }
            } => {}
            _ = async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                token.cancel();
            } => {}
        }
        let rest: Vec<MessageChunk> = stream.by_ref().map(|e| e.unwrap()).collect().await;
        assert!(rest.contains(&MessageChunk::Cancelled));
        drop(stream);
        drop(agent);

        let events = drain(&mut rx).await;
        assert!(matches!(names(&events).as_slice(), [.., "run.ended"]));
        match react_event(&**events.last().unwrap()) {
            ReActEvent::RunEnded { error, .. } => {
                assert_eq!(error, &Some(AgentError::Cancelled));
            }
            _ => panic!("expected RunEnded"),
        }
    }

    // ---- Observability spans: structure and fields (asserted with a
    // collecting subscriber) ----

    use std::collections::HashMap;
    use std::sync::atomic::AtomicU64;
    use tracing::field::{Field, Visit};
    use tracing::subscriber::Subscriber;
    use tracing::{Event, Id, Level, Metadata};

    /// Creation info of one span (name / level / field values at
    /// creation).
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct SpanInfo {
        name: &'static str,
        level: Level,
        fields: Vec<(String, String)>,
    }

    /// The operation sequence received by the subscriber (enter / exit /
    /// close / field record).
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Op {
        Enter(String),
        Exit(String),
        Close(String),
        Record(String, String),
    }

    /// Collecting subscriber: gathers span creation, enter/exit, and field
    /// records into Vecs for the tests to assert hierarchy and fields.
    /// tokio::test defaults to current_thread, so event order is
    /// deterministic.
    #[derive(Debug, Default)]
    struct CollectSubscriber {
        spans: std::sync::Mutex<Vec<SpanInfo>>,
        ops: std::sync::Mutex<Vec<Op>>,
        names: std::sync::Mutex<HashMap<Id, String>>,
        next_id: AtomicU64,
    }

    /// Collect field values as debug text (all of Visit's default methods
    /// land in record_debug).
    struct FieldCollector<'a>(&'a mut Vec<(String, String)>);

    impl Visit for FieldCollector<'_> {
        fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
            self.0
                .push((field.name().to_string(), format!("{value:?}")));
        }
    }

    impl Subscriber for CollectSubscriber {
        fn enabled(&self, _metadata: &Metadata<'_>) -> bool {
            true
        }

        fn new_span(&self, span: &tracing::span::Attributes<'_>) -> Id {
            // fetch_add returns the old value (0 on the first call), and
            // tracing Ids must be non-zero.
            let id = Id::from_u64(self.next_id.fetch_add(1, Ordering::Relaxed) + 1);
            let mut fields = Vec::new();
            span.record(&mut FieldCollector(&mut fields));
            self.spans.lock().unwrap().push(SpanInfo {
                name: span.metadata().name(),
                level: *span.metadata().level(),
                fields,
            });
            self.names
                .lock()
                .unwrap()
                .insert(id.clone(), span.metadata().name().to_string());
            id
        }

        fn record(&self, id: &Id, values: &tracing::span::Record<'_>) {
            let name = self.names.lock().unwrap().get(id).cloned();
            let Some(name) = name else { return };
            let mut fields = Vec::new();
            values.record(&mut FieldCollector(&mut fields));
            let mut ops = self.ops.lock().unwrap();
            for (field, value) in fields {
                ops.push(Op::Record(name.clone(), format!("{field}={value}")));
            }
        }

        fn enter(&self, id: &Id) {
            let name = self.names.lock().unwrap().get(id).cloned();
            if let Some(name) = name {
                self.ops.lock().unwrap().push(Op::Enter(name));
            }
        }

        fn exit(&self, id: &Id) {
            let name = self.names.lock().unwrap().get(id).cloned();
            if let Some(name) = name {
                self.ops.lock().unwrap().push(Op::Exit(name));
            }
        }

        fn try_close(&self, id: Id) -> bool {
            let name = self.names.lock().unwrap().get(&id).cloned();
            if let Some(name) = name {
                self.ops.lock().unwrap().push(Op::Close(name));
            }
            true
        }

        fn clone_span(&self, id: &Id) -> Id {
            id.clone()
        }

        fn event(&self, _event: &Event<'_>) {}
        fn record_follows_from(&self, _span: &Id, _follows: &Id) {}
    }

    /// Swap the current thread's default dispatch to the collecting
    /// subscriber (the guard is held across awaits).
    fn collect_guard(sub: &Arc<CollectSubscriber>) -> tracing::dispatcher::DefaultGuard {
        tracing::dispatcher::set_default(&tracing::Dispatch::new(sub.clone()))
    }

    fn enter_names(ops: &[Op]) -> Vec<String> {
        ops.iter()
            .filter_map(|op| match op {
                Op::Enter(name) => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    fn records_of(ops: &[Op], span: &str) -> Vec<String> {
        ops.iter()
            .filter_map(|op| match op {
                Op::Record(s, kv) if s == span => Some(kv.clone()),
                _ => None,
            })
            .collect()
    }

    /// Assert hierarchy invariants (shared by both paths): (1) whenever a
    /// span is entered, `agent.run` must be on the enter/exit stack (no
    /// orphan spans — llm/tool must render under run);
    /// (2) no span with the same name on the stack (the same span can't be
    /// entered twice); (3) enter/exit strictly pair up.
    fn assert_nesting_invariants(ops: &[Op]) {
        let mut stack: Vec<String> = Vec::new();
        for op in ops {
            match op {
                Op::Enter(name) => {
                    if name != "agent.run" {
                        assert!(
                            stack.contains(&"agent.run".to_string()),
                            "span {name} requires agent.run on the stack when entering (stack: {stack:?})"
                        );
                    }
                    assert!(
                        !stack.contains(name),
                        "same span entered twice (double instrumenting): {name} (stack: {stack:?})"
                    );
                    stack.push(name.clone());
                }
                Op::Exit(name) => {
                    assert_eq!(
                        stack.pop().as_deref(),
                        Some(name.as_str()),
                        "exit must pair with enter: {name}"
                    );
                }
                _ => {}
            }
        }
    }

    /// Non-streaming path: span tree agent.run → llm_request / tool (two
    /// levels, grouped by the round attribute); hierarchy invariants hold
    /// (no orphans, no double enters); llm_request records usage at wrap-up
    /// (dual channel); run.id is consistent across the tree.
    #[tokio::test]
    async fn trace_span_tree_non_stream() {
        let sub = Arc::new(CollectSubscriber::default());
        let _guard = collect_guard(&sub);

        let (calc, _calls) = FakeTool::new("calc", "42");
        let mut registry = ToolRegistry::new();
        registry.register(calc);
        let fake = SharedFake::new([
            FakeReply::WithUsage {
                reply: Box::new(FakeReply::ToolCalls {
                    content: "".into(),
                    calls: vec![call("c1", "calc", "{}")],
                }),
                usage: Usage::new(10, 2),
            },
            FakeReply::text_with_usage("42", Usage::new(20, 5)),
        ]);
        let mut agent = agent_with_registry(fake, registry, AgentConfig::default());
        assert_eq!(agent.run("Compute").await.unwrap(), "42");

        let ops = sub.ops.lock().unwrap().clone();
        // Hierarchy invariants (no orphan spans, no double enters, paired
        // exits).
        assert_nesting_invariants(&ops);
        // Structure: two rounds, one llm_request each (round one also has a
        // tool). Note every instrumented future enters and exits twice (one
        // poll + one more enter inside Instrumented's drop; tracing
        // guarantees inner's Drop also runs in the span context), so the
        // assertions use "first-appearance order" and lower-bound counts
        // rather than exact enter/exit sequences.
        let enters = enter_names(&ops);
        let mut first_seen = Vec::new();
        for name in enters.iter() {
            if !first_seen.contains(name) {
                first_seen.push(name.clone());
            }
        }
        assert_eq!(first_seen, ["agent.run", "llm_request", "tool"]);
        assert!(enters.iter().filter(|n| *n == "llm_request").count() >= 2);
        assert!(enters.iter().filter(|n| *n == "tool").count() >= 1);

        // usage via both channels: llm_request records on the span at
        // wrap-up (the usage injected per round).
        assert_eq!(
            records_of(&ops, "llm_request"),
            [
                "usage.prompt_tokens=10",
                "usage.completion_tokens=2",
                "usage.prompt_tokens=20",
                "usage.completion_tokens=5",
            ]
        );

        // Levels: agent.run is INFO (skeleton visible by default), detail
        // spans are DEBUG.
        let spans = sub.spans.lock().unwrap().clone();
        let run_span = spans.iter().find(|s| s.name == "agent.run").unwrap();
        assert_eq!(run_span.level, Level::INFO);
        assert_eq!(
            spans
                .iter()
                .find(|s| s.name == "llm_request")
                .unwrap()
                .level,
            Level::DEBUG
        );
        assert_eq!(
            spans.iter().find(|s| s.name == "tool").unwrap().level,
            Level::DEBUG
        );

        // round attribute: llm_request carries the round number (two
        // rounds = 1, 2 — the grouping key).
        let llm_rounds: Vec<u64> = spans
            .iter()
            .filter(|s| s.name == "llm_request")
            .map(|s| {
                s.fields
                    .iter()
                    .find(|(f, _)| f == "round")
                    .map(|(_, v)| v.parse().unwrap())
                    .unwrap()
            })
            .collect();
        assert_eq!(llm_rounds, vec![1, 2]);

        // run.id: every span carries the same run id (the correlation key
        // with the event stream's RunStarted).
        let run_ids: Vec<String> = spans
            .iter()
            .map(|s| {
                s.fields
                    .iter()
                    .find(|(f, _)| f == "run.id")
                    .map(|(_, v)| v.clone())
                    .unwrap_or_else(|| panic!("span {} must carry a run.id field", s.name))
            })
            .collect();
        assert!(run_ids.iter().all(|id| id == &run_ids[0]));
        assert!(run_ids[0].starts_with("run-"));
    }

    /// Streaming path: the same span structure; the run span enters on
    /// every poll (covering the whole consumption period); usage is
    /// recorded at wrap-up when the Done event arrives.
    #[tokio::test]
    async fn trace_span_tree_stream() {
        let sub = Arc::new(CollectSubscriber::default());
        let _guard = collect_guard(&sub);

        let (calc, _calls) = FakeTool::new("calc", "42");
        let mut registry = ToolRegistry::new();
        registry.register(calc);
        let fake = SharedFake::new([
            FakeReply::WithUsage {
                reply: Box::new(FakeReply::ToolCalls {
                    content: "".into(),
                    calls: vec![call("c1", "calc", "{}")],
                }),
                usage: Usage::new(10, 2),
            },
            FakeReply::text_with_usage("42", Usage::new(20, 5)),
        ]);
        let mut agent = agent_with_registry(fake, registry, AgentConfig::default());
        agent
            .run_stream("Compute")
            .await
            .unwrap()
            .for_each(|_| async {})
            .await;

        let ops = sub.ops.lock().unwrap().clone();
        // Hierarchy invariants: any change that breaks the span hierarchy
        // (e.g. losing an instrument on the tool path) is caught by this
        // assertion.
        assert_nesting_invariants(&ops);
        // First-appearance enter order = the hierarchy; per-await
        // instrumenting inside the generator makes the same span enter and
        // exit many times (across polls + drops), so only first-appearance
        // order and multiple enters of run are asserted.
        let enters = enter_names(&ops);
        let mut first_seen = Vec::new();
        for name in enters.iter() {
            if !first_seen.contains(name) {
                first_seen.push(name.clone());
            }
        }
        assert_eq!(first_seen, ["agent.run", "llm_request", "tool"]);
        // The run span enters on every poll: it enters many times during
        // stream consumption, far beyond the 2 of poll+drop.
        assert!(enters.iter().filter(|n| *n == "agent.run").count() > 2);

        // usage recorded at wrap-up (the per-round injected usage).
        assert_eq!(
            records_of(&ops, "llm_request"),
            [
                "usage.prompt_tokens=10",
                "usage.completion_tokens=2",
                "usage.prompt_tokens=20",
                "usage.completion_tokens=5",
            ]
        );
    }

    /// Two runs: run.id differs (per-instance increment); the event stream's
    /// RunStarted carries the same id — the correlation key between trace
    /// and the event stream.
    #[tokio::test]
    async fn trace_run_id_differs_between_runs() {
        let sub = Arc::new(CollectSubscriber::default());
        let _guard = collect_guard(&sub);

        let fake = SharedFake::new([FakeReply::Text("hi".into()), FakeReply::Text("bye".into())]);
        let (mut agent, mut rx) = attach_channel(agent(fake, ""));
        agent.run("one").await.unwrap();
        agent.run("two").await.unwrap();
        drop(agent);

        let spans = sub.spans.lock().unwrap().clone();
        let run_ids: Vec<String> = spans
            .iter()
            .filter(|s| s.name == "agent.run")
            .map(|s| {
                s.fields
                    .iter()
                    .find(|(f, _)| f == "run.id")
                    .map(|(_, v)| v.clone())
                    .unwrap()
            })
            .collect();
        assert_eq!(run_ids.len(), 2);
        assert_ne!(run_ids[0], run_ids[1]);

        // Event-stream side: RunStarted carries the same run id as the span
        // (one per run).
        let events = drain(&mut rx).await;
        let event_ids: Vec<String> = events
            .iter()
            .filter_map(|e| match react_event(&**e) {
                ReActEvent::RunStarted { run_id, .. } => Some(run_id.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(event_ids, run_ids);
    }

    /// Run failure: the agent.run span records an error at wrap-up
    /// (observability spots the failed run at a glance).
    #[tokio::test]
    async fn trace_run_error_recorded() {
        let sub = Arc::new(CollectSubscriber::default());
        let _guard = collect_guard(&sub);

        let (calc, _calls) = FakeTool::new("calc", "42");
        let mut registry = ToolRegistry::new();
        registry.register(calc);
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "calc", "{}")],
            },
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c2", "calc", "{}")],
            },
        ]);
        let mut agent = agent_with_registry(
            fake,
            registry,
            AgentConfig {
                max_tool_rounds: 2,
                ..Default::default()
            },
        );
        assert!(matches!(
            agent.run("Keep computing").await,
            Err(AgentError::TooManyToolRounds(2))
        ));

        let ops = sub.ops.lock().unwrap().clone();
        // Values are Debug-formatted, so strings carry quotes; the error
        // text is asserted with starts_with (extensible, not locked to the
        // full text).
        let records = records_of(&ops, "agent.run");
        assert_eq!(records.len(), 1);
        assert!(
            records[0].starts_with("error=\"model requested tools for more than 2 rounds"),
            "unexpected records: {records:?}"
        );
    }

    /// Tool failure: the tool span records an error (the failure is fed
    /// back as text; the run ends normally, agent.run has no error).
    #[tokio::test]
    async fn trace_tool_error_recorded() {
        struct FailingTool;
        #[async_trait::async_trait]
        impl Tool for FailingTool {
            fn schema(&self) -> ToolSchema {
                ToolSchema {
                    name: "boom".into(),
                    description: "Tool that always fails".into(),
                    parameters: serde_json::json!({}),
                }
            }
            async fn call(
                &self,
                _arguments: serde_json::Value,
                _state: &SharedState,
            ) -> Result<String, ToolError> {
                Err(ToolError::Execution("internal error".into()))
            }
        }

        let sub = Arc::new(CollectSubscriber::default());
        let _guard = collect_guard(&sub);

        let mut registry = ToolRegistry::new();
        registry.register(FailingTool);
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "boom", "{}")],
            },
            FakeReply::Text("Got it".into()),
        ]);
        let mut agent = agent_with_registry(fake, registry, AgentConfig::default());
        agent.run("Trigger failure").await.unwrap();

        let ops = sub.ops.lock().unwrap().clone();
        // The tool span records the failure (the registry classification's
        // Display); the run ends normally, with no error.
        let tool_errors = records_of(&ops, "tool");
        assert_eq!(tool_errors.len(), 1);
        assert!(tool_errors[0].contains("internal error"));
        assert!(records_of(&ops, "agent.run").is_empty());
    }

    /// Provider failure (non-streaming): the llm_request span records the
    /// error — observability can pinpoint which call of which round failed;
    /// the run span records too (the whole run failed).
    #[tokio::test]
    async fn trace_llm_error_recorded() {
        let sub = Arc::new(CollectSubscriber::default());
        let _guard = collect_guard(&sub);

        // Script exhausted: the second run's conversation fails outright.
        let fake = SharedFake::new([FakeReply::Text("hi".into())]);
        let mut agent = agent(fake, "");
        agent.run("one").await.unwrap();
        let err = agent.run("two").await.unwrap_err();
        assert!(matches!(err, AgentError::Provider(_)));

        let ops = sub.ops.lock().unwrap().clone();
        // The failed call: the llm_request span has an error record (the
        // first run's successful call only has usage records; the two spans'
        // records land in the same collector).
        let llm_records = records_of(&ops, "llm_request");
        assert!(
            llm_records.iter().any(|r| r.starts_with("error=")),
            "the failed round's llm span should have an error: {llm_records:?}"
        );
        // The run span also records the failed outcome.
        assert!(
            records_of(&ops, "agent.run")
                .iter()
                .any(|r| r.starts_with("error="))
        );
    }

    /// Provider failure mid-stream (streaming): the consumption loop's Err
    /// event records the error on the llm_request span, and the run span
    /// records in sync; hierarchy invariants still hold.
    #[tokio::test]
    async fn trace_stream_llm_error_recorded() {
        struct FailInStream;
        #[async_trait::async_trait]
        impl Provider for FailInStream {
            async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, ProviderError> {
                unreachable!("this test uses streaming path only")
            }
            async fn stream_chat(
                &self,
                _request: ChatRequest,
            ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError>
            {
                Ok(Box::pin(futures::stream::iter(vec![
                    Ok(StreamEvent::Delta("hi".into())),
                    Err(ProviderError::Api {
                        status: 0,
                        message: "boom".into(),
                    }),
                ])))
            }
        }

        let sub = Arc::new(CollectSubscriber::default());
        let _guard = collect_guard(&sub);

        let mut agent = ReActAgent::new(FailInStream, ToolRegistry::new(), "");
        let mut stream = agent.run_stream("two").await.unwrap();
        while stream.next().await.is_some() {}

        let ops = sub.ops.lock().unwrap().clone();
        assert_nesting_invariants(&ops);
        assert!(
            records_of(&ops, "llm_request")
                .iter()
                .any(|r| r.contains("boom"))
        );
        assert!(
            records_of(&ops, "agent.run")
                .iter()
                .any(|r| r.contains("boom"))
        );
    }

    // ---------- Skill assembly ----------

    /// Test skill: builds a minimal valid SKILL.md.
    fn skill(name: &str, description: &str, body: &str) -> crate::skill::Skill {
        crate::skill::Skill::parse(&format!(
            "---\nname: {name}\ndescription: {description}\n---\n{body}"
        ))
        .unwrap()
    }

    /// A registry containing multiple skills.
    fn skill_registry(skills: &[(&str, &str, &str)]) -> SkillRegistry {
        let registry = SkillRegistry::new();
        for (name, description, body) in skills {
            registry.add(skill(name, description, body));
        }
        registry
    }

    /// The System message text of the most recent request.
    fn system_text(fake: &SharedFake) -> String {
        let requests = fake.requests();
        let last = requests.last().expect("expected a request");
        match &last.messages[0] {
            Message::System(s) => s.clone(),
            other => panic!("first message must be System, got: {other:?}"),
        }
    }

    /// The ToolResult content carrying the given substring in the most
    /// recent request.
    fn tool_result_contains(fake: &SharedFake, needle: &str) {
        let requests = fake.requests();
        let last = requests.last().expect("expected a request");
        let content = last
            .messages
            .iter()
            .find_map(|m| match m {
                Message::ToolResult { content, .. } => Some(content.clone()),
                _ => None,
            })
            .expect("expected ToolResult");
        assert!(
            content.contains(needle),
            "ToolResult should contain {needle:?}, got: {content:?}"
        );
    }

    #[tokio::test]
    async fn with_skills_adds_menu_to_system_prompt() {
        let fake = SharedFake::new([FakeReply::Text("OK".into())]);
        let mut agent = agent(fake.clone(), "You are an assistant").with_skills(skill_registry(&[
            ("code-review", "Review code", "Step one"),
            ("greet", "Say hello", "Hello"),
        ]));

        agent.run("Are you there").await.unwrap();
        let system = system_text(&fake);
        assert!(
            system.starts_with("You are an assistant"),
            "base prompt must come first: {system}"
        );
        assert!(system.contains("- code-review: Review code"));
        assert!(system.contains("- greet: Say hello"));
    }

    #[tokio::test]
    async fn with_skills_registers_load_skill() {
        let fake = SharedFake::new([FakeReply::Text("OK".into())]);
        let mut agent = agent(fake.clone(), "You are an assistant")
            .with_skills(skill_registry(&[("greet", "Say hello", "Hello body")]));

        // The tool is registered into the registry; the schema reaches the
        // request's tools, visible to the model.
        assert!(agent.registry.names().contains(&"load_skill".to_string()));
        agent.run("Are you there").await.unwrap();
        assert!(
            fake.requests()[0]
                .tools
                .iter()
                .any(|t| t.name == "load_skill")
        );
    }

    #[tokio::test]
    async fn load_skill_used_in_loop() {
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "load_skill", r#"{"name":"greet"}"#)],
            },
            FakeReply::Text("Done".into()),
        ]);
        let mut agent = agent(fake.clone(), "You are an assistant")
            .with_skills(skill_registry(&[("greet", "Say hello", "Hello body")]));

        let answer = agent.run("Say hello").await.unwrap();
        assert_eq!(answer, "Done");
        // The body is recorded as a ToolResult and sent back to the model in
        // the second request.
        let requests = fake.requests();
        assert_eq!(requests.len(), 2);
        let tool_results: Vec<&Message> = requests[1]
            .messages
            .iter()
            .filter(|m| matches!(m, Message::ToolResult { .. }))
            .collect();
        assert_eq!(tool_results.len(), 1);
        match tool_results[0] {
            Message::ToolResult { content, .. } => assert!(content.contains("Hello body")),
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn load_skill_not_found_returns_error_text() {
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "load_skill", r#"{"name":"ghost"}"#)],
            },
            FakeReply::Text("Try another".into()),
        ]);
        let mut agent = agent(fake.clone(), "You are an assistant")
            .with_skills(skill_registry(&[("greet", "Say hello", "Hello body")]));

        agent.run("Load skill").await.unwrap();
        tool_result_contains(&fake, "not found");
    }

    #[tokio::test]
    async fn load_skill_not_enabled_returns_text() {
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "load_skill", r#"{"name":"other"}"#)],
            },
            FakeReply::Text("Understood".into()),
        ]);
        let mut agent = agent(fake.clone(), "You are an assistant")
            .with_skills(skill_registry(&[
                ("greet", "Say hello", "Hello body"),
                ("other", "Another", "Other content"),
            ]))
            .with_enabled_skills(&["greet"]);

        agent.run("Load").await.unwrap();
        tool_result_contains(&fake, "not enabled");
    }

    #[tokio::test]
    async fn with_skills_inline_embeds_all_bodies() {
        let fake = SharedFake::new([FakeReply::Text("OK".into())]);
        let mut agent =
            agent(fake.clone(), "You are an assistant").with_skills_inline(skill_registry(&[
                ("greet", "Say hello", "Hello body"),
                ("other", "Another", "Other content"),
            ]));
        assert!(!agent.registry.names().contains(&"load_skill".to_string()));

        agent.run("Are you there").await.unwrap();
        let system = system_text(&fake);
        assert!(system.contains("[Skill greet]\nHello body"));
        assert!(system.contains("[Skill other]\nOther content"));
        // Inline mode has no menu.
        assert!(!system.contains("- greet:"));
    }

    #[tokio::test]
    async fn with_enabled_skills_filters_menu() {
        let fake = SharedFake::new([FakeReply::Text("OK".into())]);
        let mut agent = agent(fake.clone(), "You are an assistant")
            .with_skills(skill_registry(&[
                ("greet", "Say hello", "Hello body"),
                ("other", "Another", "Other content"),
            ]))
            .with_enabled_skills(&["greet"]);

        agent.run("Are you there").await.unwrap();
        let system = system_text(&fake);
        assert!(system.contains("- greet: Say hello"));
        assert!(!system.contains("- other:"));
    }

    #[tokio::test]
    async fn activate_skill_embeds_body_and_leaves_menu() {
        let fake = SharedFake::new([FakeReply::Text("OK".into())]);
        let mut agent = agent(fake.clone(), "You are an assistant").with_skills(skill_registry(&[
            ("greet", "Say hello", "Hello body"),
            ("other", "Another", "Other content"),
        ]));
        assert!(agent.activate_skill("greet"));

        agent.run("Are you there").await.unwrap();
        let system = system_text(&fake);
        assert!(system.contains("[Skill greet]\nHello body"));
        // Pre-activated skills leave the menu; other skills remain.
        assert!(!system.contains("- greet:"));
        assert!(system.contains("- other: Another"));
    }

    #[tokio::test]
    async fn deactivate_skill_returns_to_menu() {
        let fake = SharedFake::new([FakeReply::Text("OK".into())]);
        let mut agent = agent(fake.clone(), "You are an assistant")
            .with_skills(skill_registry(&[("greet", "Say hello", "Hello body")]));
        assert!(agent.activate_skill("greet"));
        assert!(agent.deactivate_skill("greet"));

        agent.run("Are you there").await.unwrap();
        let system = system_text(&fake);
        assert!(system.contains("- greet: Say hello"));
        assert!(!system.contains("[Skill greet]"));
        // Deactivating a skill that isn't activated: false.
        assert!(!agent.deactivate_skill("greet"));
    }

    #[tokio::test]
    async fn activate_skill_failure_paths() {
        // Nonexistent / outside the allowlist → false; repeated activation
        // is idempotent → true.
        let mut whitelisted = agent(SharedFake::new([FakeReply::Text("OK".into())]), "")
            .with_skills(skill_registry(&[("greet", "Say hello", "Hello body")]))
            .with_enabled_skills(&["greet"]);
        assert!(!whitelisted.activate_skill("ghost"));
        assert!(!whitelisted.activate_skill("other"));
        assert!(whitelisted.activate_skill("greet"));
        assert!(whitelisted.activate_skill("greet"));

        // Inline mode: pre-activation has no effect.
        let mut inline = agent(SharedFake::new([FakeReply::Text("OK".into())]), "")
            .with_skills_inline(skill_registry(&[("greet", "Say hello", "Hello body")]));
        assert!(!inline.activate_skill("greet"));
        assert!(!inline.deactivate_skill("greet"));
    }

    #[tokio::test]
    async fn empty_skills_registry_zero_cost() {
        let fake = SharedFake::new([FakeReply::Text("OK".into())]);
        let mut agent =
            agent(fake.clone(), "You are an assistant").with_skills(SkillRegistry::new());
        agent.run("Are you there").await.unwrap();
        // Empty registry: the system prompt is unchanged.
        assert_eq!(system_text(&fake), "You are an assistant");
        // load_skill is registered: hot-swapping from empty to populated
        // lets the model load immediately.
        assert!(agent.registry.names().contains(&"load_skill".to_string()));
    }

    #[tokio::test]
    async fn skills_hot_swap_takes_effect_next_request() {
        let fake = SharedFake::new([
            FakeReply::Text("first round".into()),
            FakeReply::Text("second round".into()),
        ]);
        let mut swap_agent =
            agent(fake.clone(), "You are an assistant").with_skills(SkillRegistry::new());

        swap_agent.run("Are you there").await.unwrap();
        // The application side hot-swaps via the pub skills handle: it takes
        // effect on the next request.
        swap_agent
            .skills
            .add(skill("late", "Skill added later", "Late body"));
        swap_agent.run("Once more").await.unwrap();
        assert!(system_text(&fake).contains("- late: Skill added later"));

        // load_skill looks up by name at call time: newly hot-swapped skills
        // load immediately.
        let fake2 = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "load_skill", r#"{"name":"late"}"#)],
            },
            FakeReply::Text("Done".into()),
        ]);
        let mut late_agent =
            agent(fake2.clone(), "You are an assistant").with_skills(SkillRegistry::new());
        late_agent
            .skills
            .add(skill("late", "Skill added later", "Late body"));
        late_agent.run("Load").await.unwrap();
        tool_result_contains(&fake2, "Late body");
    }

    #[tokio::test]
    async fn with_skills_inline_overrides_with_skills() {
        let fake = SharedFake::new([FakeReply::Text("OK".into())]);
        let mut agent = agent(fake.clone(), "You are an assistant")
            .with_skills(skill_registry(&[("a", "A", "Body-a")]))
            .with_skills_inline(skill_registry(&[("b", "B", "Body-b")]));
        // Dynamic → inline switch: load_skill is removed.
        assert!(!agent.registry.names().contains(&"load_skill".to_string()));

        agent.run("Are you there").await.unwrap();
        let system = system_text(&fake);
        assert!(system.contains("[Skill b]\nBody-b"));
        assert!(!system.contains("Body-a"));
    }

    #[tokio::test]
    async fn load_skill_content_survives_window_trim() {
        // Small budget window: skill bodies (protected) stay resident while
        // regular conversation rounds are trimmed as usual.
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "load_skill", r#"{"name":"greet"}"#)],
            },
            FakeReply::Text("round one".into()),
            FakeReply::Text("round two".into()),
            FakeReply::Text("round three".into()),
        ]);
        let mut agent = agent(fake.clone(), "You are an assistant")
            .with_memory(crate::memory::WindowMemory::new(3))
            .with_skills(skill_registry(&[(
                "greet",
                "Say hello",
                "Skill body content",
            )]));

        agent.run("Load").await.unwrap();
        agent.run("round two").await.unwrap();
        agent.run("round three").await.unwrap();

        // Context of the last request: the protected first round (skill
        // body) and the latest round are kept; the middle regular rounds
        // are trimmed.
        let requests = fake.requests();
        let last = requests.last().unwrap();
        let texts: Vec<String> = last
            .messages
            .iter()
            .map(|m| match m {
                Message::ToolResult { content, .. } => content.clone(),
                Message::User(blocks) => blocks
                    .iter()
                    .map(|b| match b {
                        crate::ContentBlock::Text(t) => t.clone(),
                    })
                    .collect(),
                _ => String::new(),
            })
            .collect();
        let joined = texts.join("|");
        assert!(
            joined.contains("Skill body content"),
            "skill body should stay resident: {joined}"
        );
        assert!(
            joined.contains("round three"),
            "latest round should be kept: {joined}"
        );
        assert!(
            !joined.contains("round two"),
            "middle regular rounds should be trimmed: {joined}"
        );
    }

    #[tokio::test]
    async fn enabled_skills_before_with_skills_order_independent() {
        // Allowlist set before assembly: with_skills reads the allowlist
        // when registering load_skill.
        let fake = SharedFake::new([
            FakeReply::ToolCalls {
                content: "".into(),
                calls: vec![call("c1", "load_skill", r#"{"name":"other"}"#)],
            },
            FakeReply::Text("Understood".into()),
        ]);
        let mut agent = agent(fake.clone(), "You are an assistant")
            .with_enabled_skills(&["greet"])
            .with_skills(skill_registry(&[
                ("greet", "Say hello", "Hello body"),
                ("other", "Another", "Other content"),
            ]));

        agent.run("Load").await.unwrap();
        tool_result_contains(&fake, "not enabled");
    }
}
