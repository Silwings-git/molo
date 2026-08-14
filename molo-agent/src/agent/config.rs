//! Agent construction configuration: an aggregate of optional behaviors.
//!
//! Required dependencies (Provider / ToolRegistry / system_prompt) go
//! through constructor parameters, while optional behaviors go through
//! [`AgentConfig`] — the same pattern as
//! [`ModelOptions`](crate::provider::ModelOptions): all fields optional +
//! `Default`.

use crate::provider::ModelOptions;
use serde::{Deserialize, Serialize};

/// Optional behavior configuration for an Agent.
///
/// Kept separate from the required parameters (Provider / ToolRegistry /
/// system_prompt): the required ones define what the Agent is made of, while
/// the optional ones are behavioral parameters for how it runs.
///
/// # Examples
///
/// Adjust the tool-round limit and model parameters (temperature, etc.):
///
/// ```
/// # extern crate molo_agent as molo;
/// use molo::agent::{AgentConfig, ReActAgent};
/// use molo::provider::{FakeProvider, FakeReply, ModelOptions};
/// use molo::tool::ToolRegistry;
///
/// let config = AgentConfig::default()
///     .with_max_tool_rounds(20)
///     .with_max_structured_retries(5)
///     .with_options(ModelOptions {
///         temperature: Some(0.2),
///         ..Default::default()
///     });
///
/// let agent = ReActAgent::new(
///     FakeProvider::new([FakeReply::Text("Hello".into())]),
///     ToolRegistry::new(),
///     "",
/// )
/// .with_config(config);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
#[non_exhaustive]
pub struct AgentConfig {
    /// Maximum number of rounds in which the model requests tools; if it
    /// still hasn't answered directly when this is reached, the run fails
    /// ([`AgentError::TooManyToolRounds`](crate::agent::AgentError)).
    /// Prevents the model from getting stuck requesting tools in a loop.
    /// Default: 10.
    pub(crate) max_tool_rounds: usize,
    /// Maximum number of retries after structured output validation fails;
    /// if it still fails when this is reached, the run fails
    /// ([`AgentError::StructuredRetriesExhausted`](crate::agent::AgentError)).
    /// Counted **independently** of the tool-round limit (each failure mode
    /// has its own clear error semantics). Default: 3.
    pub(crate) max_structured_retries: usize,
    /// Model parameters carried on every round of conversation (temperature /
    /// max_tokens / extra parameters);
    /// `Default` = all empty, using the vendor defaults.
    pub(crate) options: ModelOptions,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_tool_rounds: 10,
            max_structured_retries: 3,
            options: ModelOptions::default(),
        }
    }
}

impl AgentConfig {
    /// Constructs a config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Maximum tool-call rounds before a run fails.
    pub fn max_tool_rounds(&self) -> usize {
        self.max_tool_rounds
    }

    /// Returns a config with an updated tool-round limit.
    pub fn with_max_tool_rounds(mut self, max_tool_rounds: usize) -> Self {
        self.max_tool_rounds = max_tool_rounds;
        self
    }

    /// Maximum structured-output validation retries before a run fails.
    pub fn max_structured_retries(&self) -> usize {
        self.max_structured_retries
    }

    /// Returns a config with an updated structured-output retry limit.
    pub fn with_max_structured_retries(mut self, max_structured_retries: usize) -> Self {
        self.max_structured_retries = max_structured_retries;
        self
    }

    /// Model options sent on each provider request unless a run overrides them.
    pub fn options(&self) -> &ModelOptions {
        &self.options
    }

    /// Returns a config with updated model options.
    pub fn with_options(mut self, options: ModelOptions) -> Self {
        self.options = options;
        self
    }
}
