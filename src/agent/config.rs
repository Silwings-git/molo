//! Agent construction configuration: an aggregate of optional behaviors.
//!
//! Required dependencies (Provider / ToolRegistry / system_prompt) go
//! through constructor parameters, while optional behaviors go through
//! [`AgentConfig`] — the same pattern as
//! [`ModelOptions`](crate::provider::ModelOptions): all fields optional +
//! `Default`.

use crate::provider::ModelOptions;

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
/// use molo::agent::{AgentConfig, ReActAgent};
/// use molo::provider::{FakeProvider, FakeReply, ModelOptions};
/// use molo::tool::ToolRegistry;
///
/// let config = AgentConfig {
///     max_tool_rounds: 20,
///     max_structured_retries: 5,
///     options: ModelOptions {
///         temperature: Some(0.2),
///         ..Default::default()
///     },
/// };
///
/// let agent = ReActAgent::new(
///     FakeProvider::new([FakeReply::Text("Hello".into())]),
///     ToolRegistry::new(),
///     "",
/// )
/// .with_config(config);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct AgentConfig {
    /// Maximum number of rounds in which the model requests tools; if it
    /// still hasn't answered directly when this is reached, the run fails
    /// ([`AgentError::TooManyToolRounds`](crate::agent::AgentError)).
    /// Prevents the model from getting stuck requesting tools in a loop.
    /// Default: 10.
    pub max_tool_rounds: usize,
    /// Maximum number of retries after structured output validation fails;
    /// if it still fails when this is reached, the run fails
    /// ([`AgentError::StructuredRetriesExhausted`](crate::agent::AgentError)).
    /// Counted **independently** of the tool-round limit (each failure mode
    /// has its own clear error semantics). Default: 3.
    pub max_structured_retries: usize,
    /// Model parameters carried on every round of conversation (temperature /
    /// max_tokens / extra parameters);
    /// `Default` = all empty, using the vendor defaults.
    pub options: ModelOptions,
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
