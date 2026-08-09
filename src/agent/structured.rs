//! Structured output validation components: schema validation of the model's
//! answer + an independent retry budget + feedback messages.
//!
//! Used when wiring up typed output
//! ([`TypedAgent`](crate::agent::TypedAgent)) — a self-implemented Agent
//! calls [`StructuredValidator::validate`] each round inside its own
//! reasoning loop, and a single `match` replaces the hand-written
//! "validate → count → check limit → feedback message" boilerplate.

use crate::message::Message;

/// Structured output validation: the answer must parse as JSON and conform
/// to the schema.
///
/// Used by self-implemented Agents wiring up typed output
/// ([`TypedAgent`](crate::agent::TypedAgent)): on failure, record the
/// feedback text from [`structured_retry_message`] as a User message so the
/// model can correct and retry (the budget is defined by the assembler; use
/// [`StructuredValidator`] when you want a state machine with the budget
/// built in). Failures return **model-facing error text** (English — fed
/// back to the model, which corrects based on it).
///
/// The schema is compiled on every call (jsonschema caches compiled results
/// internally; regular schemas compile in under a millisecond, negligible
/// relative to model latency).
///
/// # Examples
///
/// ```
/// use molo::agent::validate_structured;
///
/// let schema = serde_json::json!({
///     "type": "object",
///     "properties": { "city": { "type": "string" } },
///     "required": ["city"],
/// });
/// assert!(validate_structured(&schema, r#"{"city":"Beijing"}"#).is_ok());
/// assert!(validate_structured(&schema, "not JSON").is_err());
/// ```
pub fn validate_structured(schema: &serde_json::Value, answer: &str) -> Result<(), String> {
    let instance: serde_json::Value =
        serde_json::from_str(answer).map_err(|e| format!("answer is not valid JSON: {e}"))?;
    let validator =
        jsonschema::validator_for(schema).map_err(|e| format!("invalid JSON schema: {e}"))?;
    if let Err(e) = validator.validate(&instance) {
        return Err(format!("answer does not match the JSON schema: {e}"));
    }
    Ok(())
}

/// Feedback message for structured output validation failure (recorded as a
/// User message; the model retries based on it).
///
/// Complements [`validate_structured`]: record this message after a
/// validation failure, and the model sees the error details in the next
/// round and corrects its answer.
pub fn structured_retry_message(error: &str) -> Message {
    Message::user(format!(
        "your previous answer failed JSON schema validation: {error}; \
         please reply with a single JSON value conforming to the schema"
    ))
}

/// The outcome of a single validation
/// ([`StructuredValidator::validate`] return value).
///
/// - [`Passed`](StructuredOutcome::Passed): the answer conforms to the
///   schema; the run wraps up;
/// - [`Retry`](StructuredOutcome::Retry): validation failed, carrying a
///   model-facing feedback message — recorded as a User message so the model
///   corrects itself in the next round;
/// - [`Exhausted`](StructuredOutcome::Exhausted): the retry budget is
///   exhausted; the run fails.
///
/// The enum carries `#[non_exhaustive]`: future outcomes won't be a breaking
/// change; matches should include a wildcard arm.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum StructuredOutcome {
    /// Passed: the answer conforms to the schema.
    Passed,
    /// Validation failed, carrying a feedback message (model-facing,
    /// recorded as a User message so the model corrects itself).
    Retry {
        /// Model-facing feedback message (English, recorded as a User
        /// message).
        message: Message,
    },
    /// Retry budget exhausted, carrying the limit.
    Exhausted {
        /// The retry budget limit (matches the `max_retries` used to
        /// construct `StructuredValidator`).
        max_retries: usize,
    },
}

/// Structured output validator: schema validation of the model's answer +
/// **independent retry budget** + feedback messages.
///
/// Used by self-implemented Agents wiring up typed output
/// ([`TypedAgent`](crate::agent::TypedAgent)): call
/// [`validate`](StructuredValidator::validate) each round inside the loop,
/// and the component owns the budget counting — user code no longer writes
/// the "validate → count → check limit" boilerplate.
///
/// Budget semantics: up to `max_retries` retries after a failed validation;
/// the `max_retries + 1`-th failure returns
/// [`Exhausted`](StructuredOutcome::Exhausted).
///
/// # Examples
///
/// ```
/// use molo::agent::{StructuredOutcome, StructuredValidator};
///
/// let schema = serde_json::json!({
///     "type": "object",
///     "properties": { "city": { "type": "string" } },
///     "required": ["city"],
/// });
/// let mut validator = StructuredValidator::new(schema, 3);
///
/// assert!(matches!(validator.validate("bad"), StructuredOutcome::Retry { .. }));
/// assert!(matches!(
///     validator.validate(r#"{"city":"Beijing"}"#),
///     StructuredOutcome::Passed
/// ));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredValidator {
    schema: serde_json::Value,
    /// Maximum number of retries after a validation failure.
    max_retries: usize,
    /// Number of retries already used.
    retries_used: usize,
}

impl StructuredValidator {
    /// Construct with a schema and a retry budget; `max_retries` has the
    /// same semantics as
    /// [`AgentConfig::max_structured_retries`](crate::agent::AgentConfig)
    /// (the built-in assembly defaults to 3).
    pub fn new(schema: serde_json::Value, max_retries: usize) -> Self {
        Self {
            schema,
            max_retries,
            retries_used: 0,
        }
    }

    /// Validate one answer: passed →
    /// [`Passed`](StructuredOutcome::Passed); failed with budget remaining →
    /// [`Retry`](StructuredOutcome::Retry) (carrying a feedback message,
    /// recorded so the model corrects itself); failed with budget exhausted
    /// → [`Exhausted`](StructuredOutcome::Exhausted).
    pub fn validate(&mut self, answer: &str) -> StructuredOutcome {
        match validate_structured(&self.schema, answer) {
            Ok(()) => StructuredOutcome::Passed,
            Err(error) => {
                self.retries_used += 1;
                if self.retries_used > self.max_retries {
                    StructuredOutcome::Exhausted {
                        max_retries: self.max_retries,
                    }
                } else {
                    StructuredOutcome::Retry {
                        message: structured_retry_message(&error),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ContentBlock;

    fn schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        })
    }

    /// Three-state outcomes: failure within budget → Retry (carrying a
    /// feedback message), exhausted → Exhausted, passed → Passed.
    #[test]
    fn validator_three_outcomes() {
        let mut validator = StructuredValidator::new(schema(), 1);
        match validator.validate("bad") {
            StructuredOutcome::Retry { message } => {
                let Message::User(blocks) = message else {
                    panic!("retry message must be a user message")
                };
                assert!(blocks.iter().any(|b| matches!(
                    b,
                    ContentBlock::Text(t) if t.contains("JSON schema validation")
                )));
            }
            other => panic!("expected Retry, got {other:?}"),
        }
        // Budget exhausted (didn't pass within 1 retry; the 2nd failure hits
        // the limit).
        assert!(matches!(
            validator.validate("bad2"),
            StructuredOutcome::Exhausted { max_retries: 1 }
        ));
    }

    /// Passing does not consume budget; a failed answer retried after
    /// correction → Passed.
    #[test]
    fn validator_passes_after_retry() {
        let mut validator = StructuredValidator::new(schema(), 3);
        assert!(matches!(
            validator.validate("bad"),
            StructuredOutcome::Retry { .. }
        ));
        assert!(matches!(
            validator.validate(r#"{"city":"Beijing"}"#),
            StructuredOutcome::Passed
        ));
        // Re-validating after a pass still passes (counting doesn't affect
        // the success path).
        assert!(matches!(
            validator.validate(r#"{"city":"Shanghai"}"#),
            StructuredOutcome::Passed
        ));
    }
}
