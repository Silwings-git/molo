//! Summarization trim strategy: uses an LLM to compact over-budget old
//! messages into a single summary.
//!
//! [`SummarizeStrategy`] is a ready-to-use implementation of
//! [`TrimStrategy`](crate::memory::TrimStrategy) that holds a
//! [`Provider`](crate::Provider) to call a summarizer model: when over budget
//! it keeps the most recent rounds verbatim (the tail of the conversation
//! carries the strongest signal and is not compacted), textifies the earlier
//! messages, and hands them to the summarizer; the output replaces the
//! compacted portion as a single **System summary message** and is
//! materialized (`replace: true`) — one compaction covers many rounds, until
//! new messages breach the budget again.
//!
//! The summary message sits at the front of the result sequence and uses the
//! System role: a summary is not "what the user said" — it is
//! framework-injected history compaction, and the model should treat it as
//! context background rather than conversation content. The summary message
//! also obeys Memory's trim rules and participates as input in the next
//! compaction when over budget again (incremental merge — the summary is never
//! compacted repeatedly).
//!
//! A failed summarization (network / rate limit etc.) does not interrupt the
//! conversation: it degrades to window dropping (projection — storage keeps
//! the original text and summarization is retried on the next over-budget
//! retrieval), logging a single warn.

use std::fmt;

use super::window::{count_message, keep_rounds, split_rounds};
use super::{Budget, MemoryError, TokenCounter, TrimResult, TrimStrategy, WindowDrop};
use crate::message::{ContentBlock, Message};
use crate::provider::{ChatRequest, ModelOptions, Provider};

/// Content prefix of the summary message: lets the model and downstream tools
/// recognize "this is a historical summary" (fed as the prior-summary input on
/// the next compaction, never re-summarized).
const SUMMARY_PREFIX: &str = "Summary of prior context:\n";

/// Default summarization prompt: instruction + requirements, sent to the
/// summarizer model.
const DEFAULT_SUMMARY_PROMPT: &str = "\
You are a conversation-history compressor. Compress the provided conversation history into a concise summary so the model can recover context in subsequent conversation.

Requirements:
- Preserve the task goal, current state, decisions made, open items, and next steps;
- Keep proper nouns such as code identifiers, file paths, URLs, and numbers verbatim;
- The history may contain a \"Summary of prior context\" (system message): first understand the prior summary, then merge it with the new history and output a single integrated summary; do not re-summarize the prior summary itself;
- Output only the summary body, with no explanations or surrounding text.";

/// Summarization trim strategy: over-budget old messages → summarizer → one
/// System summary message.
///
/// # Behavior
///
/// - **Keep the most recent rounds verbatim**: keep complete rounds from the
///   tail forward until the budget is approached (the budget first deducts a
///   reservation for the summary output, see
///   [`with_summary_max_tokens`](SummarizeStrategy::with_summary_max_tokens));
///   at least the most recent round is kept (kept even when a single round
///   exceeds the budget; nothing more can be compacted);
/// - **Incremental compaction**: when over budget again, the previous summary
///   message goes to the summarizer as input (prior summary), producing an
///   integrated new summary — the summary is never compacted repeatedly
///   (semantic drift), and information already compacted is not lost;
/// - **Materialization**: `replace: true`, the compaction result is written
///   back to storage with zero recomputation afterwards;
/// - **Failure degradation**: when the summarizer call fails, degrade to
///   window dropping (projection — storage keeps the original text, retried on
///   the next over-budget retrieval); the conversation is not interrupted.
///
/// The summary message uses the System role and sits at the front of the
/// sequence (a summary is not conversation content but history compaction; the
/// System role also avoids consecutive User messages after compaction). The
/// summary message's tokens count toward the budget and obey the window trim
/// rules like any other message.
///
/// # Comparison
///
/// Compared to the default [`WindowDrop`]: window dropping is a lossless
/// projection with cheap per-turn recomputation, suited to ad-hoc inspection;
/// this strategy trades one summarizer call for a longer compactable span,
/// suited to long conversations that must not exceed limits, at the cost of
/// the summarizer call (cost + latency) and information loss.
///
/// # Example
///
/// Inject the summarization strategy (using
/// [`FakeProvider`](crate::FakeProvider) in place of a real summarizer to
/// demonstrate the full flow):
///
/// ```rust
/// use std::sync::Arc;
/// use molo::memory::{Memory, SummarizeStrategy, WindowMemory};
/// use molo::{FakeProvider, FakeReply, Message};
///
/// #[tokio::main]
/// async fn main() -> Result<(), molo::memory::MemoryError> {
///     let fake = Arc::new(FakeProvider::new([FakeReply::Text("Key points from earlier rounds".into())]));
///     let mut memory = WindowMemory::new(30)
///         .with_strategy(Arc::new(SummarizeStrategy::new(fake)));
///     for i in 1..=4 {
///         memory.record(Message::user(format!("Question from round {i}"))).await?;
///         memory.record(Message::assistant(format!("Answer from round {i}"))).await?;
///     }
///
///     // 44 tokens total > budget 30: compacted to [summary, most recent round].
///     let context = memory.context().await?;
///     assert_eq!(context.len(), 3);
///     assert!(matches!(context[0], Message::System(_)));
///     assert_eq!(context[1], Message::user("Question from round 4"));
///     Ok(())
/// }
/// ```
pub struct SummarizeStrategy {
    provider: Box<dyn Provider>,
    prompt: String,
    /// Token cap for the summary output (also the budget reservation for kept
    /// rounds).
    summary_max_tokens: u32,
}

impl fmt::Debug for SummarizeStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Provider is a trait object and cannot derive Debug: print the
        // comparable fields and mark the Provider's type.
        f.debug_struct("SummarizeStrategy")
            .field("provider", &"Box<dyn Provider>")
            .field("prompt_len", &self.prompt.len())
            .field("summary_max_tokens", &self.summary_max_tokens)
            .finish()
    }
}

impl SummarizeStrategy {
    /// Creates a summarization strategy with the default prompt and a default
    /// summary output cap (1024 tokens).
    pub fn new(provider: impl Provider + 'static) -> Self {
        Self {
            provider: Box::new(provider),
            prompt: DEFAULT_SUMMARY_PROMPT.into(),
            summary_max_tokens: 1024,
        }
    }

    /// Replaces the default summarization prompt (instruction + requirements;
    /// the history text is appended by the strategy).
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    /// Sets the token cap for the summary output (default 1024).
    ///
    /// The value doubles as the budget reservation for kept rounds: the kept
    /// rounds' token total stays within "budget − summary cap", leaving room
    /// for the summary output.
    pub fn with_summary_max_tokens(mut self, max_tokens: u32) -> Self {
        self.summary_max_tokens = max_tokens;
        self
    }

    /// Shared implementation of the summarization call: decide the kept rounds
    /// → send the compacted part to the summarizer → assemble
    /// [summary message, kept rounds].
    async fn trim_impl(
        &self,
        messages: &[Message],
        counts: &[usize],
        budget: &Budget,
        counter: &dyn TokenCounter,
        fallback: &WindowDrop,
    ) -> Result<TrimResult, MemoryError> {
        // Keep the most recent rounds: budget minus the reservation for the
        // summary output; at least 1 round (fallback).
        let rounds = split_rounds(messages, counts);
        let keep = keep_rounds(&rounds, budget, self.summary_max_tokens as usize);
        let cut = rounds[rounds.len() - keep].0;
        if cut == 0 {
            // Everything is kept (single round over budget, or within budget):
            // nothing to compact.
            return Ok(TrimResult {
                messages: messages.to_vec(),
                replace: false,
            });
        }

        let request = ChatRequest {
            messages: vec![
                Message::system(self.prompt.clone()),
                Message::user(messages_to_text(&messages[..cut])),
            ],
            tools: Vec::new(),
            options: ModelOptions {
                max_tokens: Some(self.summary_max_tokens),
                ..Default::default()
            },
        };
        let summary = match self.provider.chat(request).await {
            Ok(response) => {
                // The summarizer should reply with an Assistant message; other
                // roles degrade defensively.
                let Message::Assistant { content, .. } = &response.message else {
                    return fallback
                        .trim_with_counts(messages, counts, budget, counter)
                        .await;
                };
                if content.trim().is_empty() {
                    None
                } else {
                    Some(content.trim().to_string())
                }
            }
            Err(error) => {
                // On failure, degrade to window dropping (projection): the
                // conversation continues, storage keeps the original text, and
                // summarization is retried on the next over-budget retrieval.
                tracing::warn!(%error, "summarize failed, falling back to window drop");
                None
            }
        };
        let Some(summary) = summary else {
            return fallback
                .trim_with_counts(messages, counts, budget, counter)
                .await;
        };

        let mut result = Vec::with_capacity(keep + 1);
        result.push(Message::system(format!("{SUMMARY_PREFIX}{summary}")));
        result.extend_from_slice(&messages[cut..]);
        // Compaction event (observability): compacted count, kept count,
        // summary output token count.
        // The summary message's tokens count toward the budget and may leave
        // the materialized result over budget (the next retrieval compacts
        // again) — the log carries this info for diagnosis.
        let summary_tokens = count_message(counter, &result[0]).await?;
        tracing::info!(
            compressed = cut,
            kept = messages.len() - cut,
            summary_tokens,
            "context summarized by LLM"
        );
        Ok(TrimResult {
            messages: result,
            replace: true,
        })
    }
}

#[async_trait::async_trait]
impl TrimStrategy for SummarizeStrategy {
    async fn trim(
        &self,
        messages: &[Message],
        budget: &Budget,
        counter: &dyn TokenCounter,
    ) -> Result<TrimResult, MemoryError> {
        let mut counts = Vec::with_capacity(messages.len());
        for m in messages {
            counts.push(count_message(counter, m).await?);
        }
        self.trim_with_counts(messages, &counts, budget, counter)
            .await
    }

    async fn trim_with_counts(
        &self,
        messages: &[Message],
        counts: &[usize],
        budget: &Budget,
        counter: &dyn TokenCounter,
    ) -> Result<TrimResult, MemoryError> {
        if messages.is_empty() {
            return Ok(TrimResult {
                messages: Vec::new(),
                replace: false,
            });
        }
        debug_assert_eq!(messages.len(), counts.len());
        let fallback = WindowDrop;
        self.trim_impl(messages, counts, budget, counter, &fallback)
            .await
    }
}

/// Textifies a message sequence as input for the summarizer.
///
/// One `role: content` line per message: System verbatim, User joins content
/// blocks, Assistant takes the body plus tool calls (name + arguments; the
/// reasoning trace is not carried), ToolResult takes the result text.
/// The summary message (prior summary) appears here as a `system:` line, which
/// is how the summarizer recognizes it.
fn messages_to_text(messages: &[Message]) -> String {
    let mut lines = Vec::with_capacity(messages.len());
    for message in messages {
        match message {
            Message::System(s) => lines.push(format!("system: {s}")),
            Message::User(blocks) => {
                let text: String = blocks
                    .iter()
                    .map(|block| match block {
                        ContentBlock::Text(t) => t.clone(),
                        // The summary keeps a placeholder so the context
                        // "an image was there" survives compression (the
                        // bytes themselves cannot be summarized).
                        ContentBlock::Image(image) => format!("[image: {}]", image.mime_type),
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                lines.push(format!("user: {text}"));
            }
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                if tool_calls.is_empty() {
                    lines.push(format!("assistant: {content}"));
                } else {
                    let calls = tool_calls
                        .iter()
                        .map(|tc| format!("{} {}", tc.name, tc.arguments))
                        .collect::<Vec<_>>()
                        .join("; ");
                    lines.push(format!("assistant: {content}\ntool_calls: {calls}"));
                }
            }
            Message::ToolResult { content, .. } => lines.push(format!("tool_result: {content}")),
        }
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{CharTokenCounter, Memory, WindowMemory};
    use crate::message::ToolCall;
    use crate::provider::{FakeProvider, FakeReply, ProviderError};
    use std::sync::Arc;

    /// Builds a conversation (round `start` through round `n`; about 11 tokens
    /// per round).
    async fn record_rounds(memory: &mut WindowMemory, start: usize, n: usize) {
        for i in start..=n {
            memory
                .record(Message::user(format!("Question from round {i}")))
                .await
                .unwrap();
            memory
                .record(Message::assistant(format!("Answer from round {i}")))
                .await
                .unwrap();
        }
    }

    /// Builds the strategy with a shared instance: Arc<FakeProvider> implements
    /// Provider, so tests can assert the request history from the original
    /// instance.
    fn strategy(fake: Arc<FakeProvider>) -> Arc<SummarizeStrategy> {
        Arc::new(SummarizeStrategy::new(fake))
    }

    /// Basic compaction: over budget → [summary (System), most recent round];
    /// the summarizer only receives the compacted part (kept rounds excluded).
    #[tokio::test]
    async fn summarizes_old_rounds_and_keeps_recent() {
        let fake = Arc::new(FakeProvider::new([FakeReply::Text(
            "Key points from earlier rounds".into(),
        )]));
        let mut memory = WindowMemory::new(30).with_strategy(strategy(fake.clone()));
        record_rounds(&mut memory, 1, 4).await;

        let context = memory.context().await.unwrap();
        assert_eq!(context.len(), 3);
        assert_eq!(
            context[0],
            Message::system(format!("{SUMMARY_PREFIX}Key points from earlier rounds"))
        );
        assert_eq!(context[1], Message::user("Question from round 4"));
        assert_eq!(context[2], Message::assistant("Answer from round 4"));

        // Request = [prompt (system), compacted-part text (user)]; kept rounds
        // excluded.
        let requests = fake.requests();
        assert_eq!(requests.len(), 1);
        let messages = &requests[0].messages;
        assert!(
            matches!(&messages[0], Message::System(p) if p.contains("conversation-history compressor"))
        );
        let Message::User(blocks) = &messages[1] else {
            panic!("expected user message");
        };
        let ContentBlock::Text(text) = &blocks[0] else {
            panic!("expected a text block");
        };
        assert!(text.contains("Question from round 1") && text.contains("Question from round 3"));
        assert!(!text.contains("Question from round 4"));
    }

    /// The summary output cap doubles as the budget reservation for kept
    /// rounds: a smaller cap → more rounds kept.
    #[tokio::test]
    async fn summary_budget_reserves_space_for_recent_rounds() {
        let fake = Arc::new(FakeProvider::new([FakeReply::Text("Highlights".into())]));
        // Budget 30, summary cap 4: 26 tokens available for kept rounds. Each
        // round is 11 tokens ("Question/Answer from round N"): keep the recent
        // 2 rounds (22); adding a 3rd (33 > 26) stops. Result: [summary,
        // round 3, round 4].
        let mut memory = WindowMemory::new(30).with_strategy(Arc::new(
            SummarizeStrategy::new(fake.clone()).with_summary_max_tokens(4),
        ));
        record_rounds(&mut memory, 1, 4).await;

        let context = memory.context().await.unwrap();
        assert_eq!(context.len(), 5);
        assert_eq!(context[1], Message::user("Question from round 3"));
        assert_eq!(context[3], Message::user("Question from round 4"));
    }

    /// Incremental compaction: on the second over-budget event, the previous
    /// summary is merged into the new one as input.
    #[tokio::test]
    async fn incremental_summary_reuses_previous_summary() {
        let fake = Arc::new(FakeProvider::new([
            FakeReply::Text("first segment highlights".into()),
            FakeReply::Text("merged highlights".into()),
        ]));
        let mut memory = WindowMemory::new(30).with_strategy(Arc::new(
            SummarizeStrategy::new(fake.clone()).with_summary_max_tokens(4),
        ));
        record_rounds(&mut memory, 1, 4).await;
        memory.context().await.unwrap(); // first compaction → [summary, u3, a3, u4, a4]

        // Append rounds 5 and 6: over budget again → second compaction.
        record_rounds(&mut memory, 5, 6).await;
        let context = memory.context().await.unwrap();
        assert_eq!(
            context[0],
            Message::system(format!("{SUMMARY_PREFIX}merged highlights"))
        );
        // Budget 30 minus reserve 4 → 26: keep the 2 most recent rounds
        // (11 + 11); the earlier messages (including the old summary) go to the
        // second summarization. Result: [new summary, u5, a5, u6, a6].
        assert_eq!(context.len(), 5);
        assert_eq!(context[1], Message::user("Question from round 5"));

        // The second request's input text contains the first summary message
        // (incremental merge).
        let requests = fake.requests();
        assert_eq!(requests.len(), 2);
        let Message::User(blocks) = &requests[1].messages[1] else {
            panic!("expected user message");
        };
        let ContentBlock::Text(text) = &blocks[0] else {
            panic!("expected a text block");
        };
        assert!(text.contains("first segment highlights"));
    }

    /// Failure degradation: summarizer call fails → window-dropped view
    /// (projection, storage unchanged), the conversation continues; the next
    /// over-budget retrieval retries summarization.
    #[tokio::test]
    async fn provider_failure_falls_back_to_window_drop() {
        let fake = Arc::new(FakeProvider::new([
            FakeReply::Error(ProviderError::Network("mock network error".into())),
            FakeReply::Text("second attempt succeeded".into()),
        ]));
        let mut memory = WindowMemory::new(30).with_strategy(strategy(fake.clone()));
        record_rounds(&mut memory, 1, 4).await;

        // First time: failure → degrade to window dropping (budget 30 keeps the
        // 2 most recent rounds, no summary message).
        let context = memory.context().await.unwrap();
        assert_eq!(
            context,
            vec![
                Message::user("Question from round 3"),
                Message::assistant("Answer from round 3"),
                Message::user("Question from round 4"),
                Message::assistant("Answer from round 4"),
            ]
        );

        // Storage was not materialized: still over budget on the second call →
        // the strategy runs again, this time succeeding.
        let context = memory.context().await.unwrap();
        assert_eq!(
            context[0],
            Message::system(format!("{SUMMARY_PREFIX}second attempt succeeded"))
        );
        assert_eq!(fake.requests().len(), 2);
    }

    /// Empty summary output (the model produced no text) → likewise falls back
    /// to window dropping.
    #[tokio::test]
    async fn empty_summary_falls_back() {
        let fake = Arc::new(FakeProvider::new([FakeReply::Text("".into())]));
        let mut memory = WindowMemory::new(30).with_strategy(strategy(fake.clone()));
        record_rounds(&mut memory, 1, 4).await;

        let context = memory.context().await.unwrap();
        assert_eq!(context.len(), 4); // window dropping: budget 30 keeps the 2 most recent rounds
        assert!(!context.iter().any(|m| matches!(m, Message::System(_))));
    }

    /// Single-round over-budget fallback: nothing to compact, returns as-is
    /// without calling the summarizer.
    #[tokio::test]
    async fn single_round_over_budget_returns_as_is() {
        let fake = Arc::new(FakeProvider::new([FakeReply::Text("summary".into())]));
        let mut memory = WindowMemory::new(5).with_strategy(strategy(fake.clone()));
        memory
            .record(Message::user(
                "An extremely long user message, over budget in a single round",
            ))
            .await
            .unwrap();
        memory.record(Message::assistant("Reply")).await.unwrap();

        let context = memory.context().await.unwrap();
        assert_eq!(context.len(), 2);
        assert!(
            fake.requests().is_empty(),
            "with nothing to compact the summarizer must not be called"
        );
    }

    /// Defensive: an empty message list is returned as-is.
    #[tokio::test]
    async fn empty_input_returns_empty() {
        let fake = Arc::new(FakeProvider::new([FakeReply::Text("summary".into())]));
        let strategy = strategy(fake);
        let result = strategy
            .trim(&[], &Budget::tokens(10), &CharTokenCounter)
            .await
            .unwrap();
        assert!(result.messages.is_empty());
        assert!(!result.replace);
    }

    /// Tool-call round textification: the assistant message carries the tool
    /// name and arguments, with the tool result on its own line.
    #[tokio::test]
    async fn messages_to_text_includes_tool_calls_and_results() {
        let messages = vec![
            Message::system("setup"),
            Message::user("calculate"),
            Message::Assistant {
                content: "".into(),
                reasoning: Some("reasoning trace is not included".into()),
                tool_calls: vec![ToolCall {
                    id: "t1".into(),
                    name: "calc".into(),
                    arguments: r#"{"expr":"1+1"}"#.into(),
                }],
            },
            Message::tool_result("t1", "2"),
        ];
        let text = messages_to_text(&messages);
        assert!(text.contains("system: setup"));
        assert!(text.contains("user: calculate"));
        assert!(text.contains(r#"tool_calls: calc {"expr":"1+1"}"#));
        assert!(text.contains("tool_result: 2"));
        // The reasoning trace never enters the summary input.
        assert!(!text.contains("reasoning trace is not included"));
    }

    /// The textified input counts against the kept-round budget (same
    /// convention as window trimming).
    #[tokio::test]
    async fn trim_counts_consistent_with_window() {
        let fake = Arc::new(FakeProvider::new([FakeReply::Text("summary".into())]));
        let strategy = strategy(fake);
        let messages = vec![
            Message::user("u1"),
            Message::assistant("a1"),
            Message::user("u2"),
            Message::assistant("a2"),
        ];
        let counter = CharTokenCounter;
        let result = strategy
            .trim_with_counts(&messages, &[1, 1, 1, 1], &Budget::tokens(2), &counter)
            .await
            .unwrap();
        // Budget 2 − reserve 1024 → only 1 round kept: summary + u2/a2.
        assert_eq!(result.messages.len(), 3);
        assert!(result.replace);
    }
}
