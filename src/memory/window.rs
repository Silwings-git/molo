//! Window memory: an in-memory implementation with a budget and trim strategy.
//!
//! When the conversation history exceeds the budget, the retrieved context is
//! trimmed to a recent window so the model's context window is never exceeded.
//! This module provides three pieces:
//!
//! - [`Budget`] — a two-dimensional budget of token and round limits;
//! - [`TrimStrategy`] — the trim strategy trait; the default [`WindowDrop`]
//!   drops the earliest messages by round; heavier strategies such as
//!   summarization are injected by the user;
//! - [`WindowMemory`] — the [`Memory`] implementation combining budget,
//!   counting, and strategy.
//!
//! This implementation does not manage System messages: the system prompt is
//! the Agent layer's responsibility (assembled per request); Memory only
//! manages the conversation history. If a user records System messages
//! themselves, trimming treats them like any other message, with no special
//! case.

use std::fmt;
use std::sync::{Arc, Mutex};

use super::{Memory, MemoryError};
use crate::message::{ContentBlock, Message};

/// Window budget: token and round limits (both optional; when both are set, the
/// smaller window wins).
///
/// - `max_tokens`: total token budget for the context ([`WindowMemory::new`]
///   requires it);
/// - `max_rounds`: maximum number of complete rounds to keep. A round is one
///   User message plus everything after it until the next User message (a tool
///   message belongs to the same round as its Assistant).
///
/// The two dimensions can be used independently or together: when both are
/// set, the kept result is the smaller window satisfying **both** (trim by
/// tokens first, then constrained by rounds).
///
/// # Example
///
/// ```rust
/// use molo::memory::Budget;
///
/// // Token-only budget: window of at most 4096 tokens, no round limit.
/// let by_tokens = Budget::tokens(4096);
/// // Rounds-only budget: keep at most 8 rounds.
/// let by_rounds = Budget::rounds(8);
/// // Both set: the smaller window wins.
/// let both = Budget::both(4096, 8);
///
/// assert_eq!(by_tokens.max_tokens, Some(4096));
/// assert_eq!(by_tokens.max_rounds, None);
/// assert_eq!(by_rounds.max_rounds, Some(8));
/// assert!(both.max_tokens.is_some() && both.max_rounds.is_some());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// Token budget limit; `None` = unlimited.
    pub max_tokens: Option<usize>,
    /// Round limit; `None` = unlimited.
    pub max_rounds: Option<usize>,
}

impl Default for Budget {
    /// Unlimited in both dimensions (consistent with "no trimming by default").
    fn default() -> Self {
        Self {
            max_tokens: None,
            max_rounds: None,
        }
    }
}

impl Budget {
    /// Token-only budget (no round limit).
    pub fn tokens(max_tokens: usize) -> Self {
        Self {
            max_tokens: Some(max_tokens),
            max_rounds: None,
        }
    }

    /// Rounds-only budget (no token limit).
    ///
    /// `max_rounds == 0` is treated as 1 (the window always keeps the most
    /// recent round, avoiding an empty window).
    pub fn rounds(max_rounds: usize) -> Self {
        Self {
            max_tokens: None,
            max_rounds: Some(max_rounds),
        }
    }

    /// Sets both the token budget and the round limit (the smaller window wins
    /// when both are set).
    ///
    /// `max_rounds == 0` is treated as 1 (see [`rounds`](Budget::rounds)).
    pub fn both(max_tokens: usize, max_rounds: usize) -> Self {
        Self {
            max_tokens: Some(max_tokens),
            max_rounds: Some(max_rounds),
        }
    }
}

/// Counts the token number of a text.
///
/// Memory does not know the model; "exact" token counts are user-side
/// knowledge, so this trait carries no model or provider information:
/// - single-model setups: inject a counter built for that model when
///   constructing [`WindowMemory`] (e.g., wiring in tiktoken-rs); the counting
///   convention is bound at construction;
/// - multi-model dynamic routing: the custom implementation holds shared state
///   internally and switches conventions when the application switches models.
///
/// The default implementation [`CharTokenCounter`] is a heuristic
/// approximation that depends on no model.
///
/// The trait is `async`: it supports remote counting (e.g., calling a vendor's
/// counting API); local implementations just return `Ok(approx)`. All call
/// sites (record / context / trim) are already async, so remote counting costs
/// nothing extra.
///
/// # Example
///
/// Inject a custom counter (e.g., trimming by message count):
///
/// ```rust
/// use molo::memory::{Memory, MemoryError, TokenCounter, WindowMemory};
/// use molo::Message;
///
/// // Each message counts as exactly 1 token: the token budget degenerates to
/// // a message-count limit.
/// #[derive(Default)]
/// struct OnePerMessage;
///
/// #[molo::async_trait]
/// impl TokenCounter for OnePerMessage {
///     async fn count(&self, _text: &str) -> Result<usize, MemoryError> {
///         Ok(1)
///     }
/// }
///
/// #[tokio::main]
/// async fn main() -> Result<(), MemoryError> {
///     let mut memory =
///         WindowMemory::new(2).with_token_counter(Box::new(OnePerMessage));
///     memory.record(Message::user("u1")).await?;
///     memory.record(Message::assistant("a1")).await?;
///     memory.record(Message::user("u2")).await?;
///     memory.record(Message::assistant("a2")).await?;
///
///     // 4 messages > 2 tokens: trimmed to the most recent round.
///     assert_eq!(memory.context().await?.len(), 2);
///     Ok(())
/// }
/// ```
#[async_trait::async_trait]
pub trait TokenCounter: Send + Sync {
    /// Counts the token number of a text.
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError::TokenCount`] when counting fails (e.g., a remote
    /// counting API is unavailable).
    async fn count(&self, text: &str) -> Result<usize, MemoryError>;
}

/// Default counter: CJK characters count as 1 token each, other characters
/// count as 1 token per 4 characters (rounded up).
///
/// The common chars/4 approximation underestimates Chinese by roughly 4x
/// (Chinese is roughly 1 token per character), so this implementation
/// distinguishes CJK to stay closer to reality; it is still an approximation —
/// inject a custom implementation when exact counts are needed.
///
/// # Example
///
/// ```rust
/// # #[tokio::main]
/// # async fn main() -> Result<(), molo::memory::MemoryError> {
/// use molo::memory::{CharTokenCounter, TokenCounter};
///
/// let counter = CharTokenCounter;
/// // 5 characters → 2 tokens (1 token per 4 characters, rounded up).
/// assert_eq!(counter.count("hello").await?, 2);
/// // 4 characters → exactly 1 token.
/// assert_eq!(counter.count("abcd").await?, 1);
/// // 7 characters → 2 tokens (rounded up).
/// assert_eq!(counter.count("morning").await?, 2);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CharTokenCounter;

#[async_trait::async_trait]
impl TokenCounter for CharTokenCounter {
    async fn count(&self, text: &str) -> Result<usize, MemoryError> {
        let (mut cjk, mut other) = (0usize, 0usize);
        for c in text.chars() {
            if is_cjk(c) {
                cjk += 1;
            } else {
                other += 1;
            }
        }
        Ok(cjk + other.div_ceil(4))
    }
}

/// CJK ideographs (unified ideographs plus extensions A–F and compatibility
/// ideographs; kana / Hangul excluded).
fn is_cjk(c: char) -> bool {
    matches!(
        c as u32,
        0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0x20000..=0x2FA1F
    )
}

/// Token count of a message: the sum over all its text blocks.
///
/// The Assistant's `reasoning` and `tool_calls` arguments are counted too —
/// they are sent back to the API verbatim and occupy the window; thinking-model
/// reasoning is long, and undercounting would really blow the limit.
pub(crate) async fn count_message(
    counter: &dyn TokenCounter,
    message: &Message,
) -> Result<usize, MemoryError> {
    match message {
        Message::System(s) => counter.count(s).await,
        Message::User(blocks) => {
            let mut total = 0usize;
            for b in blocks {
                match b {
                    ContentBlock::Text(t) => total += counter.count(t).await?,
                    // Images and pass-through blocks carry no text to count;
                    // the message stays in the window whole so the content
                    // still reaches the provider.
                    ContentBlock::Image(_) | ContentBlock::Wire(_) => {}
                }
            }
            Ok(total)
        }
        Message::Assistant {
            content,
            reasoning,
            tool_calls,
        } => {
            let mut total = counter.count(content).await?;
            if let Some(r) = reasoning {
                total += counter.count(r).await?;
            }
            for tc in tool_calls {
                total += counter.count(&tc.arguments).await?;
            }
            Ok(total)
        }
        Message::ToolResult { content, .. } => counter.count(content).await,
    }
}

/// Output of a trim strategy: the trimmed message sequence + how the result is
/// handled.
///
/// # Comparison: materialization vs. projection
///
/// - `replace: true` (**materialized**): the result is written back to
///   storage; the replaced old messages are no longer kept. Subsequent
///   retrievals recompute nothing until new messages breach the budget again
///   (the strategy input = previous result + new messages). Heavy operations
///   such as LLM summarization should pick this semantic to avoid
///   recomputation every turn.
/// - `replace: false` (**projection**): the result is only visible for this
///   retrieval; storage is unchanged and nothing is lost — raising the budget
///   restores all trimmed history. The cost is recomputation whenever over
///   budget, so it suits cheap operations (e.g., window dropping).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrimResult {
    /// The trimmed message sequence.
    pub messages: Vec<Message>,
    /// Whether the result is materialized back into storage (semantics in the
    /// type docs).
    pub replace: bool,
}

/// Trim strategy: decides "how to trim" — window dropping, summarization, LLM
/// compaction, etc. Users can inject custom implementations.
///
/// The strategy is fully responsible for the trim result: the output must
/// satisfy the message interface constraints (role alternation etc.), and
/// round boundaries are the strategy's own responsibility; `budget` and
/// `counter` are provided for the strategy to compute how much to compact.
#[async_trait::async_trait]
pub trait TrimStrategy: Send + Sync {
    /// Trims a message sequence against a budget and counter, returning the
    /// trim result (semantics in [`TrimResult`]).
    ///
    /// # Errors
    ///
    /// Returns [`MemoryError`] when counting or trimming fails (e.g., remote
    /// counting is unavailable).
    async fn trim(
        &self,
        messages: &[Message],
        budget: &Budget,
        counter: &dyn TokenCounter,
    ) -> Result<TrimResult, MemoryError>;

    /// Trims with per-message token counts; the default implementation
    /// delegates to [`trim`](TrimStrategy::trim), so custom strategies need not
    /// know about it. `WindowDrop` overrides it to amortized O(1) — counts are
    /// cached by Memory, avoiding per-message recounts inside the strategy (and
    /// saving IO with remote counters).
    async fn trim_with_counts(
        &self,
        messages: &[Message],
        _counts: &[usize],
        budget: &Budget,
        counter: &dyn TokenCounter,
    ) -> Result<TrimResult, MemoryError> {
        self.trim(messages, budget, counter).await
    }
}

/// Default trim strategy: drops the earliest complete rounds until the
/// remaining sequence fits the budget.
///
/// Uses projection semantics (`replace: false`): lossless — raising the budget
/// restores the trimmed history, and each recomputation is cheap. Guarantees:
///
/// - at least the most recent round is kept (kept even when a single round
///   exceeds the budget; nothing more can be trimmed);
/// - a tool message and its Assistant message share a round and are kept as a
///   pair;
/// - recorded System messages are treated like any other message and may be
///   trimmed (the system prompt is managed by the Agent layer; Memory does not
///   manage System).
///
/// # Comparison
///
/// When a heavy strategy such as summarization or LLM compaction is needed,
/// inject a custom [`TrimStrategy`] that declares materialization
/// (`replace: true`), trading the materialized zero-recompute for a longer
/// context; light and lossless round-wise dropping is this implementation.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WindowDrop;

#[async_trait::async_trait]
impl TrimStrategy for WindowDrop {
    async fn trim(
        &self,
        messages: &[Message],
        budget: &Budget,
        counter: &dyn TokenCounter,
    ) -> Result<TrimResult, MemoryError> {
        // Fall back to the self-counting async path (when counts are not
        // cached).
        let mut counts = Vec::with_capacity(messages.len());
        for m in messages {
            counts.push(count_message(counter, m).await?);
        }
        Ok(TrimResult {
            messages: window_from_counts(messages, &counts, budget),
            replace: false,
        })
    }

    async fn trim_with_counts(
        &self,
        messages: &[Message],
        counts: &[usize],
        budget: &Budget,
        _counter: &dyn TokenCounter,
    ) -> Result<TrimResult, MemoryError> {
        Ok(TrimResult {
            messages: window_from_counts(messages, counts, budget),
            replace: false,
        })
    }
}

/// A round's boundaries and token total: (start index, end index, tokens).
/// A round = one User message plus everything after it until the next User;
/// a leading non-User message (defensive) belongs to the first round.
pub(crate) type Round = (usize, usize, usize);

/// Splits a message sequence into complete rounds (shared convention for both
/// window trimming and summarization).
pub(crate) fn split_rounds(messages: &[Message], counts: &[usize]) -> Vec<Round> {
    debug_assert_eq!(messages.len(), counts.len());
    if messages.is_empty() {
        return Vec::new();
    }
    let mut rounds = Vec::new();
    let mut start = 0usize;
    let mut tokens = counts[0];
    for (i, message) in messages.iter().enumerate().skip(1) {
        if matches!(message, Message::User(_)) {
            rounds.push((start, i, tokens));
            start = i;
            tokens = counts[i];
        } else {
            tokens += counts[i];
        }
    }
    rounds.push((start, messages.len(), tokens));
    rounds
}

/// Keeps complete rounds from the tail forward, returning the number of rounds
/// kept (at least 1; a single over-budget round is still kept); `reserved` is
/// the portion of the budget set aside in advance (e.g., room for the summary
/// output).
///
/// Rules: the most recent round is kept unconditionally; later rounds are kept
/// only while "adding them stays within budget"; `max_rounds` then caps the
/// count (at most the recent N rounds, at least 1).
pub(crate) fn keep_rounds(rounds: &[Round], budget: &Budget, reserved: usize) -> usize {
    let mut keep = 0usize;
    let mut sum = 0usize;
    for &(_, _, t) in rounds.iter().rev() {
        if keep > 0
            && let Some(limit) = budget.max_tokens
            && sum + t > limit.saturating_sub(reserved)
        {
            break;
        }
        sum += t;
        keep += 1;
    }
    if let Some(limit) = budget.max_rounds {
        keep = keep.min(limit.max(1));
    }
    keep
}

/// Window view: trims from the earliest complete round, returning the kept
/// sequence (at least the most recent round).
/// Pure-synchronous variant: per-message token counts come from the caller
/// (amortized O(1) on cache hit).
fn window_from_counts(messages: &[Message], counts: &[usize], budget: &Budget) -> Vec<Message> {
    if messages.is_empty() {
        return Vec::new();
    }
    let rounds = split_rounds(messages, counts);
    let keep = keep_rounds(&rounds, budget, 0);
    let start_index = rounds[rounds.len() - keep].0;
    messages[start_index..].to_vec()
}

/// Window memory: stores all messages; `context()` trims to the budget on
/// retrieval.
///
/// - Within budget: returns a full clone as-is;
/// - Over budget: calls the [`TrimStrategy`] — the default [`WindowDrop`]
///   drops the earliest messages by round (projection, lossless); an injected
///   heavy strategy can declare `replace: true` to materialize (write back to
///   storage, zero recomputation afterwards).
///
/// Counting is lazy: after a materialized write-back nothing is counted until
/// the first async call (record / context) recomputes in one pass —
/// [`TokenCounter`] may involve remote IO, and the synchronous path never
/// touches it. Materialization happens inside [`context`](Memory::context) on
/// `&self`, using an internal `std::sync::Mutex` for interior mutability; the
/// critical section contains no await, so single-threaded runtimes are not
/// blocked.
///
/// # Comparison
///
/// Compared to [`InMemoryMemory`](crate::memory::InMemoryMemory): the latter
/// stores all messages verbatim and never trims, suiting small conversations
/// without budget control; use this type when context size must be controlled
/// (the model's context window is finite).
///
/// # Panics
///
/// Panics only when the internal lock is poisoned (a panic occurred while
/// holding it); never in normal use.
///
/// # Examples
///
/// Basic usage:
///
/// ```rust
/// # #[tokio::main]
/// # async fn main() -> Result<(), molo::memory::MemoryError> {
/// use molo::memory::{Memory, WindowMemory};
///
/// let mut memory = WindowMemory::new(100);
/// memory.record(molo::message::Message::user("hello")).await?;
/// assert_eq!(memory.context().await?.len(), 1);
/// # Ok(())
/// # }
/// ```
///
/// Trims by round when over budget, keeping at least the most recent round:
///
/// ```rust
/// # #[tokio::main]
/// # async fn main() -> Result<(), molo::memory::MemoryError> {
/// use molo::memory::{Memory, WindowMemory};
///
/// let mut memory = WindowMemory::new(3);
/// memory.record(molo::message::Message::user("u1")).await?;
/// memory.record(molo::message::Message::assistant("a1")).await?;
/// memory.record(molo::message::Message::user("u2")).await?;
/// memory.record(molo::message::Message::assistant("a2")).await?;
///
/// // 4 tokens total, over the 3-token budget: the earliest round is dropped.
/// let context = memory.context().await?;
/// assert_eq!(context.len(), 2);
/// assert_eq!(context[0], molo::message::Message::user("u2"));
/// # Ok(())
/// # }
/// ```
pub struct WindowMemory {
    inner: Mutex<WindowInner>,
    budget: Budget,
    counter: Box<dyn TokenCounter>,
    strategy: Arc<dyn TrimStrategy>,
}

struct WindowInner {
    messages: Vec<Message>,
    /// Protection flags parallel to `messages`: protected messages (e.g.,
    /// skill bodies) are exempt from trimming.
    protected: Vec<bool>,
    /// Per-message token counts parallel to `messages` (used directly by the
    /// default strategy's trim, avoiding rescans; invalidated together with
    /// `total_tokens`).
    tokens: Vec<usize>,
    /// Total token count of all messages; `None` = not yet counted (after a
    /// materialized write-back / after swapping the counter), recomputed on the
    /// first async call.
    total_tokens: Option<usize>,
    user_count: usize,
}

impl fmt::Debug for WindowMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The counter and strategy are trait objects and cannot derive Debug:
        // print the budget and message count, and mark the other fields' types.
        let message_count = self
            .inner
            .lock()
            .expect("WindowMemory internal lock poisoned")
            .messages
            .len();
        f.debug_struct("WindowMemory")
            .field("budget", &self.budget)
            .field("message_count", &message_count)
            .field("counter", &"Box<dyn TokenCounter>")
            .field("strategy", &"Arc<dyn TrimStrategy>")
            .finish()
    }
}

impl WindowMemory {
    /// Creates an empty session with a context budget of `max_tokens` tokens.
    ///
    /// Counting and trimming default to [`CharTokenCounter`] and
    /// [`WindowDrop`]; replace them with
    /// [`with_token_counter`](WindowMemory::with_token_counter) and
    /// [`with_strategy`](WindowMemory::with_strategy) respectively.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            inner: Mutex::new(WindowInner {
                messages: Vec::new(),
                protected: Vec::new(),
                tokens: Vec::new(),
                total_tokens: Some(0),
                user_count: 0,
            }),
            budget: Budget::tokens(max_tokens),
            counter: Box::new(CharTokenCounter),
            strategy: Arc::new(WindowDrop),
        }
    }

    /// Adds a "keep at most the recent N rounds" limit (unlimited by default;
    /// the smaller window wins when set together with the token budget).
    pub fn with_max_rounds(mut self, max_rounds: usize) -> Self {
        self.budget.max_rounds = Some(max_rounds);
        self
    }

    /// Replaces the default heuristic counting with exact counting (matching
    /// your own model).
    ///
    /// Swapping changes the counting convention and invalidates previously
    /// counted results: the first async call (record / context) recomputes
    /// with the new counter.
    pub fn with_token_counter(mut self, counter: Box<dyn TokenCounter>) -> Self {
        self.counter = counter;
        // The counting convention changed: invalidate and recompute lazily via
        // ensure_counts.
        let mut inner = self
            .inner
            .lock()
            .expect("WindowMemory internal lock poisoned");
        inner.total_tokens = None;
        drop(inner);
        self
    }

    /// Injects a custom trim strategy (summarization, LLM compaction, etc.);
    /// defaults to [`WindowDrop`].
    ///
    /// # Example
    ///
    /// A summarization strategy: replaces everything but the most recent round
    /// with a summary message and materializes it (`replace: true`):
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use molo::memory::{
    ///     Budget, Memory, MemoryError, TokenCounter, TrimResult, TrimStrategy, WindowMemory,
    /// };
    /// use molo::Message;
    ///
    /// #[derive(Default)]
    /// struct Summarize;
    ///
    /// #[molo::async_trait]
    /// impl TrimStrategy for Summarize {
    ///     async fn trim(
    ///         &self,
    ///         messages: &[Message],
    ///         _budget: &Budget,
    ///         _counter: &dyn TokenCounter,
    ///     ) -> Result<TrimResult, MemoryError> {
    ///         // Keep the most recent round; replace earlier messages with a
    ///         // summary message.
    ///         // The summary uses the System role so no consecutive User
    ///         // messages appear after replacement (role-alternation
    ///         // constraint).
    ///         let pos = messages
    ///             .iter()
    ///             .rposition(|m| matches!(m, Message::User(_)))
    ///             .unwrap_or(0);
    ///         let mut result = Vec::with_capacity(messages.len() - pos + 1);
    ///         result.push(Message::system("prior summary"));
    ///         result.extend_from_slice(&messages[pos..]);
    ///         Ok(TrimResult { messages: result, replace: true })
    ///     }
    /// }
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let mut memory = WindowMemory::new(7).with_strategy(Arc::new(Summarize));
    ///     for i in 1..=4 {
    ///         memory.record(Message::user(format!("u{i}"))).await.unwrap();
    ///         memory.record(Message::assistant(format!("a{i}"))).await.unwrap();
    ///     }
    ///
    ///     // 8 > 7: compacted to [summary, most recent round].
    ///     let context = memory.context().await.unwrap();
    ///     assert_eq!(context.len(), 3);
    ///     assert!(matches!(context[0], Message::System(_)));
    /// }
    /// ```
    pub fn with_strategy(mut self, strategy: Arc<dyn TrimStrategy>) -> Self {
        self.strategy = strategy;
        self
    }

    /// Adjusts the token budget at runtime; takes effect on the next context
    /// retrieval.
    ///
    /// With a projection strategy, raising the budget restores the trimmed
    /// history; with a materializing strategy, old messages are gone and a
    /// higher budget cannot restore them.
    pub fn set_max_tokens(&mut self, max_tokens: usize) {
        self.budget.max_tokens = Some(max_tokens);
    }

    /// Adjusts the round limit at runtime (`None` = unlimited); takes effect
    /// on the next context retrieval.
    pub fn set_max_rounds(&mut self, max_rounds: Option<usize>) {
        self.budget.max_rounds = max_rounds;
    }

    /// Lazily recomputes counts (when not counted): awaits outside the lock
    /// (remote counting may do IO), writes back inside it.
    async fn ensure_counts(&self) -> Result<(), MemoryError> {
        let missing: Option<Vec<Message>> = {
            let inner = self
                .inner
                .lock()
                .expect("WindowMemory internal lock poisoned");
            if inner.total_tokens.is_none() {
                Some(inner.messages.clone())
            } else {
                None
            }
        };
        let Some(messages) = missing else {
            return Ok(());
        };
        if messages.is_empty() {
            let mut inner = self
                .inner
                .lock()
                .expect("WindowMemory internal lock poisoned");
            inner.total_tokens = Some(0);
            return Ok(());
        }
        let mut total = 0usize;
        let mut tokens = Vec::with_capacity(messages.len());
        for m in &messages {
            let t = count_message(&*self.counter, m).await?;
            total += t;
            tokens.push(t);
        }
        let user_count = messages
            .iter()
            .filter(|m| matches!(m, Message::User(_)))
            .count();
        let mut inner = self
            .inner
            .lock()
            .expect("WindowMemory internal lock poisoned");
        inner.total_tokens = Some(total);
        inner.tokens = tokens;
        inner.user_count = user_count;
        Ok(())
    }
}

#[async_trait::async_trait]
impl Memory for WindowMemory {
    async fn record(&mut self, message: Message) -> Result<(), MemoryError> {
        self.record_impl(message, false).await
    }

    async fn record_protected(&mut self, message: Message) -> Result<(), MemoryError> {
        self.record_impl(message, true).await
    }

    async fn context(&self) -> Result<Vec<Message>, MemoryError> {
        self.ensure_counts().await?;
        // Snapshot and budget check: the critical section only clones, no
        // await.
        let (over_budget, snapshot, protected, tokens) = {
            let inner = self
                .inner
                .lock()
                .expect("WindowMemory internal lock poisoned");
            let total = inner
                .total_tokens
                .expect("counts guaranteed by ensure_counts");
            let over = self.budget.max_tokens.is_some_and(|limit| total > limit)
                || self
                    .budget
                    .max_rounds
                    .is_some_and(|limit| inner.user_count > limit);
            (
                over,
                inner.messages.clone(),
                inner.protected.clone(),
                inner.tokens.clone(),
            )
        };
        if !over_budget {
            return Ok(snapshot);
        }

        // Rounds containing protected messages are exempt as a whole: they are
        // pulled out of the candidate set, and the strategy only handles
        // trimmable messages (transparent to the strategy; custom strategies
        // get the exemption for free). The pulled-out part goes first in the
        // result, naturally aligned with the window trim's "keep recent
        // rounds" tail semantics.
        let protected_set: std::collections::HashSet<usize> =
            protected_round_indices(&snapshot, &protected)
                .into_iter()
                .collect();
        let mut kept: Vec<Message> = Vec::with_capacity(snapshot.len());
        let mut candidates: Vec<Message> = Vec::new();
        let mut candidate_tokens: Vec<usize> = Vec::new();
        for (i, message) in snapshot.into_iter().enumerate() {
            if protected_set.contains(&i) {
                kept.push(message);
            } else {
                candidates.push(message);
                candidate_tokens.push(tokens[i]);
            }
        }

        let result = self
            .strategy
            .trim_with_counts(&candidates, &candidate_tokens, &self.budget, &*self.counter)
            .await?;
        if result.replace {
            // Materialize: write back to storage (protected part + trimmed
            // candidates); the replaced old messages are no longer kept; counts
            // are invalidated and recomputed next time.
            let protected_len = kept.len();
            kept.extend(result.messages);
            let mut inner = self
                .inner
                .lock()
                .expect("WindowMemory internal lock poisoned");
            inner.messages = kept.clone();
            inner.protected = (0..kept.len()).map(|i| i < protected_len).collect();
            inner.total_tokens = None;
            inner.tokens.clear();
            inner.user_count = 0;
        } else {
            kept.extend(result.messages);
        }
        Ok(kept)
    }
}

impl WindowMemory {
    /// Shared implementation of record / record_protected: count + append +
    /// protection flag.
    async fn record_impl(&mut self, message: Message, protected: bool) -> Result<(), MemoryError> {
        self.ensure_counts().await?;
        let tokens = count_message(&*self.counter, &message).await?;
        let mut inner = self
            .inner
            .lock()
            .expect("WindowMemory internal lock poisoned");
        let total = inner
            .total_tokens
            .as_mut()
            .expect("counts guaranteed by ensure_counts");
        *total += tokens;
        if matches!(message, Message::User(_)) {
            inner.user_count += 1;
        }
        inner.messages.push(message);
        inner.protected.push(protected);
        inner.tokens.push(tokens);
        Ok(())
    }
}

/// Indices of the rounds that contain protected messages (a round = one User
/// message plus everything until the next User; a leading non-User message
/// belongs to the first round — same round-splitting convention as window
/// trimming).
fn protected_round_indices(messages: &[Message], protected: &[bool]) -> Vec<usize> {
    let mut result = Vec::new();
    let mut round_start = 0usize;
    for (i, message) in messages.iter().enumerate().skip(1) {
        if matches!(message, Message::User(_)) {
            if protected[round_start..i].iter().any(|&p| p) {
                result.extend(round_start..i);
            }
            round_start = i;
        }
    }
    if protected[round_start..].iter().any(|&p| p) {
        result.extend(round_start..messages.len());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{ContentBlock, ToolCall};

    fn tool_result(id: &str, content: &str) -> Message {
        Message::ToolResult {
            id: id.into(),
            content: content.into(),
        }
    }

    /// Test fake summarization strategy: replaces everything but the most
    /// recent round with a summary message (materializing); records the call
    /// count and the message count of the last input.
    #[derive(Debug, Default)]
    struct FakeSummarizer {
        calls: std::sync::atomic::AtomicUsize,
        last_input_len: std::sync::atomic::AtomicUsize,
        last_input_has_summary: std::sync::atomic::AtomicBool,
    }

    #[async_trait::async_trait]
    impl TrimStrategy for FakeSummarizer {
        async fn trim(
            &self,
            messages: &[Message],
            _budget: &Budget,
            _counter: &dyn TokenCounter,
        ) -> Result<TrimResult, MemoryError> {
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.last_input_len
                .store(messages.len(), std::sync::atomic::Ordering::Relaxed);
            let has_summary = messages
                .iter()
                .any(|m| matches!(m, Message::System(s) if s.starts_with("prior summary")));
            self.last_input_has_summary
                .store(has_summary, std::sync::atomic::Ordering::Relaxed);

            // Keep the most recent round; replace earlier messages with a
            // summary message.
            // The summary uses the System role: avoids consecutive User
            // messages after replacement (wire role-alternation constraint).
            let result = match messages.iter().rposition(|m| matches!(m, Message::User(_))) {
                Some(pos) => {
                    let mut result = Vec::with_capacity(messages.len() - pos + 1);
                    result.push(Message::system("prior summary"));
                    result.extend_from_slice(&messages[pos..]);
                    result
                }
                None => messages.to_vec(),
            };
            Ok(TrimResult {
                messages: result,
                replace: true,
            })
        }
    }

    #[tokio::test]
    async fn within_budget_returns_all() {
        let mut memory = WindowMemory::new(1000);
        memory.record(Message::user("hello")).await.unwrap();
        memory.record(Message::assistant("hello!")).await.unwrap();

        let context = memory.context().await.unwrap();
        assert_eq!(context.len(), 2);
        assert_eq!(context[0], Message::user("hello"));
        assert_eq!(context[1], Message::assistant("hello!"));
    }

    /// Over budget: drops the earliest complete round, keeps the most recent
    /// one; the result starts with a User message.
    #[tokio::test]
    async fn drops_earliest_rounds_when_over_budget() {
        let mut memory = WindowMemory::new(3);
        // Each round is about 2 tokens (ASCII: 4 chars = 1 token): User 1 +
        // Asst 1.
        memory.record(Message::user("u1")).await.unwrap();
        memory.record(Message::assistant("a1")).await.unwrap();
        memory.record(Message::user("u2")).await.unwrap();
        memory.record(Message::assistant("a2")).await.unwrap();
        memory.record(Message::user("u3")).await.unwrap();
        memory.record(Message::assistant("a3")).await.unwrap();

        let context = memory.context().await.unwrap();
        assert_eq!(context, vec![Message::user("u3"), Message::assistant("a3")]);
    }

    /// Wire constraint: a tool message and its Assistant message are trimmed
    /// as a pair, never split.
    #[tokio::test]
    async fn tool_messages_trimmed_with_assistant() {
        let mut memory = WindowMemory::new(8);
        // Round 1: User + Assistant (with tool calls) + ToolResult ×2 — a
        // complete round (calculate 3 + empty content 0 + args 1 + 1 + results
        // 1 + 1 = 7 tokens).
        memory.record(Message::user("calculate")).await.unwrap();
        memory
            .record(Message::Assistant {
                content: "".into(),
                reasoning: None,
                tool_calls: vec![
                    ToolCall {
                        id: "t1".into(),
                        name: "calc".into(),
                        arguments: "1+1".into(),
                    },
                    ToolCall {
                        id: "t2".into(),
                        name: "calc".into(),
                        arguments: "2+2".into(),
                    },
                ],
            })
            .await
            .unwrap();
        memory.record(tool_result("t1", "2")).await.unwrap();
        memory.record(tool_result("t2", "4")).await.unwrap();
        // Round 2: User + Assistant (continue 2 + okay 1 = 3 tokens).
        memory.record(Message::user("continue")).await.unwrap();
        memory.record(Message::assistant("okay")).await.unwrap();

        let context = memory.context().await.unwrap();
        assert_eq!(
            context,
            vec![Message::user("continue"), Message::assistant("okay")]
        );
        // The trimmed round is gone entirely: no Assistant with tool_calls and
        // no ToolResult.
        assert!(
            !context.iter().any(
                |m| matches!(m, Message::Assistant { tool_calls, .. } if !tool_calls.is_empty())
            )
        );
        assert!(
            !context
                .iter()
                .any(|m| matches!(m, Message::ToolResult { .. }))
        );
    }

    /// Memory does not manage System: a recorded System message is treated
    /// like any other and may be trimmed when over budget (the system prompt
    /// is the Agent layer's responsibility).
    #[tokio::test]
    async fn recorded_system_can_be_trimmed() {
        let mut memory = WindowMemory::new(3);
        memory.record(Message::system("setup")).await.unwrap();
        memory.record(Message::user("u1")).await.unwrap();
        memory.record(Message::assistant("a1")).await.unwrap();
        memory.record(Message::user("u2")).await.unwrap();
        memory.record(Message::assistant("a2")).await.unwrap();

        // 6 tokens total > 3: the first round (including System) is dropped,
        // keeping the most recent round.
        let context = memory.context().await.unwrap();
        assert_eq!(context, vec![Message::user("u2"), Message::assistant("a2")]);
    }

    /// A single round over budget: the most recent round is kept as a fallback
    /// (nothing more can be trimmed).
    #[tokio::test]
    async fn keeps_last_round_even_when_over_budget() {
        let mut memory = WindowMemory::new(3);
        memory
            .record(Message::user(
                "An extremely long user message, over budget in a single round",
            ))
            .await
            .unwrap();
        memory.record(Message::assistant("Reply")).await.unwrap();
        memory.record(Message::user("u2")).await.unwrap();
        memory.record(Message::assistant("a2")).await.unwrap();

        let context = memory.context().await.unwrap();
        assert_eq!(context, vec![Message::user("u2"), Message::assistant("a2")]);
    }

    /// max_rounds: keeps only the most recent N rounds.
    #[tokio::test]
    async fn max_rounds_window() {
        let mut memory = WindowMemory::new(1000).with_max_rounds(2);
        for i in 1..=4 {
            memory.record(Message::user(format!("u{i}"))).await.unwrap();
            memory
                .record(Message::assistant(format!("a{i}")))
                .await
                .unwrap();
        }

        let context = memory.context().await.unwrap();
        assert_eq!(
            context,
            vec![
                Message::user("u3"),
                Message::assistant("a3"),
                Message::user("u4"),
                Message::assistant("a4")
            ]
        );
    }

    /// max_tokens and max_rounds both set: the smaller window wins.
    #[tokio::test]
    async fn tokens_and_rounds_take_smaller() {
        let mut memory = WindowMemory::new(3).with_max_rounds(3);
        for i in 1..=4 {
            memory.record(Message::user(format!("u{i}"))).await.unwrap();
            memory
                .record(Message::assistant(format!("a{i}")))
                .await
                .unwrap();
        }
        // max_rounds=3 would allow 3 rounds, but max_tokens=3 fits only 1
        // round → the smaller one wins.
        let context = memory.context().await.unwrap();
        assert_eq!(context, vec![Message::user("u4"), Message::assistant("a4")]);
    }

    /// Projection: raising the budget restores the trimmed history (lossless).
    #[tokio::test]
    async fn raising_budget_restores_history() {
        let mut memory = WindowMemory::new(3);
        memory.record(Message::user("u1")).await.unwrap();
        memory.record(Message::assistant("a1")).await.unwrap();
        memory.record(Message::user("u2")).await.unwrap();
        memory.record(Message::assistant("a2")).await.unwrap();

        assert_eq!(memory.context().await.unwrap().len(), 2); // most recent round

        memory.set_max_tokens(1000);
        assert_eq!(memory.context().await.unwrap().len(), 4); // history restored
    }

    /// reasoning counts toward the budget: the same content with long
    /// reasoning triggers trimming; without it, nothing is trimmed.
    #[tokio::test]
    async fn reasoning_counts_toward_budget() {
        let mut memory = WindowMemory::new(11);
        // Round 1: User + Assistant (with reasoning): 1 + 1 + 10 = 12.
        memory.record(Message::user("u1")).await.unwrap();
        memory
            .record(Message::assistant_with_reasoning(
                "hi",
                "a".repeat(40), // 40 chars = 10 tokens
            ))
            .await
            .unwrap();
        // Round 2: User + Assistant (no reasoning): 1 + 1 = 2.
        memory.record(Message::user("u2")).await.unwrap();
        memory.record(Message::assistant("hi")).await.unwrap();

        // 14 total > 11 → round 1 is trimmed.
        let context = memory.context().await.unwrap();
        assert_eq!(context, vec![Message::user("u2"), Message::assistant("hi")]);

        // Same setup without reasoning: 4 ≤ 11 → nothing is trimmed.
        let mut memory2 = WindowMemory::new(11);
        memory2.record(Message::user("u1")).await.unwrap();
        memory2.record(Message::assistant("hi")).await.unwrap();
        memory2.record(Message::user("u2")).await.unwrap();
        memory2.record(Message::assistant("hi")).await.unwrap();
        assert_eq!(memory2.context().await.unwrap().len(), 4);
    }

    /// Custom counter: each message counts as exactly 1 token (trim by message
    /// count).
    #[tokio::test]
    async fn custom_token_counter() {
        #[derive(Debug, Default)]
        struct OnePerMessage;
        #[async_trait::async_trait]
        impl TokenCounter for OnePerMessage {
            async fn count(&self, _text: &str) -> Result<usize, MemoryError> {
                Ok(1)
            }
        }

        let mut memory = WindowMemory::new(2).with_token_counter(Box::new(OnePerMessage));
        // 5 messages (5) > 2 → trimmed to the most recent 1 round (a round
        // starts at a User).
        memory.record(Message::user("u1")).await.unwrap();
        memory.record(Message::assistant("a1")).await.unwrap();
        memory.record(Message::user("u2")).await.unwrap();
        memory.record(Message::assistant("a2")).await.unwrap();
        memory.record(Message::user("u3")).await.unwrap();

        let context = memory.context().await.unwrap();
        assert_eq!(context, vec![Message::user("u3")]);
    }

    /// Materialization: `replace: true` writes back to storage; subsequent
    /// in-budget context() calls recompute nothing (the strategy is not
    /// called).
    #[tokio::test]
    async fn materialize_replaces_storage_and_skips_strategy() {
        let summarizer = Arc::new(FakeSummarizer::default());
        let mut memory = WindowMemory::new(7).with_strategy(summarizer.clone());
        for i in 1..=4 {
            memory.record(Message::user(format!("u{i}"))).await.unwrap();
            memory
                .record(Message::assistant(format!("a{i}")))
                .await
                .unwrap();
        }

        // 8 > 7 → compacted to [summary (4 tokens), u4, a4].
        let first = memory.context().await.unwrap();
        assert_eq!(
            summarizer.calls.load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(
            first,
            vec![
                Message::system("prior summary"),
                Message::user("u4"),
                Message::assistant("a4"),
            ]
        );

        // After materialization the budget holds (4 + 2 = 6 ≤ 7): the second
        // context() just clones, without calling the strategy.
        let second = memory.context().await.unwrap();
        assert_eq!(
            summarizer.calls.load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        assert_eq!(second, first);
    }

    /// Over budget again after materialization: the strategy input = previous
    /// compaction result + new messages (chained, including the summary).
    #[tokio::test]
    async fn materialize_strategy_input_is_materialized_sequence() {
        let summarizer = Arc::new(FakeSummarizer::default());
        let mut memory = WindowMemory::new(7).with_strategy(summarizer.clone());
        for i in 1..=4 {
            memory.record(Message::user(format!("u{i}"))).await.unwrap();
            memory
                .record(Message::assistant(format!("a{i}")))
                .await
                .unwrap();
        }
        memory.context().await.unwrap(); // first compaction, materialized

        // Append new messages until over budget again → the strategy is called
        // again, with the previous summary message in the input.
        memory.record(Message::user("u5")).await.unwrap();
        memory.record(Message::assistant("a5")).await.unwrap();
        memory.record(Message::user("u6")).await.unwrap();
        memory.record(Message::assistant("a6")).await.unwrap();
        memory.context().await.unwrap();

        assert_eq!(
            summarizer.calls.load(std::sync::atomic::Ordering::Relaxed),
            2
        );
        assert!(
            summarizer
                .last_input_has_summary
                .load(std::sync::atomic::Ordering::Relaxed)
        );
    }

    /// Projection: `replace: false` only affects this retrieval's view;
    /// storage is unchanged (lossless).
    #[tokio::test]
    async fn projection_keeps_storage_unchanged() {
        #[derive(Debug, Default)]
        struct KeepLastRound;
        #[async_trait::async_trait]
        impl TrimStrategy for KeepLastRound {
            async fn trim(
                &self,
                messages: &[Message],
                _budget: &Budget,
                _counter: &dyn TokenCounter,
            ) -> Result<TrimResult, MemoryError> {
                let pos = messages
                    .iter()
                    .rposition(|m| matches!(m, Message::User(_)))
                    .unwrap_or(0);
                Ok(TrimResult {
                    messages: messages[pos..].to_vec(),
                    replace: false,
                })
            }
        }

        let strategy = Arc::new(KeepLastRound);
        let mut memory = WindowMemory::new(1).with_strategy(strategy);
        memory.record(Message::user("u1")).await.unwrap();
        memory.record(Message::assistant("a1")).await.unwrap();
        memory.record(Message::user("u2")).await.unwrap();
        memory.record(Message::assistant("a2")).await.unwrap();

        let view = memory.context().await.unwrap();
        assert_eq!(view, vec![Message::user("u2"), Message::assistant("a2")]);

        // Storage unchanged: raising the budget restores everything.
        memory.set_max_tokens(1000);
        assert_eq!(memory.context().await.unwrap().len(), 4);
    }

    /// Behavior: `context()` returns a clone; mutating it does not affect the
    /// internals.
    #[tokio::test]
    async fn context_is_a_copy() {
        let mut memory = WindowMemory::new(1000);
        memory.record(Message::user("a")).await.unwrap();

        let mut context = memory.context().await.unwrap();
        context.push(Message::assistant("b"));

        assert_eq!(memory.context().await.unwrap().len(), 1);
    }

    /// Defensive: a history without User messages does not panic and is
    /// returned as-is.
    #[tokio::test]
    async fn no_user_messages_returns_all() {
        let mut memory = WindowMemory::new(1);
        memory
            .record(Message::assistant("assistant-only message"))
            .await
            .unwrap();

        let context = memory.context().await.unwrap();
        assert_eq!(context.len(), 1);
    }

    /// ContentBlock with multiple text blocks: all of them count toward the
    /// budget.
    #[tokio::test]
    async fn user_blocks_all_counted() {
        let mut memory = WindowMemory::new(2);
        // The User message has two text blocks totalling over budget → the
        // single-round fallback keeps the whole round.
        memory
            .record(Message::user_blocks(vec![
                ContentBlock::Text("aaaa".into()),
                ContentBlock::Text("bbbb".into()),
            ]))
            .await
            .unwrap();
        memory.record(Message::assistant("hi")).await.unwrap();

        let context = memory.context().await.unwrap();
        assert_eq!(context.len(), 2); // whole round kept as fallback
    }

    /// Swapping the counter: the counting convention changes, so counts are
    /// invalidated and recomputed — the new counter recounts every message.
    #[tokio::test]
    async fn changing_counter_recounts() {
        #[derive(Debug)]
        struct CountingCounter {
            calls: Arc<std::sync::atomic::AtomicUsize>,
        }
        impl CountingCounter {
            fn shared(calls: Arc<std::sync::atomic::AtomicUsize>) -> Self {
                Self { calls }
            }
        }
        #[async_trait::async_trait]
        impl TokenCounter for CountingCounter {
            async fn count(&self, _text: &str) -> Result<usize, MemoryError> {
                self.calls
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(1)
            }
        }

        let first_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut memory = WindowMemory::new(100)
            .with_token_counter(Box::new(CountingCounter::shared(first_calls.clone())));
        memory.record(Message::user("u1")).await.unwrap();
        memory.record(Message::assistant("a1")).await.unwrap();
        memory.context().await.unwrap(); // lazy recompute of the 2 messages
        assert_eq!(first_calls.load(std::sync::atomic::Ordering::Relaxed), 2);

        // Swap the counter → invalidated and recomputed: the new counter
        // counts the existing 2 messages + the new record (1).
        let second_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut memory =
            memory.with_token_counter(Box::new(CountingCounter::shared(second_calls.clone())));
        memory.record(Message::user("u2")).await.unwrap();
        assert_eq!(second_calls.load(std::sync::atomic::Ordering::Relaxed), 3);
    }

    /// Remote counting failure: MemoryError::TokenCount propagates (record
    /// returns Err).
    #[tokio::test]
    async fn token_count_failure_propagates() {
        #[derive(Debug, Default)]
        struct FailingCounter;
        #[async_trait::async_trait]
        impl TokenCounter for FailingCounter {
            async fn count(&self, _text: &str) -> Result<usize, MemoryError> {
                Err(MemoryError::TokenCount(
                    "remote counting API unavailable".into(),
                ))
            }
        }

        let mut memory = WindowMemory::new(100).with_token_counter(Box::new(FailingCounter));
        let err = memory.record(Message::user("hi")).await.unwrap_err();
        assert!(matches!(err, MemoryError::TokenCount(_)));
    }

    /// Protected messages (and their rounds) are exempt as a whole when over
    /// budget; ordinary rounds are trimmed as usual.
    #[tokio::test]
    async fn protected_round_survives_trim() {
        // Candidates (ordinary rounds) total 4 tokens > budget 3: the earliest
        // ordinary round is dropped; the protected round is exempt, so the
        // total context may exceed the budget (the inherent cost of resident
        // skill instructions).
        let mut memory = WindowMemory::new(3);
        memory.record(Message::user("u1")).await.unwrap();
        memory.record(Message::assistant("a1")).await.unwrap();
        // Round 2: the load_skill call round (its ToolResult is protected).
        memory.record(Message::user("u2")).await.unwrap();
        memory
            .record(Message::Assistant {
                content: "a2".into(),
                reasoning: None,
                tool_calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "load_skill".into(),
                    arguments: r#"{"name":"x"}"#.into(),
                }],
            })
            .await
            .unwrap();
        memory
            .record_protected(tool_result("c1", "<skill_content>body</skill_content>"))
            .await
            .unwrap();
        // Round 3 (ordinary, most recent).
        memory.record(Message::user("u3")).await.unwrap();
        memory.record(Message::assistant("a3")).await.unwrap();

        // Total tokens exceed the budget: the ordinary first round is dropped;
        // the protected second round and the most recent third round are kept.
        let context = memory.context().await.unwrap();
        assert_eq!(
            context,
            vec![
                Message::user("u2"),
                Message::Assistant {
                    content: "a2".into(),
                    reasoning: None,
                    tool_calls: vec![ToolCall {
                        id: "c1".into(),
                        name: "load_skill".into(),
                        arguments: r#"{"name":"x"}"#.into(),
                    }],
                },
                tool_result("c1", "<skill_content>body</skill_content>"),
                Message::user("u3"),
                Message::assistant("a3"),
            ]
        );
    }

    /// After a protected message is recorded, trimming never removes it;
    /// ordinary messages are unaffected.
    #[tokio::test]
    async fn protected_message_never_pruned() {
        let mut memory = WindowMemory::new(3);
        memory.record(Message::user("u1")).await.unwrap();
        memory.record(Message::assistant("a1")).await.unwrap();
        memory.record(Message::user("u2")).await.unwrap();
        memory.record(Message::assistant("a2")).await.unwrap();
        // The protected message forms a round by itself (no User precedes it;
        // a leading non-User message belongs to the first round).
        memory
            .record_protected(tool_result("c1", "skill body"))
            .await
            .unwrap();

        // After many context() rounds: the protected message is still there
        // (ordinary rounds have been trimmed away).
        let mut all_kept = true;
        for _ in 0..5 {
            let context = memory.context().await.unwrap();
            if !context.iter().any(
                |m| matches!(m, Message::ToolResult { content, .. } if content == "skill body"),
            ) {
                all_kept = false;
                break;
            }
        }
        assert!(
            all_kept,
            "the protected message must survive repeated trims"
        );
    }
}
