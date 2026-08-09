//! Retry wrapper: `RetryProvider` implements [`Provider`] by wrapping an inner
//! provider and retrying recoverable failures per policy — rate limits /
//! network / timeouts / 5xx.
//!
//! Retrying is a wrapper layer outside the Provider: the interface itself has
//! no retry logic, and any Provider implementation (OpenAi / Fake / a
//! user-written one) gains retry by being wrapped, with zero changes to
//! callers or the loop layer.
//!
//! # Examples
//!
//! ```rust
//! # #[tokio::main]
//! # async fn main() -> Result<(), molo::provider::ProviderError> {
//! use molo::message::Message;
//! use molo::provider::{FakeProvider, FakeReply, Provider, RetryProvider, ProviderError};
//!
//! // Script: the first call is rate limited, the second succeeds — RetryProvider
//! // retries automatically and the caller sees nothing.
//! let inner = FakeProvider::new([
//!     FakeReply::Error(ProviderError::RateLimited { retry_after: None }),
//!     FakeReply::Text("hi".into()),
//! ]);
//! let provider = RetryProvider::new(inner);
//! let resp = provider.chat(molo::provider::ChatRequest::default()).await?;
//! assert_eq!(resp.message, Message::assistant("hi"));
//! # Ok(())
//! # }
//! ```

use super::{ChatRequest, ChatResponse, Provider, ProviderError, StreamEvent};
use async_trait::async_trait;
use futures::stream::BoxStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Retry policy (with defaults; `Default` is "exponential backoff + jitter,
/// 3 attempts").
///
/// Default backoff: initial 0.5s / factor 2.0 / cap 10s / jitter on (full
/// jitter, to prevent thundering herds).
#[derive(Debug, Clone, PartialEq)]
pub struct RetryPolicy {
    /// Total attempts (including the first); default 3 (i.e. at most 2
    /// retries after a failure).
    pub max_attempts: usize,
    /// Backoff strategy (how long to wait before retrying after each
    /// failure).
    pub backoff: Backoff,
    /// Which errors are retryable; the default is [`Retryable::Default`].
    pub retryable: Retryable,
    /// When rate limited, prefer waiting the vendor's `Retry-After` duration
    /// (overrides backoff); default true.
    pub respect_retry_after: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            backoff: Backoff::Exponential {
                initial: Duration::from_millis(500),
                factor: 2.0,
                max: Duration::from_secs(10),
                jitter: true,
            },
            retryable: Retryable::Default,
            respect_retry_after: true,
        }
    }
}

/// Backoff strategy: how long to wait after each failure before the next
/// attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum Backoff {
    /// Fixed interval.
    Fixed(Duration),
    /// Exponential backoff: `initial * factor^attempt`, capped at `max`;
    /// when `jitter` is on, draw full jitter in `[0, computed value)`
    /// (thundering-herd protection — requests failing at the same time do
    /// not all retry at the same moment).
    Exponential {
        /// Initial wait duration.
        initial: Duration,
        /// Exponential factor.
        factor: f64,
        /// Cap duration.
        max: Duration,
        /// Whether to use full jitter (`[0, computed value)`,
        /// thundering-herd protection).
        jitter: bool,
    },
}

impl Backoff {
    /// The duration to wait after the `attempt`-th failure (0 = the first
    /// failure).
    fn delay(&self, attempt: usize) -> Duration {
        match self {
            Self::Fixed(d) => *d,
            Self::Exponential {
                initial,
                factor,
                max,
                jitter,
            } => {
                let base =
                    (initial.as_secs_f64() * factor.powf(attempt as f64)).min(max.as_secs_f64());
                if *jitter {
                    // Full jitter: LCG pseudo-random (zero new dependencies;
                    // the state is atomically shared so concurrent draws
                    // advance sequentially — concurrently failing requests
                    // get dispersed delays, making the herd protection work).
                    Duration::from_secs_f64(base * random01())
                } else {
                    Duration::from_secs_f64(base)
                }
            }
        }
    }
}

/// `Retry-After` wait cap (5 minutes): if the vendor's instruction exceeds
/// this, wait this long instead.
/// Backoff itself is capped by `max`, but the Retry-After path does not go
/// through backoff — an absurd value from a broken endpoint (e.g. 136 years)
/// would make the caller wait forever, hence the separate cap.
const RETRY_AFTER_CAP: Duration = Duration::from_secs(300);

/// Lightweight pseudo-random source (LCG; zero new dependencies, serves only
/// the thundering-herd jitter, not for cryptography).
///
/// Seed = clock nanoseconds at the first draw (full 64 bits); each subsequent
/// draw advances the LCG state (atomically shared, so concurrent draws
/// advance sequentially). The clock appears only in the initial seed and does
/// not carry the randomness itself — when a rate-limit burst hits,
/// simultaneously failing requests draw from sequentially advanced
/// independent states, so delays are highly dispersed.
fn random01() -> f64 {
    const A: u64 = 6364136223846793005; // LCG constants (Knuth / MMIX)
    const C: u64 = 1442695040888963407;
    static STATE: AtomicU64 = AtomicU64::new(0);
    loop {
        let current = STATE.load(Ordering::Relaxed);
        let next = if current == 0 {
            // First bootstrap: seed with full epoch nanoseconds (the clock
            // only contributes the initial difference and does not take part
            // in later draws), and **advance the LCG immediately** — otherwise
            // the high 24 bits of the seed are often 0 and the first draw
            // always returns 0 (the first retry would have no backoff,
            // breaking herd protection at the very first burst).
            let seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            seed.wrapping_mul(A).wrapping_add(C)
        } else {
            current.wrapping_mul(A).wrapping_add(C)
        };
        if STATE
            .compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            // Take the high 24 bits mapped to [0, 1).
            return (next >> 40) as f64 / (1u64 << 24) as f64;
        }
    }
}

/// Retry judgment: which errors are worth retrying.
///
/// `Custom` closures are not comparable, so `PartialEq` is implemented by
/// hand (only `Default` values equal each other); `Debug` is also
/// hand-written (a closure only prints its variant name).
#[derive(Clone)]
pub enum Retryable {
    /// Default: Network / Timeout / RateLimited / Api (status ≥ 500);
    /// 4xx (auth / invalid arguments / quota exhausted) are not retried —
    /// retrying would not change the outcome.
    Default,
    /// Custom judgment (e.g. only retry rate limits, or exclude certain 4xx
    /// by message content).
    Custom(Arc<dyn Fn(&ProviderError) -> bool + Send + Sync>),
}

impl std::fmt::Debug for Retryable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Default => f.write_str("Default"),
            Self::Custom(_) => f.write_str("Custom(_)"),
        }
    }
}

impl PartialEq for Retryable {
    fn eq(&self, other: &Self) -> bool {
        matches!((self, other), (Self::Default, Self::Default))
    }
}

impl Retryable {
    fn is_retryable(&self, error: &ProviderError) -> bool {
        match self {
            Self::Default => {
                matches!(
                    error,
                    ProviderError::Network(_)
                        | ProviderError::Timeout(_)
                        | ProviderError::RateLimited { .. }
                ) || matches!(error, ProviderError::Api { status, .. } if *status >= 500)
            }
            Self::Custom(f) => f(error),
        }
    }
}

/// Retry wrapper: implements [`Provider`] and retries inner failures per
/// [`RetryPolicy`].
///
/// **Streaming semantics**: only when `stream_chat` returns `Err` (connection
/// setup failure) is the whole call retried; **errors inside the stream are
/// passed through verbatim and never retried** — once the stream is
/// established this implementation cannot tell an "interruption before the
/// first event" from an "interruption after part of the output was
/// delivered", and retrying would duplicate or corrupt output.
/// Hence "retry before the first event" only covers method-level failures;
/// interruptions within the stream (including after setup but before the
/// first event) are left to the caller (the Agent) to handle.
///
/// Timeouts are the inner provider's (e.g.
/// [`OpenAiProvider`](super::OpenAiProvider)) responsibility: each attempt
/// may hit the inner `Timeout` error, and `Retryable::Default` includes
/// Timeout, so "timeouts are retried too" falls out naturally.
///
/// # Errors
///
/// After retries are exhausted, the **last** error is returned (neither the
/// first nor an aggregate); non-retryable errors are returned immediately on
/// the first failure.
///
/// # Cancellation semantics
///
/// Dropping the future while waiting on backoff cancels the whole call; no
/// further attempts are made. An already-issued inner request is cancelled
/// together with the future (whether the network request continues at the
/// transport layer depends on the underlying client).
pub struct RetryProvider<I: Provider> {
    inner: I,
    policy: RetryPolicy,
}

impl<I: Provider> std::fmt::Debug for RetryProvider<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // inner is a generic Provider without Debug; print the type name and
        // policy (in the same style as ReActAgent).
        f.debug_struct("RetryProvider")
            .field("inner", &std::any::type_name::<I>())
            .field("policy", &self.policy)
            .finish()
    }
}

impl<I: Provider> RetryProvider<I> {
    /// Wraps with the default policy (3 attempts / exponential backoff +
    /// jitter / default retry judgment).
    pub fn new(inner: I) -> Self {
        Self {
            inner,
            policy: RetryPolicy::default(),
        }
    }

    /// Replaces the retry policy.
    ///
    /// # Examples
    ///
    /// For tests / local simulation: fixed short waits, at most 2 attempts,
    /// ignoring the vendor's `Retry-After`:
    ///
    /// ```
    /// use std::time::Duration;
    /// use molo::{Backoff, FakeProvider, RetryPolicy, RetryProvider};
    ///
    /// let provider = RetryProvider::new(FakeProvider::new([])).with_policy(
    ///     RetryPolicy {
    ///         max_attempts: 2,
    ///         backoff: Backoff::Fixed(Duration::from_millis(10)),
    ///         respect_retry_after: false,
    ///         ..Default::default()
    ///     },
    /// );
    /// ```
    pub fn with_policy(mut self, policy: RetryPolicy) -> Self {
        self.policy = policy;
        self
    }
}

impl<I: Provider> RetryProvider<I> {
    /// Retry decision and wait for one failure: whether another attempt is
    /// possible, and if so how long to wait.
    ///
    /// `Some(delay)` = wait then retry; `None` = give up (not retryable or
    /// attempts exhausted).
    fn retry_decision(&self, error: &ProviderError, attempts: usize) -> Option<Duration> {
        if !self.policy.retryable.is_retryable(error) || attempts + 1 >= self.policy.max_attempts {
            return None;
        }
        let delay = match (error, self.policy.respect_retry_after) {
            // Retry-After cap: a broken endpoint may return an absurd value
            // (e.g. 136 years), and backoff's own `max` cap does not apply on
            // this path — cap at [`RETRY_AFTER_CAP`].
            (
                ProviderError::RateLimited {
                    retry_after: Some(d),
                },
                true,
            ) => (*d).min(RETRY_AFTER_CAP),
            _ => self.policy.backoff.delay(attempts),
        };
        tracing::warn!(
            attempt = attempts + 1,
            max_attempts = self.policy.max_attempts,
            delay = ?delay,
            error = %error,
            "provider call failed, retrying",
        );
        Some(delay)
    }
}

#[async_trait]
impl<I: Provider + Send + Sync> Provider for RetryProvider<I> {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ProviderError> {
        let mut attempts = 0usize;
        loop {
            match self.inner.chat(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(error) => match self.retry_decision(&error, attempts) {
                    Some(delay) => {
                        tokio::time::sleep(delay).await;
                        attempts += 1;
                    }
                    None => return Err(error),
                },
            }
        }
    }

    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError> {
        // Retry only when the method returns Err; errors inside the stream
        // are always passed through (see the type-level docs).
        let mut attempts = 0usize;
        loop {
            match self.inner.stream_chat(request.clone()).await {
                Ok(stream) => return Ok(stream),
                Err(error) => match self.retry_decision(&error, attempts) {
                    Some(delay) => {
                        tokio::time::sleep(delay).await;
                        attempts += 1;
                    }
                    None => return Err(error),
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{FakeProvider, FakeReply};
    use std::sync::Arc;

    fn rate_limited() -> ProviderError {
        ProviderError::RateLimited { retry_after: None }
    }

    fn api(status: u16) -> ProviderError {
        ProviderError::Api {
            status,
            message: "boom".into(),
        }
    }

    fn network() -> ProviderError {
        ProviderError::Network("connection refused".into())
    }

    #[tokio::test]
    async fn retries_then_succeeds() {
        // Rate-limit failure → retry succeeds; the caller only sees the
        // final result.
        let inner = Arc::new(FakeProvider::new([
            FakeReply::Error(rate_limited()),
            FakeReply::Text("ok".into()),
        ]));
        let provider = RetryProvider::new(inner.clone());
        let resp = provider.chat(ChatRequest::default()).await.unwrap();
        assert_eq!(resp.message, crate::message::Message::assistant("ok"));
        // Assert attempt count: first failure + one retry = 2 calls.
        assert_eq!(inner.requests().len(), 2);
    }

    #[tokio::test]
    async fn gives_up_after_max_attempts() {
        // Always fails: return the last error after max_attempts (default 3)
        // attempts.
        let inner = Arc::new(FakeProvider::new([
            FakeReply::Error(network()),
            FakeReply::Error(network()),
            FakeReply::Error(network()),
        ]));
        let provider = RetryProvider::new(inner.clone());
        let err = provider.chat(ChatRequest::default()).await.unwrap_err();
        assert!(matches!(err, ProviderError::Network(_)));
        assert_eq!(inner.requests().len(), 3);
    }

    #[tokio::test]
    async fn api_4xx_not_retried() {
        // 4xx business error: the default judgment does not retry, giving up
        // after a single call.
        let inner = Arc::new(FakeProvider::new([FakeReply::Error(api(400))]));
        let provider = RetryProvider::new(inner.clone());
        let err = provider.chat(ChatRequest::default()).await.unwrap_err();
        assert!(matches!(err, ProviderError::Api { status: 400, .. }));
        assert_eq!(inner.requests().len(), 1);
    }

    #[tokio::test]
    async fn api_5xx_retried() {
        let inner = Arc::new(FakeProvider::new([
            FakeReply::Error(api(503)),
            FakeReply::Text("ok".into()),
        ]));
        let provider = RetryProvider::new(inner.clone());
        provider.chat(ChatRequest::default()).await.unwrap();
        assert_eq!(inner.requests().len(), 2);
    }

    #[tokio::test]
    async fn max_attempts_one_disables_retry() {
        let inner = Arc::new(FakeProvider::new([FakeReply::Error(network())]));
        let provider = RetryProvider::new(inner.clone()).with_policy(RetryPolicy {
            max_attempts: 1,
            ..Default::default()
        });
        provider.chat(ChatRequest::default()).await.unwrap_err();
        assert_eq!(inner.requests().len(), 1);
    }

    /// Retry-After cap: a vendor instruction of 3600s is capped at
    /// `RETRY_AFTER_CAP` (300s) rather than waiting indefinitely.
    #[test]
    fn retry_after_capped_at_cap() {
        let provider = RetryProvider::new(FakeProvider::new([]));
        let delay = provider.retry_decision(
            &ProviderError::RateLimited {
                retry_after: Some(Duration::from_secs(3600)),
            },
            0,
        );
        assert_eq!(delay, Some(RETRY_AFTER_CAP));
    }

    #[tokio::test]
    async fn retry_after_respected_when_present() {
        // The vendor instructs a 1s wait: the policy prefers it over backoff;
        // the retry succeeds after the wait.
        let inner = Arc::new(FakeProvider::new([
            FakeReply::Error(ProviderError::RateLimited {
                retry_after: Some(Duration::from_millis(1)),
            }),
            FakeReply::Text("ok".into()),
        ]));
        let provider = RetryProvider::new(inner.clone());
        provider.chat(ChatRequest::default()).await.unwrap();
        assert_eq!(inner.requests().len(), 2);
    }

    /// Verifies the Retry-After **priority** mechanism: backoff is 50ms
    /// (factor 1.0) while Retry-After is 200ms — the actual wait must be
    /// ≈200ms rather than 50ms (if respect were broken, the retry would finish
    /// after 50ms). 200ms is the shortest reliably measurable window for the
    /// suite (tokio timers only fire late, never early; the 150ms lower bound
    /// always holds; the 600ms upper bound leaves 400ms of slack).
    #[tokio::test]
    async fn retry_after_delay_overrides_backoff() {
        use std::time::Instant;

        let inner = Arc::new(FakeProvider::new([
            FakeReply::Error(ProviderError::RateLimited {
                retry_after: Some(Duration::from_millis(200)),
            }),
            FakeReply::Text("ok".into()),
        ]));
        let provider = RetryProvider::new(inner.clone()).with_policy(RetryPolicy {
            backoff: Backoff::Exponential {
                initial: Duration::from_millis(50),
                factor: 1.0,
                max: Duration::from_secs(1),
                jitter: false,
            },
            ..Default::default()
        });
        let start = Instant::now();
        provider.chat(ChatRequest::default()).await.unwrap();
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(150) && elapsed < Duration::from_millis(600),
            "should wait Retry-After (200ms) rather than 50ms backoff, elapsed {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn custom_retryable_predicate() {
        // Custom judgment: retry only 4xx other than 429 (reversing the
        // default).
        let inner = Arc::new(FakeProvider::new([
            FakeReply::Error(api(400)),
            FakeReply::Text("ok".into()),
        ]));
        let provider = RetryProvider::new(inner.clone()).with_policy(RetryPolicy {
            retryable: Retryable::Custom(Arc::new(
                |e| matches!(e, ProviderError::Api { status, .. } if *status == 400),
            )),
            ..Default::default()
        });
        provider.chat(ChatRequest::default()).await.unwrap();
        assert_eq!(inner.requests().len(), 2);
    }

    #[tokio::test]
    async fn stream_retries_only_before_first_event() {
        // Streaming: first setup failure → retry the whole call; after
        // success, events inside the stream pass through.
        let inner = Arc::new(FakeProvider::new([
            FakeReply::Error(network()),
            FakeReply::Text("ok".into()),
        ]));
        let provider = RetryProvider::new(inner.clone());
        let mut stream = provider.stream_chat(ChatRequest::default()).await.unwrap();
        use futures::StreamExt;
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first, StreamEvent::Delta("ok".into()));
        assert_eq!(inner.requests().len(), 2);
    }

    #[tokio::test]
    async fn stream_failure_after_max_attempts() {
        let inner = Arc::new(FakeProvider::new([FakeReply::Error(network())]));
        let provider = RetryProvider::new(inner.clone()).with_policy(RetryPolicy {
            max_attempts: 1,
            ..Default::default()
        });
        // BoxStream has no Debug, so unwrap_err is unavailable; match to get
        // the error.
        let err = match provider.stream_chat(ChatRequest::default()).await {
            Err(e) => e,
            Ok(_) => panic!("expected error"),
        };
        assert!(matches!(err, ProviderError::Network(_)));
        assert_eq!(inner.requests().len(), 1);
    }

    #[test]
    fn exponential_backoff_grows_and_caps() {
        let backoff = Backoff::Exponential {
            initial: Duration::from_millis(500),
            factor: 2.0,
            max: Duration::from_secs(10),
            jitter: false,
        };
        assert_eq!(backoff.delay(0), Duration::from_millis(500));
        assert_eq!(backoff.delay(1), Duration::from_secs(1));
        assert_eq!(backoff.delay(2), Duration::from_secs(2));
        // Cap: even after 5 failures the delay never exceeds max.
        assert_eq!(backoff.delay(10), Duration::from_secs(10));
    }

    #[test]
    fn jitter_stays_within_bounds() {
        let backoff = Backoff::Exponential {
            initial: Duration::from_secs(1),
            factor: 1.0,
            max: Duration::from_secs(10),
            jitter: true,
        };
        // Full jitter in [0, base): base is always 1s (factor 1.0), so the
        // upper bound must be < 1s — a limit of 10s would let a degenerate
        // implementation (e.g. always returning 0.9) pass.
        for attempt in 0..50 {
            let d = backoff.delay(attempt);
            assert!(d < Duration::from_secs(1), "jitter out of bounds: {d:?}");
        }
    }

    /// Verifies jitter's randomness mechanics: the first draw (LCG bootstrap)
    /// is not always 0 — if it were, the first retry in the process would
    /// have no backoff; multiple draws must yield multiple distinct values
    /// (state advancing).
    #[test]
    fn jitter_first_draw_nonzero_and_dispersed() {
        let first = random01();
        assert!(
            first > 0.0,
            "first draw must not be 0 (bootstrap must advance LCG first)"
        );
        assert!(first < 1.0);

        let mut seen = std::collections::HashSet::new();
        for _ in 0..64 {
            let d = random01();
            assert!(d > 0.0 && d < 1.0);
            seen.insert(d.to_bits());
        }
        assert!(seen.len() > 1, "LCG draws must be dispersed (state advancing)");
    }
}
