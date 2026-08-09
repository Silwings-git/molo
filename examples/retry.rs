//! RetryProvider example: how to use the retry wrapper.
//!
//! [`RetryProvider`](molo::provider::RetryProvider) implements
//! [`Provider`](molo::provider::Provider) and wraps any Provider — one wrapper
//! layer gives you retry, and anything that consumes a Provider (ReActAgent,
//! hand-written loops) works unchanged. Timeouts are the inner Provider's job
//! (OpenAiProvider's constructor has built-in defaults); the default retry
//! decision covers the `Timeout` / `RateLimited` / `Network` error classes
//! plus 5xx.
//!
//! Self-contained: errors are injected via
//! [`FakeProvider`](molo::provider::FakeProvider) scripts, with no real API
//! dependency.
//!
//! Run: `cargo run --example retry`

use molo::provider::{
    Backoff, FakeProvider, FakeReply, OpenAiProvider, Provider, ProviderError, RetryPolicy,
    RetryProvider, Retryable,
};
use std::sync::Arc;
use std::time::Duration;

/// Usage 1: wrap with the default policy — a rate-limited failure retries automatically, invisible to the caller.
async fn default_policy_retries_rate_limited() {
    // Script: the first call is rate-limited, the second succeeds.
    let inner = Arc::new(FakeProvider::new([
        FakeReply::Error(ProviderError::RateLimited { retry_after: None }),
        FakeReply::Text("retry succeeded".into()),
    ]));
    let provider = RetryProvider::new(inner.clone());

    let resp = provider
        .chat(molo::provider::ChatRequest::default())
        .await
        .unwrap();
    let text = match resp.message {
        molo::Message::Assistant { content, .. } => content,
        _ => String::new(),
    };
    println!("default policy: first call rate-limited, retried automatically → {text}");

    // requests() asserts the attempt count: one failure + one retry = 2 calls.
    println!("attempt count (requests): {}", inner.requests().len());
    assert_eq!(inner.requests().len(), 2);
}

/// Usage 2: a custom policy — 5 attempts, a slower backoff, and only rate-limited errors are retried.
async fn custom_policy() {
    let policy = RetryPolicy {
        max_attempts: 5,
        backoff: Backoff::Exponential {
            initial: Duration::from_millis(200),
            factor: 2.0,
            max: Duration::from_secs(5),
            jitter: false,
        },
        // Custom decision: only retry rate limiting; network / 5xx give up immediately.
        retryable: Retryable::Custom(Arc::new(|e| matches!(e, ProviderError::RateLimited { .. }))),
        ..Default::default()
    };

    let inner = Arc::new(FakeProvider::new([
        FakeReply::Error(ProviderError::RateLimited { retry_after: None }),
        FakeReply::Error(ProviderError::RateLimited { retry_after: None }),
        FakeReply::Text("succeeded after two rate limits".into()),
    ]));
    let provider = RetryProvider::new(inner.clone()).with_policy(policy);
    let resp = provider
        .chat(molo::provider::ChatRequest::default())
        .await
        .unwrap();
    let text = match resp.message {
        molo::Message::Assistant { content, .. } => content,
        _ => String::new(),
    };
    println!("custom policy: two rate limits, retried automatically → {text}");
}

/// Usage 3: streaming retry — retries only when the failure happens before the first event; once the stream is established, an interruption passes through as-is.
async fn streaming_retry_before_first_event() {
    use futures::StreamExt;

    // The first establishment fails (network), the second succeeds.
    let inner = Arc::new(FakeProvider::new([
        FakeReply::Error(ProviderError::Network("connection refused".into())),
        FakeReply::Text("streaming succeeded".into()),
    ]));
    let provider = RetryProvider::new(inner.clone());

    let mut stream = provider
        .stream_chat(molo::provider::ChatRequest::default())
        .await
        .unwrap();
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            molo::provider::StreamEvent::Delta(text) => println!("streaming retry: delta {text}"),
            molo::provider::StreamEvent::Done { .. } => {}
            _ => {}
        }
    }
    println!("streaming retry attempt count: {}", inner.requests().len());
}

/// Usage 4: timeout configuration (OpenAiProvider has built-in defaults,
/// chainable) — assembly only, no real request is made.
fn timeout_configuration() {
    // Defaults: connect 30s / non-streaming 600s / streaming event idle 60s; override chained:
    let _provider = OpenAiProvider::new("https://api.openai.com/v1", "sk-xxx", "gpt-4o-mini")
        .with_connect_timeout(Duration::from_secs(10))
        .with_request_timeout(Duration::from_secs(120))
        .with_idle_timeout(Duration::from_secs(30));
    println!("timeout config: connect 10s / request 120s / idle 30s (effective immediately)");
}

#[tokio::main]
async fn main() {
    default_policy_retries_rate_limited().await;
    custom_policy().await;
    streaming_retry_before_first_event().await;
    timeout_configuration();
    println!("all demos done (no real API needed)");
}
