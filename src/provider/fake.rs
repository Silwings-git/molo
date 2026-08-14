//! Test helper: a programmable fake Provider.
//!
//! A script is a sequence of per-turn replies ([`FakeReply`]); each `chat` /
//! `stream_chat` call consumes one turn, and the received requests are
//! recorded so tests can assert the Agent loop's process behavior (whether
//! tool results are fed back, whether history is appended in order, etc.).
//!
//! It is positioned as a test helper, not a production component: unit tests
//! of the Agent loop use it to inject deterministic replies without depending
//! on a real API; users testing their own Agent loops can do the same.

use crate::message::{Message, ToolCall};
use crate::provider::{
    ChatRequest, ChatResponse, FinishReason, Provider, ProviderCapabilities, ProviderError,
    ProviderRequestContext, StreamEvent, TimeoutStage, Usage,
};
use futures::stream::BoxStream;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// The error message returned once the script is exhausted.
const EXHAUSTED_MSG: &str = "fake provider: script exhausted";

/// One turn of reply from the script.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FakeReply {
    /// Plain text reply (`FinishReason::Stop`).
    Text(String),
    /// Text + reasoning (thinking-model scenario; reasoning is carried back
    /// verbatim in the history).
    TextWithReasoning {
        /// Reply text.
        content: String,
        /// Reasoning.
        reasoning: String,
    },
    /// Text + requests to call several tools (multiple requests from the same
    /// turn stay in one message); the text may be empty.
    ToolCalls {
        /// Reply text (may be empty).
        content: String,
        /// Tool calls requested this turn.
        calls: Vec<ToolCall>,
    },
    /// Specifies this turn's usage (default zero); wraps a base reply
    /// variant. Use it when a test needs to assert concrete token counts
    /// (e.g. the Agent's summation); existing script styles keep working
    /// unchanged.
    WithUsage {
        /// The wrapped base reply.
        reply: Box<FakeReply>,
        /// This turn's usage.
        usage: Usage,
    },
    /// This turn fails outright; the error is returned to the caller as-is.
    Error(ProviderError),
}

impl FakeReply {
    /// Convenience constructor: text reply + specified usage.
    pub fn text_with_usage(content: impl Into<String>, usage: Usage) -> Self {
        Self::WithUsage {
            reply: Box::new(Self::Text(content.into())),
            usage,
        }
    }
}

/// A programmable fake [`Provider`] — a script is a sequence of per-turn
/// replies, consumed in order by `chat` / `stream_chat`.
///
/// - Each call consumes one script turn and records the request; **once the
///   script is exhausted, calls return [`ProviderError::Protocol`] and do
///   not replay** — an Agent running an extra turn fails explicitly right
///   away in tests;
/// - `chat` and `stream_chat` consume the same script with the same
///   semantics, differing only in how the reply is delivered (an event stream
///   = several increments + one `Done`);
/// - [`requests`](FakeProvider::requests) returns a snapshot of the received
///   request history for tests to assert what the Agent sent to the model —
///   since a fake is lenient (e.g. if the Agent forgets to feed back tool
///   results, a real API would error but the fake would not), this is the
///   only way to catch such process bugs.
///
/// # Examples
///
/// ```rust
/// # #[tokio::main]
/// # async fn main() -> Result<(), molo::ProviderError> {
/// use molo::message::Message;
/// use molo::provider::{ChatRequest, FakeProvider, FakeReply, Provider};
///
/// let fake = FakeProvider::new([
///     FakeReply::Text("hi".into()),
///     FakeReply::Text("bye".into()),
/// ]);
///
/// let r = fake.chat(ChatRequest::default()).await?;
/// assert_eq!(r.message, Message::assistant("hi"));
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct FakeProvider {
    /// The script to consume; each call pops from the front of the queue.
    replies: Mutex<VecDeque<FakeReply>>,
    /// Received request history (recorded whether or not the script is
    /// exhausted).
    requests: Mutex<Vec<ChatRequest>>,
}

impl Clone for FakeProvider {
    /// Clones are independent copies: each holds and consumes its own script
    /// queue and request history without affecting the other (script calls
    /// made before the clone still consume from the start in the clone).
    fn clone(&self) -> Self {
        // std Mutex has no Clone: copy the contents under the lock and
        // rebuild (a std Mutex cannot clone two locks at once).
        let replies = self
            .replies
            .lock()
            .expect("FakeProvider internal lock poisoned")
            .clone();
        let requests = self
            .requests
            .lock()
            .expect("FakeProvider internal lock poisoned")
            .clone();
        Self {
            replies: Mutex::new(replies),
            requests: Mutex::new(requests),
        }
    }
}

impl FakeProvider {
    /// Constructs from a script sequence; `chat` / `stream_chat` consume it
    /// in order and error once exhausted.
    pub fn new(replies: impl IntoIterator<Item = FakeReply>) -> Self {
        Self {
            replies: Mutex::new(replies.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Appends one reply to the end of the script (for staged injection in
    /// tests).
    pub fn push(&self, reply: FakeReply) {
        self.replies
            .lock()
            .expect("FakeProvider internal lock poisoned")
            .push_back(reply);
    }

    /// A snapshot of the received request history, for tests to assert what
    /// the Agent sent to the model.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), molo::ProviderError> {
    /// use molo::message::Message;
    /// use molo::provider::{ChatRequest, FakeProvider, FakeReply, Provider};
    ///
    /// let fake = FakeProvider::new([FakeReply::Text("hi".into())]);
    /// fake.chat(ChatRequest {
    ///     messages: vec![Message::user("hi")],
    ///     ..Default::default()
    /// })
    /// .await?;
    ///
    /// assert_eq!(fake.requests()[0].messages, vec![Message::user("hi")]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn requests(&self) -> Vec<ChatRequest> {
        self.requests
            .lock()
            .expect("FakeProvider internal lock poisoned")
            .clone()
    }
}

#[async_trait::async_trait]
impl Provider for FakeProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            streaming: true,
            reasoning: true,
            tool_calls: true,
            parallel_tool_calls: true,
            structured_output: true,
            usage: true,
            context_cancellation: true,
            context_deadline: true,
        }
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ProviderError> {
        self.requests
            .lock()
            .expect("FakeProvider internal lock poisoned")
            .push(request);
        match self
            .replies
            .lock()
            .expect("FakeProvider internal lock poisoned")
            .pop_front()
        {
            None => Err(ProviderError::Protocol {
                message: EXHAUSTED_MSG.to_string(),
            }),
            Some(reply) => chat_response(reply),
        }
    }

    async fn chat_with_context(
        &self,
        request: ChatRequest,
        context: &ProviderRequestContext,
    ) -> Result<ChatResponse, ProviderError> {
        check_context(context)?;
        self.chat(request).await
    }

    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError> {
        self.requests
            .lock()
            .expect("FakeProvider internal lock poisoned")
            .push(request);
        // Unwrap the usage override recursively first (same as chat:
        // WithUsage only changes this turn's usage, not the reply body;
        // nested WithUsage also unwraps correctly).
        let (reply, usage_override) = match self
            .replies
            .lock()
            .expect("FakeProvider internal lock poisoned")
            .pop_front()
        {
            None => {
                return Err(ProviderError::Protocol {
                    message: EXHAUSTED_MSG.to_string(),
                });
            }
            Some(reply) => unwrap_usage(reply),
        };
        let mut events = stream_events(reply)?;
        if let Some(usage) = usage_override {
            // The last event in the sequence is always Done; replace its
            // usage.
            if let Some(Ok(StreamEvent::Done {
                usage: done_usage, ..
            })) = events.last_mut()
            {
                *done_usage = Some(usage);
            }
        }
        Ok(Box::pin(futures::stream::iter(events)))
    }

    async fn stream_chat_with_context(
        &self,
        request: ChatRequest,
        context: &ProviderRequestContext,
    ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError> {
        check_context(context)?;
        self.stream_chat(request).await
    }
}

fn check_context(context: &ProviderRequestContext) -> Result<(), ProviderError> {
    if context.is_cancelled() {
        Err(ProviderError::Cancelled)
    } else if context.is_expired() {
        Err(ProviderError::Timeout(TimeoutStage::Request))
    } else {
        Ok(())
    }
}

/// Converts one script reply into a non-streaming response.
///
/// Usage defaults to zero; `WithUsage` wrappers are unwrapped recursively and
/// overridden with the injected value (matching the streaming path).
fn chat_response(reply: FakeReply) -> Result<ChatResponse, ProviderError> {
    match reply {
        // Boundary behavior consistent with chat: this turn's failure is
        // returned as Err directly.
        FakeReply::Error(e) => Err(e),
        FakeReply::WithUsage { reply, usage } => {
            let mut response = chat_response(*reply)?;
            response.usage = usage;
            Ok(response)
        }
        FakeReply::Text(content) => Ok(ChatResponse {
            message: Message::assistant(content),
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        }),
        FakeReply::TextWithReasoning { content, reasoning } => Ok(ChatResponse {
            message: Message::assistant_with_reasoning(content, reasoning),
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        }),
        FakeReply::ToolCalls { content, calls } => Ok(ChatResponse {
            message: Message::Assistant {
                content,
                reasoning: None,
                tool_calls: calls,
            },
            finish_reason: FinishReason::Stop,
            usage: Usage::default(),
        }),
    }
}

/// Recursively unwraps `WithUsage` (matching the chat path): the outer usage
/// wins, the inner one is the fallback — nested `WithUsage(WithUsage(...))`
/// also unwraps correctly without panicking.
fn unwrap_usage(reply: FakeReply) -> (FakeReply, Option<Usage>) {
    match reply {
        FakeReply::WithUsage { reply, usage } => {
            let (inner, inner_usage) = unwrap_usage(*reply);
            (inner, Some(usage).or(inner_usage))
        }
        other => (other, None),
    }
}

/// Converts one script reply into a stream of events.
///
/// Unwrapped replies always carry zero usage (`Some`) — the fake mimics an
/// endpoint with `include_usage` on, so streaming-summary tests can assert
/// stably; `WithUsage` must be unwrapped by the caller first (this function
/// does not handle it, and its fallback arm returns an error rather than
/// panicking on an unwrapped call).
fn stream_events(
    reply: FakeReply,
) -> Result<Vec<Result<StreamEvent, ProviderError>>, ProviderError> {
    match reply {
        // Boundary behavior consistent with chat: this turn's failure makes
        // the method return Err directly.
        FakeReply::Error(e) => Err(e),
        FakeReply::Text(content) => Ok(vec![
            Ok(StreamEvent::Delta(content)),
            Ok(StreamEvent::Done {
                reason: FinishReason::Stop,
                usage: Some(Usage::default()),
            }),
        ]),
        // Scripted order: the content Delta first, then the Reasoning
        // fragment, then Done.
        FakeReply::TextWithReasoning { content, reasoning } => Ok(vec![
            Ok(StreamEvent::Delta(content)),
            Ok(StreamEvent::Reasoning(reasoning)),
            Ok(StreamEvent::Done {
                reason: FinishReason::Stop,
                usage: Some(Usage::default()),
            }),
        ]),
        FakeReply::ToolCalls { content, calls } => {
            let mut events = Vec::new();
            if !content.is_empty() {
                events.push(Ok(StreamEvent::Delta(content)));
            }
            events.extend(calls.into_iter().map(|c| {
                Ok(StreamEvent::ToolCall {
                    id: c.id,
                    name: c.name,
                    arguments: c.arguments,
                })
            }));
            events.push(Ok(StreamEvent::Done {
                reason: FinishReason::Stop,
                usage: Some(Usage::default()),
            }));
            Ok(events)
        }
        // Defensive fallback: on the normal path the caller (stream_chat) has
        // already unwrapped recursively, so this arm should not be reached;
        // return an error rather than panicking on an unwrapped call.
        FakeReply::WithUsage { .. } => Err(ProviderError::Protocol {
            message: "internal error: WithUsage must be unwrapped before stream_events".into(),
        }),
    }
}

/// `Arc<FakeProvider>` is also a Provider: tests share one instance behind an
/// Arc (used when wrapper tests assert internal request counts, such as
/// RetryProvider's attempt count).
#[async_trait::async_trait]
impl Provider for Arc<FakeProvider> {
    fn capabilities(&self) -> ProviderCapabilities {
        self.as_ref().capabilities()
    }

    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ProviderError> {
        self.as_ref().chat(request).await
    }

    async fn chat_with_context(
        &self,
        request: ChatRequest,
        context: &ProviderRequestContext,
    ) -> Result<ChatResponse, ProviderError> {
        self.as_ref().chat_with_context(request, context).await
    }

    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError> {
        self.as_ref().stream_chat(request).await
    }

    async fn stream_chat_with_context(
        &self,
        request: ChatRequest,
        context: &ProviderRequestContext,
    ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError> {
        self.as_ref()
            .stream_chat_with_context(request, context)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;
    use futures::StreamExt;

    #[tokio::test]
    async fn chat_consumes_script_in_order() {
        let fake = FakeProvider::new([FakeReply::Text("a".into()), FakeReply::Text("b".into())]);

        let r1 = fake.chat(ChatRequest::default()).await.unwrap();
        assert_eq!(r1.message, Message::assistant("a"));
        assert_eq!(r1.finish_reason, FinishReason::Stop);

        let r2 = fake.chat(ChatRequest::default()).await.unwrap();
        assert_eq!(r2.message, Message::assistant("b"));
    }

    #[tokio::test]
    async fn chat_returns_error_when_script_exhausted() {
        let fake = FakeProvider::new([FakeReply::Text("only".into())]);
        fake.chat(ChatRequest::default()).await.unwrap();

        let err = fake.chat(ChatRequest::default()).await.unwrap_err();
        assert!(matches!(err, ProviderError::Protocol { message: m } if m.contains("exhausted")));

        // The exhausted call still records its request, so assertions can
        // still see what the Agent last sent.
        assert_eq!(fake.requests().len(), 2);
    }

    #[tokio::test]
    async fn chat_with_text_and_reasoning() {
        let fake = FakeProvider::new([FakeReply::TextWithReasoning {
            content: "answer".into(),
            reasoning: "think".into(),
        }]);

        let r = fake.chat(ChatRequest::default()).await.unwrap();
        assert_eq!(
            r.message,
            Message::assistant_with_reasoning("answer", "think")
        );
    }

    #[tokio::test]
    async fn chat_with_tool_calls() {
        let calls = vec![ToolCall {
            id: "c1".into(),
            name: "add".into(),
            arguments: r#"{"a":1,"b":2}"#.into(),
        }];
        let fake = FakeProvider::new([FakeReply::ToolCalls {
            content: String::new(),
            calls: calls.clone(),
        }]);

        let r = fake.chat(ChatRequest::default()).await.unwrap();
        match r.message {
            Message::Assistant {
                content,
                reasoning,
                tool_calls,
            } => {
                assert_eq!(content, "");
                assert_eq!(reasoning, None);
                assert_eq!(tool_calls, calls);
            }
            other => panic!("expected assistant, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn chat_with_usage_override() {
        let usage = Usage::new(7, 3);
        let fake = FakeProvider::new([FakeReply::text_with_usage("hi", usage)]);

        let r = fake.chat(ChatRequest::default()).await.unwrap();
        assert_eq!(r.message, Message::assistant("hi"));
        assert_eq!(r.usage, usage);
    }

    #[tokio::test]
    async fn stream_with_usage_override() {
        let usage = Usage::new(7, 3);
        let fake = FakeProvider::new([FakeReply::text_with_usage("hi", usage)]);

        let mut stream = fake.stream_chat(ChatRequest::default()).await.unwrap();
        let events: Vec<StreamEvent> = stream.by_ref().map(|e| e.unwrap()).collect().await;
        // The last event is always Done, with the usage per the injected
        // value.
        assert_eq!(
            events.last(),
            Some(&StreamEvent::Done {
                reason: FinishReason::Stop,
                usage: Some(usage),
            })
        );
    }

    #[tokio::test]
    async fn stream_nested_usage_override_no_panic() {
        // Nested WithUsage(WithUsage(Text)): the streaming path unwraps
        // recursively, the outer usage wins, no panic.
        let usage = Usage::new(7, 3);
        let fake = FakeProvider::new([FakeReply::WithUsage {
            reply: Box::new(FakeReply::WithUsage {
                reply: Box::new(FakeReply::Text("hi".into())),
                usage: Usage::new(1, 1),
            }),
            usage,
        }]);

        let mut stream = fake.stream_chat(ChatRequest::default()).await.unwrap();
        let events: Vec<StreamEvent> = stream.by_ref().map(|e| e.unwrap()).collect().await;
        assert_eq!(events[0], StreamEvent::Delta("hi".into()));
        assert_eq!(
            events.last(),
            Some(&StreamEvent::Done {
                reason: FinishReason::Stop,
                usage: Some(usage),
            })
        );
    }

    #[tokio::test]
    async fn error_reply_passthrough_and_script_continues() {
        let fake = FakeProvider::new([
            FakeReply::Error(ProviderError::RateLimited { retry_after: None }),
            FakeReply::Text("ok".into()),
        ]);

        // This turn fails: the error is returned as-is.
        let err = fake.chat(ChatRequest::default()).await.unwrap_err();
        assert!(matches!(err, ProviderError::RateLimited { .. }));

        // The failed turn has been consumed; the next turn continues the
        // script.
        let r = fake.chat(ChatRequest::default()).await.unwrap();
        assert_eq!(r.message, Message::assistant("ok"));
    }

    #[tokio::test]
    async fn stream_chat_text_reply() {
        let fake = FakeProvider::new([FakeReply::TextWithReasoning {
            content: "hi".into(),
            reasoning: "think".into(),
        }]);

        let mut stream = fake.stream_chat(ChatRequest::default()).await.unwrap();
        // Scripted order: the content Delta first, then the Reasoning
        // fragment, then Done.
        assert_eq!(
            stream.next().await.unwrap().unwrap(),
            StreamEvent::Delta("hi".into())
        );
        assert_eq!(
            stream.next().await.unwrap().unwrap(),
            StreamEvent::Reasoning("think".into())
        );
        assert_eq!(
            stream.next().await.unwrap().unwrap(),
            StreamEvent::Done {
                reason: FinishReason::Stop,
                usage: Some(Usage::default()),
            }
        );
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn stream_chat_tool_call_reply() {
        let fake = FakeProvider::new([FakeReply::ToolCalls {
            content: "thinking aloud".into(),
            calls: vec![ToolCall {
                id: "c1".into(),
                name: "add".into(),
                arguments: r#"{"a":1}"#.into(),
            }],
        }]);

        let mut stream = fake.stream_chat(ChatRequest::default()).await.unwrap();
        // Non-empty text comes first as a Delta, then the tool-call event and
        // the closing Done.
        assert_eq!(
            stream.next().await.unwrap().unwrap(),
            StreamEvent::Delta("thinking aloud".into())
        );
        assert_eq!(
            stream.next().await.unwrap().unwrap(),
            StreamEvent::ToolCall {
                id: "c1".into(),
                name: "add".into(),
                arguments: r#"{"a":1}"#.into(),
            }
        );
        assert_eq!(
            stream.next().await.unwrap().unwrap(),
            StreamEvent::Done {
                reason: FinishReason::Stop,
                usage: Some(Usage::default()),
            }
        );
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn stream_chat_error_reply_returns_err() {
        let fake = FakeProvider::new([FakeReply::Error(ProviderError::Api {
            status: 400,
            code: None,
            message: "boom".into(),
        })]);
        // Boundary behavior consistent with chat: this turn's failure makes
        // the method return Err directly.
        match fake.stream_chat(ChatRequest::default()).await {
            Err(ProviderError::Api {
                status: 400,
                code: None,
                message: m,
            }) => assert_eq!(m, "boom"),
            Ok(_) => panic!("expected error"),
            Err(other) => panic!("expected Api error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stream_chat_exhausted_returns_err() {
        let fake = FakeProvider::new([FakeReply::Text("only".into())]);
        // Consume the only script turn first; a stream that is never polled
        // would be pointless, so drop it explicitly here.
        drop(fake.stream_chat(ChatRequest::default()).await.unwrap());

        match fake.stream_chat(ChatRequest::default()).await {
            Err(ProviderError::Protocol { message: m }) => assert!(m.contains("exhausted")),
            Ok(_) => panic!("expected error"),
            Err(other) => panic!("expected Protocol error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn requests_records_messages_passed() {
        let fake = FakeProvider::new([FakeReply::Text("hi".into())]);
        let messages = vec![Message::system("sys"), Message::user("hello")];

        fake.chat(ChatRequest {
            messages: messages.clone(),
            ..Default::default()
        })
        .await
        .unwrap();

        let requests = fake.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].messages, messages);
    }

    #[tokio::test]
    async fn push_appends_to_script() {
        let fake = FakeProvider::new([FakeReply::Text("first".into())]);
        fake.push(FakeReply::Text("second".into()));

        let r1 = fake.chat(ChatRequest::default()).await.unwrap();
        assert_eq!(r1.message, Message::assistant("first"));
        let r2 = fake.chat(ChatRequest::default()).await.unwrap();
        assert_eq!(r2.message, Message::assistant("second"));
    }

    #[tokio::test]
    async fn works_as_boxed_dyn_provider() {
        // The Agent holds providers as Box<dyn Provider>; the fake must
        // satisfy that shape.
        let fake: Box<dyn Provider> = Box::new(FakeProvider::new([FakeReply::Text("hi".into())]));
        let r = fake.chat(ChatRequest::default()).await.unwrap();
        assert_eq!(r.message, Message::assistant("hi"));
    }
}
