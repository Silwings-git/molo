//! Provider implementation for OpenAI-compatible APIs.
//!
//! The `base_url` constructor argument can point at OpenAI-compatible
//! endpoints such as OpenAI / DeepSeek / Zhipu / Moonshot, as well as Ollama's
//! OpenAI-compatible interface (pass an empty string for `api_key`).

use crate::message::{ContentBlock, Message, ToolCall};
use crate::provider::{
    ChatRequest, ChatResponse, FinishReason, ModelOptions, Provider, ProviderError, StreamEvent,
    TimeoutStage, Usage,
};
use crate::tool::ToolSchema;
use async_trait::async_trait;
use base64::Engine;
use futures::stream::{BoxStream, Stream, StreamExt, unfold};
use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Provider implementation for OpenAI-compatible APIs.
///
/// Binds base_url / api_key / model at construction; one instance serves one
/// model.
///
/// Timeouts are transport concerns owned by the implementation layer: `new`
/// ships defaults — connect 30s / non-streaming total 600s / streaming event
/// interval 60s / streaming total 30min, tunable via the chained
/// [`with_connect_timeout`](OpenAiProvider::with_connect_timeout) /
/// [`with_request_timeout`](OpenAiProvider::with_request_timeout) /
/// [`with_idle_timeout`](OpenAiProvider::with_idle_timeout) /
/// [`with_stream_timeout`](OpenAiProvider::with_stream_timeout).
///
/// Streaming has two timeouts with different roles: the idle timeout resets
/// on every received chunk and can only detect a "silent" connection; the
/// total duration is a wall-clock deadline — only it can terminate an active
/// but never-ending stream (an endpoint that keeps sending keep-alive lines).
///
/// # Examples
///
/// ```rust
/// use molo::provider::OpenAiProvider;
///
/// // Construction makes no network request; base_url can point at compatible
/// // endpoints like DeepSeek / Ollama (pass an empty api_key for Ollama).
/// let provider = OpenAiProvider::new(
///     "https://api.deepseek.com",
///     "sk-your-key",
///     "deepseek-chat",
/// );
/// # let _ = provider;
/// ```
#[derive(Clone)]
pub struct OpenAiProvider {
    base_url: String,
    api_key: String,
    model: String,
    client: reqwest::Client,
    /// Request timeout: for non-streaming, covers until the response body is
    /// fully read; for streaming, covers connect and response-header waiting
    /// (the body-reading phase is covered by the idle / streaming-total
    /// timeouts; long SSE streams are not subject to this limit).
    request_timeout: Duration,
    /// Streaming event-interval idle timeout: no data between events means
    /// the connection is dead (reqwest has no read-timeout, so timing is done
    /// per chunk at the SSE parsing layer).
    idle_timeout: Duration,
    /// Streaming total duration budget (wall-clock deadline); default 30min.
    /// Divides labor with idle — idle is reset by every chunk, so only the
    /// total duration can terminate an active but never-ending stream (an
    /// endpoint that keeps sending keep-alive comment lines).
    stream_timeout: Duration,
    /// Transport shape for structured output (default
    /// [`StructuredOutputMode::Native`]).
    structured_mode: StructuredOutputMode,
}

/// Transport shape for structured output (an OpenAiProvider construction
/// setting): decides how `response_format` is sent.
///
/// Only affects the **endpoint-side** best-effort constraint strength —
/// structural conformance is always guaranteed by framework-side validation
/// (jsonschema validation of the final answer + failure-feedback retries; see
/// [`ReActAgent`](crate::ReActAgent) and
/// [`ModelOptions::structured`](crate::provider::ModelOptions::structured)).
///
/// # Choosing a mode
///
/// - [`Native`](StructuredOutputMode::Native): OpenAI-native structured output
///   (the `json_schema` shape); when the schema satisfies the strict-mode
///   preconditions, `strict: true` is enabled automatically — the strongest
///   endpoint-side constraint;
/// - [`JsonObject`](StructuredOutputMode::JsonObject): constrains only "must
///   be valid JSON", not the structure — use it when older compatible
///   endpoints do not support the `json_schema` shape; structure relies on
///   framework-side validation (which may cost extra retry rounds);
/// - [`Off`](StructuredOutputMode::Off): does not send `response_format` —
///   no structured constraint in the prompt; structure relies entirely on
///   framework-side validation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StructuredOutputMode {
    /// Sends the `json_schema` shape of `response_format` (default).
    #[default]
    Native,
    /// Sends the `json_object` shape of `response_format`.
    JsonObject,
    /// Does not send `response_format`.
    Off,
}

impl std::fmt::Debug for OpenAiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Hand-written Debug that masks api_key: derive would print the
        // credential in plain text, leaking the key through any {:?} logs or
        // panic messages.
        f.debug_struct("OpenAiProvider")
            .field("base_url", &self.base_url)
            .field("api_key", &"***")
            .field("model", &self.model)
            .field("request_timeout", &self.request_timeout)
            .field("idle_timeout", &self.idle_timeout)
            .field("stream_timeout", &self.stream_timeout)
            .field("structured_mode", &self.structured_mode)
            .finish_non_exhaustive()
    }
}

impl OpenAiProvider {
    /// Constructs with base_url / api_key / model; one instance serves one
    /// model.
    ///
    /// `base_url` may point at compatible endpoints such as OpenAI / DeepSeek
    /// / Zhipu / Moonshot, as well as Ollama's OpenAI-compatible interface
    /// (pass an empty string for `api_key`).
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            model: model.into(),
            client: build_client(Duration::from_secs(30)),
            request_timeout: Duration::from_secs(600),
            idle_timeout: Duration::from_secs(60),
            stream_timeout: Duration::from_secs(1800),
            structured_mode: StructuredOutputMode::default(),
        }
    }

    /// Connect timeout (Client-level, shared by streaming / non-streaming);
    /// default 30s.
    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.client = build_client(timeout);
        self
    }

    /// Request timeout; default 600s. For non-streaming, covers until the
    /// response body is fully read; for streaming, covers connect and
    /// response-header waiting, while the body-reading phase is covered by
    /// [`with_idle_timeout`](Self::with_idle_timeout) /
    /// [`with_stream_timeout`](Self::with_stream_timeout).
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    /// Streaming event-interval idle timeout; default 60s. Deltas keep
    /// arriving during generation; an interval far exceeding this value means
    /// the connection is dead.
    pub fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = timeout;
        self
    }

    /// Streaming total duration budget; default 30min. A wall-clock deadline
    /// starting when the stream is established, dividing labor with the idle
    /// timeout: the idle timeout resets on every received chunk and can only
    /// detect a "silent" connection; only the total duration can terminate an
    /// active but never-ending stream (an endpoint that keeps sending
    /// keep-alive lines). When the deadline hits, the stream terminates with
    /// a [`ProviderError::Timeout`](crate::provider::ProviderError::Timeout)
    /// error; complete lines already buffered before the deadline are
    /// delivered normally.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use molo::provider::OpenAiProvider;
    /// use std::time::Duration;
    ///
    /// let provider = OpenAiProvider::new("https://api.example.com", "sk", "model")
    ///     .with_connect_timeout(Duration::from_secs(10))
    ///     .with_idle_timeout(Duration::from_secs(120))
    ///     .with_stream_timeout(Duration::from_secs(900));
    /// # let _ = provider;
    /// ```
    pub fn with_stream_timeout(mut self, timeout: Duration) -> Self {
        self.stream_timeout = timeout;
        self
    }

    /// Transport shape for structured output; default
    /// [`Native`](StructuredOutputMode::Native).
    ///
    /// When older compatible endpoints do not support the `json_schema`
    /// shape, fall back to [`JsonObject`](StructuredOutputMode::JsonObject)
    /// (constrains only "is JSON") or [`Off`](StructuredOutputMode::Off)
    /// (do not send) — **no automatic retry-based downgrade**; transport
    /// policy and retries are decoupled. Structural conformance is always
    /// guaranteed by framework-side validation.
    pub fn with_structured_output_mode(mut self, mode: StructuredOutputMode) -> Self {
        self.structured_mode = mode;
        self
    }

    /// Builds the wire `response_format` per the transport mode (`Off` or no
    /// schema → not sent).
    fn wire_response_format(
        &self,
        schema: Option<&serde_json::Value>,
    ) -> Option<OpenAiResponseFormat> {
        match (self.structured_mode, schema) {
            (StructuredOutputMode::Off, _) | (_, None) => None,
            (StructuredOutputMode::Native, Some(schema)) => {
                Some(OpenAiResponseFormat::from_schema(schema))
            }
            (StructuredOutputMode::JsonObject, Some(_)) => {
                Some(OpenAiResponseFormat::json_object())
            }
        }
    }
}

/// Builds the HTTP client: the connect timeout is configured at the Client
/// level; no TLS customization, so construction cannot fail.
fn build_client(connect_timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(connect_timeout)
        .build()
        .expect("reqwest Client::builder cannot fail without TLS config")
}

#[async_trait]
impl Provider for OpenAiProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ProviderError> {
        let wire = OpenAiChatRequest {
            model: self.model.clone(),
            messages: request
                .messages
                .iter()
                .map(OpenAiMessage::from_message)
                .collect(),
            tools: request
                .tools
                .iter()
                .map(OpenAiToolDef::from_schema)
                .collect(),
            stream: None,
            stream_options: None,
            temperature: request.options.temperature,
            max_tokens: request.options.max_tokens,
            response_format: self.wire_response_format(request.options.structured.as_ref()),
            extra: wire_extra(&request.options),
        };

        let url = format!("{}/chat/completions", self.base_url);
        let mut req = self.client.post(url).json(&wire);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        // Non-streaming total timeout (per-request; default 600s). Timeout
        // errors are classified as the Request stage (reqwest cannot
        // distinguish them from connect timeouts; diagnose as a
        // "non-streaming request").
        req = req.timeout(self.request_timeout);

        let resp = req.send().await.map_err(|e| {
            if e.is_timeout() {
                ProviderError::Timeout(TimeoutStage::Request)
            } else {
                map_network_error(e)
            }
        })?;
        let status = resp.status();
        let headers = resp.headers().clone();
        let body = read_response_body(resp).await?;
        if !status.is_success() {
            return Err(map_status_error(status.as_u16(), &headers, &body));
        }

        parse_response(&body)
    }

    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> Result<BoxStream<'static, Result<StreamEvent, ProviderError>>, ProviderError> {
        let wire = OpenAiChatRequest {
            model: self.model.clone(),
            messages: request
                .messages
                .iter()
                .map(OpenAiMessage::from_message)
                .collect(),
            tools: request
                .tools
                .iter()
                .map(OpenAiToolDef::from_schema)
                .collect(),
            stream: Some(true),
            // Streaming usage: explicitly enable include_usage, otherwise the
            // endpoint's streamed response carries no usage.
            stream_options: Some(StreamOptions {
                include_usage: true,
            }),
            temperature: request.options.temperature,
            max_tokens: request.options.max_tokens,
            response_format: self.wire_response_format(request.options.structured.as_ref()),
            extra: wire_extra(&request.options),
        };

        let url = format!("{}/chat/completions", self.base_url);
        let mut req = self.client.post(url).json(&wire);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        // Streaming has no total timeout: reqwest's timeout spans from
        // sending the request to finishing the response body, which would kill
        // long SSE streams. Connect timeout is at the Client level, and event
        // intervals are covered by the parsing layer's idle timeout.
        // Response-header waiting gets its own timeout: an endpoint that
        // accepts the request but never sends response headers would hang
        // forever (idle timing only starts at the first body byte).
        let resp = match tokio::time::timeout(self.request_timeout, req.send()).await {
            Ok(result) => result.map_err(map_network_error)?,
            Err(_) => return Err(ProviderError::Timeout(TimeoutStage::Request)),
        };
        let status = resp.status();
        let headers = resp.headers().clone();
        if !status.is_success() {
            let body = read_response_body(resp).await?;
            return Err(map_status_error(status.as_u16(), &headers, &body));
        }

        let aggregator = Arc::new(Mutex::new(ToolCallAggregator::default()));
        Ok(Box::pin(sse_stream(
            resp.bytes_stream(),
            self.idle_timeout,
            self.stream_timeout,
            aggregator,
        )))
    }
}

/// Non-streaming response-body reading: Content-Length pre-check + streaming
/// read with cumulative truncation + total timeout.
///
/// An endpoint may lie about content-length or omit it entirely, so the read
/// still accumulates and checks ([`MAX_RESPONSE_BODY`]); exceeding the limit
/// produces a [`ProviderError::Api`] (status 0) and terminates. The total
/// timeout is [`RESPONSE_BODY_TIMEOUT`] — when an error response body drips
/// data at a very low rate, streaming requests have no per-request total
/// timeout to fall back on, so without it the call would hang forever.
async fn read_response_body(resp: reqwest::Response) -> Result<String, ProviderError> {
    let result = tokio::time::timeout(RESPONSE_BODY_TIMEOUT, async {
        if let Some(len) = resp.content_length()
            && len > MAX_RESPONSE_BODY as u64
        {
            return Err(limit_error("response body exceeds size limit"));
        }
        let mut stream = resp.bytes_stream();
        let mut buf = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(map_network_error)?;
            if buf.len() + chunk.len() > MAX_RESPONSE_BODY {
                return Err(limit_error("response body exceeds size limit"));
            }
            buf.extend_from_slice(&chunk);
        }
        Ok(String::from_utf8_lossy(&buf).into_owned())
    })
    .await;
    match result {
        Ok(r) => r,
        Err(_) => Err(ProviderError::Timeout(TimeoutStage::ResponseBody)),
    }
}

/// Total timeout for response-body reading (60s): a slowly dripping error
/// response body no longer hangs forever.
const RESPONSE_BODY_TIMEOUT: Duration = Duration::from_secs(60);

/// SSE parsing pipeline: parse line by line → aggregate → fall back to the
/// stashed Done when the stream terminates.
///
/// Error semantics: after the first Err item no success events are produced;
/// the stream terminates after transport-level errors (connection drop / idle
/// stall), while line-level errors do not terminate the stream.
fn sse_stream(
    chunks: impl Stream<Item = Result<impl AsRef<[u8]>, reqwest::Error>> + Unpin,
    idle_timeout: Duration,
    stream_timeout: Duration,
    aggregator: Arc<Mutex<ToolCallAggregator>>,
) -> impl Stream<Item = Result<StreamEvent, ProviderError>> {
    let tail = aggregator.clone();
    sse_lines(chunks, idle_timeout, stream_timeout)
        .map(move |line| match line {
            Err(e) => {
                let mut agg = aggregator.lock().expect("SSE aggregator lock poisoned");
                // Transport-level errors (connection drop / idle stall)
                // after Done has been delivered carry no information; drop
                // them — the stream terminates with either Done or Err, never
                // both.
                if agg.finished {
                    return Vec::new();
                }
                // Errors terminate the stream: set the errored latch and drop
                // the stashed Done — after an error no success events are
                // produced, and a finish_reason resent by the endpoint cannot
                // resurrect the Done.
                agg.set_errored();
                vec![Err(e)]
            }
            Ok(line) => {
                let mut agg = aggregator.lock().expect("SSE aggregator lock poisoned");
                // Data after [DONE] is dead; drop it: Done is always the last
                // success event on the stream.
                if agg.finished {
                    Vec::new()
                } else {
                    parse_sse_line(&line, &mut agg)
                }
            }
        })
        .flat_map(futures::stream::iter)
        // When the stream ends naturally (EOF without [DONE]), fall back to
        // emitting the stashed Done.
        // `once` evaluates lazily: the future only runs when the chain is
        // polled (after the stream has been consumed); with eager evaluation
        // done_reason would always be None when the chain is built, and the
        // EOF fallback would never fire.
        .chain(
            futures::stream::once(async move {
                tail.lock()
                    .expect("SSE aggregator lock poisoned")
                    .flush_done()
            })
            .flat_map(futures::stream::iter),
        )
}

/// Splits the byte stream into lines on `\n`; lines spanning chunks are
/// joined automatically.
///
/// `idle_timeout`: timeout for waiting on a new chunk — an event interval
/// exceeding it means the connection is dead; the stream terminates after
/// producing `ProviderError::Timeout` (reqwest has no read-timeout, so this
/// is done at the parsing layer).
///
/// `stream_timeout`: total stream duration budget (wall-clock deadline,
/// counted from construction) — checked before every network read; when it
/// elapses, produce `ProviderError::Timeout` and terminate the stream. It
/// divides labor with idle: idle is reset by every chunk, so only the total
/// duration can terminate an active but never-ending stream (an endpoint
/// that keeps sending keep-alive comment lines). Complete lines already
/// buffered are delivered normally before the deadline; no false kills (the
/// deadline is only checked before network reads).
///
/// The line buffer has a size limit ([`MAX_SSE_LINE`]): when a broken
/// endpoint keeps sending chunks without `\n`, exceeding the limit produces
/// an error and terminates the stream — otherwise memory would grow forever
/// and the idle timeout would never fire.
///
/// **Error semantics**: after any of the three error kinds — timeout / line
/// over-limit / network drop — the stream **terminates** (the next poll
/// returns None); no buffered leftovers are produced after the error item.
fn sse_lines(
    chunks: impl Stream<Item = Result<impl AsRef<[u8]>, reqwest::Error>> + Unpin,
    idle_timeout: Duration,
    stream_timeout: Duration,
) -> impl Stream<Item = Result<String, ProviderError>> {
    // State element 3 = terminated flag: after a timeout / over-limit error,
    // the next poll returns None and the stream ends naturally (the idle
    // timeout does not terminate the underlying stream — pending never ends,
    // so the flag stops it);
    // State element 4 = total-duration deadline (wall clock, counted from
    // construction).
    let deadline = Instant::now() + stream_timeout;
    unfold(
        (chunks, Vec::new(), false, deadline),
        move |(mut chunks, mut buf, terminated, deadline)| async move {
            if terminated {
                return None;
            }
            loop {
                // Buffered lines are produced first: deadline and idle checks
                // only happen when new data must be read.
                if let Some(line) = pop_line(&mut buf) {
                    return Some((Ok(line), (chunks, buf, false, deadline)));
                }
                // Total duration budget: an active but never-ending stream
                // is killed when it expires (idle is reset by every chunk;
                // this is the only fallback).
                if Instant::now() >= deadline {
                    return Some((
                        Err(ProviderError::Timeout(TimeoutStage::StreamTotal)),
                        (chunks, buf, true, deadline),
                    ));
                }
                let next = match tokio::time::timeout(idle_timeout, chunks.next()).await {
                    Ok(next) => next,
                    Err(_) => {
                        return Some((
                            Err(ProviderError::Timeout(TimeoutStage::Idle)),
                            (chunks, buf, true, deadline),
                        ));
                    }
                };
                match next {
                    Some(Ok(bytes)) => {
                        if buf.len() + bytes.as_ref().len() > MAX_SSE_LINE {
                            return Some((
                                Err(ProviderError::Api {
                                    status: 0,
                                    message: format!(
                                        "stream line exceeds size limit ({} bytes)",
                                        MAX_SSE_LINE
                                    ),
                                }),
                                (chunks, buf, true, deadline),
                            ));
                        }
                        buf.extend_from_slice(bytes.as_ref());
                    }
                    // Transport drop: surface the error and terminate the
                    // stream — otherwise undelivered buffered lines would
                    // appear as Ok after the error item, violating the
                    // "errors terminate the stream as an Err item" semantics.
                    Some(Err(e)) => {
                        return Some((Err(map_network_error(e)), (chunks, buf, true, deadline)));
                    }
                    None => {
                        if buf.is_empty() {
                            return None;
                        }
                        return Some((
                            Ok(to_line(std::mem::take(&mut buf))),
                            (chunks, buf, false, deadline),
                        ));
                    }
                }
            }
        },
    )
}

/// Per-line SSE buffer limit (1 MiB); exceeding it is treated as a malicious
/// / broken endpoint: an error is produced and the stream terminates.
const MAX_SSE_LINE: usize = 1 << 20;
/// Non-streaming response-body limit (16 MiB): Content-Length pre-check +
/// streaming read with cumulative truncation — a large response body from a
/// broken endpoint could exhaust memory (the total timeout only limits rate,
/// not size).
const MAX_RESPONSE_BODY: usize = 16 << 20;

/// Takes one line from the front of the buffer (including its terminator);
/// returns `None` when no complete line is buffered yet.
fn pop_line(buf: &mut Vec<u8>) -> Option<String> {
    let pos = buf.iter().position(|&b| b == b'\n')?;
    let rest = buf.split_off(pos + 1);
    let line = std::mem::replace(buf, rest);
    Some(to_line(line))
}

/// Strips the trailing `\r\n` / `\n` and converts the bytes to text.
fn to_line(mut bytes: Vec<u8>) -> String {
    while matches!(bytes.last(), Some(b'\r' | b'\n')) {
        bytes.pop();
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Parses one SSE line, producing 0..N events.
///
/// Non-data lines — empty lines / comment lines / `event:` lines — are
/// skipped; `data: [DONE]` marks the end of the stream, producing the stashed
/// [`StreamEvent::Done`] after which the stream terminates itself; when the
/// JSON after `data:` fails to parse, an error event is produced.
///
/// Tool-call fragments go into the aggregator first; when the model finishes
/// this turn (finish_reason arrives), all of this turn's calls are emitted
/// whole in complete form from the aggregator. **Done is not produced
/// here** — the streaming usage (the usage-only trailing chunk) arrives after
/// the finish_reason chunk, so Done is deferred until the stream terminates
/// ([DONE] / EOF) and is emitted together with the usage.
///
/// **Error semantics**: in-line errors (parse failure / aggregation limit)
/// are produced via [`line_error`] — it sets the errored latch and drops the
/// stashed Done, so "no success events after an error" (a finish_reason
/// resent by the endpoint cannot resurrect the Done). Line-level errors do
/// not terminate the stream (one bad line does not affect later lines).
fn parse_sse_line(
    line: &str,
    tool_calls: &mut ToolCallAggregator,
) -> Vec<Result<StreamEvent, ProviderError>> {
    let mut events = Vec::new();
    if line.is_empty() || line.starts_with(':') {
        return events;
    }
    let Some(data) = line.strip_prefix("data:") else {
        return events;
    };
    let data = data.trim_start();
    if data == "[DONE]" {
        // [DONE] = end-of-stream marker: first emit the closing item (Done,
        // or the "no finish_reason" error), then set finished — every line
        // arriving afterwards is dropped.
        let events = tool_calls.flush_done();
        tool_calls.finished = true;
        return events;
    }

    let chunk = match serde_json::from_str::<OpenAiStreamChunk>(data) {
        Ok(chunk) => chunk,
        Err(_) => {
            // A parse failure is not an HTTP source; status counts as 0.
            return line_error(
                tool_calls,
                ProviderError::Api {
                    status: 0,
                    message: format!("invalid stream event: {}", extract_error_message(data)),
                },
            );
        }
    };

    // Vendor error chunk (the OpenAI streamed-error shape `{"error": {...}}`):
    // parsed explicitly, then emitted as an Api error with the latch set.
    if let Some(error) = chunk.error {
        return line_error(
            tool_calls,
            ProviderError::Api {
                status: 0,
                message: format!("provider stream error: {}", error.message),
            },
        );
    }

    // The usage-only trailing chunk (empty choices) is stashed first; when
    // finish_reason or EOF arrives it is emitted whole with the Done. It may
    // arrive in the same chunk as finish_reason or later.
    if let Some(usage) = chunk.usage {
        tool_calls.push_usage(usage);
    }
    let Some(choice) = chunk.choices.into_iter().next() else {
        return events;
    };
    // No success events after an error: Delta obeys the same latch as other
    // branches (currently unreachable thanks to the stream termination
    // semantics; defense in depth).
    if let Some(content) = choice.delta.content
        && !tool_calls.errored
    {
        events.push(Ok(StreamEvent::Delta(content)));
    }
    if let Some(reasoning) = choice.delta.reasoning_content
        && !tool_calls.errored
    {
        // Reasoning fragments are forwarded as they arrive (like Delta), not
        // deferred to the end of the turn: consumers render thinking in real
        // time. Each fragment is bounded by the SSE line limit, so no
        // aggregation-side size check is needed.
        events.push(Ok(StreamEvent::Reasoning(reasoning)));
    }

    if let Some(calls) = choice.delta.tool_calls {
        for call in calls {
            if let Err(e) = tool_calls.push_chunk(call) {
                return line_error(tool_calls, e);
            }
        }
    }
    if let Some(finish_reason) = choice.finish_reason {
        // No success events after an error has been seen: ToolCall is
        // blocked just like Done / usage.
        if !tool_calls.errored {
            // The model has finished this turn: all tool-request fragments
            // have necessarily arrived; emit them whole.
            for call in tool_calls.take_all() {
                events.push(Ok(StreamEvent::ToolCall {
                    id: call.id,
                    name: call.name,
                    arguments: call.arguments,
                }));
            }
        }
        tool_calls.set_done_reason(map_finish_reason(&finish_reason));
    }
    events
}

/// Maps the vendor's finish_reason string to
/// [`FinishReason`](crate::provider::FinishReason).
///
/// Common reasons map to typed variants; "tool_calls" means the model ended
/// this turn by requesting tool calls, which is semantically a natural stop;
/// other vendor-specific reasons are surfaced as their raw strings.
fn map_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" | "tool_calls" => FinishReason::Stop,
        "length" => FinishReason::Length,
        other => FinishReason::Other(other.to_string()),
    }
}

/// Aggregator for streamed tool-call fragments.
///
/// The model may request several tools in one turn; fragments are attributed
/// by `index`. `arguments` fragments with the same index are concatenated in
/// arrival order; id / name are only carried by the first fragment. Reasoning
/// (thinking) is not aggregated here: reasoning fragments are forwarded as
/// they arrive (see [`parse_sse_line`]).
///
/// It also stashes the Done that only becomes complete at the end of the
/// stream: the reason is recorded when finish_reason arrives (tools are
/// emitted whole then), usage arrives later (the usage-only trailing chunk),
/// and when the stream terminates
/// ([`flush_done`](ToolCallAggregator::flush_done)) all three are combined
/// into the Done.
#[derive(Default)]
struct ToolCallAggregator {
    /// Tool calls received so far, in order of first appearance.
    calls: Vec<AccumulatedCall>,
    /// The model has finished this turn (finish_reason has arrived); the Done
    /// is emitted with the usage when the stream terminates.
    done_reason: Option<FinishReason>,
    /// This turn's usage (carried by the usage-only trailing chunk); None
    /// when the endpoint did not return it.
    usage: Option<Usage>,
    /// Error-seen latch: once set, no new Done stash / usage is accepted —
    /// no success events after an error (a finish_reason resent by the
    /// endpoint after an error cannot resurrect the Done).
    errored: bool,
    /// `[DONE]` has arrived and the Done has been produced: every line
    /// arriving afterwards is dropped — `Done` is always the last success
    /// event on the stream, regardless of whether the endpoint behaves.
    finished: bool,
}

/// Maximum number of tool calls per turn; exceeding it is treated as a
/// malicious / broken endpoint: an error is produced and the stream
/// terminates.
const MAX_TOOL_CALLS: usize = 64;
/// Cumulative limit for aggregated tool-argument text (1 MiB); exceeding it
/// produces an error and terminates the stream.
const MAX_ACCUMULATED_TEXT: usize = 1 << 20;

/// Unified error for size-limit violations (non-HTTP source; status counts
/// as 0).
fn limit_error(message: &str) -> ProviderError {
    ProviderError::Api {
        status: 0,
        message: message.into(),
    }
}

/// In-line parse error: produce an error event and **set the errored latch**
/// (dropping the stashed Done) — "no success events after an error".
/// Otherwise, a bad line after finish_reason was stashed would still let the
/// later [DONE] flush an Ok(Done).
fn line_error(
    tool_calls: &mut ToolCallAggregator,
    e: ProviderError,
) -> Vec<Result<StreamEvent, ProviderError>> {
    tool_calls.set_errored();
    vec![Err(e)]
}

/// The accumulated state of one tool call.
struct AccumulatedCall {
    index: usize,
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallAggregator {
    fn push_chunk(&mut self, chunk: OpenAiToolCallChunk) -> Result<(), ProviderError> {
        // Count and cumulative-length limits: guard against unbounded growth
        // and O(n²) degradation from broken endpoints (linear find per
        // chunk).
        let name = chunk
            .function
            .as_ref()
            .and_then(|f| f.name.clone())
            .unwrap_or_default();
        let arguments = chunk
            .function
            .as_ref()
            .and_then(|f| f.arguments.clone())
            .unwrap_or_default();
        if let Some(call) = self.calls.iter_mut().find(|c| c.index == chunk.index) {
            if call.id.is_empty() {
                call.id = chunk.id.unwrap_or_default();
            }
            if call.name.is_empty() {
                call.name = name;
            }
            if call.arguments.len() + arguments.len() > MAX_ACCUMULATED_TEXT {
                return Err(limit_error("tool arguments exceed size limit"));
            }
            call.arguments.push_str(&arguments);
        } else {
            // The count limit only applies to new calls: continuation
            // fragments for existing indices are not wrongly rejected (when
            // the 64th call exactly hits the limit, its continuations can
            // still be appended).
            if self.calls.len() >= MAX_TOOL_CALLS {
                return Err(limit_error("tool call count exceeds limit"));
            }
            // The first fragment of a new call is also subject to the text
            // limit (a soft limit not backed by the line limit).
            if arguments.len() > MAX_ACCUMULATED_TEXT {
                return Err(limit_error("tool arguments exceed size limit"));
            }
            self.calls.push(AccumulatedCall {
                index: chunk.index,
                id: chunk.id.unwrap_or_default(),
                name,
                arguments,
            });
        }
        Ok(())
    }

    /// Takes all aggregated calls from this turn and clears them.
    fn take_all(&mut self) -> Vec<AccumulatedCall> {
        std::mem::take(&mut self.calls)
    }

    /// Records this turn's end reason (Done stashed, emitted when the stream
    /// terminates); no-op after an error has been seen.
    fn set_done_reason(&mut self, reason: FinishReason) {
        if !self.errored {
            self.done_reason = Some(reason);
        }
    }

    /// Stashes this turn's usage (carried by the usage-only trailing chunk;
    /// within one turn, a later one overrides an earlier one); no-op after an
    /// error has been seen.
    fn push_usage(&mut self, usage: OpenAiUsage) {
        if !self.errored {
            self.usage = Some(map_usage(usage));
        }
    }

    /// Emits the closing event when the stream terminates; four states:
    /// - errored (an Err has been produced) → empty: the error is delivered;
    ///   nothing more is produced;
    /// - finished ([DONE] processed) → empty: the [DONE] branch has already
    ///   produced the closing item; the EOF fallback produces nothing
    ///   (otherwise the taken done_reason would wrongly yield "without
    ///   finish_reason");
    /// - done_reason stashed → `Ok(Done{reason, usage})` (normal / EOF
    ///   fallback);
    /// - never stashed → `Err("stream ended without finish_reason")`: a
    ///   truncated / empty stream is an error, no longer silently treated as
    ///   a successful turn end (EOF strictness, matching the chat path's
    ///   error on empty choices).
    fn flush_done(&mut self) -> Vec<Result<StreamEvent, ProviderError>> {
        if self.errored || self.finished {
            return Vec::new();
        }
        let Some(reason) = self.done_reason.take() else {
            return vec![Err(ProviderError::Api {
                status: 0,
                message: "stream ended without finish_reason".into(),
            })];
        };
        let usage = self.usage.take();
        vec![Ok(StreamEvent::Done { reason, usage })]
    }

    /// When the stream terminates with an error, sets the latch and drops the
    /// stashed Done: no success events after an error (a finish_reason resent
    /// by the endpoint cannot resurrect the Done).
    fn set_errored(&mut self) {
        self.errored = true;
        self.done_reason = None;
    }
}

/// Maps the vendor's wire usage to [`Usage`](crate::provider::Usage).
fn map_usage(wire: OpenAiUsage) -> Usage {
    Usage {
        prompt_tokens: wire.prompt_tokens,
        completion_tokens: wire.completion_tokens,
        total_tokens: wire.total_tokens,
    }
}

/// Maps a transport error to
/// [`ProviderError`](crate::provider::ProviderError); details are carried as
/// text, without introducing reqwest's concrete error types.
/// Timeouts get their own category,
/// [`ProviderError::Timeout`](crate::provider::ProviderError::Timeout), which
/// is retryable and distinct from business errors.
fn map_network_error(e: reqwest::Error) -> ProviderError {
    if e.is_timeout() {
        ProviderError::Timeout(TimeoutStage::Transport)
    } else {
        ProviderError::Network(e.to_string())
    }
}

/// Non-success status codes → [`ProviderError`](crate::provider::ProviderError):
/// 429 is split by vendor semantics — a message containing
/// `insufficient_quota` (quota exhausted; retrying is pointless) goes to
/// `Api`, other rate limits go to `RateLimited` carrying the wait parsed
/// from the `Retry-After` header (numeric seconds);
/// all other status codes are carried as `Api`.
fn map_status_error(
    status: u16,
    headers: &reqwest::header::HeaderMap,
    body: &str,
) -> ProviderError {
    let message = extract_error_message(body);
    if status == 429 && !message.contains("insufficient_quota") {
        ProviderError::RateLimited {
            retry_after: extract_retry_after(headers),
        }
    } else {
        ProviderError::Api { status, message }
    }
}

/// Parses the `Retry-After` response header (numeric seconds); HTTP date
/// formats are not parsed (treated as None).
fn extract_retry_after(headers: &reqwest::header::HeaderMap) -> Option<Duration> {
    headers
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

/// Extracts the vendor's error.message from an error response body; returns
/// the body verbatim when extraction fails.
fn extract_error_message(body: &str) -> String {
    serde_json::from_str::<OpenAiErrorBody>(body)
        .ok()
        .and_then(|body| body.error)
        .map(|error| error.message)
        .unwrap_or_else(|| body.to_string())
}

/// Parses a vendor response into [`ChatResponse`](crate::provider::ChatResponse).
///
/// One wire reply (text + reasoning + multiple tool requests from the same
/// turn) maps to **one** [`Message::Assistant`] message, one-to-one with the
/// vendor wire structure; splitting it apart would break conversation-history
/// validation at vendors like DeepSeek.
fn parse_response(body: &str) -> Result<ChatResponse, ProviderError> {
    // A parse failure is not an HTTP source; status counts as 0.
    let parsed: OpenAiChatResponse =
        serde_json::from_str(body).map_err(|e| ProviderError::Api {
            status: 0,
            message: format!("invalid provider response: {e}"),
        })?;
    let choice = parsed
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| ProviderError::Api {
            status: 0,
            message: "provider returned empty choices".to_string(),
        })?;

    let wire = choice.message;
    // Reply content is currently text; future multimodal replies fall back
    // to serializing as text.
    let content = match wire.content {
        Some(serde_json::Value::String(s)) => s,
        Some(other) => other.to_string(),
        None => String::new(),
    };
    let raw_tool_calls = wire.tool_calls.unwrap_or_default();
    // Count limit: same constant as the streaming aggregator — a malicious
    // endpoint could pack a huge response body with massive tool calls, each
    // executing user code.
    if raw_tool_calls.len() > MAX_TOOL_CALLS {
        return Err(ProviderError::Api {
            status: 0,
            message: format!("too many tool calls ({})", raw_tool_calls.len()),
        });
    }
    let tool_calls: Vec<ToolCall> = raw_tool_calls
        .into_iter()
        .map(|call| ToolCall {
            id: call.id,
            name: call.function.name,
            arguments: call.function.arguments,
        })
        .collect();
    // When the model produced nothing it is still an empty Assistant,
    // keeping the convention that each turn has exactly one reply.
    // Usage is always present: compatible endpoints that omit usage in
    // non-streaming responses count as zero (consumers need not handle
    // Option).
    Ok(ChatResponse {
        message: Message::Assistant {
            content,
            reasoning: wire.reasoning_content,
            tool_calls,
        },
        finish_reason: map_finish_reason(&choice.finish_reason),
        usage: parsed.usage.map(map_usage).unwrap_or_default(),
    })
}

/// The wire request format for OpenAI-compatible APIs (sent only, never
/// parsed).
#[derive(serde::Serialize)]
struct OpenAiChatRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    /// Tool definitions; not sent when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAiToolDef>,
    /// Must be explicitly enabled for streaming requests; not sent for
    /// non-streaming requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    /// Explicitly enables usage reporting for streaming requests (otherwise
    /// streamed endpoint responses never carry usage); sent only when
    /// streaming.
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    /// Sampling temperature; vendor default when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    /// Maximum number of tokens for the reply; vendor default when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    /// Structured output: endpoint-side best-effort constraint (the
    /// OpenAI-compatible `json_schema` shape); unsupported endpoints ignore
    /// it or error, with framework-side validation as the fallback.
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<OpenAiResponseFormat>,
    /// Vendor extension parameters: keys are wire field names, serialized
    /// into the request body verbatim.
    #[serde(flatten)]
    extra: BTreeMap<String, serde_json::Value>,
}

/// Wire shapes of `response_format`: `json_schema` (OpenAI-native structured
/// output, optionally strict) or `json_object` (the degraded shape that only
/// constrains "is valid JSON").
#[derive(Serialize)]
struct OpenAiResponseFormat {
    #[serde(rename = "type")]
    kind: &'static str,
    /// Strict-mode switch (the `strict` parameter of OpenAI-native structured
    /// output); not carried by the `json_object` shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
    /// Schema container; not carried by the `json_object` shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    json_schema: Option<OpenAiJsonSchema>,
}

#[derive(Serialize)]
struct OpenAiJsonSchema {
    name: &'static str,
    schema: serde_json::Value,
}

impl OpenAiResponseFormat {
    /// Builds the `json_schema` shape from the schema in
    /// ModelOptions.structured.
    ///
    /// **strict strengthening**: when the schema satisfies OpenAI's
    /// strict-mode preconditions (top-level object with `required` covering
    /// all properties; the `schemars` generation path satisfies this
    /// naturally), add `additionalProperties: false` and set `strict: true`
    /// — the endpoint constraint is upgraded from best-effort to strict; when
    /// the preconditions are not met, send non-strict (hand-written schemas
    /// are not reshaped automatically; conformance is guaranteed by
    /// framework-side validation retries).
    fn from_schema(schema: &serde_json::Value) -> Self {
        if schema.get("type").and_then(|t| t.as_str()) == Some("object")
            && let (Some(properties), Some(required)) = (
                schema.get("properties").and_then(|p| p.as_object()),
                schema.get("required").and_then(|r| r.as_array()),
            )
            && required.len() == properties.len()
            && required.iter().all(|k| k.is_string())
        {
            let mut strict_schema = schema.clone();
            strict_schema["additionalProperties"] = serde_json::Value::Bool(false);
            Self {
                kind: "json_schema",
                strict: Some(true),
                json_schema: Some(OpenAiJsonSchema {
                    name: "output",
                    schema: strict_schema,
                }),
            }
        } else {
            Self {
                kind: "json_schema",
                strict: None,
                json_schema: Some(OpenAiJsonSchema {
                    name: "output",
                    schema: schema.clone(),
                }),
            }
        }
    }

    /// The `json_object` shape: constrains only "is valid JSON" (the degraded
    /// shape for older compatible endpoints).
    fn json_object() -> Self {
        Self {
            kind: "json_object",
            strict: None,
            json_schema: None,
        }
    }
}

/// Extracts vendor extension parameters from [`ModelOptions`].
///
/// Filters out framework-managed fields (model / messages / tools / stream /
/// stream_options / temperature / max_tokens); on name collision the
/// framework field wins.
fn wire_extra(options: &ModelOptions) -> BTreeMap<String, serde_json::Value> {
    options
        .extra
        .iter()
        .filter(|(key, _)| {
            !matches!(
                key.as_str(),
                "model"
                    | "messages"
                    | "tools"
                    | "stream"
                    | "stream_options"
                    | "temperature"
                    | "max_tokens"
            )
        })
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

/// Usage switch for streaming requests (the OpenAI `stream_options`
/// parameter).
#[derive(serde::Serialize)]
struct StreamOptions {
    include_usage: bool,
}

/// The wire message format for OpenAI-compatible APIs (shared by requests
/// and responses).
#[derive(serde::Serialize, serde::Deserialize)]
struct OpenAiMessage {
    role: String,
    /// Content: a string (single text block) or an array of content blocks
    /// (multiple blocks, e.g. image-text mixes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    content: Option<serde_json::Value>,
    /// The model's reasoning; provided by thinking models like DeepSeek /
    /// Qwen3.
    ///
    /// Must be carried back verbatim when the history is sent again, or the
    /// API rejects the request; the `alias` accommodates endpoints like
    /// Ollama that use the `reasoning` field name.
    #[serde(default, skip_serializing_if = "Option::is_none", alias = "reasoning")]
    reasoning_content: Option<String>,
    /// The model's tool-call requests; carried only by assistant messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAiToolCall>>,
    /// Which call the tool result corresponds to; carried only by messages
    /// with role=tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

impl OpenAiMessage {
    fn from_message(message: &Message) -> Self {
        match message {
            Message::System(content) => Self::text("system", content),
            Message::User(blocks) => Self {
                role: "user".to_string(),
                content: user_content(blocks),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
            },
            Message::Assistant {
                content,
                reasoning,
                tool_calls,
            } => Self {
                role: "assistant".to_string(),
                // A pure-tool turn (no text) does not send an empty content
                // field, matching vendor wire conventions.
                content: (!content.is_empty()).then(|| serde_json::Value::String(content.clone())),
                reasoning_content: reasoning.clone(),
                tool_calls: (!tool_calls.is_empty()).then(|| {
                    tool_calls
                        .iter()
                        .map(|call| OpenAiToolCall {
                            id: call.id.clone(),
                            kind: "function".to_string(),
                            function: OpenAiFunctionCall {
                                name: call.name.clone(),
                                arguments: call.arguments.clone(),
                            },
                        })
                        .collect()
                }),
                tool_call_id: None,
            },
            Message::ToolResult { id, content } => Self {
                role: "tool".to_string(),
                content: Some(serde_json::Value::String(content.clone())),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: Some(id.clone()),
            },
        }
    }

    fn text(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: Some(serde_json::Value::String(content.to_string())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

/// Maps the content blocks of a user message to the wire content field.
///
/// A single text block stays a string (compatible with existing wire
/// shapes); multiple blocks (mixed text / image) serialize to an array of
/// content blocks `[{"type": "text", "text": "..."}, {"type": "image_url",
/// "image_url": {"url": "data:image/png;base64,..."}}]`.
fn user_content(blocks: &[ContentBlock]) -> Option<serde_json::Value> {
    match blocks {
        [ContentBlock::Text(text)] => Some(serde_json::Value::String(text.clone())),
        blocks => {
            let parts: Vec<serde_json::Value> = blocks.iter().map(content_block_wire).collect();
            (!parts.is_empty()).then_some(serde_json::Value::Array(parts))
        }
    }
}

/// One content block → the wire content-block shape.
///
/// Images are carried as the OpenAI-compatible `image_url` block with a
/// base64 data URL built from the raw bytes (no URL passed through: the
/// framework only ships images it holds itself; callers with a hosted URL
/// can still reference it via a text block).
fn content_block_wire(block: &ContentBlock) -> serde_json::Value {
    match block {
        ContentBlock::Text(text) => serde_json::json!({"type": "text", "text": text}),
        ContentBlock::Image(image) => serde_json::json!({
            "type": "image_url",
            "image_url": {
                "url": format!(
                    "data:{};base64,{}",
                    image.mime_type,
                    base64::engine::general_purpose::STANDARD.encode(&image.data)
                )
            }
        }),
    }
}

/// One tool call in the wire format.
#[derive(serde::Serialize, serde::Deserialize)]
struct OpenAiToolCall {
    id: String,
    /// Tool type; always "function".
    #[serde(rename = "type")]
    kind: String,
    function: OpenAiFunctionCall,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct OpenAiFunctionCall {
    name: String,
    /// Arguments generated by the model (JSON text).
    arguments: String,
}

/// A tool definition in the wire format.
#[derive(serde::Serialize)]
struct OpenAiToolDef {
    #[serde(rename = "type")]
    kind: &'static str,
    function: OpenAiFunctionDef,
}

#[derive(serde::Serialize)]
struct OpenAiFunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

impl OpenAiToolDef {
    fn from_schema(schema: &ToolSchema) -> Self {
        Self {
            kind: "function",
            function: OpenAiFunctionDef {
                name: schema.name.clone(),
                description: schema.description.clone(),
                parameters: schema.parameters.clone(),
            },
        }
    }
}

/// The wire usage format for OpenAI-compatible APIs (shared by non-streaming
/// responses and streamed trailing chunks).
#[derive(serde::Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// The wire response format for OpenAI-compatible APIs (only the needed
/// fields).
#[derive(serde::Deserialize)]
struct OpenAiChatResponse {
    choices: Vec<OpenAiChoice>,
    /// Non-streaming usage; compatible endpoints that omit it count as zero
    /// (non-streaming always has it; consumers need not handle `Option`).
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(serde::Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
    finish_reason: String,
}

/// The streamed response chunk format for OpenAI-compatible APIs (only the
/// needed fields).
#[derive(serde::Deserialize)]
struct OpenAiStreamChunk {
    /// Empty by default: a usage-only trailing chunk may omit the choices key
    /// (a compatible shape with the same semantics as "choices empty").
    #[serde(default)]
    choices: Vec<OpenAiStreamChoice>,
    /// Streamed usage: with include_usage on, the usage-only trailing chunk
    /// (empty choices) at the end of the stream carries this turn's totals;
    /// None for most chunks.
    #[serde(default)]
    usage: Option<OpenAiUsage>,
    /// Vendor error chunk (`{"error": {"message": ...}}`; the OpenAI
    /// streamed-error shape, parsed explicitly rather than inferred from
    /// missing fields).
    #[serde(default)]
    error: Option<OpenAiStreamError>,
}

/// A vendor streamed error.
#[derive(serde::Deserialize)]
struct OpenAiStreamError {
    message: String,
}

#[derive(serde::Deserialize)]
struct OpenAiStreamChoice {
    delta: OpenAiStreamDelta,
    /// null in most chunks; the end reason is only given in the last chunk.
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(serde::Deserialize)]
struct OpenAiStreamDelta {
    /// Incremental content; null in most chunks (the first chunk only carries
    /// role).
    #[serde(default)]
    content: Option<String>,
    /// Incremental reasoning fragments; thinking models emit reasoning first,
    /// then content or tool requests.
    /// The `alias` accommodates endpoints like Ollama that use the
    /// `reasoning` field name.
    #[serde(default, alias = "reasoning")]
    reasoning_content: Option<String>,
    /// Tool-call fragments; the same call is distinguished by `index`, and
    /// `arguments` are fragments to be concatenated.
    #[serde(default)]
    tool_calls: Option<Vec<OpenAiToolCallChunk>>,
}

/// One tool-call fragment in a streamed response.
#[derive(serde::Deserialize)]
struct OpenAiToolCallChunk {
    /// Which call this fragment belongs to; fragments with the same index
    /// belong to the same call.
    index: usize,
    /// Call id; carried only by the first fragment.
    #[serde(default)]
    id: Option<String>,
    /// Function info; some fragments only carry an arguments increment.
    #[serde(default)]
    function: Option<OpenAiFunctionChunk>,
}

#[derive(serde::Deserialize)]
struct OpenAiFunctionChunk {
    /// Function name; carried only by the first fragment.
    #[serde(default)]
    name: Option<String>,
    /// An arguments fragment (part of the JSON text); fragments with the same
    /// index are concatenated in order into the complete JSON.
    #[serde(default)]
    arguments: Option<String>,
}

/// Error response body: { "error": { "message": "..." } }.
#[derive(serde::Deserialize)]
struct OpenAiErrorBody {
    error: Option<OpenAiErrorDetail>,
}

#[derive(serde::Deserialize)]
struct OpenAiErrorDetail {
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ImageContent;
    use reqwest::header::{HeaderMap, RETRY_AFTER};
    use serde_json::json;

    // —— 429 classification rules ——

    #[test]
    fn status_429_quota_goes_to_api() {
        // Quota exhausted (message contains insufficient_quota): a business
        // error, not retried.
        let body = r#"{"error":{"message":"You exceeded your current quota: insufficient_quota"}}"#;
        let err = map_status_error(429, &HeaderMap::new(), body);
        assert!(matches!(err, ProviderError::Api { status: 429, .. }));
    }

    #[test]
    fn status_429_rate_limited_with_retry_after() {
        let mut headers = HeaderMap::new();
        headers.insert(RETRY_AFTER, "5".parse().unwrap());
        let body = r#"{"error":{"message":"rate limit exceeded"}}"#;
        let err = map_status_error(429, &headers, body);
        assert!(matches!(
            err,
            ProviderError::RateLimited { retry_after: Some(d) } if d == Duration::from_secs(5)
        ));
    }

    #[test]
    fn status_429_without_retry_after_header() {
        let body = r#"{"error":{"message":"rate limit exceeded"}}"#;
        let err = map_status_error(429, &HeaderMap::new(), body);
        assert!(matches!(
            err,
            ProviderError::RateLimited { retry_after: None }
        ));
    }

    #[test]
    fn status_5xx_goes_to_api_with_status() {
        let body = r#"{"error":{"message":"server overloaded"}}"#;
        let err = map_status_error(503, &HeaderMap::new(), body);
        assert!(matches!(err, ProviderError::Api { status: 503, .. }));
    }

    #[test]
    fn maps_message_to_wire() {
        let message = Message::user("hi");
        let wire = OpenAiMessage::from_message(&message);
        assert_eq!(
            serde_json::to_value(&wire).unwrap(),
            json!({"role": "user", "content": "hi"})
        );
    }

    #[test]
    fn maps_multi_block_user_content_to_wire_parts() {
        // Multiple content blocks serialize to the wire's content-block
        // array; a single text block stays a string (see maps_message_to_wire).
        let message = Message::user_blocks(vec![
            ContentBlock::Text("take a look:".into()),
            ContentBlock::Text("see the attachment".into()),
        ]);
        assert_eq!(
            serde_json::to_value(OpenAiMessage::from_message(&message)).unwrap(),
            json!({
                "role": "user",
                "content": [
                    {"type": "text", "text": "take a look:"},
                    {"type": "text", "text": "see the attachment"}
                ]
            })
        );
    }

    #[test]
    fn maps_image_block_to_wire_data_url() {
        // An image block serializes to the OpenAI-compatible image_url block
        // with a base64 data URL; a single image block forces the array shape
        // (a bare string would not be a valid image carrier).
        let message = Message::user_blocks(vec![ContentBlock::Image(ImageContent::new(
            "image/png",
            b"fake-png-bytes".to_vec(),
        ))]);
        let expected_b64 = base64::engine::general_purpose::STANDARD.encode(b"fake-png-bytes");
        assert_eq!(
            serde_json::to_value(OpenAiMessage::from_message(&message)).unwrap(),
            json!({
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {"url": format!("data:image/png;base64,{expected_b64}")}
                }]
            })
        );
    }

    #[test]
    fn maps_mixed_text_and_image_blocks() {
        // Text + image in one user message: both serialize into the content
        // block array, text as text and image as image_url (the wire shape
        // consumed by qwen-vl / gpt-4o-class multimodal models).
        let message = Message::user_blocks(vec![
            ContentBlock::Text("what is this?".into()),
            ContentBlock::Image(ImageContent::new("image/jpeg", vec![1, 2, 3])),
        ]);
        assert_eq!(
            serde_json::to_value(OpenAiMessage::from_message(&message)).unwrap(),
            json!({
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AQID"}}
                ]
            })
        );
    }

    #[test]
    fn maps_system_and_assistant_roles() {
        assert_eq!(
            OpenAiMessage::from_message(&Message::system("s")).role,
            "system"
        );
        assert_eq!(
            OpenAiMessage::from_message(&Message::assistant("a")).role,
            "assistant"
        );
    }

    #[test]
    fn maps_tool_call_to_wire() {
        let message = Message::Assistant {
            content: String::new(),
            reasoning: None,
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "calculator".into(),
                arguments: r#"{"expression":"1+1"}"#.into(),
            }],
        };
        assert_eq!(
            serde_json::to_value(OpenAiMessage::from_message(&message)).unwrap(),
            json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": "{\"expression\":\"1+1\"}"}
                }]
            })
        );
    }

    #[test]
    fn maps_tool_result_to_wire() {
        let message = Message::ToolResult {
            id: "call_1".into(),
            content: "2".into(),
        };
        assert_eq!(
            serde_json::to_value(OpenAiMessage::from_message(&message)).unwrap(),
            json!({"role": "tool", "tool_call_id": "call_1", "content": "2"})
        );
    }

    #[test]
    fn maps_reasoning_to_wire() {
        let message = Message::assistant_with_reasoning("2", "thinking process");
        assert_eq!(
            serde_json::to_value(OpenAiMessage::from_message(&message)).unwrap(),
            json!({
                "role": "assistant",
                "content": "2",
                "reasoning_content": "thinking process"
            })
        );
    }

    #[test]
    fn omits_empty_reasoning_in_wire() {
        let message = Message::assistant("2");
        assert_eq!(
            serde_json::to_value(OpenAiMessage::from_message(&message)).unwrap(),
            json!({"role": "assistant", "content": "2"})
        );
    }

    #[test]
    fn parses_reasoning_content() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"2","reasoning_content":"thinking process"},"finish_reason":"stop"}]}"#;
        let response = parse_response(body).unwrap();
        assert_eq!(
            response.message,
            Message::assistant_with_reasoning("2", "thinking process")
        );
    }

    #[test]
    fn parses_reasoning_with_tool_call() {
        // The model requests a tool after reasoning: content is empty and the
        // reasoning rides along with the tool call.
        let body = r#"{"choices":[{"message":{"role":"assistant","content":null,"reasoning_content":"decided","tool_calls":[{"id":"call_1","type":"function","function":{"name":"calculator","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#;
        let response = parse_response(body).unwrap();
        assert_eq!(
            response.message,
            Message::Assistant {
                content: String::new(),
                reasoning: Some("decided".into()),
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "calculator".into(),
                    arguments: "{}".into(),
                }],
            }
        );
    }

    #[test]
    fn accepts_ollama_reasoning_field() {
        // Ollama's qwen3 uses the `reasoning` field name; deserialization
        // must accommodate it.
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"2","reasoning":"thinking"},"finish_reason":"stop"}]}"#;
        let response = parse_response(body).unwrap();
        assert_eq!(
            response.message,
            Message::assistant_with_reasoning("2", "thinking")
        );
    }

    #[test]
    fn serializes_tool_defs() {
        let schema = ToolSchema {
            name: "calculator".into(),
            description: "Evaluate expressions".into(),
            parameters: json!({"type": "object"}),
        };
        let def = OpenAiToolDef::from_schema(&schema);
        assert_eq!(
            serde_json::to_value(&def).unwrap(),
            json!({
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate expressions",
                    "parameters": {"type": "object"}
                }
            })
        );
    }

    #[test]
    fn omits_empty_tools_in_request() {
        let request = OpenAiChatRequest {
            model: "m".into(),
            messages: vec![OpenAiMessage::from_message(&Message::user("hi"))],
            tools: vec![],
            stream: None,
            stream_options: None,
            temperature: None,
            max_tokens: None,
            response_format: None,
            extra: BTreeMap::new(),
        };
        assert_eq!(
            serde_json::to_value(&request).unwrap(),
            json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        );
    }

    #[test]
    fn serializes_structured_output_as_response_format() {
        // Top-level object with required covering all properties → strict
        // mode triggers: additionalProperties: false is added and strict:
        // true is sent.
        let schema = json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        });
        let options = ModelOptions {
            structured: Some(schema.clone()),
            ..Default::default()
        };
        let request = OpenAiChatRequest {
            model: "m".into(),
            messages: vec![OpenAiMessage::from_message(&Message::user("hi"))],
            tools: vec![],
            stream: None,
            stream_options: None,
            temperature: None,
            max_tokens: None,
            response_format: options
                .structured
                .as_ref()
                .map(OpenAiResponseFormat::from_schema),
            extra: BTreeMap::new(),
        };
        let mut expected_schema = schema.clone();
        expected_schema["additionalProperties"] = json!(false);
        assert_eq!(
            serde_json::to_value(&request).unwrap(),
            json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {
                    "type": "json_schema",
                    "strict": true,
                    "json_schema": { "name": "output", "schema": expected_schema }
                }
            })
        );
    }

    #[test]
    fn strict_mode_skipped_when_required_incomplete() {
        // Optional properties exist (required does not cover all) → the
        // strict-mode preconditions are not met: send non-strict
        // (best-effort), with the schema passed through unchanged.
        let schema = json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" },
                "country": { "type": "string" },
            },
            "required": ["city"],
        });
        let request = OpenAiChatRequest {
            model: "m".into(),
            messages: vec![OpenAiMessage::from_message(&Message::user("hi"))],
            tools: vec![],
            stream: None,
            stream_options: None,
            temperature: None,
            max_tokens: None,
            response_format: Some(OpenAiResponseFormat::from_schema(&schema)),
            extra: BTreeMap::new(),
        };
        assert_eq!(
            serde_json::to_value(&request).unwrap(),
            json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": { "name": "output", "schema": schema }
                }
            })
        );
    }

    #[test]
    fn json_object_mode_omits_schema() {
        // The degraded shape: constrains only "is valid JSON", without schema
        // / strict.
        let wire = OpenAiResponseFormat::json_object();
        assert_eq!(
            serde_json::to_value(&wire).unwrap(),
            json!({ "type": "json_object" })
        );
    }

    #[test]
    fn response_format_mode_gating() {
        // Transport-mode gating: Off sends nothing; JsonObject sends
        // json_object; Native sends json_schema (default); no schema means
        // nothing is ever sent.
        let schema = json!({ "type": "object" });
        let provider = OpenAiProvider::new("http://x", "k", "m");
        assert!(provider.wire_response_format(None).is_none());
        assert!(matches!(
            provider.wire_response_format(Some(&schema)),
            Some(OpenAiResponseFormat {
                kind: "json_schema",
                ..
            })
        ));

        let off = OpenAiProvider::new("http://x", "k", "m")
            .with_structured_output_mode(StructuredOutputMode::Off);
        assert!(off.wire_response_format(Some(&schema)).is_none());

        let json_object = OpenAiProvider::new("http://x", "k", "m")
            .with_structured_output_mode(StructuredOutputMode::JsonObject);
        assert!(matches!(
            json_object.wire_response_format(Some(&schema)),
            Some(OpenAiResponseFormat {
                kind: "json_object",
                ..
            })
        ));
    }

    #[test]
    fn serializes_model_options_in_request() {
        let request = OpenAiChatRequest {
            model: "m".into(),
            messages: vec![OpenAiMessage::from_message(&Message::user("hi"))],
            tools: vec![],
            stream: None,
            stream_options: None,
            temperature: Some(0.5), // exactly representable in f32, avoiding precision differences
            max_tokens: Some(128),
            response_format: None,
            extra: BTreeMap::new(),
        };
        assert_eq!(
            serde_json::to_value(&request).unwrap(),
            json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.5,
                "max_tokens": 128
            })
        );
    }

    #[test]
    fn passes_through_extra_options() {
        // Vendor-specific parameters pass through extra verbatim, no
        // framework update needed.
        let mut extra = BTreeMap::new();
        extra.insert("top_p".into(), serde_json::json!(0.9));
        extra.insert(
            "response_format".into(),
            serde_json::json!({"type": "json_object"}),
        );
        let request = OpenAiChatRequest {
            model: "m".into(),
            messages: vec![OpenAiMessage::from_message(&Message::user("hi"))],
            tools: vec![],
            stream: None,
            stream_options: None,
            temperature: None,
            max_tokens: None,
            response_format: None,
            extra,
        };
        assert_eq!(
            serde_json::to_value(&request).unwrap(),
            json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "top_p": 0.9,
                "response_format": {"type": "json_object"}
            })
        );
    }

    #[test]
    fn extra_keys_colliding_with_managed_fields_are_filtered() {
        // Extra keys colliding with framework-managed fields are filtered
        // out in favor of the typed fields.
        let mut extra = BTreeMap::new();
        extra.insert("temperature".into(), serde_json::json!(1.0));
        extra.insert("top_p".into(), serde_json::json!(0.9));
        let options = ModelOptions {
            temperature: Some(0.5),
            max_tokens: None,
            extra,
            structured: None,
        };
        let request = OpenAiChatRequest {
            model: "m".into(),
            messages: vec![OpenAiMessage::from_message(&Message::user("hi"))],
            tools: vec![],
            stream: None,
            stream_options: None,
            temperature: options.temperature,
            max_tokens: options.max_tokens,
            response_format: options
                .structured
                .as_ref()
                .map(OpenAiResponseFormat::from_schema),
            extra: wire_extra(&options),
        };
        assert_eq!(
            serde_json::to_value(&request).unwrap(),
            json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.5,
                "top_p": 0.9
            })
        );
    }

    #[test]
    fn parses_chat_response() {
        let body = r#"{"id":"chatcmpl-1","object":"chat.completion","created":1,"model":"gpt-4o-mini","choices":[{"index":0,"message":{"role":"assistant","content":"hi!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}"#;
        let response = parse_response(body).unwrap();
        assert_eq!(response.message, Message::assistant("hi!"));
        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert_eq!(response.usage, Usage::new(5, 3));
    }

    #[test]
    fn parses_tool_call_response() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"calculator","arguments":"{\"expression\": \"1+1\"}"}}]},"finish_reason":"tool_calls"}]}"#;
        let response = parse_response(body).unwrap();
        assert_eq!(
            response.message,
            Message::Assistant {
                content: String::new(),
                reasoning: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "calculator".into(),
                    arguments: r#"{"expression": "1+1"}"#.into(),
                }],
            }
        );
        assert_eq!(response.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn parses_text_with_tool_calls() {
        // Text and tool requests in the same turn: merged into one Assistant
        // message, preserving the wire structure.
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"let me compute","tool_calls":[{"id":"call_1","type":"function","function":{"name":"calculator","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#;
        let response = parse_response(body).unwrap();
        assert_eq!(
            response.message,
            Message::Assistant {
                content: "let me compute".into(),
                reasoning: None,
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "calculator".into(),
                    arguments: "{}".into(),
                }],
            }
        );
    }

    #[test]
    fn parses_multiple_tool_calls_in_one_message() {
        // Multiple tool requests from the same turn must stay in one
        // Assistant message; they must not be split apart.
        let body = r#"{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"calculator","arguments":"{}"}},{"id":"b","type":"function","function":{"name":"calculator","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#;
        let response = parse_response(body).unwrap();
        assert_eq!(
            response.message,
            Message::Assistant {
                content: String::new(),
                reasoning: None,
                tool_calls: vec![
                    ToolCall {
                        id: "a".into(),
                        name: "calculator".into(),
                        arguments: "{}".into(),
                    },
                    ToolCall {
                        id: "b".into(),
                        name: "calculator".into(),
                        arguments: "{}".into(),
                    },
                ],
            }
        );
    }

    #[test]
    fn maps_length_finish_reason() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":""},"finish_reason":"length"}]}"#;
        let response = parse_response(body).unwrap();
        assert_eq!(response.message, Message::assistant(""));
        assert_eq!(response.finish_reason, FinishReason::Length);
    }

    #[test]
    fn maps_unknown_finish_reason_to_other() {
        // Vendor-specific reasons are surfaced as raw strings, never silently
        // dropped.
        let body = r#"{"choices":[{"message":{"role":"assistant","content":""},"finish_reason":"content_filter"}]}"#;
        let response = parse_response(body).unwrap();
        assert_eq!(
            response.finish_reason,
            FinishReason::Other("content_filter".into())
        );
    }

    #[test]
    fn rejects_empty_choices() {
        let body = r#"{"choices":[]}"#;
        assert!(parse_response(body).is_err());
    }

    #[test]
    fn missing_usage_defaults_to_zero() {
        // Compatible endpoints that omit usage count as zero — non-streaming
        // always has usage, so consumers need not handle Option.
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}"#;
        let response = parse_response(body).unwrap();
        assert_eq!(response.usage, Usage::default());
    }

    #[test]
    fn extracts_error_message() {
        let body = r#"{"error":{"message":"Incorrect API key","type":"invalid_request_error"}}"#;
        assert_eq!(extract_error_message(body), "Incorrect API key");
    }

    #[test]
    fn falls_back_to_raw_body() {
        let body = "not json";
        assert_eq!(extract_error_message(body), "not json");
    }

    /// Test helper: unwraps all parse results so event sequences can be
    /// asserted directly.
    fn parse_events(line: &str, aggregator: &mut ToolCallAggregator) -> Vec<StreamEvent> {
        parse_sse_line(line, aggregator)
            .into_iter()
            .map(|event| event.expect("test input must not produce an error"))
            .collect()
    }

    #[test]
    fn stream_parses_delta() {
        let line = r#"data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#;
        assert_eq!(
            parse_events(line, &mut ToolCallAggregator::default()),
            vec![StreamEvent::Delta("hi".to_string())]
        );
    }

    #[test]
    fn stream_parses_finish_reason() {
        let line = r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        let mut aggregator = ToolCallAggregator::default();
        // The reason is stashed, not emitted immediately (it comes with usage
        // when the stream terminates; see the usage-only chunk test).
        assert!(parse_events(line, &mut aggregator).is_empty());
        assert_eq!(
            aggregator.flush_done(),
            vec![Ok(StreamEvent::Done {
                reason: FinishReason::Stop,
                usage: None,
            })]
        );
    }

    #[test]
    fn stream_maps_length_finish_reason() {
        let line = r#"data: {"choices":[{"delta":{},"finish_reason":"length"}]}"#;
        let mut aggregator = ToolCallAggregator::default();
        assert!(parse_events(line, &mut aggregator).is_empty());
        assert_eq!(
            aggregator.flush_done(),
            vec![Ok(StreamEvent::Done {
                reason: FinishReason::Length,
                usage: None,
            })]
        );
    }

    #[test]
    fn stream_maps_tool_calls_finish_reason() {
        let line = r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}"#;
        let mut aggregator = ToolCallAggregator::default();
        assert!(parse_events(line, &mut aggregator).is_empty());
        assert_eq!(
            aggregator.flush_done(),
            vec![Ok(StreamEvent::Done {
                reason: FinishReason::Stop,
                usage: None,
            })]
        );
    }

    #[test]
    fn stream_parses_usage_only_tail_chunk() {
        // With include_usage on, the stream ends with a usage-only trailing
        // chunk (empty choices): usage is stashed and emitted with the Done
        // on flush.
        let mut aggregator = ToolCallAggregator::default();
        let done_line = r#"data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}"#;
        let usage_line = r#"data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}"#;

        assert_eq!(
            parse_events(done_line, &mut aggregator),
            vec![StreamEvent::Delta("ok".to_string())]
        );
        assert!(parse_events(usage_line, &mut aggregator).is_empty());
        assert_eq!(
            aggregator.flush_done(),
            vec![Ok(StreamEvent::Done {
                reason: FinishReason::Stop,
                usage: Some(Usage::new(5, 3)),
            })]
        );
    }

    #[test]
    fn stream_parses_usage_in_finish_chunk() {
        // Compatible endpoints carry usage in the same chunk as finish_reason:
        // it is stashed the same way and emitted with the Done.
        let line = r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}"#;
        let mut aggregator = ToolCallAggregator::default();
        assert!(parse_events(line, &mut aggregator).is_empty());
        assert_eq!(
            aggregator.flush_done(),
            vec![Ok(StreamEvent::Done {
                reason: FinishReason::Stop,
                usage: Some(Usage::new(5, 3)),
            })]
        );
    }

    #[test]
    fn stream_skips_irrelevant_lines() {
        let mut aggregator = ToolCallAggregator::default();
        assert!(parse_sse_line("", &mut aggregator).is_empty());
        assert!(parse_sse_line(": keep-alive", &mut aggregator).is_empty());
        assert!(parse_sse_line("event: message", &mut aggregator).is_empty());
        // [DONE] without finish_reason = truncation error (EOF strictness).
        let events = parse_sse_line("data: [DONE]", &mut aggregator);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Err(ProviderError::Api { status: 0, .. })
        ));
        assert!(parse_sse_line(r#"data: {"choices":[]}"#, &mut aggregator).is_empty());
        assert!(
            parse_sse_line(
                r#"data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}"#,
                &mut aggregator
            )
            .is_empty()
        );
    }

    #[test]
    fn stream_rejects_invalid_json() {
        let line = "data: not-json";
        let events = parse_sse_line(line, &mut ToolCallAggregator::default());
        match events.as_slice() {
            [Err(ProviderError::Api { status: 0, message })] => {
                assert_eq!(message, "invalid stream event: not-json")
            }
            _ => panic!("expected single api error, got {events:?}"),
        }
    }

    #[test]
    fn stream_extracts_provider_error() {
        let line = r#"data: {"error":{"message":"rate limited"}}"#;
        let events = parse_sse_line(line, &mut ToolCallAggregator::default());
        let err = events.into_iter().next().unwrap().unwrap_err();
        assert!(err.to_string().contains("rate limited"));
    }

    #[test]
    fn stream_forwards_reasoning_chunks() {
        // Reasoning fragments are forwarded as they arrive (not deferred to
        // the end of the turn): each line's fragment is emitted immediately.
        let mut aggregator = ToolCallAggregator::default();
        let line1 =
            r#"data: {"choices":[{"delta":{"reasoning_content":"first"},"finish_reason":null}]}"#;
        let line2 = r#"data: {"choices":[{"delta":{"reasoning_content":" multiply"},"finish_reason":null}]}"#;
        let line3 = r#"data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}"#;

        assert_eq!(
            parse_events(line1, &mut aggregator),
            vec![StreamEvent::Reasoning("first".to_string())]
        );
        assert_eq!(
            parse_events(line2, &mut aggregator),
            vec![StreamEvent::Reasoning(" multiply".to_string())]
        );
        assert_eq!(
            parse_events(line3, &mut aggregator),
            vec![StreamEvent::Delta("answer".to_string())]
        );
        assert_eq!(
            aggregator.flush_done(),
            vec![Ok(StreamEvent::Done {
                reason: FinishReason::Stop,
                usage: None,
            })]
        );
    }

    #[test]
    fn stream_reasoning_with_tool_call() {
        // Reasoning fragments arrive before the tool request and are forwarded
        // immediately; the tool call is still emitted whole at the end of the
        // turn, followed by Done.
        let mut aggregator = ToolCallAggregator::default();
        let line1 =
            r#"data: {"choices":[{"delta":{"reasoning_content":"all"},"finish_reason":null}]}"#;
        let line2 =
            r#"data: {"choices":[{"delta":{"reasoning_content":" set"},"finish_reason":null}]}"#;
        let line3 = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"calculator","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#;

        assert_eq!(
            parse_events(line1, &mut aggregator),
            vec![StreamEvent::Reasoning("all".to_string())]
        );
        assert_eq!(
            parse_events(line2, &mut aggregator),
            vec![StreamEvent::Reasoning(" set".to_string())]
        );
        assert_eq!(
            parse_events(line3, &mut aggregator),
            vec![StreamEvent::ToolCall {
                id: "call_1".to_string(),
                name: "calculator".to_string(),
                arguments: "{}".to_string(),
            },]
        );
        assert_eq!(
            aggregator.flush_done(),
            vec![Ok(StreamEvent::Done {
                reason: FinishReason::Stop,
                usage: None,
            })]
        );
    }

    #[test]
    fn stream_aggregates_tool_call_chunks() {
        // The model requests calculator: fragments arrive across three lines;
        // arguments are fragments and id/name only come with the first.
        let mut aggregator = ToolCallAggregator::default();
        let line1 = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"calculator","arguments":"{\"expression\": \""}}]},"finish_reason":null}]}"#;
        let line2 = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1+1"}}]},"finish_reason":null}]}"#;
        let line3 = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"}"}}]},"finish_reason":"tool_calls"}]}"#;

        assert!(parse_events(line1, &mut aggregator).is_empty());
        assert!(parse_events(line2, &mut aggregator).is_empty());
        assert_eq!(
            parse_events(line3, &mut aggregator),
            vec![StreamEvent::ToolCall {
                id: "call_1".to_string(),
                name: "calculator".to_string(),
                arguments: "{\"expression\": \"1+1\"}".to_string(),
            },]
        );
        assert_eq!(
            aggregator.flush_done(),
            vec![Ok(StreamEvent::Done {
                reason: FinishReason::Stop,
                usage: None,
            })]
        );
    }

    #[test]
    fn stream_handles_multiple_interleaved_tool_calls() {
        // Two tool calls with interleaved fragments: each is concatenated by
        // index and emitted whole in order of first appearance.
        let mut aggregator = ToolCallAggregator::default();
        let line1 = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"now","arguments":"{}"}},{"index":1,"id":"call_2","function":{"name":"calculator","arguments":"{\"exp"}}]},"finish_reason":null}]}"#;
        let line2 = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"ression\":\"1\"}"}}]},"finish_reason":"tool_calls"}]}"#;

        assert!(parse_events(line1, &mut aggregator).is_empty());
        assert_eq!(
            parse_events(line2, &mut aggregator),
            vec![
                StreamEvent::ToolCall {
                    id: "call_1".to_string(),
                    name: "now".to_string(),
                    arguments: "{}".to_string(),
                },
                StreamEvent::ToolCall {
                    id: "call_2".to_string(),
                    name: "calculator".to_string(),
                    arguments: "{\"expression\":\"1\"}".to_string(),
                },
            ]
        );
        assert_eq!(
            aggregator.flush_done(),
            vec![Ok(StreamEvent::Done {
                reason: FinishReason::Stop,
                usage: None,
            })]
        );
    }

    #[test]
    fn stream_keeps_delta_and_tool_call_order() {
        // Text and tool requests in the same turn: Deltas are emitted as they
        // arrive, tool calls whole at the end of the turn.
        let mut aggregator = ToolCallAggregator::default();
        let line1 =
            r#"data: {"choices":[{"delta":{"content":"let me compute"},"finish_reason":null}]}"#;
        let line2 = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"calculator","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#;

        assert_eq!(
            parse_events(line1, &mut aggregator),
            vec![StreamEvent::Delta("let me compute".to_string())]
        );
        assert_eq!(
            parse_events(line2, &mut aggregator),
            vec![StreamEvent::ToolCall {
                id: "call_1".to_string(),
                name: "calculator".to_string(),
                arguments: "{}".to_string(),
            },]
        );
        assert_eq!(
            aggregator.flush_done(),
            vec![Ok(StreamEvent::Done {
                reason: FinishReason::Stop,
                usage: None,
            })]
        );
    }

    // The idle timeout uses tokio::time::timeout, which needs a tokio runtime
    // (tokio::test).
    #[tokio::test]
    async fn sse_lines_joins_chunks_and_strips_terminators() {
        let stream = sse_lines(
            futures::stream::iter([
                Ok(b"data: a\r".to_vec()),
                Ok(b"\ndata: b\ndata: c".to_vec()),
            ]),
            Duration::from_secs(60),
            Duration::from_secs(60),
        );
        let lines: Vec<_> = stream.map(|line| line.unwrap()).collect().await;
        assert_eq!(
            lines,
            vec![
                "data: a".to_string(),
                "data: b".to_string(),
                "data: c".to_string()
            ]
        );
    }

    #[tokio::test]
    async fn sse_lines_emits_tail_without_newline() {
        let stream = sse_lines(
            futures::stream::iter([Ok(b"data: x".to_vec())]),
            Duration::from_secs(60),
            Duration::from_secs(60),
        );
        let lines: Vec<_> = stream.map(|line| line.unwrap()).collect().await;
        assert_eq!(lines, vec!["data: x".to_string()]);
    }

    #[tokio::test]
    async fn sse_lines_idle_timeout_emits_timeout() {
        // The endpoint stops producing data (pending): a Timeout is produced
        // after the idle duration and the stream terminates.
        // (sse_lines' Unfold is not Unpin, so StreamExt::next needs Box::pin)
        let mut stream = Box::pin(sse_lines(
            futures::stream::pending::<Result<Vec<u8>, reqwest::Error>>(),
            Duration::from_millis(10),
            Duration::from_secs(60),
        ));
        let first = stream.next().await.unwrap();
        assert!(matches!(first, Err(ProviderError::Timeout(_))));
        assert!(stream.next().await.is_none());
    }

    /// Total-duration budget: an active but never-ending stream (constantly
    /// sending keep-alive comment lines) is killed when the deadline hits.
    /// The idle timeout is reset by every chunk and never fires for such
    /// streams — the total deadline is the only fallback.
    /// Lines received before the deadline are produced normally (the deadline
    /// is only checked before network reads; no false kills).
    #[tokio::test]
    async fn sse_lines_total_deadline_kills_keepalive_stream() {
        // An infinite stream of comment lines: always active (idle never
        // fires), only the total deadline can terminate it.
        let mut stream = Box::pin(sse_lines(
            futures::stream::iter(std::iter::repeat_with(|| Ok(b": keep-alive\n".to_vec()))),
            Duration::from_secs(60),
            Duration::from_millis(50),
        ));
        // Before the deadline: lines are produced normally; at the deadline:
        // Err(Timeout) and termination.
        let mut lines = 0usize;
        loop {
            match stream.next().await {
                Some(Ok(_)) => lines += 1,
                Some(Err(ProviderError::Timeout(_))) => break,
                other => panic!("expected line or Timeout, got {other:?}"),
            }
        }
        assert!(lines > 0, "lines must be produced before the deadline");
        assert!(
            stream.next().await.is_none(),
            "stream must terminate after an error"
        );
    }

    /// The total deadline does not false-kill: with a lenient deadline, lines
    /// of an infinite keep-alive stream are produced normally (the deadline
    /// is only checked before network reads; buffered lines are unaffected).
    #[tokio::test]
    async fn sse_lines_keepalive_lines_delivered_within_deadline() {
        let mut stream = Box::pin(sse_lines(
            futures::stream::iter(std::iter::repeat_with(|| Ok(b": keep-alive\n".to_vec()))),
            Duration::from_secs(60),
            Duration::from_secs(60),
        ));
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first, ": keep-alive");
    }

    /// Line-buffer over-limit: the stream terminates after Err(Api) (nothing
    /// more is produced; the buffer is dropped).
    #[tokio::test]
    async fn sse_lines_over_limit_terminates_stream() {
        let mut stream = Box::pin(sse_lines(
            futures::stream::iter([Ok(vec![b'x'; MAX_SSE_LINE + 1])]),
            Duration::from_secs(60),
            Duration::from_secs(60),
        ));
        let first = stream.next().await.unwrap();
        assert!(matches!(first, Err(ProviderError::Api { status: 0, .. })));
        assert!(
            stream.next().await.is_none(),
            "stream must terminate after exceeding the limit"
        );
    }

    /// A bad in-line JSON line: produces Err and drops the stashed Done —
    /// afterwards [DONE] no longer produces an Ok(Done).
    #[test]
    fn bad_line_discards_pending_done() {
        let mut aggregator = ToolCallAggregator::default();
        let finish = r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        assert!(parse_sse_line(finish, &mut aggregator).is_empty()); // stashes done_reason

        let bad = "data: not-json";
        let events = parse_sse_line(bad, &mut aggregator);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Err(ProviderError::Api { status: 0, .. })
        ));

        // After the bad line, [DONE] no longer flushes a Done (the stash was
        // dropped).
        assert!(parse_sse_line("data: [DONE]", &mut aggregator).is_empty());
    }

    /// Tool-call count over-limit: the 65th **new** index errors and drops
    /// the stashed Done;
    /// continuations of existing indices are unaffected by the count limit.
    #[test]
    fn tool_call_count_limit_only_for_new_calls() {
        let mut aggregator = ToolCallAggregator::default();
        for i in 0..MAX_TOOL_CALLS {
            let line = format!(
                r#"data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":{i},"id":"c{i}","function":{{"name":"f","arguments":"{{}}"}}}}]}},"finish_reason":null}}]}}"#
            );
            assert!(
                parse_sse_line(&line, &mut aggregator).is_empty(),
                "tool call {i} must be accepted"
            );
        }
        // The limit is already reached: the 65th new index errors.
        let over = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":64,"id":"c64","function":{"name":"f","arguments":"{}"}}]},"finish_reason":null}]}"#;
        let events = parse_sse_line(over, &mut aggregator);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Err(ProviderError::Api { status: 0, .. })
        ));
        // Continuations of existing indices can still be appended (find comes
        // before the count check).
        let cont = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"more\":true}"}}]},"finish_reason":null}]}"#;
        assert!(parse_sse_line(cont, &mut aggregator).is_empty());
    }

    /// Aggregated-text over-limit: an arguments first fragment over 1MiB
    /// errors (the first fragment of a new call is also checked against the
    /// limit).
    #[test]
    fn tool_arguments_size_limit() {
        let mut aggregator = ToolCallAggregator::default();
        let huge = "x".repeat(MAX_ACCUMULATED_TEXT + 1);
        let line = format!(
            r#"data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"id":"c0","function":{{"name":"f","arguments":"{huge}"}}}}]}},"finish_reason":null}}]}}"#
        );
        let events = parse_sse_line(&line, &mut aggregator);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Err(ProviderError::Api { status: 0, .. })
        ));
    }

    /// Debug-mask verification: api_key never appears in any Debug output
    /// (prevents derive from printing it in plain text).
    #[test]
    fn debug_masks_api_key() {
        let provider = OpenAiProvider::new("https://api.example.com", "sk-super-secret", "model");
        let debug = format!("{provider:?}");
        assert!(
            !debug.contains("sk-super-secret"),
            "Debug leaked api_key: {debug}"
        );
        assert!(
            debug.contains("\"***\""),
            "mask placeholder missing: {debug}"
        );
    }

    /// A continuation appended to an existing index exceeding the text limit
    /// → error (continuations are checked against the cumulative limit too).
    #[test]
    fn tool_arguments_continuation_size_limit() {
        let mut aggregator = ToolCallAggregator::default();
        // The first fragment is near the limit (MAX_ACCUMULATED_TEXT - 1);
        // the continuation overshoots by 2 bytes → over-limit.
        let first = "x".repeat(MAX_ACCUMULATED_TEXT - 1);
        let line1 = format!(
            r#"data: {{"choices":[{{"delta":{{"tool_calls":[{{"index":0,"id":"c0","function":{{"name":"f","arguments":"{first}"}}}}]}},"finish_reason":null}}]}}"#
        );
        assert!(parse_sse_line(&line1, &mut aggregator).is_empty());
        let cont = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"yy"}}]},"finish_reason":null}]}"#;
        let events = parse_sse_line(cont, &mut aggregator);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Err(ProviderError::Api { status: 0, .. })
        ));
    }

    /// The errored latch: after an in-line error the endpoint resends
    /// finish_reason + [DONE] — no Ok(Done) is produced.
    #[test]
    fn errored_latch_blocks_resurrected_done() {
        let mut aggregator = ToolCallAggregator::default();
        // Bad line: set the latch.
        let events = parse_sse_line("data: not-json", &mut aggregator);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Err(ProviderError::Api { status: 0, .. })
        ));
        // A malicious endpoint resends finish_reason: set_done_reason is a
        // no-op.
        let finish = r#"data: {"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        assert!(parse_sse_line(finish, &mut aggregator).is_empty());
        // [DONE]: flush_done finds nothing; no Done resurrection.
        assert!(parse_sse_line("data: [DONE]", &mut aggregator).is_empty());
    }

    /// EOF-fallback end-to-end: on EOF without [DONE], the stashed Done is
    /// produced (including usage).
    #[tokio::test]
    async fn sse_stream_eof_flushes_pending_done() {
        let aggregator = Arc::new(Mutex::new(ToolCallAggregator::default()));
        let chunks = futures::stream::iter([
            Ok(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n".to_vec()),
            Ok(b"data: {\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n".to_vec()),
        ]);
        let mut stream = Box::pin(sse_stream(
            chunks,
            Duration::from_secs(60),
            Duration::from_secs(60),
            aggregator,
        ));
        let events: Vec<_> = stream.by_ref().map(|e| e.unwrap()).collect().await;
        assert_eq!(
            events,
            vec![StreamEvent::Done {
                reason: FinishReason::Stop,
                usage: Some(Usage::new(1, 2)),
            }]
        );
    }

    /// The normal [DONE] path (sse_stream): finish_reason → [DONE] produces
    /// the Done and the stream ends.
    #[tokio::test]
    async fn sse_stream_done_marker_flushes() {
        let aggregator = Arc::new(Mutex::new(ToolCallAggregator::default()));
        let chunks = futures::stream::iter([
            Ok(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n".to_vec()),
            Ok(b"data: [DONE]\n".to_vec()),
        ]);
        let mut stream = Box::pin(sse_stream(
            chunks,
            Duration::from_secs(60),
            Duration::from_secs(60),
            aggregator,
        ));
        let events: Vec<_> = stream.by_ref().map(|e| e.unwrap()).collect().await;
        assert_eq!(
            events,
            vec![StreamEvent::Done {
                reason: FinishReason::Stop,
                usage: None
            }]
        );
    }

    /// No Done after an error, end-to-end: finish_reason → bad-line Err →
    /// [DONE]: only the Err is produced, no Done.
    #[tokio::test]
    async fn sse_stream_error_then_no_done() {
        let aggregator = Arc::new(Mutex::new(ToolCallAggregator::default()));
        let chunks = futures::stream::iter([
            Ok(b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}".to_vec()),
            Ok(b"data: not-json".to_vec()),
            Ok(b"data: [DONE]".to_vec()),
        ]);
        let mut stream = Box::pin(sse_stream(
            chunks,
            Duration::from_secs(60),
            Duration::from_secs(60),
            aggregator,
        ));
        let events: Vec<Result<StreamEvent, ProviderError>> = stream.by_ref().collect().await;
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Err(ProviderError::Api { status: 0, .. })
        ));
    }

    /// HTTP integration: a minimal local TcpListener server verifies the wire
    /// request (method / path / auth header / body) and response parsing.
    #[tokio::test]
    async fn chat_over_real_http_wire() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 8192];
            let n = socket.read(&mut buf).await.unwrap();
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            // A single read usually contains headers + the body (small
            // requests); assert the wire shape item by item.
            assert!(
                req.starts_with("POST /chat/completions HTTP/1.1"),
                "method/path: {req}"
            );
            assert!(
                req.contains("authorization: Bearer sk-test"),
                "auth header: {req}"
            );
            assert!(
                req.contains(r#""model":"deepseek-chat""#),
                "body model: {req}"
            );
            // With stream None, serialization omits it (non-streaming).
            // Replay a minimal OpenAI response.
            let body = r#"{"choices":[{"message":{"role":"assistant","content":"hi","reasoning_content":null,"tool_calls":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(resp.as_bytes()).await.unwrap();
        });

        let provider = OpenAiProvider::new(format!("http://{addr}"), "sk-test", "deepseek-chat");
        let resp = provider
            .chat(ChatRequest {
                messages: vec![crate::message::Message::user("hi")],
                tools: vec![],
                options: Default::default(),
            })
            .await
            .unwrap();
        assert_eq!(resp.message, crate::message::Message::assistant("hi"));
        assert_eq!(resp.usage, Usage::new(1, 2));
        server.await.unwrap();
    }

    /// Non-streaming request total timeout: the server accepts the connection
    /// but never responds; after request_timeout elapses, chat returns
    /// ProviderError::Timeout (instead of hanging forever).
    #[tokio::test]
    async fn chat_request_timeout_fires() {
        use tokio::io::AsyncReadExt;
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server = tokio::spawn(async move {
            // Accept the connection and stall silently, never writing a
            // response.
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 8192];
            let _ = socket.read(&mut buf).await;
            std::future::pending::<()>().await;
        });

        let provider = OpenAiProvider::new(format!("http://{addr}"), "sk-test", "m")
            .with_request_timeout(Duration::from_millis(100));
        let err = provider
            .chat(ChatRequest {
                messages: vec![crate::message::Message::user("hi")],
                tools: vec![],
                options: Default::default(),
            })
            .await
            .unwrap_err();
        assert!(matches!(err, ProviderError::Timeout(TimeoutStage::Request)));
        server.abort();
    }

    /// Connect timeout: connecting to a TEST-NET black-hole address
    /// (192.0.2.0/24, guaranteed unreachable by RFC 5737); after
    /// connect_timeout elapses, chat returns ProviderError::Timeout.
    #[tokio::test]
    async fn chat_connect_timeout_fires() {
        let provider = OpenAiProvider::new("http://192.0.2.1/mcp", "sk-test", "m")
            .with_connect_timeout(Duration::from_millis(200));
        let err = provider
            .chat(ChatRequest {
                messages: vec![crate::message::Message::user("hi")],
                tools: vec![],
                options: Default::default(),
            })
            .await
            .unwrap_err();
        // A black-hole address may fail fast with "unreachable" (Network) or
        // wait until the timeout (Timeout): both are correct classifications
        // of a connection failure, and the test pins down "connection fails".
        assert!(
            matches!(&err, ProviderError::Timeout(_) | ProviderError::Network(_)),
            "connection failure should classify as Timeout or Network, got: {err:?}"
        );
    }

    /// EOF strictness: an empty stream (0 events) → Err("stream ended without
    /// finish_reason"), never silently treated as success.
    #[tokio::test]
    async fn sse_stream_empty_stream_is_error() {
        let aggregator = Arc::new(Mutex::new(ToolCallAggregator::default()));
        let mut stream = Box::pin(sse_stream(
            futures::stream::iter([Ok(b"\n".to_vec())]),
            Duration::from_secs(60),
            Duration::from_secs(60),
            aggregator,
        ));
        let events: Vec<Result<StreamEvent, ProviderError>> = stream.by_ref().collect().await;
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Err(ProviderError::Api { status: 0, ref message }) if message.contains("without finish_reason")
        ));
    }

    /// Data after [DONE] is dead: Delta lines arriving after the marker are
    /// dropped, and Done stays the last success event on the stream.
    #[tokio::test]
    async fn sse_stream_drops_data_after_done_marker() {
        let aggregator = Arc::new(Mutex::new(ToolCallAggregator::default()));
        let chunks = futures::stream::iter([
            Ok(b"data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n".to_vec()),
            Ok(b"data: [DONE]\n".to_vec()),
            // Malicious / abnormal data after [DONE]: must be dropped.
            Ok(b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n".to_vec()),
        ]);
        let mut stream = Box::pin(sse_stream(
            chunks,
            Duration::from_secs(60),
            Duration::from_secs(60),
            aggregator,
        ));
        let events: Vec<_> = stream.by_ref().map(|e| e.unwrap()).collect().await;
        assert_eq!(
            events,
            vec![
                StreamEvent::Delta("ok".into()),
                StreamEvent::Done {
                    reason: FinishReason::Stop,
                    usage: None
                },
            ]
        );
    }

    /// After a network drop: the stream terminates after the Err item, and
    /// buffered leftovers are no longer produced as Ok.
    #[tokio::test]
    async fn network_error_terminates_and_drops_buffered_tail() {
        // reqwest::Error has no public constructor; get a real Builder error
        // by building with an invalid URL.
        let conn_err = reqwest::Client::new()
            .get("http://[::1")
            .build()
            .expect_err("invalid URL must fail to build");
        let mut stream = Box::pin(sse_lines(
            futures::stream::iter([Ok(b"data: partial".to_vec()), Err(conn_err)]),
            Duration::from_secs(60),
            Duration::from_secs(60),
        ));
        let first = stream.next().await.unwrap();
        assert!(matches!(first, Err(ProviderError::Network(_))));
        // Terminates after Err: the partial buffered line is not produced;
        // the stream ends.
        assert!(stream.next().await.is_none());
    }
}
