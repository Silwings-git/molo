use futures::StreamExt;
use molo::{
    ChatRequest, FakeProvider, FakeReply, FinishReason, Message, ModelOptions, Provider,
    ProviderError, ProviderRequestContext, RunContext, StreamEvent, ToolCall, ToolSchema, Usage,
};
use std::time::{Duration, Instant};

fn text_request() -> ChatRequest {
    ChatRequest {
        messages: vec![Message::user("hello")],
        ..Default::default()
    }
}

#[tokio::test]
async fn fake_capabilities_match_supported_cases() {
    let fake = FakeProvider::new([]);
    let capabilities = fake.capabilities();

    assert!(capabilities.streaming);
    assert!(capabilities.reasoning);
    assert!(capabilities.tool_calls);
    assert!(capabilities.parallel_tool_calls);
    assert!(capabilities.structured_output);
    assert!(capabilities.usage);
    assert!(capabilities.context_cancellation);
    assert!(capabilities.context_deadline);
}

#[tokio::test]
async fn fake_non_streaming_text_usage_and_finish_reason() {
    let usage = Usage::new(7, 3);
    let fake = FakeProvider::new([FakeReply::text_with_usage("hi", usage)]);

    let response = fake.chat(text_request()).await.unwrap();

    assert_eq!(response.message, Message::assistant("hi"));
    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert_eq!(response.usage, usage);
    assert_eq!(fake.requests(), vec![text_request()]);
}

#[tokio::test]
async fn fake_streaming_text_matches_non_streaming_text() {
    let usage = Usage::new(5, 4);
    let fake = FakeProvider::new([FakeReply::text_with_usage("streamed", usage)]);

    let mut stream = fake.stream_chat(text_request()).await.unwrap();
    let mut text = String::new();
    let mut done = None;
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            StreamEvent::Delta(delta) => text.push_str(&delta),
            StreamEvent::Done { reason, usage } => done = Some((reason, usage)),
            other => panic!("unexpected stream event: {other:?}"),
        }
    }

    assert_eq!(text, "streamed");
    assert_eq!(done, Some((FinishReason::Stop, Some(usage))));
}

#[tokio::test]
async fn fake_reasoning_stream_is_separate_from_answer() {
    let fake = FakeProvider::new([FakeReply::TextWithReasoning {
        content: "answer".into(),
        reasoning: "think".into(),
    }]);

    let mut stream = fake.stream_chat(text_request()).await.unwrap();
    let mut text = String::new();
    let mut reasoning = String::new();
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            StreamEvent::Delta(delta) => text.push_str(&delta),
            StreamEvent::Reasoning(delta) => reasoning.push_str(&delta),
            StreamEvent::Done { .. } => {}
            other => panic!("unexpected stream event: {other:?}"),
        }
    }

    assert_eq!(text, "answer");
    assert_eq!(reasoning, "think");
}

#[tokio::test]
async fn fake_single_and_parallel_tool_calls_preserve_order() {
    let calls = vec![
        ToolCall {
            id: "call-1".into(),
            name: "search".into(),
            arguments: r#"{"q":"one"}"#.into(),
        },
        ToolCall {
            id: "call-2".into(),
            name: "read".into(),
            arguments: r#"{"path":"src/lib.rs"}"#.into(),
        },
    ];
    let fake = FakeProvider::new([FakeReply::ToolCalls {
        content: "checking".into(),
        calls: calls.clone(),
    }]);

    let response = fake.chat(text_request()).await.unwrap();

    match response.message {
        Message::Assistant {
            content,
            reasoning,
            tool_calls,
        } => {
            assert_eq!(content, "checking");
            assert_eq!(reasoning, None);
            assert_eq!(tool_calls, calls);
        }
        other => panic!("expected assistant, got {other:?}"),
    }
}

#[tokio::test]
async fn fake_streaming_tool_calls_preserve_order() {
    let fake = FakeProvider::new([FakeReply::ToolCalls {
        content: String::new(),
        calls: vec![
            ToolCall {
                id: "call-1".into(),
                name: "search".into(),
                arguments: r#"{"q":"one"}"#.into(),
            },
            ToolCall {
                id: "call-2".into(),
                name: "read".into(),
                arguments: r#"{"path":"src/lib.rs"}"#.into(),
            },
        ],
    }]);

    let mut stream = fake.stream_chat(text_request()).await.unwrap();
    let mut tool_calls = Vec::new();
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            StreamEvent::ToolCall {
                id,
                name,
                arguments,
            } => tool_calls.push((id, name, arguments)),
            StreamEvent::Done { .. } => {}
            other => panic!("unexpected stream event: {other:?}"),
        }
    }

    assert_eq!(
        tool_calls,
        vec![
            ("call-1".into(), "search".into(), r#"{"q":"one"}"#.into()),
            (
                "call-2".into(),
                "read".into(),
                r#"{"path":"src/lib.rs"}"#.into()
            ),
        ]
    );
}

#[tokio::test]
async fn fake_records_structured_schema_and_tools_without_mutating_request() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "city": { "type": "string" } },
        "required": ["city"],
    });
    let tool = ToolSchema::new(
        "lookup_city",
        "Look up a city",
        serde_json::json!({"type": "object"}),
    );
    let request = ChatRequest {
        messages: vec![Message::user("where?")],
        tools: vec![tool.clone()],
        options: ModelOptions {
            structured: Some(schema.clone()),
            ..Default::default()
        },
    };
    let fake = FakeProvider::new([FakeReply::Text(r#"{"city":"Paris"}"#.into())]);

    let _ = fake.chat(request.clone()).await.unwrap();

    assert_eq!(request.tools, vec![tool]);
    assert_eq!(request.options.structured, Some(schema.clone()));
    assert_eq!(fake.requests(), vec![request]);
}

#[tokio::test]
async fn fake_context_cancellation_and_deadline_are_classified() {
    let cancelled = RunContext::new("run-cancelled");
    cancelled.cancellation.cancel();
    let provider_context =
        ProviderRequestContext::from_run_context("run-cancelled-model-1", &cancelled);
    let fake = FakeProvider::new([FakeReply::Text("unused".into())]);

    let err = fake
        .chat_with_context(text_request(), &provider_context)
        .await
        .unwrap_err();
    assert_eq!(err, ProviderError::Cancelled);
    assert!(fake.requests().is_empty());

    let expired =
        RunContext::new("run-expired").with_deadline(Instant::now() - Duration::from_secs(1));
    let provider_context =
        ProviderRequestContext::from_run_context("run-expired-model-1", &expired);
    let err = match fake
        .stream_chat_with_context(text_request(), &provider_context)
        .await
    {
        Err(error) => error,
        Ok(_) => panic!("expected expired provider context to fail"),
    };
    assert!(matches!(err, ProviderError::Timeout(_)));
}

#[tokio::test]
async fn fake_script_exhaustion_is_protocol_error() {
    let fake = FakeProvider::new([]);

    let err = fake.chat(text_request()).await.unwrap_err();

    assert!(matches!(
        err,
        ProviderError::Protocol { message } if message.contains("script exhausted")
    ));
}

#[cfg(feature = "openai")]
mod openai_compatible {
    use super::*;
    use molo::provider::TimeoutStage;
    use molo::{OpenAiProvider, StructuredOutputMode};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn serve_once(response: String) -> (String, tokio::task::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut socket).await;
            socket.write_all(response.as_bytes()).await.unwrap();
            request
        });
        (format!("http://{addr}"), handle)
    }

    async fn serve_stalled() -> (String, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let _ = read_http_request(&mut socket).await;
            std::future::pending::<()>().await;
        });
        (format!("http://{addr}"), handle)
    }

    async fn read_http_request(socket: &mut tokio::net::TcpStream) -> String {
        let mut bytes = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let n = socket.read(&mut buf).await.unwrap();
            if n == 0 {
                break;
            }
            bytes.extend_from_slice(&buf[..n]);
            let Some(headers_end) = find_header_end(&bytes) else {
                continue;
            };
            let request = String::from_utf8_lossy(&bytes).to_string();
            let content_length = request
                .lines()
                .find_map(|line| line.strip_prefix("content-length: "))
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if bytes.len() >= headers_end + content_length {
                break;
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    fn find_header_end(bytes: &[u8]) -> Option<usize> {
        bytes
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .map(|index| index + 4)
    }

    fn response(status: &str, headers: &[(&str, &str)], body: &str) -> String {
        let mut response = format!("HTTP/1.1 {status}\r\n");
        response.push_str(&format!("content-length: {}\r\n", body.len()));
        for (name, value) in headers {
            response.push_str(name);
            response.push_str(": ");
            response.push_str(value);
            response.push_str("\r\n");
        }
        response.push_str("\r\n");
        response.push_str(body);
        response
    }

    fn json_response(body: &str) -> String {
        response("200 OK", &[("content-type", "application/json")], body)
    }

    #[tokio::test]
    async fn openai_non_streaming_text_usage_finish_reason_and_structured_mapping() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"{\"ok\":true}","reasoning_content":"because","tool_calls":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}"#;
        let (base_url, server) = serve_once(json_response(body)).await;
        let provider = OpenAiProvider::new(base_url, "sk-test-secret", "mock-model")
            .with_structured_output_mode(StructuredOutputMode::Native);
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "ok": { "type": "boolean" } },
            "required": ["ok"]
        });

        let response = provider
            .chat(ChatRequest {
                messages: vec![Message::user("return json")],
                options: ModelOptions {
                    structured: Some(schema),
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .unwrap();

        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert_eq!(response.usage, Usage::new(3, 4));
        assert_eq!(
            response.message,
            Message::assistant_with_reasoning(r#"{"ok":true}"#, "because")
        );
        let request = server.await.unwrap();
        assert!(request.starts_with("POST /chat/completions HTTP/1.1"));
        assert!(request.contains("authorization: Bearer sk-test-secret"));
        assert!(request.contains(r#""model":"mock-model""#));
        assert!(request.contains(r#""response_format""#));
        assert!(request.contains(r#""json_schema""#));
    }

    #[tokio::test]
    async fn openai_tool_calls_and_parallel_order_are_preserved() {
        let body = r#"{"choices":[{"message":{"role":"assistant","content":"checking","tool_calls":[{"id":"call-1","type":"function","function":{"name":"search","arguments":"{\"q\":\"one\"}"}},{"id":"call-2","type":"function","function":{"name":"read","arguments":"{\"path\":\"src/lib.rs\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}"#;
        let (base_url, _server) = serve_once(json_response(body)).await;
        let provider = OpenAiProvider::new(base_url, "sk-test", "mock-model");

        let response = provider.chat(text_request()).await.unwrap();

        match response.message {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                assert_eq!(content, "checking");
                assert_eq!(
                    tool_calls,
                    vec![
                        ToolCall {
                            id: "call-1".into(),
                            name: "search".into(),
                            arguments: r#"{"q":"one"}"#.into(),
                        },
                        ToolCall {
                            id: "call-2".into(),
                            name: "read".into(),
                            arguments: r#"{"path":"src/lib.rs"}"#.into(),
                        },
                    ]
                );
            }
            other => panic!("expected assistant message, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn openai_api_rate_limit_and_error_mapping() {
        let rate_limit_body = r#"{"error":{"message":"rate limit exceeded"}}"#;
        let (base_url, _server) = serve_once(response(
            "429 Too Many Requests",
            &[("content-type", "application/json"), ("retry-after", "5")],
            rate_limit_body,
        ))
        .await;
        let provider = OpenAiProvider::new(base_url, "sk-test", "mock-model");
        let err = provider.chat(text_request()).await.unwrap_err();
        assert!(matches!(
            err,
            ProviderError::RateLimited { retry_after: Some(duration) }
                if duration == Duration::from_secs(5)
        ));

        let api_body = r#"{"error":{"message":"bad auth","code":"invalid_api_key"}}"#;
        let (base_url, _server) = serve_once(response(
            "401 Unauthorized",
            &[("content-type", "application/json")],
            api_body,
        ))
        .await;
        let provider = OpenAiProvider::new(base_url, "sk-test", "mock-model");
        let err = provider.chat(text_request()).await.unwrap_err();
        assert!(matches!(
            err,
            ProviderError::Api { status: 401, code: None, message }
                if message.contains("bad auth")
        ));
    }

    #[tokio::test]
    async fn openai_decode_protocol_timeout_and_oversized_body_mapping() {
        let (base_url, _server) = serve_once(json_response("not-json")).await;
        let provider = OpenAiProvider::new(base_url, "sk-test", "mock-model");
        let err = provider.chat(text_request()).await.unwrap_err();
        assert!(matches!(err, ProviderError::Decode { .. }));

        let (base_url, _server) = serve_once(json_response(r#"{"choices":[]}"#)).await;
        let provider = OpenAiProvider::new(base_url, "sk-test", "mock-model");
        let err = provider.chat(text_request()).await.unwrap_err();
        assert!(matches!(err, ProviderError::Protocol { .. }));

        let oversized =
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 16777217\r\n\r\n";
        let (base_url, _server) = serve_once(oversized.to_string()).await;
        let provider = OpenAiProvider::new(base_url, "sk-test", "mock-model");
        let err = provider.chat(text_request()).await.unwrap_err();
        assert!(matches!(
            err,
            ProviderError::ResponseTooLarge {
                limit_bytes: 16777216
            }
        ));

        let (base_url, server) = serve_stalled().await;
        let provider = OpenAiProvider::new(base_url, "sk-test", "mock-model")
            .with_request_timeout(Duration::from_millis(50));
        let err = provider.chat(text_request()).await.unwrap_err();
        assert!(matches!(err, ProviderError::Timeout(TimeoutStage::Request)));
        server.abort();
    }

    #[tokio::test]
    async fn openai_context_cancellation_is_observed_before_request() {
        let provider = OpenAiProvider::new("http://127.0.0.1:9", "sk-test", "mock-model");
        let run = RunContext::new("run-openai-cancelled");
        run.cancellation.cancel();
        let context = ProviderRequestContext::from_run_context("model-1", &run);

        let err = provider
            .chat_with_context(text_request(), &context)
            .await
            .unwrap_err();
        assert!(matches!(err, ProviderError::Cancelled));

        let err = match provider
            .stream_chat_with_context(text_request(), &context)
            .await
        {
            Err(err) => err,
            Ok(_) => panic!("expected cancelled stream setup"),
        };
        assert!(matches!(err, ProviderError::Cancelled));
    }

    #[tokio::test]
    async fn openai_streaming_text_usage_and_malformed_stream_mapping() {
        let stream_body = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n",
            "data: [DONE]\n\n",
        );
        let (base_url, _server) = serve_once(response(
            "200 OK",
            &[("content-type", "text/event-stream")],
            stream_body,
        ))
        .await;
        let provider = OpenAiProvider::new(base_url, "sk-test", "mock-model");
        let mut stream = provider.stream_chat(text_request()).await.unwrap();
        let mut text = String::new();
        let mut done = None;
        while let Some(event) = stream.next().await {
            match event.unwrap() {
                StreamEvent::Delta(delta) => text.push_str(&delta),
                StreamEvent::Done { reason, usage } => done = Some((reason, usage)),
                other => panic!("unexpected stream event: {other:?}"),
            }
        }
        assert_eq!(text, "hello");
        assert_eq!(done, Some((FinishReason::Stop, Some(Usage::new(1, 2)))));

        let (base_url, _server) = serve_once(response(
            "200 OK",
            &[("content-type", "text/event-stream")],
            "data: not-json\n\n",
        ))
        .await;
        let provider = OpenAiProvider::new(base_url, "sk-test", "mock-model");
        let mut stream = provider.stream_chat(text_request()).await.unwrap();
        let err = stream.next().await.unwrap().unwrap_err();
        assert!(matches!(err, ProviderError::Decode { .. }));
        assert!(stream.next().await.is_none());
    }

    #[test]
    fn openai_debug_masks_api_key() {
        let provider = OpenAiProvider::new("https://api.example.invalid", "sk-test-secret", "m");
        let debug = format!("{provider:?}");
        assert!(!debug.contains("sk-test-secret"));
        assert!(debug.contains("***"));
    }
}
