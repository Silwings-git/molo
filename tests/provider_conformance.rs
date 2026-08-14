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
