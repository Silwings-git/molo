//! Runtime benchmarks for core agent data paths.
//!
//! These benches use deterministic in-process fixtures only. They do not
//! contact a live provider and should be suitable for release preflight trend
//! recording.

use criterion::{Criterion, criterion_group, criterion_main};
use futures::StreamExt;
use molo::{
    BroadcastEventChannel, ChatRequest, ContentBlock, EventChannel, FakeProvider, FakeReply,
    Memory, Message, Provider, ReActEvent, SharedState, Tool, ToolContext, ToolError,
    ToolNamespace, ToolNamespaceKind, ToolRegistry, ToolResult, ToolSchema, ToolSource,
    ToolTrustLevel, WindowMemory,
};
use serde_json::json;
use std::hint::black_box;
use std::sync::Arc;

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("benchmark runtime")
}

fn message_fixture(rounds: usize) -> Vec<Message> {
    let mut messages = Vec::with_capacity(rounds * 4 + 1);
    messages.push(Message::system("You are benchmarking message assembly."));
    for i in 0..rounds {
        messages.push(Message::user_blocks(vec![
            ContentBlock::Text(format!("user request {i}")),
            ContentBlock::Wire(json!({"kind": "fixture", "round": i})),
        ]));
        messages.push(Message::Assistant {
            content: format!("assistant response {i}"),
            reasoning: Some(format!("reasoning trace {i}")),
            tool_calls: vec![molo::ToolCall {
                id: format!("call-{i}"),
                name: "lookup".to_string(),
                arguments: format!(r#"{{"round":{i},"query":"fixture"}}"#),
            }],
        });
        messages.push(Message::tool_result(format!("call-{i}"), "tool result"));
    }
    messages
}

fn bench_message_assembly(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime_message");
    group.bench_function("message_assembly_100_rounds", |b| {
        b.iter(|| {
            let messages = message_fixture(100);
            let bytes = serde_json::to_vec(&messages).expect("serialize messages");
            black_box(bytes);
        })
    });
    group.finish();
}

fn bench_memory_trimming(c: &mut Criterion) {
    let runtime = rt();
    let mut group = c.benchmark_group("runtime_memory");
    group.bench_function("window_trim_200_rounds", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let mut memory = WindowMemory::new(512);
                for message in message_fixture(200) {
                    memory.record(message).await.expect("record message");
                }
                black_box(memory.context().await.expect("trim context"));
            })
        })
    });
    group.finish();
}

#[derive(Debug)]
struct NamedNoopTool {
    name: String,
}

#[molo::async_trait]
impl Tool for NamedNoopTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema::new(
            self.name.clone(),
            "benchmark noop",
            json!({"type": "object"}),
        )
    }

    async fn call(
        &self,
        _arguments: serde_json::Value,
        _context: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError> {
        Ok("ok".into())
    }
}

fn registry_fixture(size: usize) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    for i in 0..size {
        let name = format!("tool_{i}");
        registry
            .register_with_source(
                NamedNoopTool { name: name.clone() },
                ToolSource::new(
                    ToolNamespace::new(ToolNamespaceKind::Custom("bench".to_string()), "runtime"),
                    name.clone(),
                    name,
                )
                .with_trust(ToolTrustLevel::Trusted),
            )
            .expect("register benchmark tool");
    }
    registry
}

fn bench_tool_lookup(c: &mut Criterion) {
    let runtime = rt();
    let mut group = c.benchmark_group("runtime_tool_registry");
    group.bench_function("source_aware_lookup_512_tools", |b| {
        let registry = registry_fixture(512);
        let state = SharedState::new();
        let run = molo::RunContext::new("bench-tool-lookup");
        b.iter(|| {
            runtime.block_on(async {
                black_box(
                    registry
                        .call_named("tool_400", "{}", &run, &state)
                        .await
                        .expect("call tool"),
                );
            })
        })
    });
    group.bench_function("namespace_subset_512_tools", |b| {
        let registry = registry_fixture(512);
        let namespace =
            ToolNamespace::new(ToolNamespaceKind::Custom("bench".to_string()), "runtime");
        b.iter(|| {
            black_box(
                registry
                    .subset_by_namespace(&namespace)
                    .expect("subset namespace")
                    .schemas(),
            );
        })
    });
    group.finish();
}

fn bench_event_publish(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime_event_publish");
    group.bench_function("broadcast_publish_with_subscriber", |b| {
        let channel = BroadcastEventChannel::new(1024);
        let _receiver = channel.subscribe();
        b.iter(|| {
            channel.publish(Arc::new(ReActEvent::Delta {
                text: "delta".to_string(),
            }));
            black_box(channel.stats());
        })
    });
    group.finish();
}

fn bench_provider_stream(c: &mut Criterion) {
    let runtime = rt();
    let mut group = c.benchmark_group("runtime_provider_stream");
    group.bench_function("fake_stream_parallel_tool_calls_64", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let calls = (0..64)
                    .map(|i| molo::ToolCall {
                        id: format!("call-{i}"),
                        name: "tool".to_string(),
                        arguments: format!(r#"{{"i":{i}}}"#),
                    })
                    .collect();
                let provider = FakeProvider::new([FakeReply::ToolCalls {
                    content: "planning".to_string(),
                    calls,
                }]);
                let mut stream = provider
                    .stream_chat(ChatRequest::default())
                    .await
                    .expect("stream chat");
                let mut count = 0usize;
                while let Some(event) = stream.next().await {
                    black_box(event.expect("stream event"));
                    count += 1;
                }
                black_box(count);
            })
        })
    });
    group.finish();
}

#[cfg(feature = "structured")]
fn bench_structured_validation(c: &mut Criterion) {
    use molo::{StructuredOutcome, StructuredValidator};

    let mut group = c.benchmark_group("runtime_structured_validation");
    let schema = json!({
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "changed_files": {
                "type": "array",
                "items": {"type": "string"}
            },
            "passed": {"type": "boolean"}
        },
        "required": ["summary", "changed_files", "passed"]
    });
    group.bench_function("valid_output", |b| {
        b.iter(|| {
            let mut validator = StructuredValidator::new(schema.clone(), 3);
            assert!(matches!(
                validator
                    .validate(r#"{"summary":"ok","changed_files":["src/lib.rs"],"passed":true}"#),
                StructuredOutcome::Passed
            ));
        })
    });
    group.bench_function("invalid_output_failure_path", |b| {
        b.iter(|| {
            let mut validator = StructuredValidator::new(schema.clone(), 3);
            black_box(validator.validate(r#"{"summary": 7}"#));
        })
    });
    group.finish();
}

#[cfg(feature = "structured")]
criterion_group!(
    benches,
    bench_message_assembly,
    bench_memory_trimming,
    bench_tool_lookup,
    bench_event_publish,
    bench_provider_stream,
    bench_structured_validation
);

#[cfg(not(feature = "structured"))]
criterion_group!(
    benches,
    bench_message_assembly,
    bench_memory_trimming,
    bench_tool_lookup,
    bench_event_publish,
    bench_provider_stream
);

criterion_main!(benches);
