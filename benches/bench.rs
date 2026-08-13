//! Hot-path throughput benchmarks: the "lightweight" figures backing the README's performance claims.
//!
//! They measure the **framework's own overhead** (FakeProvider does no networking, tools have no logic),
//! reflecting the fixed cost of each loop iteration; in real scenarios model latency (seconds) dominates,
//! so the framework overhead is negligible.
//!
//! Run with: `cargo bench`
//!
//! Reference numbers on this machine (record the machine model and toolchain; results vary with hardware):
//! - Apple Silicon (M-series): `agent_run::5_rounds` ≈ a few µs/round (loop + recording +
//!   counting), `tool_call` ≈ ~1 µs, `window_memory_trim_50_rounds` ≈ tens of µs.

use criterion::{Criterion, criterion_group, criterion_main};
use molo::message::Message;
use molo::provider::{FakeProvider, FakeReply};
use molo::tool::{SharedState, ToolRegistry};
use molo::{
    Agent, Memory, ReActAgent, Tool, ToolContext, ToolError, ToolOutput, ToolResult, ToolSchema,
    WindowMemory,
};

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn bench_agent_run(c: &mut Criterion) {
    let runtime = rt();
    let mut group = c.benchmark_group("agent_run");
    // A short 5-round conversation: loop termination check, Message recording, token counting, events (no
    // construction when no channel is attached).
    group.bench_function("5_rounds", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let provider = FakeProvider::new(vec![FakeReply::Text("hi".into()); 5]);
                let mut agent = ReActAgent::new(provider, ToolRegistry::new(), "");
                for _ in 0..5 {
                    agent.run("hi").await.unwrap();
                }
            })
        })
    });
    // Streaming: 20 rounds, consuming MessageChunk blocks one by one (Delta + Done per round).
    group.bench_function("stream_20_rounds", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let provider = FakeProvider::new(vec![FakeReply::Text("hi".into()); 20]);
                let mut agent = ReActAgent::new(provider, ToolRegistry::new(), "");
                let mut stream = agent.run_stream("hi").await.unwrap();
                while let Some(Ok(_)) = futures::StreamExt::next(&mut stream).await {}
            })
        })
    });
    group.finish();
}

fn bench_tool_call(c: &mut Criterion) {
    // A zero-overhead tool: static schema, call returns directly.
    struct Noop;
    #[molo::async_trait]
    impl Tool for Noop {
        fn schema(&self) -> ToolSchema {
            ToolSchema::new("noop", "noop", serde_json::json!({}))
        }
        async fn call(
            &self,
            _arguments: serde_json::Value,
            _context: ToolContext<'_>,
        ) -> Result<ToolResult, ToolError> {
            Ok(ToolOutput::text("ok").into())
        }
    }

    let runtime = rt();
    let mut group = c.benchmark_group("tool_call");
    group.bench_function("registry_call", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let mut registry = ToolRegistry::new();
                registry.register(Noop);
                let state = SharedState::new();
                registry
                    .call_named("noop", "{}", &molo::RunContext::new("bench"), &state)
                    .await
                    .unwrap();
            })
        })
    });
    group.finish();
}

fn bench_window_memory(c: &mut Criterion) {
    let runtime = rt();
    let mut group = c.benchmark_group("window_memory");
    // 50 rounds of conversation writes + over-budget trimming (default WindowDrop): the amortized cost of record's
    // incremental counting and the cached trimming path of context.
    group.bench_function("trim_50_rounds", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let mut memory = WindowMemory::new(200);
                for i in 0..50 {
                    memory
                        .record(Message::user(format!("user message number {i}")))
                        .await
                        .unwrap();
                    memory
                        .record(Message::assistant(format!("assistant reply number {i}")))
                        .await
                        .unwrap();
                }
                let _ = memory.context().await.unwrap();
            })
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_agent_run,
    bench_tool_call,
    bench_window_memory
);
criterion_main!(benches);
