//! Phase 9 benchmarks for governed effect lifecycle hot paths.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use molo::{
    AlwaysAllowApprovalBroker, AuditEvent, AuditSink, BasicHarness, DefaultPolicyEngine,
    DisplayFormat, DisplayOutput, EffectKind, EffectRequest, EffectStatus, Harness, HarnessConfig,
    NetworkPolicy, OutputLimit, PatternRedactor, RawEffectOutput, RunContext, RunMetadata,
    RunRequest, SandboxPolicy, StaticEffectExecutor, TranscriptRecord, TranscriptStore,
    VecAuditSink, VecTranscriptStore,
};
use serde_json::json;
use std::hint::black_box;
use std::time::Duration;

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("benchmark runtime")
}

fn bench_transcript_append(c: &mut Criterion) {
    let runtime = rt();
    let mut group = c.benchmark_group("runtime_transcript");
    group.bench_function("vec_transcript_append_100_records", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let store = VecTranscriptStore::new();
                let context = RunContext::new("bench-transcript");
                for i in 0..100 {
                    store
                        .append(
                            TranscriptRecord::EffectObservation {
                                run_id: context.run_id.clone(),
                                effect_id: format!("effect-{i}"),
                                status: EffectStatus::Succeeded,
                            },
                            &context,
                        )
                        .await
                        .expect("append transcript");
                }
                black_box(store.records());
            })
        })
    });
    group.finish();
}

fn bench_audit_append(c: &mut Criterion) {
    let runtime = rt();
    let mut group = c.benchmark_group("runtime_audit");
    group.bench_function("vec_audit_append_100_records", |b| {
        b.iter(|| {
            runtime.block_on(async {
                let sink = VecAuditSink::new();
                let context = RunContext::new("bench-audit");
                for i in 0..100 {
                    sink.record(
                        AuditEvent::EffectRequested {
                            effect_id: format!("effect-{i}"),
                            kind: EffectKind::ReadFile,
                            description: "read fixture".to_string(),
                            risk: molo::RiskLevel::Low,
                        },
                        &context,
                    )
                    .await
                    .expect("append audit");
                }
                black_box(sink.events());
            })
        })
    });
    group.finish();
}

fn harness_fixture() -> (
    BasicHarness<
        StaticEffectExecutor,
        DefaultPolicyEngine,
        AlwaysAllowApprovalBroker,
        VecAuditSink,
        VecTranscriptStore,
    >,
    EffectRequest,
    RunContext,
) {
    let mut metadata = RunMetadata::new();
    metadata.insert("debug".to_string(), json!("secret-token in metadata"));
    let raw = RawEffectOutput::text(format!(
        "secret-token {}\n{}",
        "visible",
        "x".repeat(128 * 1024)
    ))
    .with_display(DisplayOutput::new(
        DisplayFormat::PlainText,
        "secret-token display",
    ))
    .with_metadata(metadata);
    let harness = BasicHarness::new(
        StaticEffectExecutor::new().with_output("effect-1", raw),
        DefaultPolicyEngine,
        AlwaysAllowApprovalBroker,
        VecAuditSink::new(),
        VecTranscriptStore::new(),
    )
    .with_config(HarnessConfig {
        default_sandbox: SandboxPolicy::ReadOnly,
        default_network: NetworkPolicy::Deny,
        default_timeout: Duration::from_secs(5),
        output_limit: OutputLimit {
            model_bytes: 4096,
            display_bytes: 4096,
            debug_bytes: 1024,
        },
        ..HarnessConfig::default()
    })
    .with_redactor(PatternRedactor::new(["secret-token"]));
    let request =
        EffectRequest::new(EffectKind::ReadFile, "read fixture", json!({})).with_id("effect-1");
    (harness, request, RunContext::new("bench-harness"))
}

fn bench_harness_execute(c: &mut Criterion) {
    let runtime = rt();
    let mut group = c.benchmark_group("runtime_harness");
    group.bench_function("effect_lifecycle_with_limit_redaction_audit", |b| {
        b.iter_batched(
            harness_fixture,
            |(harness, request, context)| {
                runtime.block_on(async {
                    let observation = harness
                        .execute(request, &context)
                        .await
                        .expect("execute effect");
                    black_box(observation);
                })
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn bench_transcript_record_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime_transcript");
    let record = TranscriptRecord::RunStarted {
        run_id: "bench-run".to_string(),
        request: RunRequest::text("benchmark request"),
    };
    group.bench_function("record_json_serialize", |b| {
        b.iter(|| {
            black_box(serde_json::to_vec(&record).expect("serialize transcript"));
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_transcript_append,
    bench_audit_append,
    bench_harness_execute,
    bench_transcript_record_serialization
);
criterion_main!(benches);
