# Runtime Benchmarks

Phase 9 benchmarks measure framework hot paths only. They use generated or
scripted fixtures, never live providers, API keys, private source code, or raw
user transcripts.

Recommended commands:

```bash
cargo bench -p molo --features full
cargo bench -p molo --features full --no-run
```

Release gate policy:

- normal CI compiles the benchmark suite with `--no-run`;
- pre-freeze records a full local benchmark summary under
  `benches/baselines/`;
- release-candidate comparison investigates hot-path regressions over 20% or
  allocation/count blow-ups before publishing;
- baseline updates must state whether the change came from implementation
  changes, fixture changes, or an accepted performance tradeoff.

Do not commit raw Criterion output or machine-specific target directories.
Only commit redacted summaries that identify the benchmark names, toolchain,
feature set, and interpretation.
