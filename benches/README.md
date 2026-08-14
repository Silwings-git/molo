# Runtime Benchmarks

These benchmarks measure framework hot paths only. They use generated or
scripted fixtures, never live providers, API keys, private source code, or raw
user transcripts.

Recommended commands:

```bash
cargo bench -p molo --features full
cargo bench -p molo --features full --no-run
```

Benchmark workflow:

- normal CI compiles the benchmark suite with `--no-run`;
- baseline updates record a full local benchmark summary under
  `benches/baselines/`;
- regression review investigates hot-path regressions over 20% or
  allocation/count blow-ups before publishing;
- baseline updates must state whether the change came from implementation
  changes, fixture changes, or an accepted performance tradeoff.

Do not commit raw Criterion output or machine-specific target directories.
Only commit redacted summaries that identify the benchmark names, toolchain,
feature set, and interpretation.
