# Coding Task Evals

Coding task evals are development and release-validation fixtures. Their
manifests are not a stable public API and should not capture raw prompts, raw
command output, API keys, auth headers, environment values, or private source
code.

Useful commands:

```bash
cargo run -p molo-eval-runner -- --validate-dir evals/cases/coding
cargo run -p molo-eval-runner -- --manifest evals/cases/coding/edit-function/eval.json
```

Results are written to `evals/results/` and ignored by default. Commit only
redacted summaries when comparing eval behavior across changes.
