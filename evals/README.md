# Coding Task Evals

Phase 9 evals are internal release-gate fixtures. They are not a stable public
API and should not capture raw prompts, raw command output, API keys, auth
headers, environment values, or private source code.

Useful commands:

```bash
cargo run -p molo-eval-runner -- --validate-dir evals/cases/coding
cargo run -p molo-eval-runner -- --manifest evals/cases/coding/edit-function/eval.json
```

Results are written to `evals/results/` and ignored by default. Commit only
redacted summaries when a release review needs a baseline.
