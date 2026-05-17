# Grader scripts

Runnable utilities under `grader/scripts/`. Invoke from **repository root**:

```bash
uv run python grader/scripts/<script>.py --help
```

## Benchmark entrypoints

| Script | Purpose |
|--------|---------|
| [`bench_infer.py`](bench_infer.py) | Wall time for `Pipeline.predict_batch` on fixed fixture strings. |
| [`bench_ingest_smoke.py`](bench_ingest_smoke.py) | Smoke `DiscogsIngester` with `--dry-run` / `--cache-only` (no CI token required). |

Full environment notes and results tables: [`grader/docs/benchmarks/README.md`](../docs/benchmarks/README.md).
