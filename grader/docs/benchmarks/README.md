# Grader benchmarks

Performance and smoke benchmarks for the vinyl grader package. Run commands from the **repository root** unless noted.

## Environment (fill when recording)

| Field | Example |
|-------|---------|
| OS | macOS 14.x |
| CPU | Apple M2 |
| GPU | `nvidia-smi` or N/A |
| Python | `uv run python -V` |
| Commit | `git rev-parse --short HEAD` |

---

## 1. Inference (`Pipeline.predict_batch`)

**Command:**

```bash
uv run python grader/scripts/bench_infer.py
```

Optional: `uv run python grader/scripts/bench_infer.py --config path/to/grader.yaml`

**Results**

| date | commit | metric | value | notes |
|------|--------|--------|-------|-------|
| — | — | wall_s_predict_batch | N/A | First run after clone; fill on first measurement |

Cold start includes lazy load of preprocessor / model / rule engine.

---

## 2. Training

**Command (stub):** short train or subset flags when available; otherwise document manual / Colab.

**Results**

| date | commit | metric | value | notes |
|------|--------|--------|-------|-------|
| — | — | — | N/A | Template — safe CI subset TBD |

---

## 3. Discogs ingest

**Command:**

```bash
uv run python grader/scripts/bench_ingest_smoke.py --dry-run --cache-only
```

Throughput with a live token is **local-only**; do not run rate-limited benchmarks in CI without mocks.

**Results**

| date | commit | metric | value | notes |
|------|--------|--------|-------|-------|
| — | — | smoke_wall_s | N/A | dry-run + cache-only smoke |

---

## 4. Eval tools (label audit / annotation review)

**Command (stub):** manual Streamlit / LLM throughput; automate later.

**Results**

| date | commit | metric | value | notes |
|------|--------|--------|-------|-------|
| — | — | — | N/A | Template |

---

## Policy

- PRs that change measured algorithm / batching / concurrency **must** add a row to the relevant table and bump notes.
- Split-only refactors with no perf claim **do not** require new rows unless they add or rename a benchmark script (then update this README and `grader/scripts/README.md`).
