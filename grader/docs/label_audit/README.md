# Label audit — operational runbook

This doc is the **step-by-step command reference** for the LLM label-audit pipeline and for **monitoring baselines** after audits change label distributions. Higher-level context lives in [`../../README.md`](../../README.md) (section “LLM label-audit workflow”).

Run commands from the **repository root** via `uv run` unless noted.

---

## Prerequisites

- **`uv`** environment synced (`uv sync` from repo root; use `--extra test` if you rely on cleanlab-backed flows).
- **API keys** for the LLM provider you use in `label_audit_run_llm` (commonly Gemini). Load from **`.env`** or your shell; see [`label_audit_backend`](../../src/eval/label_audit_backend.py) / runner docs for provider-specific variables.
- Cleanlab audit **CSV** paths (for queue build) if you start from Cleanlab exports.

Default SQLite queue path used below: **`grader/reports/label_audit_queue.sqlite`**.

---

## Pipeline (canonical order)

### 1. Build the audit queue

From one or more Cleanlab CSV files:

```bash
uv run python -m grader.src.eval.label_audit_queue_build \
  --config grader/configs/grader.yaml \
  --db grader/reports/label_audit_queue.sqlite \
  --csv path/to/cleanlab_audit_a.csv path/to/cleanlab_audit_b.csv
```

Optional: `--default-split train|val|test` when split cannot be inferred.

---

### 2. Run the LLM over the queue

Uses grading guidelines aligned with production rules:

```bash
uv run python -m grader.src.eval.label_audit_run_llm \
  --config grader/configs/grader.yaml \
  --guidelines grader/configs/grading_guidelines.yaml \
  --db grader/reports/label_audit_queue.sqlite
```

Useful flags (see module help): `--limit N`, `--gating-pass 1|2|3`, `--splits train val test`, `--targets sleeve media`, `--sleep-seconds`, `--require-disagree`.

```bash
uv run python -m grader.src.eval.label_audit_run_llm --help
```

---

### 3. Human review (Streamlit)

```bash
uv run streamlit run grader/src/eval/annotation_review_app.py
```

---

### 4. Export reviewed rows to CSV

```bash
uv run python -m grader.src.eval.label_audit_export_reviewed \
  --db grader/reports/label_audit_queue.sqlite \
  --output grader/reports/label_audit_reviewed_export.csv
```

---

### 5. Commit reviewed decisions into label patches

Appends into **`grader/data/label_patches.jsonl`** (idempotent guards):

```bash
uv run python -m grader.src.eval.label_audit_commit_patches \
  --db grader/reports/label_audit_queue.sqlite \
  --label-patches grader/data/label_patches.jsonl
```

---

## Optional: calibrate auto-review policy

After enough human-reviewed rows exist in the queue DB (defaults require **≥100** reviewed rows and **≥30** per target unless `--force`):

Dry-run report:

```bash
uv run python -m grader.src.eval.label_audit_calibrate_policy \
  --db grader/reports/label_audit_queue.sqlite \
  --dry-run
```

Apply policy (example):

```bash
uv run python -m grader.src.eval.label_audit_calibrate_policy \
  --db grader/reports/label_audit_queue.sqlite \
  --apply
```

See `--help` for `--mode`, `--holdout-ratio`, `--policy-version`, etc.

---

## Optional: critic evaluation (first pass vs critic)

Requires **≥20** reviewed rows in the queue:

```bash
uv run python -m grader.src.eval.label_audit_eval_critic \
  --db grader/reports/label_audit_queue.sqlite \
  --env-file .env \
  --report-path grader/reports/critic_eval_report.json
```

---

## After an audit: monitoring and baseline hygiene

Label audits **intentionally change** label distributions and sometimes feature/queue composition. Drift monitors that compare “today” to an **old** reference will alarm even when nothing is broken.

### Before a large audit

- Record the **monitoring baseline identity** you use in production (for example GCS path + tag or `MONITORING_REFERENCE_TAG` once VinylIQ/grader monitoring ships).

### During the campaign

- Optionally **silence** scheduled drift jobs (repo variable, workflow toggle, or pause cron) for the affected vertical so expected shifts do not page.

### After labels stabilize

1. **Retrain** models if training labels changed materially; prediction distributions may move until then.
2. **Promote a new reference snapshot** for drift monitoring (new Parquet/JSON artifact version on GCS or your bucket)—do not compare post-audit traffic to a pre-audit baseline indefinitely.
3. Attach a one-line note to the audit ticket: baseline version bumped from **A → B**.

Audit-export CSVs are **not** a random sample of production traffic unless you designed them that way—do not feed them into **production traffic drift** suites without documenting the cohort.

---

## Related tests

```bash
uv run pytest grader/tests/test_label_audit_backend.py grader/tests/test_label_audit_calibrate_policy.py grader/tests/test_label_audit_critic.py -v
```

---

## See also

- [`../../README.md`](../../README.md) — LLM label-audit workflow overview and evaluation loop.
- [`../../src/eval/`](../../src/eval/) — implementation modules (`label_audit_*`).
