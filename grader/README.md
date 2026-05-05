# Vinyl condition grader (`grader/`)

Supervised NLP plus a rule layer that predicts **sleeve** and **media** condition from seller notes (Discogs / eBay JP style), with optional CoreML export for on-device iOS use.

This package is part of **vinyl_management_system**. Paths in [`grader/configs/grader.yaml`](grader/configs/grader.yaml) are relative to the **current working directory**; run commands from the **repo root** unless you use absolute paths.

---

## Repository structure

```
grader/
├── configs/
│   ├── grader.yaml              # Main pipeline paths, ingest, models, MLflow
│   ├── grading_guidelines.yaml  # Grade schema, rules, eBay JP harmonization
│   └── transformer_tune.yaml    # Hyperparameter sweep presets (optional)
├── src/
│   ├── data/                    # ingest_discogs, ingest_ebay, ingest_sale_history,
│   │                            # harmonize_labels, preprocess
│   ├── features/                # tfidf_features
│   ├── models/                  # baseline_model, transformer, transformer_tune
│   ├── eval/                    # export_mispredictions, …
│   └── pipeline.py              # train (steps 1–9), predict
├── data/                        # raw + processed + splits (gitignored as configured)
├── artifacts/                   # model weights, vectorizers (gitignored)
├── reports/                     # class distribution, tuning CSV, etc.
├── experiments/                 # MLflow SQLite + run dirs (see below)
├── discogs_release_marketplace_workflow.md
└── README.md
```

**Config:** [`grader/configs/grader.yaml`](grader/configs/grader.yaml) merges with [`configs/base.yaml`](../configs/base.yaml). Secrets use env vars (for example `DISCOGS_TOKEN`, eBay keys) — never commit tokens.

---

## Prerequisites

- **Discogs:** [personal access token](https://www.discogs.com/settings/developers). Set `DISCOGS_TOKEN` in the **repo-root** `.env` or export it in the shell. Ingest modules load `.env` via `load_project_dotenv` (searching upward from cwd and from `grader/src/`). Existing shell variables are not overwritten.
- **eBay JP ingest (optional):** `EBAY_CLIENT_ID` and `EBAY_CLIENT_SECRET` when you do not pass `--skip-ebay-ingest` and want `ebay_processed.jsonl` merged at harmonization.
- **Sale history (optional):** SQLite path defaults under `data.sale_history` in `grader.yaml` (for example `price_estimator/data/cache/sale_history.sqlite`). Populate it with your sale-history pipeline before `ingest_sale_history`.

---

## Setup and run

From **project root** (`vinyl_management_system`):

```bash
uv sync --extra test
```

Use `uv run python …` so the workspace package resolves without manual `PYTHONPATH`.

---

## Training workflow

The orchestrator is **`python -m grader.src.pipeline train`**. It runs, in order:

1. **Data ingestion** — Discogs inventory API, optional eBay Browse API, then label patches and optional post-patch vinyl filter.
2. **Label harmonization** — merge sources into `unified.jsonl`.
3. **Preprocessing and splitting** — train / val / test JSONL under `paths.splits`, adequacy reports.
4. **TF-IDF feature extraction** — sparse matrices and encoders under `paths.artifacts`.
5. **Baseline (TF-IDF + logistic regression)** — train or load from disk when `--skip-baseline`.
6. **Transformer (DistilBERT)** — unless `--baseline-only` / `--skip-transformer`.
7. **Model comparison** — baseline vs transformer on test.
8. **Calibration evaluation** — reliability / ECE-style diagnostics.
9. **Rule engine evaluation** — on `test` and `test_thin`.

**MLflow:** Tracking metadata defaults to **`grader/experiments/mlflow.db`** (SQLite). Pass **`--no-mlflow`** to disable. ETL CLIs may open MLflow runs when configured; **`--no-mlflow`** on `pipeline train` disables tracking for the full train pass.

**Thin seller notes:** With `preprocessing.description_adequacy` and `drop_insufficient_from_training: true`, inadequate rows are excluded from train/val/test but kept in **`grader/data/splits/test_thin.jsonl`** for evaluation. Inference adds `needs_richer_note`, `description_quality_gaps`, and `description_quality_prompts` in prediction `metadata`. See `grader/reports/description_adequacy_summary.txt` after preprocess.

### One command (full pipeline)

Discogs only, baseline without transformer (fast smoke on real data):

```bash
uv run python -m grader.src.pipeline train --skip-ebay-ingest --baseline-only
```

Full train including eBay and transformer (needs all tokens and time):

```bash
uv run python -m grader.src.pipeline train
```

Reuse everything on disk and only re-run modeling steps (steps 5–9; still reads splits/features from `grader.yaml` paths):

```bash
uv run python -m grader.src.pipeline train \
  --skip-ingest \
  --skip-harmonize \
  --skip-preprocess \
  --skip-features
```

Common flags (combine as needed): `--skip-ebay-ingest`, `--skip-ingest`, `--skip-harmonize`, `--skip-preprocess`, `--skip-features`, `--skip-baseline`, `--baseline-only` / `--skip-transformer`, `--no-mlflow`, `--mlflow-no-artifacts`, `--no-register`.

---

### 1) Discogs marketplace ingest

Fetches seller inventory via **`GET /users/{username}/inventory`** (not `/marketplace/search`, which returns 404). Writes raw pages under `grader/data/raw/discogs/inventory/` and processed **`grader/data/processed/discogs_processed.jsonl`** (paths from `data.discogs` in `grader.yaml`).

```bash
# Auth + parse only; no writes, no MLflow
uv run python -m grader.src.data.ingest_discogs --dry-run

# Full pull (respects target_per_grade and sellers in grader.yaml — can take a long time)
uv run python -m grader.src.data.ingest_discogs

# Smoke: lower per-grade cap
uv run python -m grader.src.data.ingest_discogs --target-per-grade 80

# Override format filter (YAML default is often Vinyl)
uv run python -m grader.src.data.ingest_discogs --format Vinyl

# Only parse cached raw pages; skip missing pages (no new HTTP)
uv run python -m grader.src.data.ingest_discogs --cache-only
```

**Details (same behavior as before):** Seller list: `data.discogs.inventory_sellers` in `grader.yaml`. **Pagination:** Discogs returns **403** after **page 100** for shops that are not yours (~10k items at 100 rows/page). **Page size:** The ingester sends `per_page` / `limit` / `format` per YAML; if the API responds with `pagination.per_page: 100`, you may see a warning (API vs seller UI). **Generic notes:** Rows matching `generic_note_filter` patterns can be dropped unless item-specific hints, Mint, or preservation phrases apply; optional **strip_boilerplate** trims noise from `text`. Tune in `grader.yaml` or set `generic_note_filter.enabled: false`. Raw pages are resume-safe; delete seller folders under `grader/data/raw/discogs/inventory/` to force a full re-fetch.

---

### 2) eBay JP ingest (optional)

When you want `ebay_processed.jsonl` merged at harmonization:

```bash
uv run python -m grader.src.data.ingest_ebay --dry-run
uv run python -m grader.src.data.ingest_ebay
```

Or omit eBay in the full pipeline with **`--skip-ebay-ingest`**.

---

### 3) Sale history → grader JSONL (optional)

Exports completed sales from SQLite into the same record shape as Discogs ingest, for harmonization. Defaults: `data.sale_history` in `grader.yaml` (`sqlite_path`, `processed_jsonl`).

The exporter joins each `release_id` to **`releases_features`** in the price-estimator feature store (`feature_store_path`, e.g. `price_estimator/data/feature_store.sqlite`) to set `release_format` / `release_description`, then applies the same **physical-vinyl** filter as `discogs_processed.jsonl` (see `apply_vinyl_filter` and `vinyl_format.py`). Optional `enrich_missing_from_discogs` uses the Discogs API for releases missing from the feature store. `on_missing_release: keep` retains rows with no format data (the default); `drop` removes them.

```bash
uv run python -m grader.src.data.ingest_sale_history --dry-run
uv run python -m grader.src.data.ingest_sale_history
```

Useful flags: `--sale-db`, `--out`, `--limit N`, `--require-fetch-ok` (only releases with `sale_history_fetch_status.status = ok`), `--dry-run`.

The full training pipeline runs the same export at the **start** of ingest (before Discogs/eBay) so `label_patches` can merge into a fresh `discogs_sale_history.jsonl` after the write. Use **`python -m grader.src.pipeline train --skip-sale-history`** to skip it (for example if the sale DB is empty or you want a faster train).

Run **before** harmonize so `discogs_sale_history.jsonl` exists when `LabelHarmonizer` runs.

---

### 4) Release marketplace JSONL (optional)

If present, **`grader/data/processed/discogs_release_marketplace.jsonl`** is merged when building `unified.jsonl`. See [`grader/discogs_release_marketplace_workflow.md`](discogs_release_marketplace_workflow.md) for how to produce that file.

---

### 5) Harmonize labels

Merges Discogs (and optional eBay, sale history, marketplace JSONL) into **`grader/data/processed/unified.jsonl`** (see `data.harmonization.output_path`). Writes **`grader/reports/class_distribution.txt`**.

```bash
uv run python -m grader.src.data.harmonize_labels --dry-run
uv run python -m grader.src.data.harmonize_labels
```

---

### 6) Preprocess and split

Reads `unified.jsonl`, applies cleaning / adequacy rules, writes split JSONL files under **`grader/data/splits/`** (`paths.splits`).

```bash
uv run python -m grader.src.data.preprocess --dry-run
uv run python -m grader.src.data.preprocess
```

---

### 7) TF-IDF features

Builds sparse feature matrices and encoders under **`grader/artifacts/`** (`paths.artifacts`).

```bash
uv run python -m grader.src.features.tfidf_features --dry-run
uv run python -m grader.src.features.tfidf_features
```

---

### 8) Baseline, transformer, comparison, calibration, rules

There is **no** separate `python -m` entrypoint for the baseline-only or rule-evaluation substeps; they run inside **`pipeline train`**.

After steps **1–7** above (manual ETL + features), run modeling and evaluation without redoing data:

```bash
uv run python -m grader.src.pipeline train \
  --skip-ingest \
  --skip-harmonize \
  --skip-preprocess \
  --skip-features
```

**Load baseline from disk** (skip step 5 training; still evaluates if artifacts exist):

```bash
uv run python -m grader.src.pipeline train \
  --skip-ingest \
  --skip-harmonize \
  --skip-preprocess \
  --skip-features \
  --skip-baseline
```

**Transformer only on top of existing baseline artifacts** is still invoked through the same command with appropriate skips; for a **single** DistilBERT train without the full pipeline, you can run:

```bash
uv run python -m grader.src.models.transformer
# Optional: --skip-mlflow
```

---

## Rule-engine iteration loop (Track B)

**`guidelines_version`** (top of `grading_guidelines.yaml`, format `YYYY.MM.DD` or `YYYY.MM.DD.n`) must be bumped whenever rule semantics or YAML schema shape changes. The **committed** `grader/reports/rule_engine_baseline.json` must list the same `guidelines_version` and `canonical_grades_sha256` (regenerate via pipeline eval or update in the same PR). `pytest` enforces pairing when the baseline file is present. Training logs an MLflow tag `guidelines_version` and a `training_rubric_manifest` artifact; the FastAPI grader compares the tag to runtime YAML (see `grader/serving/README.md` and `GRADER_STRICT_GUIDELINES_PAIRING`).

**Change tier (for PRs):** state **A** (signal/threshold only), **B** (logic/ownership, no new labels), or **C** (grade schema / maps) in the description and use the **Tier C** issue template for schema changes that need stakeholder sign-off (`.github/ISSUE_TEMPLATE/grader_tier_c_rubric.md`).

`grading_guidelines.yaml` evolves in short, measurable iterations. In `grader/configs/grader.yaml`, **`rules.allow_excellent_soft_override`** defaults to **`false`**, so the post-model rule stack does not assign the **Excellent** grade via soft override unless you opt in (useful when training/eval have no gold Excellent). With that default, the rule engine **remaps any final Excellent sleeve or media grade to Near Mint** (confidence mass folded into Near Mint), including when the model predicted Excellent and on the contradiction early-return path, so shipped labels stay off Excellent. Each loop has two committed outputs: the **baseline snapshot** (`grader/reports/rule_engine_baseline.json`, which includes per-target `delta_macro_f1` and `delta_accuracy` alongside override stats, also mirrored to MLflow tags under `rule_baseline_*`) and the **override audit / rule-owned slice sections** appended to `grader/reports/grade_analysis_{split}.txt` by `_run_rule_engine_evaluation`.

**One iteration:**

1. **Reproduce the rule-eval report** on the latest artifacts (no re-train needed if features + baseline are current):

   ```bash
   uv run python -m grader.src.pipeline train \
     --skip-ingest --skip-harmonize --skip-preprocess --skip-features \
     --skip-baseline --skip-transformer
   ```

   If `test_thin` is huge, set `evaluation.rule_eval_splits: [test]` in `grader.yaml` (uncomment the stub there) so step 9 only scores `test` and finishes quickly; restore `test` + `test_thin` for full coverage before you commit.

2. **Regenerate the diagnostic CSVs** for the split under review:

   ```bash
   uv run python -m grader.src.eval.analyze_harmful_overrides
   uv run python -m grader.src.eval.analyze_missed_rule_owned
   ```

   Outputs land under `grader/reports/harmful_overrides_{split}.csv` and `grader/reports/missed_rule_owned_{split}_{target}.csv`. The missed-rows exporter emits two diagnostic booleans — `hard_signal_pre_forbidden` (would any hard signal match, ignoring forbiddens?) and `hard_signal_post_forbidden` (what the engine actually did) — so rows categorize cleanly into *missing pattern* (both false) vs *over-eager forbidden* (pre true, post false).

3. **Diff the baseline** against the prior iteration's `rule_engine_baseline.json` (git history or a saved copy) and open MLflow tags keyed `rule_baseline_*` in parallel. Use `python -m grader.src.eval.diff_rule_engine_baseline <before> <after>`; it understands the on-disk `splits` shape from `_write_rule_engine_baseline`. Focus on the rule-owned `by_after` buckets (`Poor`, `Generic`) and `slice_recall` / `recall_adjusted`.

  ```bash
  python -m grader.src.eval.diff_rule_engine_baseline <before> <after>
  ```


4. **Edit `grader/configs/grading_guidelines.yaml`**. For every change, pair a positive and an adversarial fixture in [`grader/tests/test_grade_analysis_slice.py`](grader/tests/test_grade_analysis_slice.py) (`TestRegressionGuidelineFixtures`) or [`test_rule_engine.py`](grader/tests/test_rule_engine.py). The `hard_signals_strict` / `hard_signals_cosignal` split (and the per-target `_sleeve` / `_media` variants) is the primary tuning surface — prefer demoting phrases to `cosignal` over widening `forbidden_signals`, because cosignal requires *evidence* rather than *absence of counter-evidence*.

5. **Green tests + re-run step 1**. Commit only if:
   - every rule-owned `slice_recall_adjusted` is non-decreasing,
   - every rule-owned `by_after` bucket's `override_precision` is non-decreasing (2 pp slack on small splits),
   - global rule-adjusted macro-F1 is within 1 pp of the prior iteration.

Each harmful row is categorized using a four-class rubric:

- **False hard** — the rule fired and forced a hard override but the row was actually fine; usually a `forbidden_signals` term that needs demoting to `hard_signals_cosignal` (evidence-required) or removing.
- **Missing exception** — the rule fired correctly but the row had a legitimate exception that should have suppressed the override; widen `safe_signals` or `cosignal_keywords` rather than narrowing the trigger.
- **Contradiction candidate** — both pro- and anti-condition signals are present in the same comment; current heuristic suppresses overrides on these rather than scanning for negation.
- **Ambiguous** — the gold label itself is unclear; flag for label audit (see below) rather than tuning the rule.

We deliberately do not run negation-aware contradiction scanning at this iteration — the precision/recall trade-off is unfavorable on the current label distribution, and label-quality work via the audit pipeline subsumes most of the same wins.

### LLM label-audit workflow

For systematically improving the training label quality, the canonical pipeline is:

1. [`grader/src/eval/label_audit_queue_build.py`](src/eval/label_audit_queue_build.py) — build a candidate review queue from disagreements between baseline / transformer / rules and from cleanlab signals.
2. [`grader/src/eval/label_audit_run_llm.py`](src/eval/label_audit_run_llm.py) — score each candidate with an LLM using the same grading guidelines the rules use.
3. [`grader/src/eval/annotation_review_app.py`](src/eval/annotation_review_app.py) — Streamlit UI for human review on top of the LLM proposals (`uv run streamlit run grader/src/eval/annotation_review_app.py`).
4. [`grader/src/eval/label_audit_export_reviewed.py`](src/eval/label_audit_export_reviewed.py) — export reviewed rows back to a label-patch CSV.
5. [`grader/src/eval/label_audit_commit_patches.py`](src/eval/label_audit_commit_patches.py) — apply the patches to the training SQLite, with idempotency guards so re-runs are safe.

Calibration metrics for the LLM critic itself live in [`label_audit_calibrate_policy.py`](src/eval/label_audit_calibrate_policy.py) and [`label_audit_eval_critic.py`](src/eval/label_audit_eval_critic.py); rerun those after any prompt or model swap before trusting fresh patches.

---

## Quick eval (synthetic / resume)

Same data recipe as `grader/tests/conftest.py` — fast, reproducible:

```bash
uv run python scripts/grader_eval_resume.py
```

Writes `artifacts/grader_eval_resume.json` with macro-F1, accuracy, and ECE. Use the `disclaimer` field: cite as **synthetic pipeline validation**, not real marketplace accuracy.

---

## DistilBERT hyperparameter tuning

Presets in [`grader/configs/transformer_tune.yaml`](configs/transformer_tune.yaml) merge onto `models.transformer` in `grader.yaml`. Each run writes weights under `grader/artifacts/tuning/<preset_key>/` and appends to `grader/reports/transformer_tune_results.csv`.

```bash
uv run python -m grader.src.models.transformer_tune --presets partial1_low_lr --dry-run
uv run python -m grader.src.models.transformer_tune --presets all --skip-mlflow
uv run python -m grader.src.models.transformer_tune --promote partial1_low_lr
```

Copy winning numeric settings from the CSV into `grader.yaml` under `models.transformer` for reproducible defaults.

---

## Misclassification sample for review

```bash
uv run python -m grader.src.eval.export_mispredictions \
  --split test --n 120 --stratify \
  --output grader/reports/mispredictions_sample_test.csv
```

Uses transformer weights under `grader/artifacts/` unless `--artifact-subdir tuning/<preset>` is set.

---

## Tests

```bash
uv run pytest grader/tests
```

---

## HTTP API (monorepo)

`POST /api/condition` on the FastAPI app calls the **baseline** grader in the default personal-use setup. See the root [`README.md`](../README.md) for the JSON contract.

---

## MLflow migration note

To migrate old runs from a legacy `./mlruns` file store, use [mlflow-export-import](https://github.com/mlflow/mlflow-export-import).
