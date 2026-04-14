# VinylIQ — Project Plan
> Chrome extension for ML-powered vinyl price estimation on Discogs

---

## 1. Product Overview

A Chrome extension that overlays price estimates directly on Discogs release pages. The user provides a release's media and sleeve condition grades; the backend ML model returns a condition-adjusted fair market value estimate.

**Two user modes:**
- **Buyers** — understand true collection value at a glance
- **Sellers** — get suggested pricing for inventory without manual research

---

## 2. Presentation Outline

### Slide 1 — Project Description
- Chrome extension overlaying price estimates on Discogs release pages
- Inputs: release ID + media/sleeve condition grades
- Outputs: estimated fair market value with confidence range
- Backend ML model served via FastAPI; predictions injected as a lightweight DOM overlay

### Slide 2 — Competitive Landscape (SWOT)

| Strengths | Weaknesses |
|---|---|
| Condition-aware pricing (not just median) | No access to raw transaction history via API |
| Embedded directly in Discogs UX | Chrome-only at MVP stage |
| Serves both buyers and sellers | Cold-start problem for rare/niche pressings |

| Opportunities | Threats |
|---|---|
| Growing vinyl market; no dominant ML pricing tool | Discogs could productize this themselves |
| Want/have ratio is a strong scarcity signal | API terms may tighten further |
| LLM-assisted dev accelerates build speed | Cross-site data matching too unreliable |

**Competitor comparison:**
- **Discogs built-in pricing suggestion** — simple median of recent sales, no ML, no condition weighting
- **Popsike** — manual lookup, no Discogs integration, scraping-only
- **Our edge** — condition × rarity × metadata signals, embedded where users already are

### Slide 3 — Data Strategy

| Source | What You Get | Status |
|---|---|---|
| Discogs API `/marketplace/stats/{release_id}` | Current lowest price, median price, # for sale | ✅ Permitted |
| Discogs Monthly Data Dumps | Full catalog: 15M+ releases, want/have counts, genre, label, format, year, country | ✅ CC0 license |
| Authenticated API (OAuth) | User's own collection + inventory with condition grades | ✅ Permitted for own data |

**Why not scrape?** Discogs ToS explicitly prohibits automated scraping and using service content to train ML models. Cross-site matching (e.g. Popsike) introduces release ID ambiguity that corrupts training labels.

### Slide 4 — Goals & 7-Week Timeline

**Success metrics:**
- Price estimate within ~15% of actual marketplace median
- Prediction loads in under 2 seconds in-extension
- Works across at least 3 genres (rock, jazz, electronic)

| Week | Focus | Milestone |
|---|---|---|
| **Week 1** | API + OAuth setup; data dump EDA; feature engineering; train baseline model; MLflow tracking; FastAPI endpoint | 🏁 **Remote Tracking & Online API** |
| **Week 2** | Model evaluation (MAE/RMSE); iterate on features; document system architecture | 🏁 **System Design + Accomplishments** |
| **Week 3** | Chrome extension scaffold; connect to FastAPI; basic overlay UI | — |
| **Week 4** | Collection valuation feature; UI polish; user-facing narrative and demo prep | 🏁 **Product Pitch** |
| **Week 5** | Deep-dive model choices, feature engineering, evaluation metrics; technical narrative | 🏁 **Technical Pitch** |
| **Week 6** | End-to-end demo recording; final UI polish; wrap-up | 🏁 **Video Demo + Product Pitch** |
| **Week 7** | Buffer — model improvements, edge cases, presentation polish | — |

---

## 3. Data Strategy (Detailed)

### 3.1 The Core Constraint
The data dump does **not** include any pricing data — only catalog metadata and community `want`/`have` counts. Price labels must be collected actively by calling `/marketplace/stats` per release ID and storing results. This is your training dataset bottleneck — **start the collection script on Day 1**.

At ~25 authenticated requests/minute, collecting 10,000 labeled examples takes several hours. Let it run in the background while you set up other parts of the stack.

**Training label queue (coverage vs. popularity):** You only need `/marketplace/stats` for IDs you intend to label—not the full dump. Combine **(1)** IDs sorted by community **have** / **want** (liquid head of the catalog) with **(2)** **stratified** draws (e.g. decade × genre from `releases_features`) so decades and genres are not all “top hits.” The repo script **`scripts/build_stats_collection_queue.py`** merges those streams into one deduped file for **`collect_marketplace_stats.py`**. Order-of-magnitude targets and rationale: see **`README.md` → “Data collection strategy.”**

### 3.2 Data Sources

**Discogs Monthly Data Dump** (`data.discogs.com`)
- Format: gzipped XML (~5GB compressed, ~32GB unpacked for releases)
- Use a streaming SAX parser — do not load into memory all at once
- Recommended tool: [`discogs-xml2db`](https://github.com/philipmat/discogs-xml2db) to parse into SQLite/PostgreSQL
- Extract per release: `release_id`, `master_id`, `title`, `artists`, `labels`, `genres`, `styles`, `year`, `country`, `format`, `want_count`, `have_count`

**Discogs API — Price Labels**
- `GET /marketplace/stats/{release_id}` → `lowest_price`, `median_price`, `num_for_sale`
- This is your **training label** (`median_price`) and baseline anchor at inference
- Cache responses to avoid redundant calls (SQLite or Redis)

**Multi-Dump Temporal Features (stretch goal)**
- Download several months of historical dumps from the archive
- Extract `want`/`have` counts per release per dump month
- Engineer trend features: `want_growth_rate`, `have_growth_rate`, `want_have_ratio_trend`
- Prices still come from the live API — temporal signals enrich features only
- Storage tip: don't store full dumps — extract just the signals into a narrow table: `release_id | dump_month | wants | haves`

### 3.3 Feature Engineering

**Condition grades** — encode as ordinal integers:
```
Media/Sleeve: M=8, NM=7, VG+=6, VG=5, G+=4, G=3, F=2, P=1
Sleeve-only extras: Generic=0, Not Graded=-1, No Cover=-2
```

**Feature set:**

| Feature | Derivation | Signal |
|---|---|---|
| `media_grade` | ordinal encoding | Core condition input |
| `sleeve_grade` | ordinal encoding | Core condition input |
| `condition_discount` | media_grade / 8.0 | How far below mint |
| `want_have_ratio` | wants / haves | Scarcity/demand proxy |
| `log_have_count` | log(haves) | Normalised supply |
| `log_num_for_sale` | log(num_for_sale + 1) | Current market supply |
| `is_original_pressing` | release year == master year | Originals trade at premium |
| `label_tier` | categorical encode top labels | Blue Note ≠ generic reissue |
| `is_colored_vinyl` | from format description | Special formats = premium |
| `is_picture_disc` | from format description | Special formats = premium |
| `is_promo` | from format description | Collector premium |
| `genre_encoded` | label/target encode | Genre affects price range |
| `decade` | floor(year/10)*10 | Decade > exact year |
| `want_growth_rate` | % Δ wants across dumps | Momentum signal (stretch) |
| `want_have_ratio_trend` | slope of ratio over time | Heating up or cooling off (stretch) |

**Target variable:** `log1p(median_price)` — log-transform to handle heavy right skew (most records $5–30, some $500+)

---

## 4. Model Strategy

### 4.1 Baseline Model
**XGBoost or LightGBM** — recommended because:
- Handles mixed numeric + categorical features natively
- Robust to missing values (many releases have incomplete metadata)
- Fast to train and iterate on
- Explainable via feature importance (useful for technical pitch)

**Train/test split:** by `release_id`, not random rows — avoids leakage across pressings of the same master.

### 4.2 Condition Multiplier Layer
After the base model predicts a price, apply a condition adjustment learned from the data. Derive from training data: what is the average ratio of VG+ price to NM price for the same release? This makes condition sensitivity interpretable and easy to explain.

### 4.3 MLflow Tracking
Log per run:
- Hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`
- Metrics: MAE, RMSE, MAPE on held-out set
- Feature importance plot as artifact
- Trained model as a registered model

```python
import mlflow
import mlflow.xgboost

with mlflow.start_run():
    mlflow.log_params({"n_estimators": 300, "max_depth": 6})
    model = xgb.train(params, dtrain)
    mlflow.log_metric("mae", mae)
    mlflow.xgboost.log_model(model, "price_estimator")
```

### 4.4 Known Risks

| Risk | Mitigation |
|---|---|
| Data collection is the bottleneck | Start API collection script Day 1 |
| Rate limits (~25 req/min authenticated) | Throttle requests; cache all responses |
| Rare releases have no marketplace stats | Fall back to genre/decade/format median |
| User condition grades are self-reported and optimistic | Add calibration note in UI |
| Cold start for very obscure releases | Use master release median as fallback |

---

## 5. System Architecture

### 5.1 FastAPI Backend

Three endpoints:

```
POST /estimate
  body:    { release_id, media_condition, sleeve_condition }
  returns: { estimated_price, confidence_interval, baseline_median }

POST /collection/value
  body:    { username, items: [{release_id, media_condition, sleeve_condition}] }
  returns: { total_estimated_value, per_item_breakdown }

GET /health
  returns: { status: "ok" }
```

**Inference flow:**
1. Receive `release_id` + condition grades
2. Fetch live `/marketplace/stats` for that release (or from cache)
3. Look up static features from precomputed feature store (SQLite table from data dump)
4. Assemble feature vector → run XGBoost model → apply condition multiplier
5. Return estimate + confidence interval

### 5.2 Chrome Extension
Keep it thin — all ML logic lives server-side. The extension only needs to:
1. **Detect** when user is on a release page (`discogs.com/release/...`) — parse release ID from URL
2. **Read** condition grades from the page DOM (present in marketplace listing HTML)
3. **Call** FastAPI `/estimate` and inject the overlay card into the DOM

### 5.3 Feature Store
A lightweight SQLite table built once from the data dump and refreshed monthly:
```
releases_features(release_id, want_count, have_count, want_have_ratio,
                  genre, decade, country, label_tier, is_original,
                  is_colored_vinyl, is_picture_disc, is_promo)
```

---

## 6. Immediate Next Steps (Week 1 Priority Order)

1. Register Discogs developer app → get API credentials + OAuth token
2. Write and run data collection script: iterate release IDs from dump, call `/marketplace/stats`, store to SQLite
3. Download latest data dump; set up streaming parser (use `discogs-xml2db`)
4. EDA: distribution of prices, condition grade frequency, want/have ratio spread
5. Build feature engineering pipeline
6. Train baseline XGBoost model; evaluate MAE/RMSE
7. Set up MLflow tracking server
8. Wrap model in FastAPI with `/estimate` and `/health` endpoints
9. Deploy FastAPI (even locally via ngrok is fine for the milestone)
