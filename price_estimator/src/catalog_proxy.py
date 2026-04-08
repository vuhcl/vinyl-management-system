"""
Catalog-based proxy score for ordering releases without community have/want.

Various-artist compilations: Discogs uses artist id **194** for "Various" and
often puts "Various" / "Various Artists" in the first ``artists_json`` entry.
Releases whose primary (first) artist is **Unknown Artist** are excluded too.
Those rows are dropped from proxy ranking and from master/artist fan-out
counts so compilations and placeholders do not dominate the queue.

**Digital / File:** Releases with Discogs format name **File** (see
``formats_json`` or ``format_desc``) are excluded from the stats-collection
queue.

**Vinyl:** LP / 7\" / 10\" / 12\" and Discogs format name **vinyl** (and related)
are treated as vinyl for sorting and optional quota sampling.
"""
from __future__ import annotations

# Discogs database artist id for "Various" (canonical compilation artist).
DISCOGS_VARIOUS_ARTIST_IDS_SQL = "('194')"

# Score = w_master * master_sibling_count
#   + w_artist * primary_artist catalog count.


def sql_exclude_various_primary_artist(artists_json_col: str) -> str:
    """
    SQL predicate: TRUE for rows to **keep** (drop unwanted primary artists).

    Excludes the first ``artists_json`` entry when it is Discogs Various (id
    **194**), when the name contains ``various`` (case-insensitive), or when the
    trimmed name equals **Unknown Artist** (case-insensitive).

    *artists_json_col* is a qualified column, e.g. ``artists_json`` or
    ``rf.artists_json``. Rows with NULL/empty ``artists_json`` are kept.
    """
    n0 = (
        f"LOWER(TRIM(COALESCE(json_extract({artists_json_col}, '$[0].name'), '')))"
    )
    return f"""(
  {artists_json_col} IS NULL OR TRIM({artists_json_col}) = ''
  OR (
    {n0} NOT LIKE '%various%'
    AND {n0} != 'unknown artist'
    AND CAST(COALESCE(json_extract({artists_json_col}, '$[0].id'), '')
      AS TEXT) NOT IN {DISCOGS_VARIOUS_ARTIST_IDS_SQL}
  )
)"""


def sql_exclude_file_format_releases(
    formats_json_col: str,
    format_desc_col: str,
) -> str:
    """
    SQL predicate: TRUE for rows to **keep** (drop Discogs **File** / digital).

    Uses ``formats_json`` array entries' ``name`` and common ``format_desc``
    patterns (``File``, ``File, ...``, ``..., File``).
    """
    fj = formats_json_col
    fd = format_desc_col
    return f"""(
  (
    {fj} IS NULL OR TRIM({fj}) = '' OR TRIM({fj}) = '[]'
    OR NOT EXISTS (
      SELECT 1 FROM json_each({fj}) AS _jfile
      WHERE LOWER(TRIM(COALESCE(json_extract(_jfile.value, '$.name'), '')))
        = 'file'
    )
  )
  AND (
    {fd} IS NULL OR TRIM({fd}) = ''
    OR NOT (
      LOWER(TRIM({fd})) = 'file'
      OR LOWER(TRIM({fd})) LIKE 'file,%'
      OR LOWER(TRIM({fd})) LIKE '%, file'
      OR LOWER(TRIM({fd})) LIKE '%, file,%'
    )
  )
)"""


def sql_vinyl_preference_key(formats_json_col: str, format_desc_col: str) -> str:
    """
    SQL expression: **1** if release looks like a vinyl **medium** (LP / singles),
    else **0**.

    Uses ``format_desc`` and structured ``formats_json`` ``name`` (and optional
    ``descriptions`` strings). Sort **DESC** to favor vinyl after the main score.
    """
    fj = formats_json_col
    fd = format_desc_col
    # Word-ish LP: pad with spaces so we do not match arbitrary substrings like "help".
    fd_lp = (
        f"(' ' || LOWER(REPLACE(REPLACE(COALESCE({fd}, ''), ',', ' '), ';', ' ')) "
        f"|| ' ')"
    )
    return f"""(CASE WHEN (
  (
    {fd} IS NOT NULL AND TRIM({fd}) != ''
    AND (
      INSTR(LOWER({fd}), 'vinyl') > 0
      OR INSTR(LOWER({fd}), '7"') > 0
      OR INSTR(LOWER({fd}), '10"') > 0
      OR INSTR(LOWER({fd}), '12"') > 0
      OR INSTR(LOWER({fd}), '7 inch') > 0
      OR INSTR(LOWER({fd}), '10 inch') > 0
      OR INSTR(LOWER({fd}), '12 inch') > 0
      OR INSTR({fd_lp}, ' lp ') > 0
      OR LOWER(TRIM({fd})) IN ('lp', 'vinyl')
    )
  )
  OR (
    {fj} IS NOT NULL AND TRIM({fj}) != '' AND TRIM({fj}) != '[]'
    AND EXISTS (
      SELECT 1 FROM json_each({fj}) AS _jvin
      WHERE LOWER(TRIM(COALESCE(json_extract(_jvin.value, '$.name'), ''))) IN (
        'vinyl', 'lp', '7"', '10"', '12"'
      )
      OR EXISTS (
        SELECT 1 FROM json_each(
          CASE
            WHEN json_type(_jvin.value, '$.descriptions') = 'array'
            THEN json_extract(_jvin.value, '$.descriptions')
            ELSE '[]' END
        ) AS _jdesc
        WHERE LOWER(COALESCE(_jdesc.value, '')) LIKE '%vinyl%'
           OR LOWER(COALESCE(_jdesc.value, '')) LIKE '%7"%'
           OR LOWER(COALESCE(_jdesc.value, '')) LIKE '%10"%'
           OR LOWER(COALESCE(_jdesc.value, '')) LIKE '%12"%'
           OR INSTR(' ' || LOWER(COALESCE(_jdesc.value, '')) || ' ', ' lp ') > 0
      )
    )
  )
) THEN 1 ELSE 0 END)"""


_NOT_VARIOUS_RF = sql_exclude_various_primary_artist("rf.artists_json")
_NOT_VARIOUS_PLAIN = sql_exclude_various_primary_artist("artists_json")
_NO_FILE_RF = sql_exclude_file_format_releases("rf.formats_json", "rf.format_desc")
_NO_FILE_PLAIN = sql_exclude_file_format_releases("formats_json", "format_desc")
_VINYL_KEY_RF = sql_vinyl_preference_key("rf.formats_json", "rf.format_desc")
_QUEUE_WHERE_RF = f"({_NOT_VARIOUS_RF}) AND ({_NO_FILE_RF})"

MASTER_COUNTS_CTE = f"""
master_counts AS (
  SELECT master_id, COUNT(*) AS mc
  FROM releases_features
  WHERE master_id IS NOT NULL AND TRIM(CAST(master_id AS TEXT)) != ''
    AND ({_NOT_VARIOUS_PLAIN})
    AND ({_NO_FILE_PLAIN})
  GROUP BY master_id
)"""

ARTIST_COUNTS_CTE = f"""
artist_counts AS (
  SELECT json_extract(artists_json, '$[0].id') AS aid, COUNT(*) AS ac
  FROM releases_features
  WHERE artists_json IS NOT NULL AND TRIM(artists_json) != ''
    AND json_extract(artists_json, '$[0].id') IS NOT NULL
    AND ({_NOT_VARIOUS_PLAIN})
    AND ({_NO_FILE_PLAIN})
  GROUP BY json_extract(artists_json, '$[0].id')
)"""


def catalog_proxy_join_body() -> str:
    """FROM/JOIN attaching mc and ac to each row alias ``rf``."""
    return """
FROM releases_features rf
LEFT JOIN master_counts m ON rf.master_id = m.master_id
LEFT JOIN artist_counts a
  ON json_extract(rf.artists_json, '$[0].id') = a.aid
"""


def catalog_proxy_score_expr() -> str:
    """Weighted score; use two placeholders for w_master and w_artist."""
    return "(? * COALESCE(m.mc, 0) + ? * COALESCE(a.ac, 0))"


def catalog_proxy_order_by_tail(alias_rf: str = "rf") -> str:
    """Tie-breakers after proxy score and vinyl preference."""
    return (
        f"{alias_rf}.year DESC NULLS LAST, "
        f"COALESCE({alias_rf}.label_tier, 0) DESC, "
        f"{alias_rf}.release_id ASC"
    )


def _proxy_order_by_clause() -> str:
    score = catalog_proxy_score_expr()
    tail = catalog_proxy_order_by_tail()
    return f"{score} DESC, {_VINYL_KEY_RF} DESC, {tail}"


def sql_select_release_ids_ordered_by_catalog_proxy() -> str:
    """Full SELECT: ordered release_ids; params (w_master, w_artist)."""
    ob = _proxy_order_by_clause()
    join = catalog_proxy_join_body()
    return f"""
WITH {MASTER_COUNTS_CTE.strip()},
{ARTIST_COUNTS_CTE.strip()}
SELECT rf.release_id
{join}
WHERE {_QUEUE_WHERE_RF}
ORDER BY {ob}
"""


def sql_select_release_id_and_primary_artist_ordered_by_catalog_proxy() -> str:
    """
    Same order as ``sql_select_release_ids_ordered_by_catalog_proxy`` but also
    returns primary Discogs artist id (text) and vinyl-medium flag (0/1).

    Params: (w_master, w_artist).
    """
    ob = _proxy_order_by_clause()
    join = catalog_proxy_join_body()
    return f"""
WITH {MASTER_COUNTS_CTE.strip()},
{ARTIST_COUNTS_CTE.strip()}
SELECT rf.release_id,
       CAST(COALESCE(json_extract(rf.artists_json, '$[0].id'), '') AS TEXT) AS aid,
       CAST({_VINYL_KEY_RF} AS INTEGER) AS _vinyl
{join}
WHERE {_QUEUE_WHERE_RF}
ORDER BY {ob}
"""


def _stratify_partition_rf(stratify_by: str) -> str:
    if stratify_by == "decade":
        return "rf.decade"
    return "rf.decade, COALESCE(rf.genre, '')"


def _stratify_bucket_g_expr(stratify_by: str) -> str:
    """Second bucket column: constant for decade-only stratification."""
    if stratify_by == "decade":
        return "''"
    return "COALESCE(rf.genre, '')"


def stratified_vinyl_bucket_counts(
    per_bucket: int, target_vinyl_fraction: float
) -> tuple[int, int]:
    """
    Split *per_bucket* into (vinyl_slots, non_vinyl_slots) from a target fraction.

    Slots sum to *per_bucket*; uses rounding so small buckets stay non-empty when
    possible.
    """
    pb = max(0, int(per_bucket))
    if pb == 0:
        return (0, 0)
    f = max(0.0, min(1.0, float(target_vinyl_fraction)))
    kv = int(round(pb * f))
    kv = max(0, min(pb, kv))
    return (kv, pb - kv)


_STRAT_CAP_ORDER = (
    "pscore DESC, _vinyl DESC, _y DESC NULLS LAST, _lt DESC, release_id ASC"
)


def sql_stratified_release_ids_catalog_proxy(
    stratify_by: str,
    *,
    max_per_primary_artist_per_bucket: int = 0,
    vinyl_bucket_split: tuple[int, int] | None = None,
) -> str:
    """
    Window query: up to ``per_bucket`` rows per stratification bucket.

    If ``vinyl_bucket_split`` is ``(kv, kn)``, each bucket takes at most *kv* vinyl
    and *kn* non-vinyl rows (same proxy ordering within each stratum).

    If ``max_per_primary_artist_per_bucket`` <= 0: global proxy order within each
    bucket (mega-artists can fill the bucket).

    If > 0: at most that many releases per primary artist **within each bucket**
    before ranking, so buckets stay mixed.

    Params:
      split None, max <= 0: (w_master, w_artist, per_bucket)
      split (kv, kn), max <= 0: (w_master, w_artist, kv, kn)
      split None, max > 0: (w_master, w_artist, max_per_primary_artist, per_bucket)
      split (kv, kn), max > 0: (w_master, w_artist, max_per_primary_artist, kv, kn)
    """
    score = catalog_proxy_score_expr()
    join = catalog_proxy_join_body()
    part = _stratify_partition_rf(stratify_by)
    ob_inner = _proxy_order_by_clause()
    gexpr = _stratify_bucket_g_expr(stratify_by)
    mpp = max(0, int(max_per_primary_artist_per_bucket))
    split = vinyl_bucket_split

    if mpp <= 0 and split is None:
        return f"""
WITH {MASTER_COUNTS_CTE.strip()},
{ARTIST_COUNTS_CTE.strip()},
ranked AS (
  SELECT rf.release_id,
         row_number() OVER (
           PARTITION BY {part}
           ORDER BY {ob_inner}
         ) AS rn
  {join}
  WHERE {_QUEUE_WHERE_RF}
)
SELECT release_id FROM ranked WHERE rn <= ?
"""

    if mpp <= 0 and split is not None:
        return f"""
WITH {MASTER_COUNTS_CTE.strip()},
{ARTIST_COUNTS_CTE.strip()},
base AS (
  SELECT rf.release_id,
         rf.decade AS _d,
         {gexpr} AS _g,
         {score} AS pscore,
         {_VINYL_KEY_RF} AS _vinyl,
         rf.year AS _y,
         COALESCE(rf.label_tier, 0) AS _lt
  {join}
  WHERE {_QUEUE_WHERE_RF}
),
vin_r AS (
  SELECT release_id,
         row_number() OVER (
           PARTITION BY _d, _g
           ORDER BY {_STRAT_CAP_ORDER}
         ) AS rn
  FROM base WHERE _vinyl = 1
),
non_r AS (
  SELECT release_id,
         row_number() OVER (
           PARTITION BY _d, _g
           ORDER BY {_STRAT_CAP_ORDER}
         ) AS rn
  FROM base WHERE _vinyl = 0
)
SELECT release_id FROM vin_r WHERE rn <= ?
UNION ALL
SELECT release_id FROM non_r WHERE rn <= ?
"""

    if mpp > 0 and split is None:
        return f"""
WITH {MASTER_COUNTS_CTE.strip()},
{ARTIST_COUNTS_CTE.strip()},
base AS (
  SELECT rf.release_id,
         rf.decade AS _d,
         {gexpr} AS _g,
         CAST(COALESCE(json_extract(rf.artists_json, '$[0].id'), '') AS TEXT)
           AS _aid,
         {score} AS pscore,
         {_VINYL_KEY_RF} AS _vinyl,
         rf.year AS _y,
         COALESCE(rf.label_tier, 0) AS _lt
  {join}
  WHERE {_QUEUE_WHERE_RF}
),
slotted AS (
  SELECT release_id, _d, _g, pscore, _vinyl, _y, _lt,
         row_number() OVER (
           PARTITION BY _d, _g, _aid
           ORDER BY pscore DESC, _vinyl DESC, _y DESC NULLS LAST, _lt DESC,
             release_id ASC
         ) AS arn
  FROM base
),
capped AS (
  SELECT release_id, _d, _g, pscore, _vinyl, _y, _lt
  FROM slotted WHERE arn <= ?
),
ranked AS (
  SELECT release_id,
         row_number() OVER (
           PARTITION BY _d, _g
           ORDER BY pscore DESC, _vinyl DESC, _y DESC NULLS LAST, _lt DESC,
             release_id ASC
         ) AS rn
  FROM capped
)
SELECT release_id FROM ranked WHERE rn <= ?
"""

    return f"""
WITH {MASTER_COUNTS_CTE.strip()},
{ARTIST_COUNTS_CTE.strip()},
base AS (
  SELECT rf.release_id,
         rf.decade AS _d,
         {gexpr} AS _g,
         CAST(COALESCE(json_extract(rf.artists_json, '$[0].id'), '') AS TEXT)
           AS _aid,
         {score} AS pscore,
         {_VINYL_KEY_RF} AS _vinyl,
         rf.year AS _y,
         COALESCE(rf.label_tier, 0) AS _lt
  {join}
  WHERE {_QUEUE_WHERE_RF}
),
slotted AS (
  SELECT release_id, _d, _g, pscore, _vinyl, _y, _lt,
         row_number() OVER (
           PARTITION BY _d, _g, _aid
           ORDER BY pscore DESC, _vinyl DESC, _y DESC NULLS LAST, _lt DESC,
             release_id ASC
         ) AS arn
  FROM base
),
capped AS (
  SELECT release_id, _d, _g, pscore, _vinyl, _y, _lt
  FROM slotted WHERE arn <= ?
),
vin_r AS (
  SELECT release_id,
         row_number() OVER (
           PARTITION BY _d, _g
           ORDER BY {_STRAT_CAP_ORDER}
         ) AS rn
  FROM capped WHERE _vinyl = 1
),
non_r AS (
  SELECT release_id,
         row_number() OVER (
           PARTITION BY _d, _g
           ORDER BY {_STRAT_CAP_ORDER}
         ) AS rn
  FROM capped WHERE _vinyl = 0
)
SELECT release_id FROM vin_r WHERE rn <= ?
UNION ALL
SELECT release_id FROM non_r WHERE rn <= ?
"""
