"""CLI entry for label audit LLM runner."""

from __future__ import annotations

import sys

from . import lib as _lib


def _sync_globals_from_lib() -> None:
    """Copy names from lib into this module so ``main()`` sees monkeypatched lib.

    Call at the start of ``main()`` so tests (or callers) that patch
    ``label_audit_run_llm.lib`` before ``main()`` runs get updated bindings.
    """
    g: dict[str, object] = sys.modules[__name__].__dict__
    for k in dir(_lib):
        if k.startswith("__"):
            continue
        g[k] = getattr(_lib, k)


def main() -> int:
    _sync_globals_from_lib()
    p = argparse.ArgumentParser(description="Run Gemini LLM over label-audit queue.")
    p.add_argument("--config", default="grader/configs/grader.yaml")
    p.add_argument(
        "--guidelines",
        default="grader/configs/grading_guidelines.yaml",
    )
    p.add_argument("--db", default="grader/reports/label_audit_queue.sqlite")
    p.add_argument(
        "--raw-dir",
        default="grader/reports/label_audit_raw",
        help="Directory for cached raw model responses.",
    )
    p.add_argument(
        "--gating-pass",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="1=cleanlab_issue, 2=low confidence, 3=disagree assigned/model",
    )
    p.add_argument(
        "--splits",
        nargs="*",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
    )
    p.add_argument(
        "--targets",
        nargs="*",
        choices=["sleeve", "media"],
        default=["sleeve", "media"],
    )
    p.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    p.add_argument(
        "--health-every",
        type=int,
        default=50,
        help=(
            "Print health heartbeat every N processed rows " "(0 disables heartbeats)."
        ),
    )
    p.add_argument(
        "--sleep-seconds",
        type=float,
        default=2.0,
        help="Baseline delay between listings/terminal failures.",
    )
    p.add_argument(
        "--rotation-cooldown-seconds",
        type=float,
        default=1.5,
        help="Delay before switching to next model on quota/unavailable errors.",
    )
    p.add_argument(
        "--min-self-confidence",
        type=float,
        default=None,
        help=(
            "Optional extra prefilter: only rows with "
            "cleanlab_self_confidence >= this value."
        ),
    )
    p.add_argument(
        "--max-self-confidence",
        type=float,
        default=None,
        help=(
            "Optional extra prefilter: only rows with "
            "cleanlab_self_confidence <= this value."
        ),
    )
    p.add_argument(
        "--require-disagree",
        action="store_true",
        help=("Optional extra prefilter: require disagree_assigned_vs_model=1."),
    )
    p.add_argument(
        "--sources",
        default="",
        help="Optional comma-separated source allowlist prefilter.",
    )
    p.add_argument(
        "--assigned-labels",
        default="",
        help="Optional comma-separated assigned_label allowlist prefilter.",
    )
    p.add_argument(
        "--exclude-patched",
        action="store_true",
        help="Exclude listings already present in label_patches JSONL.",
    )
    p.add_argument(
        "--label-patches-path",
        default="grader/data/label_patches.jsonl",
        help="Path to label_patches JSONL used when --exclude-patched is enabled.",
    )
    p.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Total shard workers for deterministic parallelization.",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index for this process.",
    )
    p.add_argument(
        "--provider",
        choices=["gemini", "openrouter", "ollama"],
        default="",
        help="LLM provider. Blank uses LLM_PROVIDER env (default openrouter).",
    )
    p.add_argument(
        "--model-id",
        default="",
        help="Override provider model env (GEMINI_MODEL or XAI_MODEL).",
    )
    p.add_argument(
        "--model-rotation",
        default="",
        help=(
            "Comma-separated fallback models used on quota/rate-limit "
            "errors. Example: model-a,model-b,model-c"
        ),
    )
    p.add_argument(
        "--tier-a-size",
        type=int,
        default=3,
        help="Number of active priority models to keep in Tier A.",
    )
    p.add_argument(
        "--model-tiers-state-path",
        default=str(MODEL_TIERS_STATE_PATH),
        help="JSON file tracking exhausted models for the current UTC day.",
    )
    p.add_argument(
        "--env-file",
        default=".env",
        help="dotenv file path for GEMINI_API_KEY/GEMINI_MODEL.",
    )
    p.add_argument(
        "--max-attempts-per-listing",
        type=int,
        default=6,
        help="Hard cap on API attempts for a listing before marking error.",
    )
    p.add_argument(
        "--max-rate-limit-wait-seconds",
        type=float,
        default=300.0,
        help="Maximum reset-based wait; longer waits stop the run.",
    )
    p.add_argument(
        "--rate-limit-jitter-seconds",
        type=float,
        default=0.5,
        help="Extra buffer added to reset-based wait.",
    )
    p.add_argument(
        "--enable-critic-pass",
        action="store_true",
        help="Run a second critic pass and write auto decision fields.",
    )
    p.add_argument(
        "--critic-k-examples",
        type=int,
        default=5,
        help="Manual examples to retrieve per critic decision.",
    )
    p.add_argument(
        "--critic-min-confidence",
        type=float,
        default=0.90,
        help="Minimum critic confidence to emit proposed_grade.",
    )
    p.add_argument(
        "--critic-model-id",
        default="",
        help="Optional model override for critic pass.",
    )
    p.add_argument(
        "--critic-policy-version",
        default="",
        help="Version tag written to auto_policy_version for critic output.",
    )
    p.add_argument(
        "--critic-include-auto-apply-gold",
        action="store_true",
        help="Include auto_apply rows when building critic example bank.",
    )
    args = p.parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1.")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < shard-count.")

    load_dotenv(args.env_file)

    provider = (
        args.provider.strip().lower()
        or (os.getenv("LLM_PROVIDER") or "openrouter").strip().lower()
    )
    if provider not in {"gemini", "openrouter", "ollama"}:
        raise ValueError("provider must be one of: gemini, openrouter, ollama")

    if provider == "gemini":
        api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY.")
        default_model = (os.getenv("GEMINI_MODEL") or "").strip()
        key_source = "GEMINI_API_KEY"
    elif provider == "openrouter":
        openrouter_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
        api_key = openrouter_key or groq_key
        if not api_key:
            raise ValueError("Missing OPENROUTER_API_KEY (or GROQ_API_KEY fallback).")
        key_source = "OPENROUTER_API_KEY" if openrouter_key else "GROQ_API_KEY"
        default_model = (os.getenv("OPENROUTER_MODEL") or "openai/gpt-4.1-mini").strip()
    else:
        # Ollama runs locally and does not require an API key.
        api_key = "local"
        key_source = "OLLAMA_LOCAL"
        default_model = (os.getenv("OLLAMA_MODEL") or "llama3.1:8b").strip()

    model_id = args.model_id.strip() or default_model
    if not model_id:
        raise ValueError("Missing model id (or pass --model-id).")
    all_models: list[str] = [model_id]
    rotation_from_env = (os.getenv("LLM_MODEL_ROTATION") or "").strip()
    rotation_raw = args.model_rotation.strip() or rotation_from_env
    if not rotation_raw and provider == "openrouter":
        rotation_raw = ",".join(OPENROUTER_FREE_MODELS)
    for m in _parse_model_rotation(rotation_raw):
        if m not in all_models:
            all_models.append(m)
    if args.tier_a_size < 1:
        raise ValueError("--tier-a-size must be >= 1.")

    tiers_state_path = Path(args.model_tiers_state_path)
    tiers_state = _load_model_tiers_state(tiers_state_path)
    provider_state = tiers_state.get(provider, {})
    if not isinstance(provider_state, dict):
        provider_state = {}
    today = _today_utc_ymd()
    exhausted_today_raw = provider_state.get(today, [])
    exhausted_today = (
        {str(x) for x in exhausted_today_raw}
        if isinstance(exhausted_today_raw, list)
        else set()
    )
    model_candidates, tier_a_models, tier_b_models = _tiered_candidates(
        all_models=all_models,
        exhausted=exhausted_today,
        tier_a_size=int(args.tier_a_size),
    )
    if not model_candidates:
        # Reset exhausted list for the day if all models were marked unavailable.
        exhausted_today = set()
        model_candidates, tier_a_models, tier_b_models = _tiered_candidates(
            all_models=all_models,
            exhausted=exhausted_today,
            tier_a_size=int(args.tier_a_size),
        )
    current_model_idx = 0
    rotations_used = 0
    startup_meta = {
        "event": "startup",
        "provider": provider,
        "active_model": model_candidates[current_model_idx],
        "model_candidates": model_candidates,
        "tier_a_models": tier_a_models,
        "tier_b_models": tier_b_models,
        "exhausted_today_models": sorted(exhausted_today),
        "api_key_source": key_source,
        "shard_count": int(args.shard_count),
        "shard_index": int(args.shard_index),
    }
    print(json.dumps(startup_meta, ensure_ascii=False), flush=True)
    STARTUP_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    STARTUP_META_PATH.write_text(
        json.dumps(startup_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if provider == "gemini":
        try:
            import google.genai as _genai  # pyright: ignore[reportMissingImports]

            _ = _genai
        except ImportError:
            raise ImportError("Install `google-genai` first.")

    _ = load_yaml_mapping(Path(args.config))
    guidelines_path = Path(args.guidelines)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    where_gate, gate_params = _gating_clause(args.gating_pass)
    selected_targets = set(args.targets)
    selected_sources = _parse_csv_values(args.sources)
    selected_assigned_labels = _parse_csv_values(args.assigned_labels)
    extra_where: list[str] = []
    extra_params: list[Any] = []
    if args.min_self_confidence is not None:
        extra_where.append(
            "cleanlab_self_confidence IS NOT NULL AND cleanlab_self_confidence >= ?"
        )
        extra_params.append(float(args.min_self_confidence))
    if args.max_self_confidence is not None:
        extra_where.append(
            "cleanlab_self_confidence IS NOT NULL AND cleanlab_self_confidence <= ?"
        )
        extra_params.append(float(args.max_self_confidence))
    if args.require_disagree:
        extra_where.append("disagree_assigned_vs_model = 1")
    if selected_sources:
        extra_where.append(f"source IN ({','.join(['?'] * len(selected_sources))})")
        extra_params.extend(selected_sources)
    if selected_assigned_labels:
        extra_where.append(
            f"assigned_label IN ({','.join(['?'] * len(selected_assigned_labels))})"
        )
        extra_params.extend(selected_assigned_labels)
    extra_where_sql = (" AND " + " AND ".join(extra_where)) if extra_where else ""
    limit_sql = f"LIMIT {int(args.limit)}" if args.limit > 0 else ""
    processed = 0
    cache_hits = 0
    errors = 0
    excluded_patched = 0
    stopped_due_to_quota = False
    quota_error_message = ""
    patched_keys: set[tuple[str, str]] = set()
    if args.exclude_patched:
        patched_keys = _load_patched_keys(Path(args.label_patches_path))

    with sqlite3.connect(Path(args.db)) as conn:
        conn.row_factory = sqlite3.Row
        critic_examples = (
            load_human_gold_examples(
                conn, include_auto_apply=bool(args.critic_include_auto_apply_gold)
            )
            if args.enable_critic_pass
            else []
        )
        critic_policy_version = (
            args.critic_policy_version.strip() or f"critic_pass_{utc_now_iso()}"
        )
        seed_rows = conn.execute(
            f"""
            SELECT split, source, item_id, target
            FROM queue
            WHERE split IN ({",".join(["?"] * len(args.splits))})
              AND target IN ({",".join(["?"] * len(args.targets))})
              AND COALESCE(llm_status, '') NOT IN ('ok')
              AND ({where_gate})
              {extra_where_sql}
            ORDER BY cleanlab_self_confidence ASC
            {limit_sql}
            """,
            tuple(args.splits)
            + tuple(args.targets)
            + gate_params
            + tuple(extra_params),
        ).fetchall()

        listings: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for r in seed_rows:
            if patched_keys and (str(r["source"]), str(r["item_id"])) in patched_keys:
                excluded_patched += 1
                continue
            key = (str(r["split"]), str(r["source"]), str(r["item_id"]))
            if key in seen:
                continue
            if not _listing_belongs_to_shard(
                split=key[0],
                source=key[1],
                item_id=key[2],
                shard_count=int(args.shard_count),
                shard_index=int(args.shard_index),
            ):
                continue
            seen.add(key)
            listings.append(key)

        media_guides = load_guideline_prompt_bits(guidelines_path, "media")
        sleeve_guides = load_guideline_prompt_bits(guidelines_path, "sleeve")

        for split, source, item_id in listings:
            rows = conn.execute(
                """
                SELECT *
                FROM queue
                WHERE split=? AND source=? AND item_id=?
                """,
                (split, source, item_id),
            ).fetchall()
            row_map = {str(r["target"]): r for r in rows}
            media_row = row_map.get("media")
            sleeve_row = row_map.get("sleeve")
            listing_payload = {
                "split": split,
                "source": source,
                "item_id": item_id,
                "text": str((rows[0]["text"] if rows else "") or ""),
                "media_assigned_label": str(_row_value(media_row, "assigned_label")),
                "media_model_pred_label": str(
                    _row_value(media_row, "model_pred_label")
                ),
                "media_cleanlab_self_confidence": _row_value(
                    media_row, "cleanlab_self_confidence", None
                ),
                "media_cleanlab_label_issue": int(
                    _row_value(media_row, "cleanlab_label_issue", 0)
                ),
                "sleeve_assigned_label": str(_row_value(sleeve_row, "assigned_label")),
                "sleeve_model_pred_label": str(
                    _row_value(sleeve_row, "model_pred_label")
                ),
                "sleeve_cleanlab_self_confidence": _row_value(
                    sleeve_row, "cleanlab_self_confidence", None
                ),
                "sleeve_cleanlab_label_issue": int(
                    _row_value(sleeve_row, "cleanlab_label_issue", 0)
                ),
            }
            msgs = _build_joint_messages(
                listing=listing_payload,
                media_guides=media_guides,
                sleeve_guides=sleeve_guides,
            )
            raw_text = ""
            used_model_id = ""
            prompt_ver = ""
            ck = ""
            attempts_for_listing = 0
            listing_failed = False
            while True:
                active_model_id = model_candidates[current_model_idx]
                attempts_for_listing += 1
                if (
                    args.max_attempts_per_listing > 0
                    and attempts_for_listing > args.max_attempts_per_listing
                ):
                    msg = "max attempts exceeded " f"({args.max_attempts_per_listing})"
                    errors += 1
                    for target_name, row_obj in row_map.items():
                        if target_name not in selected_targets:
                            continue
                        conn.execute(
                            """
                            UPDATE queue
                            SET llm_status='error', llm_error=?, updated_at=?
                            WHERE queue_row_id=?
                            """,
                            (msg, utc_now_iso(), int(row_obj["queue_row_id"])),
                        )
                    conn.commit()
                    listing_failed = True
                    print(
                        json.dumps(
                            {
                                "event": "listing_fail_max_attempts",
                                "split": split,
                                "source": source,
                                "item_id": item_id,
                                "attempts": attempts_for_listing - 1,
                                "provider": provider,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    break
                print(
                    json.dumps(
                        {
                            "event": "attempt_start",
                            "split": split,
                            "source": source,
                            "item_id": item_id,
                            "attempt": attempts_for_listing,
                            "model": active_model_id,
                            "provider": provider,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if attempts_for_listing == 1:
                    prompt_stats = _prompt_size_stats(msgs)
                    print(
                        json.dumps(
                            {
                                "event": "prompt_size",
                                "split": split,
                                "source": source,
                                "item_id": item_id,
                                "provider": provider,
                                "model": active_model_id,
                                **prompt_stats,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                prompt_ver = _joint_prompt_version(
                    model_id=active_model_id,
                    media_allowed=media_guides["allowed"],
                    sleeve_allowed=sleeve_guides["allowed"],
                    media_desc=media_guides["descriptions"],
                    sleeve_desc=sleeve_guides["descriptions"],
                    media_rubric=str(media_guides.get("rubric_text", "") or ""),
                    sleeve_rubric=str(sleeve_guides.get("rubric_text", "") or ""),
                )
                ck = _cache_key(provider, active_model_id, prompt_ver, msgs)
                cache_row = conn.execute(
                    "SELECT response_json FROM llm_cache WHERE cache_key = ?",
                    (ck,),
                ).fetchone()
                if cache_row:
                    cache_hits += 1
                    cache_obj = json.loads(str(cache_row[0]))
                    raw_text = str(cache_obj.get("raw_text", "") or "")
                    used_model_id = str(cache_obj.get("used_model_id", "") or "")
                    if not used_model_id:
                        used_model_id = active_model_id
                    break
                try:
                    raw_text, used_model_id = _query_provider(
                        provider=provider,
                        messages=msgs,
                        model_id=active_model_id,
                    )
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO llm_cache(
                            cache_key, model_id, prompt_version, response_json, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            ck,
                            f"{provider}:{(used_model_id or active_model_id)}",
                            prompt_ver,
                            json.dumps(
                                {
                                    "raw_text": raw_text,
                                    "used_model_id": used_model_id or active_model_id,
                                },
                                ensure_ascii=False,
                            ),
                            utc_now_iso(),
                        ),
                    )
                    print(
                        json.dumps(
                            {
                                "event": "attempt_success",
                                "split": split,
                                "source": source,
                                "item_id": item_id,
                                "attempt": attempts_for_listing,
                                "model": active_model_id,
                                "used_model": used_model_id or active_model_id,
                                "provider": provider,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    break
                except Exception as e:  # pragma: no cover - runtime/API path
                    for target_name, row_obj in row_map.items():
                        if target_name not in selected_targets:
                            continue
                        conn.execute(
                            """
                            UPDATE queue
                            SET llm_status='retrying',
                                llm_error=?,
                                llm_model_id=?,
                                updated_at=?
                            WHERE queue_row_id=?
                            """,
                            (
                                (
                                    f"attempt={attempts_for_listing} "
                                    f"model={active_model_id} error={str(e)}"
                                )[:1200],
                                f"{provider}:{active_model_id}",
                                utc_now_iso(),
                                int(row_obj["queue_row_id"]),
                            ),
                        )
                    conn.commit()
                    print(
                        json.dumps(
                            {
                                "event": "attempt_fail",
                                "split": split,
                                "source": source,
                                "item_id": item_id,
                                "attempt": attempts_for_listing,
                                "model": active_model_id,
                                "provider": provider,
                                "error": str(e)[:300],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    quota_hit = _is_quota_exhausted_error(e)
                    model_unavailable_hit = _is_model_unavailable_error(e)
                    daily_cap_hit = _is_daily_model_cap_error(e)
                    mark_model_exhausted = model_unavailable_hit or daily_cap_hit
                    if mark_model_exhausted:
                        exhausted_today.add(active_model_id)
                        provider_state[today] = sorted(exhausted_today)
                        tiers_state[provider] = provider_state
                        _save_model_tiers_state(tiers_state_path, tiers_state)
                        (
                            model_candidates,
                            tier_a_models,
                            tier_b_models,
                        ) = _tiered_candidates(
                            all_models=all_models,
                            exhausted=exhausted_today,
                            tier_a_size=int(args.tier_a_size),
                        )
                        if not model_candidates:
                            stopped_due_to_quota = True
                            quota_error_message = (
                                "All models exhausted/unavailable for today. "
                                "Wait for reset or refresh rotation list."
                            )
                            break
                        current_model_idx = 0
                        print(
                            json.dumps(
                                {
                                    "event": "model_exhausted_today",
                                    "provider": provider,
                                    "model": active_model_id,
                                    "reason": (
                                        "daily_cap"
                                        if daily_cap_hit
                                        else "model_unavailable"
                                    ),
                                    "tier_a_models": tier_a_models,
                                    "tier_b_models": tier_b_models,
                                    "exhausted_today_models": sorted(
                                        exhausted_today
                                    ),
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                    if (
                        quota_hit or model_unavailable_hit
                    ) and current_model_idx + 1 < len(model_candidates):
                        if args.rotation_cooldown_seconds > 0:
                            time.sleep(args.rotation_cooldown_seconds)
                        current_model_idx += 1
                        rotations_used += 1
                        continue
                    if quota_hit:
                        wait_s = _rate_limit_wait_seconds(e)
                        if wait_s is not None and wait_s <= float(
                            args.max_rate_limit_wait_seconds
                        ):
                            backoff_s = max(
                                float(args.sleep_seconds),
                                wait_s + float(args.rate_limit_jitter_seconds),
                            )
                            print(
                                json.dumps(
                                    {
                                        "event": "rate_limit_backoff",
                                        "split": split,
                                        "source": source,
                                        "item_id": item_id,
                                        "attempt": attempts_for_listing,
                                        "model": active_model_id,
                                        "provider": provider,
                                        "wait_seconds": round(backoff_s, 3),
                                    },
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
                            time.sleep(backoff_s)
                            continue
                    # Local Ollama: long generations or crashed runners — retry instead of marking error immediately.
                    if provider == "ollama" and _is_ollama_transient_error(e):
                        backoff_s = max(
                            float(args.sleep_seconds),
                            float(
                                (os.getenv("OLLAMA_TRANSIENT_RETRY_SLEEP_SECONDS") or "8").strip()
                            ),
                        )
                        print(
                            json.dumps(
                                {
                                    "event": "ollama_transient_backoff",
                                    "split": split,
                                    "source": source,
                                    "item_id": item_id,
                                    "attempt": attempts_for_listing,
                                    "model": active_model_id,
                                    "error": str(e)[:300],
                                    "retry_in_seconds": round(backoff_s, 3),
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                        time.sleep(backoff_s)
                        continue
                    errors += 1
                    for target_name, row_obj in row_map.items():
                        if target_name not in selected_targets:
                            continue
                        conn.execute(
                            """
                            UPDATE queue
                            SET llm_status='error', llm_error=?, updated_at=?
                            WHERE queue_row_id=?
                            """,
                            (str(e), utc_now_iso(), int(row_obj["queue_row_id"])),
                        )
                    conn.commit()
                    if quota_hit:
                        stopped_due_to_quota = True
                        quota_error_message = str(e)
                        break
                    time.sleep(args.sleep_seconds)
                    listing_failed = True
                    break

            if stopped_due_to_quota:
                break
            if listing_failed:
                processed += sum(
                    1 for target_name in row_map if target_name in selected_targets
                )
                if args.health_every > 0 and processed % args.health_every == 0:
                    counts = _health_counts(conn)
                    print(
                        json.dumps(
                            {
                                "event": "health",
                                "processed": processed,
                                "cache_hits": cache_hits,
                                "errors": errors,
                                "provider": provider,
                                "active_model": (
                                    model_candidates[current_model_idx]
                                    if model_candidates
                                    else ""
                                ),
                                "rotations_used": rotations_used,
                                "ok": counts["ok"],
                                "error": counts["error"],
                                "parse_error": counts["parse_error"],
                                "pending": counts["pending"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                time.sleep(args.sleep_seconds)
                continue

            safe_id = hashlib.sha256(
                f"{split}|{source}|{item_id}".encode("utf-8")
            ).hexdigest()[:12]
            raw_path = raw_dir / f"{split}_{safe_id}.json"
            raw_path.write_text(
                json.dumps(
                    {
                        "messages": msgs,
                        "raw_text": raw_text,
                        "requested_model_id": active_model_id,
                        "used_model_id": used_model_id or active_model_id,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            try:
                decisions_obj = _parse_joint_decisions(raw_text)
                run_id = f"pass{args.gating_pass}_{utc_now_iso()}"
                for target_name, row_obj in row_map.items():
                    if target_name not in selected_targets:
                        continue
                    if target_name not in decisions_obj:
                        continue
                    allowed = (
                        media_guides["allowed"]
                        if target_name == "media"
                        else sleeve_guides["allowed"]
                    )
                    decision = validate_decision(decisions_obj[target_name], allowed)
                    auto_decision = ""
                    auto_final_label = ""
                    auto_decision_score: float | None = None
                    auto_decision_reason = ""
                    auto_decision_source = ""
                    auto_policy_version = ""
                    if args.enable_critic_pass:
                        critic_model_id = args.critic_model_id.strip() or active_model_id
                        critic_examples_for_row = retrieve_examples(
                            bank=critic_examples,
                            target=target_name,
                            text=str(listing_payload.get("text", "")),
                            split_exclude=str(listing_payload.get("split", "") or ""),
                            row_id_exclude=int(row_obj["queue_row_id"]),
                            k=max(0, int(args.critic_k_examples)),
                        )
                        critic_messages = build_critic_messages(
                            target=target_name,
                            text=str(listing_payload.get("text", "")),
                            assigned_label=str(row_obj["assigned_label"]),
                            model_pred_label=str(row_obj["model_pred_label"]),
                            llm_verdict=str(decision.verdict),
                            llm_confidence=float(decision.confidence),
                            allowed_labels=allowed,
                            examples=critic_examples_for_row,
                        )
                        try:
                            critic_raw_text, critic_used_model = _query_provider(
                                provider=provider,
                                messages=critic_messages,
                                model_id=critic_model_id,
                            )
                            critic_decision = parse_critic_decision(
                                critic_raw_text, allowed
                            )
                            (
                                auto_decision,
                                auto_final_label,
                                auto_decision_score,
                                _raw_reason,
                            ) = to_auto_decision(
                                critic_decision,
                                min_confidence=float(args.critic_min_confidence),
                            )
                            auto_decision_reason = (
                                f"{_raw_reason} "
                                f"(k={len(critic_examples_for_row)}; model={critic_used_model})"
                            ).strip()[:500]
                            auto_decision_source = "llm_critic"
                            auto_policy_version = critic_policy_version
                        except Exception as critic_exc:
                            auto_decision = "needs_review"
                            auto_decision_reason = (
                                f"critic_error: {str(critic_exc)}"
                            )[:500]
                            auto_decision_source = "llm_critic"
                            auto_policy_version = critic_policy_version
                    disag = compute_disagreement_flags(
                        assigned_label=str(row_obj["assigned_label"]),
                        model_pred_label=str(row_obj["model_pred_label"]),
                        llm_verdict=decision.verdict,
                    )
                    conn.execute(
                        """
                        UPDATE queue
                        SET llm_status='ok',
                            llm_verdict=?,
                            llm_confidence=?,
                            reason_code=?,
                            llm_abstain=?,
                            llm_model_id=?,
                            prompt_version=?,
                            audit_run_id=?,
                            raw_response_path=?,
                            response_cache_key=?,
                            disagree_llm_vs_assigned=?,
                            disagree_llm_vs_model=?,
                            auto_decision=?,
                            auto_final_label=?,
                            auto_decision_score=?,
                            auto_decision_reason=?,
                            auto_decision_source=?,
                            auto_policy_version=?,
                            updated_at=?
                        WHERE queue_row_id=?
                        """,
                        (
                            decision.verdict,
                            decision.confidence,
                            decision.reason_code,
                            int(decision.abstain),
                            f"{provider}:{(used_model_id or active_model_id)}",
                            prompt_ver,
                            run_id,
                            str(raw_path),
                            ck,
                            disag["disagree_llm_vs_assigned"],
                            disag["disagree_llm_vs_model"],
                            auto_decision,
                            auto_final_label,
                            auto_decision_score,
                            auto_decision_reason,
                            auto_decision_source,
                            auto_policy_version,
                            utc_now_iso(),
                            int(row_obj["queue_row_id"]),
                        ),
                    )
            except Exception as e:
                errors += 1
                for target_name, row_obj in row_map.items():
                    if target_name not in selected_targets:
                        continue
                    conn.execute(
                        """
                        UPDATE queue
                        SET llm_status='parse_error',
                            llm_error=?,
                            raw_response_path=?,
                            updated_at=?
                        WHERE queue_row_id=?
                        """,
                        (
                            str(e),
                            str(raw_path),
                            utc_now_iso(),
                            int(row_obj["queue_row_id"]),
                        ),
                    )
            conn.commit()
            processed += sum(
                1 for target_name in row_map if target_name in selected_targets
            )
            if args.health_every > 0 and processed % args.health_every == 0:
                counts = _health_counts(conn)
                print(
                    json.dumps(
                        {
                            "event": "health",
                            "processed": processed,
                            "cache_hits": cache_hits,
                            "errors": errors,
                            "provider": provider,
                            "active_model": (
                                model_candidates[current_model_idx]
                                if model_candidates
                                else ""
                            ),
                            "rotations_used": rotations_used,
                            "ok": counts["ok"],
                            "error": counts["error"],
                            "parse_error": counts["parse_error"],
                            "pending": counts["pending"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if stopped_due_to_quota:
                break
            time.sleep(args.sleep_seconds)

    summary = {
        "processed": processed,
        "cache_hits": cache_hits,
        "errors": errors,
        "gating_pass": args.gating_pass,
        "provider": provider,
        "model_current": model_candidates[current_model_idx] if model_candidates else "",
        "model_candidates": model_candidates,
        "tier_a_models": tier_a_models,
        "tier_b_models": tier_b_models,
        "exhausted_today_models": sorted(exhausted_today),
        "rotations_used": rotations_used,
        "stopped_due_to_quota": stopped_due_to_quota,
        "exclude_patched": bool(args.exclude_patched),
        "excluded_patched": excluded_patched,
        "prefilters": {
            "min_self_confidence": args.min_self_confidence,
            "max_self_confidence": args.max_self_confidence,
            "require_disagree": bool(args.require_disagree),
            "sources": selected_sources,
            "assigned_labels": selected_assigned_labels,
        },
        "shard_count": int(args.shard_count),
        "shard_index": int(args.shard_index),
    }
    if quota_error_message:
        summary["quota_error"] = quota_error_message
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 2 if stopped_due_to_quota else 0
