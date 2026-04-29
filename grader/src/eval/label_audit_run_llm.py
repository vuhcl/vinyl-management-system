from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from grader.src.eval.label_audit_backend import (
    REASON_CODES,
    compute_disagreement_flags,
    load_guideline_prompt_bits,
    parse_llm_json,
    utc_now_iso,
    validate_decision,
)
from grader.src.eval.label_audit_critic import (
    build_critic_messages,
    load_human_gold_examples,
    parse_critic_decision,
    retrieve_examples,
    to_auto_decision,
)

OPENROUTER_FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "poolside/laguna-m.1:free",
    "inclusionai/ling-2.6-1t:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "liquid/lfm-2.5-1.2b-thinking:free",
    "baidu/qianfan-ocr-fast:free",
    "inclusionai/ling-2.6-flash:free",
    "poolside/laguna-xs.2:free",
]
STARTUP_META_PATH = Path("grader/reports/label_audit_last_startup.json")
MODEL_TIERS_STATE_PATH = Path("grader/reports/label_audit_model_tiers_state.json")


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cache_key(
    provider: str,
    model_id: str,
    prompt_version: str,
    messages: list[dict[str, str]],
) -> str:
    payload = json.dumps(
        {
            "provider": provider,
            "model": model_id,
            "prompt_version": prompt_version,
            "messages": messages,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prompt_size_stats(messages: list[dict[str, str]]) -> dict[str, int]:
    """Fast token estimate for context-length tuning in logs."""
    system_chars = 0
    user_chars = 0
    for m in messages:
        role = str(m.get("role", "")).strip().lower()
        content = str(m.get("content", ""))
        if role == "system":
            system_chars += len(content)
        elif role == "user":
            user_chars += len(content)
    total_chars = system_chars + user_chars
    # Heuristic for mixed English/JSON prompts.
    approx_tokens = max(1, int(round(total_chars / 4.0)))
    return {
        "system_chars": system_chars,
        "user_chars": user_chars,
        "total_chars": total_chars,
        "approx_tokens": approx_tokens,
    }


def _joint_prompt_version(
    *,
    model_id: str,
    media_allowed: list[str],
    sleeve_allowed: list[str],
    media_desc: list[tuple[str, str]],
    sleeve_desc: list[tuple[str, str]],
    media_rubric: str,
    sleeve_rubric: str,
) -> str:
    payload = json.dumps(
        {
            "model_id": model_id,
            "media_allowed": media_allowed,
            "sleeve_allowed": sleeve_allowed,
            "media_desc": media_desc,
            "sleeve_desc": sleeve_desc,
            "media_rubric": media_rubric,
            "sleeve_rubric": sleeve_rubric,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _build_joint_messages(
    *,
    listing: dict[str, Any],
    media_guides: dict[str, Any],
    sleeve_guides: dict[str, Any],
) -> list[dict[str, str]]:
    media_defs = "\n".join(f"- {g}: {d}" for g, d in media_guides["descriptions"])
    sleeve_defs = "\n".join(f"- {g}: {d}" for g, d in sleeve_guides["descriptions"])
    media_rubric = str(media_guides.get("rubric_text", "") or "").strip()
    sleeve_rubric = str(sleeve_guides.get("rubric_text", "") or "").strip()
    system = (
        "You audit vinyl listing labels for BOTH targets: media and sleeve.\n"
        "Return exactly one JSON object and nothing else.\n"
        "Do not include markdown fences, explanations, or any prose.\n"
        "Top-level keys must be exactly: media, sleeve.\n"
        "Each target object keys: llm_verdict, llm_confidence, reason_code, llm_abstain.\n"
        f"reason_code must be one of {list(REASON_CODES)}.\n"
        "Use exact allowed label strings for each target.\n"
        "If insufficient evidence for a target set llm_abstain=true for that target.\n"
        "Manual-audit few-shot examples (real relabel patterns):\n"
        "Example 1 input:\n"
        "- text: 'sleeve has slight yellowing at top & small crease in upper left "
        "corner; disc is glossy, appears unplayed'\n"
        "- assigned: sleeve='Very Good', media='Very Good Plus'\n"
        "- model_pred: sleeve='Very Good Plus', media='Near Mint'\n"
        "Example 1 output:\n"
        "{\n"
        '  "media": {\n'
        '    "llm_verdict": "Near Mint",\n'
        '    "llm_confidence": 0.79,\n'
        '    "reason_code": "supports_higher",\n'
        '    "llm_abstain": false\n'
        "  },\n"
        '  "sleeve": {\n'
        '    "llm_verdict": "Very Good",\n'
        '    "llm_confidence": 0.72,\n'
        '    "reason_code": "mixed_signals",\n'
        '    "llm_abstain": false\n'
        "  }\n"
        "}\n"
        "Example 2 input:\n"
        "- text: '1-2 really light marks on 2 sides Rest mint Box is fine Just a bit "
        "of wear'\n"
        "- assigned: sleeve='Very Good', media='Near Mint'\n"
        "- model_pred: sleeve='Very Good Plus', media='Very Good'\n"
        "Example 2 output:\n"
        "{\n"
        '  "media": {\n'
        '    "llm_verdict": "Near Mint",\n'
        '    "llm_confidence": 0.64,\n'
        '    "reason_code": "supports_higher",\n'
        '    "llm_abstain": false\n'
        "  },\n"
        '  "sleeve": {\n'
        '    "llm_verdict": "Very Good Plus",\n'
        '    "llm_confidence": 0.68,\n'
        '    "reason_code": "supports_higher",\n'
        '    "llm_abstain": false\n'
        "  }\n"
        "}\n"
        'If abstaining for a target, use llm_abstain=true and llm_verdict="".\n'
        f"Allowed labels for media: {media_guides['allowed']}\n"
        f"Allowed labels for sleeve: {sleeve_guides['allowed']}\n"
        "Media grade definitions:\n"
        f"{media_defs}\n"
        "Sleeve grade definitions:\n"
        f"{sleeve_defs}\n"
        "Expanded media grading rubric (extracted from grading_guidelines):\n"
        f"{media_rubric}\n"
        "Expanded sleeve grading rubric (extracted from grading_guidelines):\n"
        f"{sleeve_rubric}\n"
        "Critical boundary rules:\n"
        "1) Do not infer sleeve Generic from condition severity alone.\n"
        "   Generic requires explicit wording that the original cover/sleeve is missing\n"
        "   or replaced by generic/plain/company/replacement sleeve language.\n"
        "2) For media, do not downgrade to Good from light/faint wear wording alone.\n"
        "   If text says plays through/plays well/sounds good and no skips or no major pops,\n"
        "   keep the result at least Very Good (Very Good or Very Good Plus depending on\n"
        "   defect severity), unless stronger playback-impact evidence supports Good/Poor.\n"
        "3) Keep targets independent: sleeve damage does not force media downgrade,\n"
        "   and media defects do not force sleeve downgrade.\n"
        "4) Missing primary media components (e.g., missing LP/disc) should be treated as\n"
        "   severe media evidence (typically Poor), even if remaining inserts are perfect.\n"
        "5) Media NM vs VG+ boundary:\n"
        "   - '1-2 really light marks' can still be Near Mint when the rest of evidence is\n"
        "     effectively pristine.\n"
        "   - Do NOT generalize all 'light marks' to Near Mint. Phrases like 'a couple of\n"
        "     light marks' should usually remain Very Good Plus unless explicitly supported\n"
        "     by stronger pristine wording.\n"
        "   - Do not require explicit evidence for BOTH playback and appearance to allow\n"
        "     Near Mint; seller notes are often sparse for high-condition records.\n"
        "   - If only one dimension is described, infer cautiously from phrasing strength:\n"
        "     strong pristine wording can support Near Mint, while vague defect wording\n"
        "     should remain Very Good Plus.\n"
        "   - When uncertain at the NM/VG+ boundary, prefer Very Good Plus.\n"
        "6) For media grading generally, if visual and playback evidence conflict,\n"
        "   prioritize playback impact over raw visual defect counts.\n"
        "   - This applies across the scale, including higher conditions.\n"
        "   - Example: 'high gloss, looks pristine, very little background noise' should\n"
        "     be capped at Very Good Plus; any audible playback issue rules out Near Mint.\n"
        "   - For higher conditions, keep stricter standards: noticeable defects should\n"
        "     cap the grade even when playback is strong.\n"
        "   - Example: 'Lots of hairlines and scratches, but background noise does not\n"
        "     overpower the music' => Very Good (playback still acceptable).\n"
        "   - Example: 'Only a few hairlines and a single scratch that cause crackling\n"
        "     and pops for several rotations' => Good (playback-impact defects).\n"
    )
    payload = {
        "source": listing["source"],
        "item_id": listing["item_id"],
        "split": listing["split"],
        "text": listing.get("text", "")[:4000],
        "targets": {
            "media": {
                "assigned_label": listing.get("media_assigned_label", ""),
                "model_pred_label": listing.get("media_model_pred_label", ""),
                "cleanlab_self_confidence": listing.get(
                    "media_cleanlab_self_confidence"
                ),
                "cleanlab_label_issue": bool(listing.get("media_cleanlab_label_issue")),
                "allowed_labels": media_guides["allowed"],
            },
            "sleeve": {
                "assigned_label": listing.get("sleeve_assigned_label", ""),
                "model_pred_label": listing.get("sleeve_model_pred_label", ""),
                "cleanlab_self_confidence": listing.get(
                    "sleeve_cleanlab_self_confidence"
                ),
                "cleanlab_label_issue": bool(
                    listing.get("sleeve_cleanlab_label_issue")
                ),
                "allowed_labels": sleeve_guides["allowed"],
            },
        },
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        },
    ]


def _parse_joint_decisions(raw_text: str) -> dict[str, dict[str, Any]]:
    obj = parse_llm_json(raw_text)
    out: dict[str, dict[str, Any]] = {}
    for target in ("media", "sleeve"):
        sub = obj.get(target)
        if isinstance(sub, dict):
            out[target] = sub
    return out


def _row_value(row: sqlite3.Row | None, key: str, default: Any = "") -> Any:
    if row is None:
        return default
    try:
        return row[key]
    except Exception:
        return default


def _query_gemini(messages: list[dict[str, str]], model_id: str) -> tuple[str, str]:
    try:
        import google.genai as genai  # pyright: ignore[reportMissingImports]
        from google.genai import types  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        raise ImportError("Install `google-genai` first.") from e

    sys_msg = messages[0]["content"]
    user_msg = messages[1]["content"]

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "").strip())
    resp = client.models.generate_content(
        model=model_id,
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=sys_msg,
            temperature=0.2,
            max_output_tokens=200,
            response_mime_type="application/json",
        ),
    )
    text = str(getattr(resp, "text", "") or "").strip()
    # Gemini API uses requested model directly in this path.
    return text, model_id


def _query_openrouter(
    messages: list[dict[str, str]], model_id: str
) -> tuple[str, str]:
    try:
        from openai import OpenAI  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        raise ImportError("Install `openai` for OpenRouter provider support.") from e

    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip() or (
        os.getenv("GROQ_API_KEY") or ""
    ).strip()
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY (or GROQ_API_KEY fallback).")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    extra_headers: dict[str, str] = {}
    site_url = (os.getenv("OPENROUTER_SITE_URL") or "").strip()
    app_name = (os.getenv("OPENROUTER_APP_NAME") or "").strip()
    if site_url:
        extra_headers["HTTP-Referer"] = site_url
    if app_name:
        extra_headers["X-OpenRouter-Title"] = app_name

    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=0.2,
        max_tokens=200,
        response_format={"type": "json_object"},
        extra_headers=extra_headers or None,
    )
    choices = completion.choices or []
    if not choices:  # pragma: no cover - API path
        raise ValueError("OpenRouter response had no choices.")
    content = choices[0].message.content
    text = str(content or "").strip()
    routed_model = str(getattr(completion, "model", "") or "").strip() or model_id
    return text, routed_model


def _query_ollama(messages: list[dict[str, str]], model_id: str) -> tuple[str, str]:
    try:
        from openai import OpenAI  # pyright: ignore[reportMissingImports]
    except ImportError as e:
        raise ImportError("Install `openai` for Ollama provider support.") from e

    base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1").strip()
    ollama_temp = float((os.getenv("OLLAMA_TEMPERATURE") or "0.2").strip())
    ollama_num_ctx = int((os.getenv("OLLAMA_NUM_CTX") or "8192").strip())
    ollama_max_tokens = int((os.getenv("OLLAMA_MAX_TOKENS") or "200").strip())
    # Local large models can exceed default ~10min client timeout; Scout especially.
    timeout_s = float((os.getenv("OLLAMA_TIMEOUT_SECONDS") or "1200").strip())
    # Ollama's OpenAI-compatible endpoint does not require a real API key.
    client = OpenAI(base_url=base_url, api_key="ollama", timeout=timeout_s)
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=ollama_temp,
        max_tokens=ollama_max_tokens,
        extra_body={"options": {"num_ctx": ollama_num_ctx}},
    )
    choices = completion.choices or []
    if not choices:  # pragma: no cover - API path
        raise ValueError("Ollama response had no choices.")
    content = choices[0].message.content
    text = str(content or "").strip()
    used_model = str(getattr(completion, "model", "") or "").strip() or model_id
    return text, used_model


def _query_provider(
    *,
    provider: str,
    messages: list[dict[str, str]],
    model_id: str,
) -> tuple[str, str]:
    if provider == "gemini":
        return _query_gemini(messages, model_id)
    if provider == "openrouter":
        return _query_openrouter(messages, model_id)
    return _query_ollama(messages, model_id)


def _gating_clause(gating_pass: int) -> tuple[str, tuple]:
    if gating_pass == 1:
        return "cleanlab_label_issue = 1", ()
    if gating_pass == 2:
        return (
            "cleanlab_label_issue = 0 AND cleanlab_self_confidence IS NOT NULL "
            "AND cleanlab_self_confidence <= ?",
            (0.20,),
        )
    return (
        "cleanlab_label_issue = 0 AND disagree_assigned_vs_model = 1",
        (),
    )


def _parse_model_rotation(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _today_utc_ymd() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _load_model_tiers_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _save_model_tiers_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _tiered_candidates(
    all_models: list[str], exhausted: set[str], tier_a_size: int
) -> tuple[list[str], list[str], list[str]]:
    available = [m for m in all_models if m not in exhausted]
    tier_a = available[: max(1, tier_a_size)]
    tier_b = available[max(1, tier_a_size) :]
    return available, tier_a, tier_b


def _listing_belongs_to_shard(
    split: str, source: str, item_id: str, shard_count: int, shard_index: int
) -> bool:
    if shard_count <= 1:
        return True
    key = f"{split}|{source}|{item_id}".encode("utf-8")
    shard = int(hashlib.sha256(key).hexdigest()[:8], 16) % shard_count
    return shard == shard_index


def _parse_csv_values(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_patched_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not path.is_file():
        return keys
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            src = str(obj.get("source", "")).strip()
            iid = str(obj.get("item_id", "")).strip()
            if src and iid:
                keys.add((src, iid))
    return keys


def _is_quota_exhausted_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    signals = (
        "resource_exhausted",
        "quota",
        "rate limit",
        "too many requests",
        "429",
        "exceeded",
    )
    return any(s in msg for s in signals)


def _is_daily_model_cap_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    signals = (
        "per day",
        "daily",
        "prepayment credits are depleted",
        "generaterequestsperday",
    )
    return any(s in msg for s in signals)


def _is_ollama_transient_error(exc: Exception) -> bool:
    """Timeouts and server-side runner crashes often recover without changing model."""
    msg = str(exc).lower()
    signals = (
        "request timed out",
        "timed out",
        "timeout",
        "timedout",
        "read timed out",
        "connection reset",
        "connection refused",
        "error code: 500",
        "runner process has terminated",
        "temporarily unavailable",
        "api_error",
    )
    return any(s in msg for s in signals)


def _is_model_unavailable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    signals = (
        "no endpoints found",
        "error code: 404",
        "notfounderror",
        "model not found",
        "not a valid model id",
        "invalid model id",
        "json mode is not supported for this model",
        '"code":20024',
    )
    return any(s in msg for s in signals)


def _rate_limit_wait_seconds(exc: Exception) -> float | None:
    msg = str(exc)
    m_reset = re.search(
        r"x-ratelimit-reset['\"]?\s*[:=]\s*['\"]?(\d{10,16})",
        msg,
        flags=re.IGNORECASE,
    )
    if m_reset:
        raw = int(m_reset.group(1))
        reset_ts = raw / 1000.0 if raw > 10_000_000_000 else float(raw)
        wait_s = reset_ts - time.time()
        if wait_s > 0:
            return wait_s

    m_retry_in = re.search(
        r"retry in ([0-9]+(?:\.[0-9]+)?)s",
        msg,
        flags=re.IGNORECASE,
    )
    if m_retry_in:
        return float(m_retry_in.group(1))

    m_retry_delay = re.search(
        r"retrydelay['\"]?\s*[:=]\s*['\"]?([0-9]+)s",
        msg,
        flags=re.IGNORECASE,
    )
    if m_retry_delay:
        return float(m_retry_delay.group(1))
    return None


def _health_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT COALESCE(llm_status, '') AS s, COUNT(*) AS n
        FROM queue
        GROUP BY COALESCE(llm_status, '')
        """
    ).fetchall()
    out = {"ok": 0, "error": 0, "parse_error": 0, "pending": 0}
    for s, n in rows:
        key = str(s)
        count = int(n)
        if key == "ok":
            out["ok"] = count
        elif key == "error":
            out["error"] = count
        elif key == "parse_error":
            out["parse_error"] = count
        elif key == "":
            out["pending"] = count
    return out


def main() -> int:
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

    _ = _load_yaml(Path(args.config))
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


if __name__ == "__main__":
    raise SystemExit(main())
