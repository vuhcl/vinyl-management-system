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

from dotenv import load_dotenv

from grader.src.config_io import load_yaml_mapping

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
    guidelines_version: str = "",
) -> str:
    payload = json.dumps(
        {
            "guidelines_version": guidelines_version,
            "media_allowed": media_allowed,
            "media_desc": media_desc,
            "media_rubric": media_rubric,
            "model_id": model_id,
            "sleeve_allowed": sleeve_allowed,
            "sleeve_desc": sleeve_desc,
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

