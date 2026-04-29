from __future__ import annotations

import json
import os
import shlex
import sqlite3
import subprocess
import time
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest

import pandas as pd
import streamlit as st
from dotenv import dotenv_values

from grader.src.eval.label_audit_backend import (
    build_queue_from_cleanlab_csvs,
    commit_queue_to_label_patches,
    ensure_db,
    export_reviewed_to_csv,
    load_guideline_prompt_bits,
)
from grader.src.eval.label_audit_run_llm import (
    MODEL_TIERS_STATE_PATH,
    OPENROUTER_FREE_MODELS,
)


DEFAULT_DB = Path("grader/reports/label_audit_queue.sqlite")
DEFAULT_PATCHES = Path("grader/data/label_patches.jsonl")
RUN_STATE_PATH = Path("grader/reports/label_audit_run_state.json")
STARTUP_META_PATH = Path("grader/reports/label_audit_last_startup.json")
DEFAULT_GUIDELINES_PATH = Path("grader/configs/grading_guidelines.yaml")


def _conn(db_path: Path) -> sqlite3.Connection:
    ensure_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_last_json_line(text: str) -> dict | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # Treat zombie (<defunct>) processes as not alive.
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "stat="],
            capture_output=True,
            text=True,
        )
        stat = (proc.stdout or "").strip()
        if stat and "Z" in stat:
            return False
    except Exception:
        pass
    return True


def _read_run_state() -> dict[str, object]:
    if not RUN_STATE_PATH.is_file():
        return {}
    try:
        return json.loads(RUN_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_run_state(state: dict[str, object]) -> None:
    RUN_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _clear_run_state() -> None:
    if RUN_STATE_PATH.is_file():
        RUN_STATE_PATH.unlink()


def _check_ollama(base_url: str, model_name: str) -> dict[str, object]:
    out: dict[str, object] = {
        "reachable": False,
        "model_found": False,
        "models": [],
        "error": "",
    }
    # OpenAI-compat base is typically .../v1; native Ollama tags API lives at host root /api/tags.
    base = (base_url or "http://localhost:11434").rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")].rstrip("/") or "http://localhost:11434"
    tags_url = f"{base}/api/tags"
    try:
        req = urlrequest.Request(tags_url, method="GET")
        with urlrequest.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        obj = json.loads(body)
        models = obj.get("models", []) if isinstance(obj, dict) else []
        names = []
        for m in models if isinstance(models, list) else []:
            if isinstance(m, dict):
                n = str(m.get("name", "")).strip()
                if n:
                    names.append(n)
        out["reachable"] = True
        out["models"] = names
        if model_name:
            out["model_found"] = model_name in names
    except urlerror.URLError as e:
        out["error"] = str(e)
    except Exception as e:  # pragma: no cover - runtime-only path
        out["error"] = str(e)
    return out


def _env_default_model(provider: str) -> str:
    env = dotenv_values(".env")
    if provider == "openrouter":
        return str(env.get("OPENROUTER_MODEL", "") or "").strip()
    if provider == "gemini":
        return str(env.get("GEMINI_MODEL", "") or "").strip()
    if provider == "ollama":
        return str(env.get("OLLAMA_MODEL", "") or "").strip()
    return ""


def _env_value(name: str, fallback: str) -> str:
    env = dotenv_values(".env")
    raw = env.get(name, "")
    val = str(raw or "").strip()
    return val or fallback


def _state_runs(state: dict[str, object]) -> list[dict[str, object]]:
    runs = state.get("runs")
    if isinstance(runs, list):
        return [r for r in runs if isinstance(r, dict)]
    if "pid" in state and "log_path" in state:
        return [
            {
                "pid": int(state.get("pid", 0) or 0),
                "log_path": str(state.get("log_path", "")),
                "command": state.get("command", []),
            }
        ]
    return []


def _build_log_map(
    run_records: list[dict[str, object]],
    active_runs: list[dict[str, str]],
) -> dict[int, str]:
    log_map: dict[int, str] = {}
    for r in run_records:
        pid = int(r.get("pid", 0) or 0)
        log_path = str(r.get("log_path", "") or "")
        if pid > 0 and log_path:
            log_map[pid] = log_path

    # Also map child<->wrapper pid so grouped UI rows still resolve logs.
    for r in active_runs:
        pid_i = int(r.get("pid", 0) or 0)
        wrapper_i = int(r.get("wrapper_pid", 0) or 0)
        if pid_i > 0 and wrapper_i > 0:
            if pid_i in log_map and wrapper_i not in log_map:
                log_map[wrapper_i] = log_map[pid_i]
            if wrapper_i in log_map and pid_i not in log_map:
                log_map[pid_i] = log_map[wrapper_i]
    return log_map


def _list_active_runner_processes() -> list[dict[str, str]]:
    proc = subprocess.run(
        [
            "ps",
            "-ax",
            "-o",
            "pid=,ppid=,stat=,etime=,command=",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    rows: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        if "grader.src.eval.label_audit_run_llm" not in line:
            continue
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        rows.append(
            {
                "pid": parts[0],
                "ppid": parts[1],
                "stat": parts[2],
                "etime": parts[3],
                "command": parts[4],
            }
        )
    parent_rows = [
        r
        for r in rows
        if r["command"].startswith(
            "uv run python -m grader.src.eval.label_audit_run_llm"
        )
    ]
    grouped: list[dict[str, str]] = []
    used_child_pids: set[str] = set()

    for parent in parent_rows:
        child = next(
            (
                r
                for r in rows
                if r["ppid"] == parent["pid"]
                and "python" in r["command"]
                and "-m grader.src.eval.label_audit_run_llm" in r["command"]
            ),
            None,
        )
        if child:
            used_child_pids.add(child["pid"])
            grouped.append(
                {
                    "pid": child["pid"],
                    "ppid": parent["ppid"],
                    "stat": child["stat"],
                    "etime": child["etime"],
                    "command": child["command"],
                    "wrapper_pid": parent["pid"],
                }
            )
        else:
            grouped.append(
                {
                    "pid": parent["pid"],
                    "ppid": parent["ppid"],
                    "stat": parent["stat"],
                    "etime": parent["etime"],
                    "command": parent["command"],
                    "wrapper_pid": "",
                }
            )

    for r in rows:
        if r["pid"] in used_child_pids:
            continue
        if r["pid"] in {g["pid"] for g in grouped}:
            continue
        if r["pid"] in {g.get("wrapper_pid", "") for g in grouped}:
            continue
        grouped.append(
            {
                "pid": r["pid"],
                "ppid": r["ppid"],
                "stat": r["stat"],
                "etime": r["etime"],
                "command": r["command"],
                "wrapper_pid": "",
            }
        )

    return grouped


def _gating_sql(gating_pass: int) -> tuple[str, tuple]:
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


def _count_backlog_listings(
    conn: sqlite3.Connection,
    *,
    splits: list[str],
    targets: list[str],
    gating_pass: int,
    min_self_confidence: float | None = None,
    max_self_confidence: float | None = None,
    require_disagree: bool = False,
    source_allowlist: list[str] | None = None,
    assigned_label_allowlist: list[str] | None = None,
    exclude_patched: bool = False,
    patched_keys: set[tuple[str, str]] | None = None,
) -> int:
    where_gate, gate_params = _gating_sql(gating_pass)
    source_allowlist = source_allowlist or []
    assigned_label_allowlist = assigned_label_allowlist or []
    extra_where: list[str] = []
    extra_params: list[object] = []
    if min_self_confidence is not None:
        extra_where.append(
            "cleanlab_self_confidence IS NOT NULL AND cleanlab_self_confidence >= ?"
        )
        extra_params.append(float(min_self_confidence))
    if max_self_confidence is not None:
        extra_where.append(
            "cleanlab_self_confidence IS NOT NULL AND cleanlab_self_confidence <= ?"
        )
        extra_params.append(float(max_self_confidence))
    if require_disagree:
        extra_where.append("disagree_assigned_vs_model = 1")
    if source_allowlist:
        extra_where.append(f"source IN ({','.join(['?'] * len(source_allowlist))})")
        extra_params.extend(source_allowlist)
    if assigned_label_allowlist:
        extra_where.append(
            f"assigned_label IN ({','.join(['?'] * len(assigned_label_allowlist))})"
        )
        extra_params.extend(assigned_label_allowlist)
    extra_where_sql = (
        (" AND " + " AND ".join(extra_where))
        if extra_where
        else ""
    )
    rows = conn.execute(
        f"""
        SELECT DISTINCT split, source, item_id
        FROM queue
        WHERE split IN ({",".join(["?"] * len(splits))})
          AND target IN ({",".join(["?"] * len(targets))})
          AND COALESCE(llm_status, '') NOT IN ('ok')
          AND ({where_gate})
          {extra_where_sql}
        """,
        tuple(splits) + tuple(targets) + gate_params + tuple(extra_params),
    ).fetchall()
    if not exclude_patched:
        return len(rows)
    patched_keys = patched_keys or set()
    return sum(
        1 for r in rows if (str(r["source"]), str(r["item_id"])) not in patched_keys
    )


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


def _release_id_from_item_id(item_id: str) -> str:
    s = str(item_id or "").strip()
    if not s or ":" not in s:
        return ""
    return s.split(":", 1)[0].strip()


def page_setup(db_path: Path) -> None:
    st.header("1) Setup / Queue Build")
    st.caption("Build queue DB from cleanlab CSVs and split JSONLs.")
    csv_text = st.text_area(
        "Cleanlab CSV paths (one per line)",
        value="grader/reports/cleanlab_label_audit_media.csv\ngrader/reports/cleanlab_label_audit_sleeve.csv",
        height=120,
    )
    splits_dir = Path(
        st.text_input("Splits dir", value="grader/data/splits")
    )
    default_split = st.selectbox("Default split if missing", ["train", "val", "test"])
    if st.button("Build / Refresh queue DB", type="primary"):
        csvs = [Path(x.strip()) for x in csv_text.splitlines() if x.strip()]
        try:
            stats = build_queue_from_cleanlab_csvs(
                db_path=db_path,
                csv_paths=csvs,
                splits_dir=splits_dir,
                default_split=default_split,
            )
            st.success(f"Queue build done: {stats}")
        except Exception as e:
            st.error(str(e))

    with _conn(db_path) as conn:
        c = conn.execute(
            "SELECT split,target,COUNT(*) n FROM queue GROUP BY split,target ORDER BY split,target"
        ).fetchall()
    if c:
        st.dataframe(pd.DataFrame([dict(x) for x in c]), width="stretch")


def page_run_llm(db_path: Path) -> None:
    st.header("2) Run LLM")
    st.caption(
        "Run pass 1/2/3 with resume and cache. "
        "Uses provider env vars from .env "
        "(OPENROUTER_* or GROQ_API_KEY fallback, GEMINI_*, or OLLAMA_*)."
    )
    if "pending_run_limit" in st.session_state:
        st.session_state["run_limit"] = int(st.session_state.pop("pending_run_limit"))
    if "run_limit" not in st.session_state:
        st.session_state["run_limit"] = 0
    if "sleep_s" not in st.session_state:
        st.session_state["sleep_s"] = 2.0
    if "rotation_cooldown_s" not in st.session_state:
        st.session_state["rotation_cooldown_s"] = 1.5
    if "max_attempts_per_listing" not in st.session_state:
        st.session_state["max_attempts_per_listing"] = 6
    if "workers" not in st.session_state:
        st.session_state["workers"] = 1
    if "tier_a_size" not in st.session_state:
        st.session_state["tier_a_size"] = 3
    if "ollama_base_url" not in st.session_state:
        st.session_state["ollama_base_url"] = (
            os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
        )
    if "ollama_num_ctx" not in st.session_state:
        try:
            st.session_state["ollama_num_ctx"] = int(
                _env_value("OLLAMA_NUM_CTX", "12288")
            )
        except ValueError:
            st.session_state["ollama_num_ctx"] = 12288
    if "ollama_temperature" not in st.session_state:
        st.session_state["ollama_temperature"] = float(
            (os.getenv("OLLAMA_TEMPERATURE") or "0.2").strip()
        )
    if "ollama_max_tokens" not in st.session_state:
        st.session_state["ollama_max_tokens"] = int(
            (os.getenv("OLLAMA_MAX_TOKENS") or "200").strip()
        )
    if "ollama_timeout_seconds" not in st.session_state:
        try:
            st.session_state["ollama_timeout_seconds"] = float(
                (os.getenv("OLLAMA_TIMEOUT_SECONDS") or "1200").strip()
            )
        except ValueError:
            st.session_state["ollama_timeout_seconds"] = 1200.0
    if "ollama_transient_retry_sleep" not in st.session_state:
        try:
            st.session_state["ollama_transient_retry_sleep"] = float(
                (os.getenv("OLLAMA_TRANSIENT_RETRY_SLEEP_SECONDS") or "8").strip()
            )
        except ValueError:
            st.session_state["ollama_transient_retry_sleep"] = 8.0
    provider = st.selectbox("Provider", ["openrouter", "gemini", "ollama"], index=2)
    if provider == "ollama" and st.button(
        "Apply recommended Ollama defaults",
        help="Single-worker local defaults to reduce instability and churn.",
    ):
        st.session_state["sleep_s"] = 4.0
        st.session_state["rotation_cooldown_s"] = 0.5
        st.session_state["max_attempts_per_listing"] = 4
        st.session_state["workers"] = 1
        st.session_state["tier_a_size"] = 1
        st.session_state["ollama_num_ctx"] = 8192
        st.session_state["ollama_temperature"] = 0.2
        st.session_state["ollama_max_tokens"] = 200
        st.session_state["ollama_timeout_seconds"] = 1200.0
        st.session_state["ollama_transient_retry_sleep"] = 12.0
        st.rerun()
    gating_pass = st.selectbox("Gating pass", [1, 2, 3], index=0)
    limit = st.number_input(
        "Limit rows (0 = all matching)",
        min_value=0,
        key="run_limit",
    )
    sleep_s = st.number_input("Sleep seconds", min_value=0.0, step=0.1, key="sleep_s")
    rotation_cooldown_s = st.number_input(
        "Rotation cooldown seconds",
        min_value=0.0,
        step=0.1,
        key="rotation_cooldown_s",
        help="Pause before falling back to the next model on quota/unavailable errors.",
    )
    max_attempts_per_listing = st.number_input(
        "Max attempts per listing",
        min_value=1,
        max_value=30,
        step=1,
        key="max_attempts_per_listing",
        help="Hard cap before marking a listing error and moving on.",
    )
    workers = st.number_input(
        "Workers (parallel shards)",
        min_value=1,
        max_value=12,
        step=1,
        key="workers",
        help=(
            "Starts N shard processes with --shard-count/--shard-index. "
            "Use small values (2-4) to avoid provider rate-limit bursts."
        ),
    )
    tier_a_size = st.number_input(
        "Tier A size",
        min_value=1,
        max_value=12,
        step=1,
        key="tier_a_size",
        help="Number of active priority models; others remain in Tier B reserve.",
    )
    min_self_conf = st.number_input(
        "Min cleanlab self-confidence (optional prefilter; 0.0 disables)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )
    max_self_conf = st.number_input(
        "Max cleanlab self-confidence (optional prefilter; 1.0 disables)",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.01,
    )
    if min_self_conf > max_self_conf:
        st.warning("Min self-confidence is greater than max; adjust range.")
    require_disagree = st.checkbox(
        "Require assigned vs model disagreement prefilter",
        value=False,
    )
    source_allowlist = st.text_input(
        "Source allowlist (comma-separated, optional)",
        value="",
    )
    assigned_label_allowlist = st.text_input(
        "Assigned-label allowlist (comma-separated, optional)",
        value="",
    )
    exclude_patched = st.checkbox(
        "Exclude listings already in label_patches.jsonl",
        value=True,
    )
    label_patches_path = st.text_input(
        "label_patches path for exclusion",
        value="grader/data/label_patches.jsonl",
    )
    source_allowlist_values = [x.strip() for x in source_allowlist.split(",") if x.strip()]
    assigned_label_values = [
        x.strip() for x in assigned_label_allowlist.split(",") if x.strip()
    ]
    patched_keys = (
        _load_patched_keys(Path(label_patches_path)) if exclude_patched else set()
    )
    default_rotation = ",".join(OPENROUTER_FREE_MODELS) if provider == "openrouter" else ""
    model_rotation = st.text_input(
        "Model rotation fallback list (comma-separated, optional)",
        value=default_rotation,
        help="Auto-switches to next model on quota/rate-limit errors.",
    )
    splits = st.multiselect("Splits", ["train", "val", "test"], default=["train", "val", "test"])
    targets = st.multiselect("Targets", ["sleeve", "media"], default=["sleeve", "media"])
    if provider == "openrouter":
        model_env_name = "OPENROUTER_MODEL"
    elif provider == "gemini":
        model_env_name = "GEMINI_MODEL"
    else:
        model_env_name = "OLLAMA_MODEL"
    model_override = st.text_input(
        f"Model override (blank uses {model_env_name})",
        value="",
    )
    env_default_model = _env_default_model(provider)
    effective_model = model_override.strip() or env_default_model
    if effective_model:
        st.caption(
            f"Effective requested model: `{effective_model}` "
            "(override if set, otherwise .env)."
        )
    else:
        st.warning(
            f"No {model_env_name} found in .env and model override is blank."
        )
    enable_critic_pass = st.checkbox(
        "Enable critic pass (writes auto_decision as llm_critic)",
        value=False,
    )
    critic_model_id = ""
    critic_k_examples = 5
    critic_min_confidence = 0.90
    critic_policy_version = ""
    critic_include_auto_apply_gold = False
    if enable_critic_pass:
        with st.expander("Critic settings", expanded=False):
            critic_model_id = st.text_input(
                "Critic model override (blank uses primary model)",
                value="",
            )
            critic_k_examples = int(
                st.number_input(
                    "Critic retrieved examples (K)",
                    min_value=0,
                    max_value=20,
                    value=5,
                    step=1,
                )
            )
            critic_min_confidence = float(
                st.number_input(
                    "Critic minimum confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.90,
                    step=0.01,
                )
            )
            critic_policy_version = st.text_input(
                "Critic policy version (optional)",
                value="",
            )
            critic_include_auto_apply_gold = st.checkbox(
                "Include auto_apply rows in critic example bank",
                value=False,
            )
    if provider == "ollama":
        with st.expander("Ollama preflight", expanded=False):
            ollama_base_url = st.text_input(
                "Ollama base URL",
                key="ollama_base_url",
                help=(
                    "Use the OpenAI-compat root, e.g. http://localhost:11434/v1 (same as OLLAMA_BASE_URL). "
                    "The connectivity probe calls the native /api/tags on the host (if the URL ends "
                    "with /v1, that suffix is stripped for the check only)."
                ),
            )
            st.number_input(
                "Ollama context length (OLLAMA_NUM_CTX)",
                min_value=1024,
                max_value=131072,
                step=1024,
                key="ollama_num_ctx",
            )
            st.number_input(
                "Ollama temperature (OLLAMA_TEMPERATURE)",
                min_value=0.0,
                max_value=2.0,
                step=0.05,
                key="ollama_temperature",
            )
            st.number_input(
                "Ollama max tokens (OLLAMA_MAX_TOKENS)",
                min_value=32,
                max_value=4096,
                step=32,
                key="ollama_max_tokens",
            )
            st.number_input(
                "HTTP client timeout seconds (OLLAMA_TIMEOUT_SECONDS)",
                min_value=60.0,
                max_value=7200.0,
                step=30.0,
                key="ollama_timeout_seconds",
                help=(
                    "How long each completion may run before the client gives up "
                    "(separate from Ollama server limits). Large models need higher values."
                ),
            )
            st.number_input(
                "Backoff after transient errors (OLLAMA_TRANSIENT_RETRY_SLEEP_SECONDS)",
                min_value=1.0,
                max_value=120.0,
                step=1.0,
                key="ollama_transient_retry_sleep",
                help="Sleep after timeout/500/runner-crash before retrying the same listing.",
            )
            # Match label_audit_run_llm: .env is read via dotenv there and here for the
            # main caption — do not use os.getenv("OLLAMA_MODEL") (often unset in this process).
            ollama_check_model = (effective_model or "llama3.1:8b").strip()
            st.caption(
                f"Connectivity check uses: `{ollama_check_model}` "
                "(same model as above; falls back to llama3.1:8b only if "
                "`OLLAMA_MODEL` is unset and the override field is blank)."
            )
            if st.button("Check Ollama connectivity and model", key="check_ollama"):
                check = _check_ollama(ollama_base_url, ollama_check_model)
                if not check["reachable"]:
                    st.error(f"Ollama not reachable: {check['error']}")
                elif not check["model_found"]:
                    st.warning(
                        "Ollama is reachable but model not found. "
                        f"Run: `ollama pull {ollama_check_model}`"
                    )
                    models = check.get("models", [])
                    if isinstance(models, list) and models:
                        st.caption("Available local models")
                        st.code("\n".join(models[:50]))
                else:
                    st.success("Ollama reachable and model is available locally.")
    with st.expander("Daily budget planner", expanded=False):
        daily_budget = st.number_input(
            "Daily request budget",
            min_value=1,
            value=1400,
            step=50,
        )
        share1 = st.number_input("Pass 1 budget %", min_value=0, max_value=100, value=70)
        share2 = st.number_input("Pass 2 budget %", min_value=0, max_value=100, value=20)
        share3 = st.number_input("Pass 3 budget %", min_value=0, max_value=100, value=10)
        total_share = int(share1 + share2 + share3)
        if total_share != 100:
            st.warning(f"Budget shares currently sum to {total_share}%.")

        suggested_for_current_pass = 0
        if splits and targets:
            with _conn(db_path) as conn:
                p1_backlog = _count_backlog_listings(
                    conn,
                    splits=splits,
                    targets=targets,
                    gating_pass=1,
                    min_self_confidence=(
                        float(min_self_conf) if min_self_conf > 0.0001 else None
                    ),
                    max_self_confidence=(
                        float(max_self_conf) if max_self_conf < 0.9999 else None
                    ),
                    require_disagree=require_disagree,
                    source_allowlist=source_allowlist_values,
                    assigned_label_allowlist=assigned_label_values,
                    exclude_patched=exclude_patched,
                    patched_keys=patched_keys,
                )
                p2_backlog = _count_backlog_listings(
                    conn,
                    splits=splits,
                    targets=targets,
                    gating_pass=2,
                    min_self_confidence=(
                        float(min_self_conf) if min_self_conf > 0.0001 else None
                    ),
                    max_self_confidence=(
                        float(max_self_conf) if max_self_conf < 0.9999 else None
                    ),
                    require_disagree=require_disagree,
                    source_allowlist=source_allowlist_values,
                    assigned_label_allowlist=assigned_label_values,
                    exclude_patched=exclude_patched,
                    patched_keys=patched_keys,
                )
                p3_backlog = _count_backlog_listings(
                    conn,
                    splits=splits,
                    targets=targets,
                    gating_pass=3,
                    min_self_confidence=(
                        float(min_self_conf) if min_self_conf > 0.0001 else None
                    ),
                    max_self_confidence=(
                        float(max_self_conf) if max_self_conf < 0.9999 else None
                    ),
                    require_disagree=require_disagree,
                    source_allowlist=source_allowlist_values,
                    assigned_label_allowlist=assigned_label_values,
                    exclude_patched=exclude_patched,
                    patched_keys=patched_keys,
                )
            planned = [
                ("pass_1", p1_backlog, int(daily_budget * share1 / 100)),
                ("pass_2", p2_backlog, int(daily_budget * share2 / 100)),
                ("pass_3", p3_backlog, int(daily_budget * share3 / 100)),
            ]
            rows = []
            for pass_name, backlog, budget_cap in planned:
                suggested_limit = min(backlog, budget_cap)
                if pass_name == f"pass_{gating_pass}":
                    suggested_for_current_pass = suggested_limit
                rows.append(
                    {
                        "gating_pass": pass_name,
                        "backlog_listings": backlog,
                        "budget_cap": budget_cap,
                        "suggested_limit": suggested_limit,
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch")
            if st.button("Apply suggested limit to current pass"):
                st.session_state["pending_run_limit"] = int(
                    suggested_for_current_pass
                )
                st.rerun()
            st.caption(
                "Suggested limit estimates listings (1 request per listing for joint media+sleeve)."
            )
        else:
            st.info("Select at least one split and one target to compute planner.")

    run_state = st.session_state.setdefault("llm_run_state", {})
    persisted_state = _read_run_state()
    run_records = _state_runs(persisted_state)
    active_runs = _list_active_runner_processes()
    run_log_by_pid = _build_log_map(run_records, active_runs)
    any_active = bool(active_runs)
    tracked_pid = int(
        persisted_state.get("tracked_pid", persisted_state.get("pid", 0)) or 0
    )
    tracked_alive = bool(tracked_pid and _is_pid_alive(tracked_pid))
    is_running = any_active or tracked_alive

    cmd = [
        "uv",
        "run",
        "python",
        "-u",
        "-m",
        "grader.src.eval.label_audit_run_llm",
        "--db",
        str(db_path),
        "--gating-pass",
        str(gating_pass),
        "--limit",
        str(int(limit)),
        "--sleep-seconds",
        str(float(sleep_s)),
        "--rotation-cooldown-seconds",
        str(float(rotation_cooldown_s)),
        "--max-attempts-per-listing",
        str(int(max_attempts_per_listing)),
        "--tier-a-size",
        str(int(tier_a_size)),
        "--provider",
        provider,
    ]
    if effective_model:
        cmd += ["--model-id", effective_model]
    if model_rotation.strip():
        cmd += ["--model-rotation", model_rotation.strip()]
    if min_self_conf > 0.0001:
        cmd += ["--min-self-confidence", str(float(min_self_conf))]
    if max_self_conf < 0.9999:
        cmd += ["--max-self-confidence", str(float(max_self_conf))]
    if require_disagree:
        cmd += ["--require-disagree"]
    if source_allowlist.strip():
        cmd += ["--sources", source_allowlist.strip()]
    if assigned_label_allowlist.strip():
        cmd += ["--assigned-labels", assigned_label_allowlist.strip()]
    if exclude_patched:
        cmd += [
            "--exclude-patched",
            "--label-patches-path",
            str(Path(label_patches_path)),
        ]
    if splits:
        cmd += ["--splits", *splits]
    if targets:
        cmd += ["--targets", *targets]
    if enable_critic_pass:
        cmd += [
            "--enable-critic-pass",
            "--critic-k-examples",
            str(int(critic_k_examples)),
            "--critic-min-confidence",
            str(float(critic_min_confidence)),
        ]
        if critic_model_id.strip():
            cmd += ["--critic-model-id", critic_model_id.strip()]
        if critic_policy_version.strip():
            cmd += ["--critic-policy-version", critic_policy_version.strip()]
        if critic_include_auto_apply_gold:
            cmd += ["--critic-include-auto-apply-gold"]

    col1, col2, col3 = st.columns([1, 1, 2])
    start_clicked = col1.button(
        "Start background run",
        type="primary",
        disabled=is_running,
    )
    stop_clicked = col2.button("Stop tracked run", disabled=not tracked_alive)
    col3.caption(
        f"Command: `{shlex.join(cmd)}`"
    )

    stop_all_clicked = st.button("Stop all active runs", disabled=not any_active)

    if start_clicked:
        log_dir = Path("grader/reports")
        log_dir.mkdir(parents=True, exist_ok=True)
        worker_count = int(workers)
        proc_env = os.environ.copy()
        if provider == "ollama":
            proc_env["OLLAMA_BASE_URL"] = str(st.session_state["ollama_base_url"])
            proc_env["OLLAMA_NUM_CTX"] = str(int(st.session_state["ollama_num_ctx"]))
            proc_env["OLLAMA_TEMPERATURE"] = str(
                float(st.session_state["ollama_temperature"])
            )
            proc_env["OLLAMA_MAX_TOKENS"] = str(
                int(st.session_state["ollama_max_tokens"])
            )
            proc_env["OLLAMA_TIMEOUT_SECONDS"] = str(
                float(st.session_state["ollama_timeout_seconds"])
            )
            proc_env["OLLAMA_TRANSIENT_RETRY_SLEEP_SECONDS"] = str(
                float(st.session_state["ollama_transient_retry_sleep"])
            )
        started: list[dict[str, object]] = []
        run_records = _state_runs(_read_run_state())
        now_ts = int(time.time())
        for i in range(worker_count):
            worker_cmd = list(cmd)
            if worker_count > 1:
                worker_cmd += [
                    "--shard-count",
                    str(worker_count),
                    "--shard-index",
                    str(i),
                ]
            suffix = f"_w{i + 1}" if worker_count > 1 else ""
            log_path = log_dir / f"label_audit_run_{now_ts}{suffix}.log"
            log_file = log_path.open("a", encoding="utf-8")
            proc = subprocess.Popen(
                worker_cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                env=proc_env,
            )
            log_file.close()
            run_state["proc"] = proc
            run_state["log_path"] = str(log_path)
            run_state["command"] = worker_cmd
            rec = {
                "pid": int(proc.pid),
                "log_path": str(log_path),
                "command": worker_cmd,
                "started_at_unix": int(time.time()),
            }
            started.append(rec)
            run_records.append(rec)
        _write_run_state(
            {
                "tracked_pid": int(started[-1]["pid"]) if started else 0,
                "runs": run_records[-30:],
            }
        )
        if started:
            pids = ", ".join(str(int(r["pid"])) for r in started)
            st.success(
                "Started background run(s): "
                f"{worker_count} worker(s), pid(s)={pids}. "
                "You can continue reviewing while they run."
            )

    if stop_clicked and tracked_alive:
        try:
            os.kill(tracked_pid, 15)
            st.warning(f"Stop signal sent to tracked run (pid={tracked_pid}).")
        except OSError as e:
            st.error(f"Failed to stop tracked run: {e}")

    if stop_all_clicked and active_runs:
        stopped = 0
        for r in active_runs:
            try:
                os.kill(int(r["pid"]), 15)
                stopped += 1
            except OSError:
                continue
        st.warning(f"Stop signal sent to {stopped} active run(s).")

    persisted_state = _read_run_state()
    run_records = _state_runs(persisted_state)
    run_log_by_pid = _build_log_map(run_records, active_runs)
    tracked_pid = int(
        persisted_state.get("tracked_pid", persisted_state.get("pid", 0)) or 0
    )
    tracked_alive = bool(tracked_pid and _is_pid_alive(tracked_pid))
    if tracked_alive:
        st.info(
            f"Tracked background run is active (pid={tracked_pid})."
        )
    elif persisted_state:
        st.info("Tracked run is no longer active.")
        _write_run_state(
            {
                "tracked_pid": 0,
                "runs": run_records[-30:],
            }
        )

    active_runs = _list_active_runner_processes()
    # Rebuild mapping using freshly sampled active runs.
    run_log_by_pid = _build_log_map(run_records, active_runs)
    if active_runs:
        table_rows = []
        for r in active_runs:
            pid_i = int(r["pid"])
            rec = dict(r)
            rec["log_path"] = run_log_by_pid.get(pid_i, "")
            table_rows.append(rec)
        st.dataframe(pd.DataFrame(table_rows), width="stretch")
    else:
        st.caption("No active `label_audit_run_llm` processes found.")

    if STARTUP_META_PATH.is_file():
        try:
            startup_meta = json.loads(
                STARTUP_META_PATH.read_text(encoding="utf-8")
            )
            st.caption("Last runner startup metadata")
            st.json(startup_meta)
        except Exception:
            pass

    if MODEL_TIERS_STATE_PATH.is_file():
        try:
            tiers_obj = json.loads(
                MODEL_TIERS_STATE_PATH.read_text(encoding="utf-8")
            )
            provider_obj = tiers_obj.get(provider, {})
            if isinstance(provider_obj, dict) and provider_obj:
                today_key = sorted(provider_obj.keys())[-1]
                exhausted_today = provider_obj.get(today_key, [])
                st.caption(f"Exhausted models for {provider} ({today_key})")
                st.json(
                    {
                        "count": (
                            len(exhausted_today)
                            if isinstance(exhausted_today, list)
                            else 0
                        ),
                        "models": (
                            exhausted_today
                            if isinstance(exhausted_today, list)
                            else []
                        ),
                    }
                )
        except Exception:
            pass

    candidate_pids = [int(r["pid"]) for r in active_runs]
    if tracked_pid and tracked_pid not in candidate_pids:
        candidate_pids.append(tracked_pid)
    default_pid = max(candidate_pids) if candidate_pids else 0
    if (
        default_pid
        and st.session_state.get("log_view_pid") != default_pid
    ):
        st.session_state["log_view_pid"] = default_pid
    if candidate_pids:
        selected_pid = st.selectbox(
            "Log view PID",
            options=candidate_pids,
            key="log_view_pid",
        )
    else:
        selected_pid = 0
    selected_log_path = run_log_by_pid.get(selected_pid, "")
    if not selected_log_path:
        # Fallback: use the most recently started run record that has a log path.
        latest_with_log = next(
            (
                r
                for r in sorted(
                    run_records,
                    key=lambda x: int(x.get("started_at_unix", 0) or 0),
                    reverse=True,
                )
                if str(r.get("log_path", "") or "")
            ),
            None,
        )
        if latest_with_log:
            selected_log_path = str(latest_with_log.get("log_path", "") or "")
    log_path = Path(str(selected_log_path)).expanduser()
    if log_path and log_path.is_file():
        raw_text = log_path.read_text(encoding="utf-8", errors="replace")
        summary_obj = _parse_last_json_line(raw_text)
        if summary_obj is not None:
            st.json(summary_obj)
        lines = raw_text.splitlines()
        tail = "\n".join(lines[-80:]) if lines else ""
        with st.expander("LLM run log tail", expanded=False):
            st.code(tail or "(log is empty)")
    else:
        st.caption(
            "No tracked log file available. If run was started from terminal, use terminal output."
        )

    with _conn(db_path) as conn:
        summary = conn.execute(
            """
            SELECT llm_status, COUNT(*) n
            FROM queue
            GROUP BY llm_status
            ORDER BY n DESC
            """
        ).fetchall()
    if summary:
        st.dataframe(pd.DataFrame([dict(x) for x in summary]), width="stretch")


def page_review(db_path: Path) -> None:
    st.header("3) Review")
    st.caption("Filter by split/source/target and mass-review rows.")
    split = st.selectbox("Split", ["all", "train", "val", "test"], index=0)
    target = st.selectbox("Target", ["all", "sleeve", "media"], index=0)
    llm_responded_only = st.checkbox(
        "Show only rows with LLM response",
        value=True,
    )
    relabel_only = st.checkbox("Show relabel candidates only", value=False)
    decision_filter = st.selectbox(
        "Decision status",
        [
            "undecided",
            "accepted",
            "rejected",
            "decided (accepted + rejected)",
            "all",
        ],
        index=0,
    )
    auto_decision_filter = st.selectbox(
        "Auto decision",
        ["all", "proposed_grade", "needs_review"],
        index=0,
    )
    auto_source_filter = st.selectbox(
        "Auto source",
        ["all", "pred", "llm", "assigned", "llm_critic"],
        index=0,
    )
    with _conn(db_path) as conn:
        version_rows = conn.execute(
            """
            SELECT DISTINCT COALESCE(auto_policy_version, '') AS v
            FROM queue
            WHERE COALESCE(auto_policy_version, '') <> ''
            ORDER BY v DESC
            """
        ).fetchall()
    policy_options = ["all"] + [str(r["v"]) for r in version_rows]
    policy_version_filter = st.selectbox(
        "Auto policy version",
        policy_options,
        index=0,
    )
    limit = st.number_input("Rows to show", min_value=10, max_value=500, value=100, step=10)

    base_where = ["1=1"]
    base_params: list[object] = []
    if split != "all":
        base_where.append("split = ?")
        base_params.append(split)
    if target != "all":
        base_where.append("target = ?")
        base_params.append(target)
    if llm_responded_only:
        base_where.append("COALESCE(llm_status, '') = 'ok'")
        base_where.append("COALESCE(llm_verdict, '') <> ''")
    if relabel_only:
        base_where.append(
            "(cleanlab_label_issue = 1 OR disagree_llm_vs_assigned = 1 OR "
            "(llm_verdict <> '' AND assigned_label <> '' AND llm_verdict <> assigned_label))"
        )
    if auto_decision_filter != "all":
        base_where.append("COALESCE(auto_decision, '') = ?")
        base_params.append(auto_decision_filter)
    if auto_source_filter != "all":
        base_where.append("COALESCE(auto_decision_source, '') = ?")
        base_params.append(auto_source_filter)
    if policy_version_filter != "all":
        base_where.append("COALESCE(auto_policy_version, '') = ?")
        base_params.append(policy_version_filter)

    where = list(base_where)
    params: list[object] = list(base_params)
    if decision_filter == "undecided":
        where.append("COALESCE(human_action, '') = ''")
    elif decision_filter == "accepted":
        where.append("COALESCE(human_action, '') IN ('accept_llm', 'auto_apply')")
    elif decision_filter == "rejected":
        where.append("COALESCE(human_action, '') = 'keep_assigned'")
    elif decision_filter == "decided (accepted + rejected)":
        where.append(
            "COALESCE(human_action, '') IN ('accept_llm', 'auto_apply', 'keep_assigned')"
        )

    sql = (
        "SELECT queue_row_id,split,target,source,item_id,llm_status,human_action,assigned_label,model_pred_label,"
        "llm_verdict,llm_confidence,reason_code,cleanlab_self_confidence,cleanlab_label_issue,"
        "auto_decision,auto_final_label,auto_decision_score,auto_decision_reason,auto_decision_source,auto_policy_version,"
        "text FROM queue WHERE "
        + " AND ".join(where)
        + " ORDER BY cleanlab_self_confidence ASC LIMIT ?"
    )
    params.append(int(limit))
    with _conn(db_path) as conn:
        counts_row = conn.execute(
            "SELECT "
            "SUM(CASE WHEN COALESCE(human_action, '') = '' THEN 1 ELSE 0 END) AS undecided, "
            "SUM(CASE WHEN COALESCE(human_action, '') IN ('accept_llm', 'auto_apply') THEN 1 ELSE 0 END) AS accepted, "
            "SUM(CASE WHEN COALESCE(human_action, '') = 'keep_assigned' THEN 1 ELSE 0 END) AS rejected, "
            "COUNT(*) AS total "
            "FROM queue WHERE "
            + " AND ".join(base_where),
            tuple(base_params),
        ).fetchone()
        rows = conn.execute(sql, tuple(params)).fetchall()
    if counts_row is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Undecided", int(counts_row["undecided"] or 0))
        c2.metric("Accepted", int(counts_row["accepted"] or 0))
        c3.metric("Rejected", int(counts_row["rejected"] or 0))
        c4.metric("Total (current filters)", int(counts_row["total"] or 0))
    filter_bits: list[str] = []
    if split != "all":
        filter_bits.append(f"split={split}")
    if target != "all":
        filter_bits.append(f"target={target}")
    if llm_responded_only:
        filter_bits.append("llm_status=ok")
    if relabel_only:
        filter_bits.append("relabel_only")
    filter_bits.append(f"decision={decision_filter}")
    if auto_decision_filter != "all":
        filter_bits.append(f"auto={auto_decision_filter}")
    if auto_source_filter != "all":
        filter_bits.append(f"source={auto_source_filter}")
    if policy_version_filter != "all":
        filter_bits.append(f"policy={policy_version_filter}")
    st.caption("Active filters: " + ", ".join(filter_bits))
    if not rows:
        st.info("No rows match current filters.")
        return
    df = pd.DataFrame([dict(r) for r in rows])
    df["release_id"] = df["item_id"].map(_release_id_from_item_id)
    st.dataframe(
        df[
            [
                "queue_row_id",
                "split",
                "item_id",
                "release_id",
                "llm_status",
                "human_action",
                "target",
                "assigned_label",
                "model_pred_label",
                "llm_verdict",
                "auto_decision",
                "auto_final_label",
                "auto_decision_source",
                "auto_decision_score",
                "auto_policy_version",
                "text",
                "llm_confidence",
                "reason_code",
                "cleanlab_self_confidence",
                "cleanlab_label_issue",
            ]
        ],
        width="stretch",
    )

    def _apply_single_row_action(qid: int, action: str) -> None:
        with _conn(db_path) as conn:
            row = conn.execute(
                "SELECT assigned_label,llm_verdict,model_pred_label FROM queue WHERE queue_row_id=?",
                (int(qid),),
            ).fetchone()
            if row is None:
                return
            if action == "accept_llm":
                # If LLM has not responded for this row, accept falls back to model prediction.
                final_label = str(
                    row["llm_verdict"] or row["model_pred_label"] or ""
                ).strip()
            elif action == "keep_assigned":
                final_label = str(row["assigned_label"] or "").strip()
            else:
                final_label = ""
            conn.execute(
                """
                UPDATE queue
                SET human_action=?, final_label=?, updated_at=datetime('now')
                WHERE queue_row_id=?
                """,
                (action, final_label, int(qid)),
            )
            conn.commit()

    try:
        sleeve_allowed = [
            str(x).strip()
            for x in load_guideline_prompt_bits(
                DEFAULT_GUIDELINES_PATH, "sleeve"
            ).get("allowed", [])
            if str(x).strip()
        ]
        media_allowed = [
            str(x).strip()
            for x in load_guideline_prompt_bits(
                DEFAULT_GUIDELINES_PATH, "media"
            ).get("allowed", [])
            if str(x).strip()
        ]
    except Exception:
        sleeve_allowed = []
        media_allowed = []

    st.markdown("**Per-row quick actions**")
    quick_limit = min(30, len(df))
    for _, r in df.head(quick_limit).iterrows():
        qid = int(r["queue_row_id"])
        row_target = str(r.get("target", "") or "").strip().lower()
        if row_target == "sleeve":
            per_row_options = sleeve_allowed
        elif row_target == "media":
            per_row_options = [x for x in media_allowed if x.lower() != "generic"]
        else:
            per_row_options = []
        release_id = str(r.get("release_id", "") or "")
        release_part = f" | release_id={release_id}" if release_id else ""
        row_label = (
            f"#{qid} | split={r['split']} | target={r['target']} | "
            f"assigned={r['assigned_label']} | llm={r['llm_verdict']} | "
            f"pred={r['model_pred_label']}"
            f"{release_part}"
        )
        row_text = str(r.get("text", "") or "").replace("\n", " ").strip()
        c1, c2, c3, c4 = st.columns([5, 1, 1, 3])
        c1.caption(row_label)
        c1.caption(f"text: {row_text}")
        if c2.button("Accept", key=f"accept_{qid}"):
            _apply_single_row_action(qid, "accept_llm")
            st.rerun()
        if c3.button("Reject", key=f"reject_{qid}"):
            _apply_single_row_action(qid, "keep_assigned")
            st.rerun()
        if per_row_options:
            row_manual_label = c4.selectbox(
                "Manual",
                options=per_row_options,
                key=f"manual_label_{qid}",
                label_visibility="collapsed",
            )
            if c4.button("Set", key=f"set_manual_{qid}"):
                with _conn(db_path) as conn:
                    conn.execute(
                        """
                        UPDATE queue
                        SET human_action='manual_set', final_label=?, updated_at=datetime('now')
                        WHERE queue_row_id=?
                        """,
                        (str(row_manual_label).strip(), int(qid)),
                    )
                    conn.commit()
                st.rerun()
        else:
            c4.caption("Manual n/a")

    st.markdown("**Selected row text**")
    selected_text_row_id = st.selectbox(
        "Pick a queue_row_id to read full listing text",
        options=df["queue_row_id"].tolist(),
        index=0,
    )
    selected_text_row = df[df["queue_row_id"] == int(selected_text_row_id)]
    if not selected_text_row.empty:
        text_value = str(selected_text_row.iloc[0].get("text", "") or "")
        st.text_area(
            "Listing text",
            value=text_value,
            height=220,
            disabled=True,
        )
    selected = st.multiselect(
        "Select queue_row_id for bulk action",
        options=df["queue_row_id"].tolist(),
    )
    action = st.selectbox(
        "Bulk action",
        ["accept_llm", "keep_assigned", "set_manual_label", "skip"],
    )
    if target == "sleeve":
        label_options = sleeve_allowed
    elif target == "media":
        # Media must never offer Generic.
        label_options = [x for x in media_allowed if x.lower() != "generic"]
    else:
        label_options = []
    manual_label = st.selectbox(
        "Manual condition label",
        options=label_options,
        index=0 if label_options else None,
        placeholder="Pick target first (sleeve/media)",
        disabled=(target == "all"),
        help=(
            "Options follow target-specific guidelines. "
            "Set Target to sleeve or media to enable."
        ),
    )
    if st.button("Apply bulk action"):
        if not selected:
            st.warning("No rows selected.")
        elif action == "set_manual_label" and target == "all":
            st.warning(
                "Set Target to sleeve or media before using set_manual_label."
            )
        elif action == "set_manual_label" and not str(manual_label).strip():
            st.warning("Pick a manual condition label first.")
        else:
            with _conn(db_path) as conn:
                for qid in selected:
                    row = conn.execute(
                        "SELECT assigned_label,llm_verdict,model_pred_label FROM queue WHERE queue_row_id=?",
                        (int(qid),),
                    ).fetchone()
                    if action == "accept_llm":
                        final_label = str(
                            row["llm_verdict"] or row["model_pred_label"] or ""
                        ).strip()
                        applied_action = "accept_llm"
                    elif action == "keep_assigned":
                        final_label = str(row["assigned_label"] or "").strip()
                        applied_action = "keep_assigned"
                    elif action == "set_manual_label":
                        final_label = str(manual_label).strip()
                        applied_action = "manual_set"
                    else:
                        final_label = ""
                        applied_action = "skip"
                    conn.execute(
                        """
                        UPDATE queue
                        SET human_action=?, final_label=?, updated_at=datetime('now')
                        WHERE queue_row_id=?
                        """,
                        (applied_action, final_label, int(qid)),
                    )
                conn.commit()
            st.success(f"Updated {len(selected)} row(s).")

    st.markdown("**Auto policy actions**")
    if st.button("Apply auto-decision to filtered rows"):
        where_apply = list(base_where)
        params_apply: list[object] = list(base_params)
        where_apply.append("COALESCE(human_action, '') = ''")
        where_apply.append("COALESCE(auto_decision, '') = 'proposed_grade'")
        with _conn(db_path) as conn:
            n = conn.execute(
                """
                UPDATE queue
                SET human_action='auto_apply',
                    final_label=COALESCE(auto_final_label, ''),
                    updated_at=datetime('now')
                WHERE """
                + " AND ".join(where_apply),
                tuple(params_apply),
            ).rowcount
            conn.commit()
        st.success(f"Applied auto decisions to {int(n)} row(s).")

    if st.button("Clear auto decisions for filtered rows"):
        where_clear = list(base_where)
        params_clear: list[object] = list(base_params)
        with _conn(db_path) as conn:
            n = conn.execute(
                """
                UPDATE queue
                SET auto_decision='',
                    auto_final_label='',
                    auto_decision_score=NULL,
                    auto_decision_reason='',
                    auto_decision_source='',
                    auto_policy_version='',
                    updated_at=datetime('now')
                WHERE """
                + " AND ".join(where_clear),
                tuple(params_clear),
            ).rowcount
            conn.commit()
        st.success(f"Cleared auto decision fields for {int(n)} row(s).")

    with st.expander("Row text preview (legacy)"):
        rid = st.number_input(
            "queue_row_id",
            min_value=1,
            value=int(df["queue_row_id"].iloc[0]),
        )
        row = df[df["queue_row_id"] == int(rid)]
        if not row.empty:
            st.write(row.iloc[0]["text"])


def page_commit(db_path: Path) -> None:
    st.header("4) Commit patches")
    label_patches = Path(st.text_input("label_patches.jsonl path", value=str(DEFAULT_PATCHES)))
    preview_csv = Path(
        st.text_input(
            "Commit preview CSV",
            value="grader/reports/label_audit_commit_preview.csv",
        )
    )
    if st.button("Write label patches", type="primary"):
        try:
            out = commit_queue_to_label_patches(
                db_path=db_path,
                label_patches_path=label_patches,
                temp_csv_path=preview_csv,
            )
            st.success(json.dumps(out, indent=2))
        except Exception as e:
            st.error(str(e))


def page_export(db_path: Path) -> None:
    st.header("5) Export / Handoff")
    out = Path(
        st.text_input(
            "Reviewed export CSV",
            value="grader/reports/label_audit_reviewed_export.csv",
        )
    )
    if st.button("Export reviewed rows"):
        n = export_reviewed_to_csv(db_path, out)
        st.success(f"Exported {n} rows to {out}")
    st.markdown(
        """
Guided command handoff:

```bash
uv run python -m grader.src.eval.label_audit_commit_patches --db grader/reports/label_audit_queue.sqlite
uv run python -m grader.src.pipeline --config grader/configs/grader.yaml
uv run python -m grader.src.models.baseline --config grader/configs/grader.yaml
```
        """.strip()
    )


def main() -> None:
    st.set_page_config(page_title="Label Audit App", layout="wide")
    st.title("LLM-Assisted Label Audit")
    db_path = Path(
        st.sidebar.text_input("Queue DB", value=str(DEFAULT_DB))
    )
    page = st.sidebar.radio(
        "Page",
        (
            "Setup / Queue Build",
            "Run LLM",
            "Review",
            "Commit",
            "Export / Handoff",
        ),
    )
    if page == "Setup / Queue Build":
        page_setup(db_path)
    elif page == "Run LLM":
        page_run_llm(db_path)
    elif page == "Review":
        page_review(db_path)
    elif page == "Commit":
        page_commit(db_path)
    else:
        page_export(db_path)


if __name__ == "__main__":
    main()
