"""
Shared ``human_action`` value sets for label-audit SQLite queue tooling.

Keep filter lists in one place so critic / calibrate / backend stay aligned.
"""

from __future__ import annotations

# Rows used as human-reviewed gold for critic / policy calibration (no auto).
REVIEWED_HUMAN_ACTIONS: tuple[str, ...] = (
    "accept_llm",
    "keep_assigned",
    "manual_set",
)

# Same as above plus rows where automation applied a patch without manual click.
REVIEWED_HUMAN_ACTIONS_WITH_AUTO_APPLY: tuple[str, ...] = REVIEWED_HUMAN_ACTIONS + (
    "auto_apply",
)

# ``commit_queue_to_label_patches`` — includes edit path and auto-apply.
COMMIT_QUEUE_HUMAN_ACTIONS: tuple[str, ...] = (
    "accept_llm",
    "accept_edit",
    "manual_set",
    "auto_apply",
)

# Streamlit / queue browser filters (accepted = LLM path + auto-apply).
ACCEPT_LIKE_HUMAN_ACTIONS: tuple[str, ...] = (
    "accept_llm",
    "auto_apply",
)

ACCEPT_OR_REJECT_DECIDED: tuple[str, ...] = ACCEPT_LIKE_HUMAN_ACTIONS + (
    "keep_assigned",
)


def human_action_sql_in_list(actions: tuple[str, ...]) -> str:
    """Comma-separated quoted literals for ``IN (...)`` (trusted action names only)."""
    return ",".join(f"'{a}'" for a in actions)


def reviewed_actions_sql_tuple(
    actions: tuple[str, ...] = REVIEWED_HUMAN_ACTIONS,
) -> str:
    """Alias for :func:`human_action_sql_in_list` (reviewed-row filters)."""
    return human_action_sql_in_list(actions)
