"""Streamlit pages and router for label-audit review."""

from __future__ import annotations

from .lib import *  # noqa: F403

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
