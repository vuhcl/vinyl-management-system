#!/usr/bin/env python3
"""Write .notes/directory_structure.md with a pruned tree of the repo.

Run from the repository root (see OUTPUT path resolution).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".notes" / "directory_structure.md"

IGNORE_NAMES = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "node_modules",
        ".pytest_cache",
        ".mypy_cache",
        "htmlcov",
        ".tox",
        ".ruff_cache",
        ".eggs",
        "dist",
        "build",
        ".specstory",
        ".cursor",
        ".coverage",
        ".DS_Store",
        ".env",
        ".vertex_replicate_state",
    }
)

MAX_DEPTH = 5
MAX_ENTRIES = 40


def skip_path(path: Path) -> bool:
    name = path.name
    if name in IGNORE_NAMES:
        return True
    if name.endswith(".egg-info"):
        return True
    return False


def list_children(d: Path) -> list[Path]:
    try:
        items = sorted(
            d.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
    except PermissionError:
        return []
    return [p for p in items if not skip_path(p)]


def format_tree(d: Path, prefix: str = "", depth: int = 0) -> list[str]:
    lines: list[str] = []
    if depth > MAX_DEPTH:
        return lines
    children = list_children(d)
    if not children:
        return lines
    extra = 0
    if len(children) > MAX_ENTRIES:
        extra = len(children) - MAX_ENTRIES
        children = children[:MAX_ENTRIES]

    for i, p in enumerate(children):
        is_last = i == len(children) - 1 and extra == 0
        branch = "└── " if is_last else "├── "
        next_prefix = prefix + ("    " if is_last else "│   ")
        if p.is_dir():
            lines.append(f"{prefix}{branch}{p.name}/")
            lines.extend(format_tree(p, next_prefix, depth + 1))
        else:
            lines.append(f"{prefix}{branch}{p.name}")
    if extra:
        lines.append(f"{prefix}└── … ({extra} more entries not shown)")
    return lines


def main() -> None:
    excluded = ", ".join(sorted(IGNORE_NAMES))
    header = (
        "# Directory structure (generated)\n\n"
        "This file is a **machine-oriented map** of the repository tree. "
        "For **what each package does** and how components connect, read "
        "[PROJECT_STRUCTURE.md](../PROJECT_STRUCTURE.md) at the repo root.\n\n"
        f"- **Generated:** {date.today().isoformat()}\n"
        f"- **Root:** `{ROOT.name}/`\n"
        "- **Regenerate:** from repo root, "
        "`uv run python scripts/generate_notes_directory_structure.py`\n"
        f"- **Excluded:** {excluded}, `*.egg-info`, and content below depth "
        f"{MAX_DEPTH} from the repo root.\n"
        f"- **Wide directories:** At most {MAX_ENTRIES} entries are listed "
        "per folder; additional files are summarized with `… (N more)`.\n"
        "- **Note:** Gitignored paths may be missing on a fresh clone.\n\n"
        "## Tree\n\n"
        f"```\n{ROOT.name}/\n"
    )
    body = "\n".join(format_tree(ROOT, "", 0))
    footer = "\n```\n"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(header + body + footer, encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
