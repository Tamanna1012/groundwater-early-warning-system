"""Shared helper so each EDA script can own one section of docs/EDA_FINDINGS.md
without stepping on sections written by other scripts, and without piling up
duplicates when a script is re-run."""

from __future__ import annotations

from pathlib import Path

FILE_HEADER = (
    "# EDA Findings\n\n"
    "Deeper exploratory analysis beyond the initial data audit. Each section is a "
    "real, computed finding with a stated implication for modelling -- not decoration.\n"
)


def write_section(doc_path: Path, heading: str, body: str) -> None:
    """Insert/replace the `## {heading}` section in doc_path with `body`,
    leaving every other section untouched. Creates the file with the shared
    header if it doesn't exist yet."""
    marker = f"## {heading}"
    existing = doc_path.read_text() if doc_path.exists() else FILE_HEADER

    if marker in existing:
        before, rest = existing.split(marker, 1)
        # rest starts right after the heading text; find where the *next*
        # top-level section begins so we only replace this one section.
        next_marker_pos = rest.find("\n## ")
        after = rest[next_marker_pos:] if next_marker_pos != -1 else ""
        existing = before.rstrip() + "\n\n"
    else:
        existing = existing.rstrip() + "\n\n"
        after = ""

    new_content = existing + f"{marker}\n\n{body.strip()}\n" + (after if after else "")
    doc_path.write_text(new_content)
