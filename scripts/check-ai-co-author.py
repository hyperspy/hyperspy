#!/usr/bin/env python3
"""Commit-msg hook: reject Co-authored-by trailers referencing AI tools.

HyperSpy uses ``Assisted-by:`` for AI attribution.  ``Co-authored-by:`` is
reserved for human co-authors.  Several AI tools (Claude Code, VS Code with
Copilot, Cursor) add ``Co-authored-by:`` to commits by default — this hook
blocks them before the commit is created.

To use this hook, install pre-commit with the ``commit-msg`` stage::

    pre-commit install --hook-type commit-msg

This hook is registered in ``.pre-commit-config.yaml`` and runs on every
``git commit`` after installation.
"""

import re
import sys
from pathlib import Path
from typing import List

# Patterns that match AI tools in a Co-authored-by trailer line.
# Case-insensitive; word-boundary anchored where applicable.
AI_PATTERNS: List[str] = [
    r"claude",
    r"copilot",
    r"cursor",
    r"\bcodex\b",
    r"gemini",
    r"\baider\b",
    r"\bcline\b",
    r"windsurf",
    r"\bopencode\b",
    r"anthropic",
    r"openai",
    r"\bgpt\b",
    r"\bqwen\b",
    r"deepseek",
    r"mistral",
    r"\bamp\b",  # Amazon Q Developer
]

CO_AUTHORED_RE = re.compile(r"^Co-authored-by:", re.IGNORECASE)


def check_commit_message(filepath: str) -> int:
    """Return 0 if OK, 1 if prohibited trailers found."""
    path = Path(filepath)
    if not path.exists():
        # Pre-commit may pass a non-existent path on empty commits
        return 0

    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()

    violations: List[str] = []
    for line in lines:
        if CO_AUTHORED_RE.match(line):
            for pattern in AI_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(line.strip())
                    break

    if violations:
        print(
            "ERROR: Prohibited Co-authored-by trailers found:",
            file=sys.stderr,
        )
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        print(file=sys.stderr)
        print(
            "HyperSpy uses Assisted-by: for AI tool attribution, not Co-authored-by:.",
            file=sys.stderr,
        )
        print(
            "Several AI tools add Co-authored-by: by default "
            "(Claude Code, VS Code Copilot, Cursor).",
            file=sys.stderr,
        )
        print("Please strip it and use instead:", file=sys.stderr)
        print(file=sys.stderr)
        print(
            "  Assisted-by: <tool-name>:<model-version>",
            file=sys.stderr,
        )
        print(
            "  Example: Assisted-by: Claude:claude-sonnet-4-6",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(check_commit_message(sys.argv[1]))
