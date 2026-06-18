#!/usr/bin/env python3
"""CI check: scan all commits in a PR for prohibited Co-authored-by AI trailers.

This is the CI counterpart of ``scripts/check-ai-co-author.py`` (the pre-commit
hook).  Pre-commit catches them at commit time, but commits bypassed with
``--no-verify`` will be caught here.  Fails the CI job if any commit in the PR
contains a ``Co-authored-by:`` trailer referencing a known AI tool.

Usage (in CI)::

    ci-check-ai-trailers.py BASE_REF HEAD_REF

Exit 0 if clean, 1 if violations found.
"""

import re
import subprocess
import sys
from typing import List

# Mirror of scripts/check-ai-co-author.py — keep in sync.
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


def get_commit_messages(base: str, head: str) -> List[str]:
    result = subprocess.run(
        ["git", "log", f"{base}..{head}", "--format=%B"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def check_commits(base: str, head: str) -> int:
    lines = get_commit_messages(base, head)
    violations: List[str] = []

    for line in lines:
        if CO_AUTHORED_RE.match(line):
            for pattern in AI_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append(line.strip())
                    break

    if violations:
        print(
            "ERROR: Prohibited Co-authored-by trailers found in this pull request:",
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
            "(Claude Code, VS Code Copilot, Cursor, OpenCode).",
            file=sys.stderr,
        )
        print(file=sys.stderr)
        print("Fix instructions:", file=sys.stderr)
        print(file=sys.stderr)
        print(
            "  1. Rebase your branch and amend every offending commit to replace",
            file=sys.stderr,
        )
        print(
            "     Co-authored-by: <AI tool> with Assisted-by: <tool>:<model>.",
            file=sys.stderr,
        )
        print(
            "  2. Disable auto-injection in your tool's settings "
            "(see doc/dev_guide/coding_with_ai.rst).",
            file=sys.stderr,
        )
        print(
            "  3. Force-push the corrected branch.",
            file=sys.stderr,
        )
        return 1

    print("All commits clean — no prohibited AI Co-authored-by trailers.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} BASE_REF HEAD_REF", file=sys.stderr)
        sys.exit(2)

    sys.exit(check_commits(sys.argv[1], sys.argv[2]))
