#!/usr/bin/env python3
"""CI check: require a changelog entry when source files are modified.

HyperSpy uses towncrier for changelog management — every user-facing change
must include a file in ``upcoming_changes/``.  This script fails the CI job
if the PR modifies Python source under ``hyperspy/`` (excluding tests and
the ``upcoming_changes/`` directory itself) without an accompanying
changelog entry.

Usage (in CI)::

    ci-check-changelog.py BASE_REF HEAD_REF

Exit 0 if clean, 1 if changelog entry is missing.
"""

import subprocess
import sys

CHANGELOG_DIR = "upcoming_changes"
"""Directory where changelog entries live."""

SOURCE_DIR = "hyperspy"
"""Directory containing the main library source."""

SKIP_FILES = {f"{CHANGELOG_DIR}/README.rst", f"{CHANGELOG_DIR}/AGENTS.md"}
"""Files in the changelog directory that are infrastructure, not entries."""


def get_changed_files(base: str, head: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", base, head],
        capture_output=True,
        text=True,
        check=True,
    )
    return [f for f in result.stdout.splitlines() if f]


def has_source_changes(files: list[str]) -> bool:
    for f in files:
        if f.startswith(f"{SOURCE_DIR}/") and f.endswith(".py"):
            return True
    return False


def has_changelog_entry(files: list[str]) -> bool:
    for f in files:
        if f.startswith(f"{CHANGELOG_DIR}/") and f not in SKIP_FILES:
            return True
    return False


def check(base: str, head: str) -> int:
    files = get_changed_files(base, head)

    if not has_source_changes(files):
        print("No source changes detected — changelog entry not required.")
        return 0

    if has_changelog_entry(files):
        print("Changelog entry found.")
        return 0

    print(
        "ERROR: Source files modified without a changelog entry.",
        file=sys.stderr,
    )
    print(file=sys.stderr)
    print(
        "Every user-facing change to hyperspy/ must include a file in "
        f"{CHANGELOG_DIR}/.",
        file=sys.stderr,
    )
    print(file=sys.stderr)
    print("Fix instructions:", file=sys.stderr)
    print(file=sys.stderr)
    print(
        f"  1. Create a file named {CHANGELOG_DIR}/<issue>.<type>.rst "
        "where <type> is one of:",
        file=sys.stderr,
    )
    print(
        "     new, bugfix, doc, deprecation, enhancements, api, maintenance.",
        file=sys.stderr,
    )
    print(
        f"  2. See {CHANGELOG_DIR}/README.rst for the expected format.",
        file=sys.stderr,
    )
    print("  3. Commit and push the new file.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} BASE_REF HEAD_REF", file=sys.stderr)
        sys.exit(2)

    sys.exit(check(sys.argv[1], sys.argv[2]))
