#!/usr/bin/env python3
"""Validate changelog fragments before CI or doc builds.

This script checks that ``upcoming_changes/`` entries use the expected
Towncrier filename format and that ``towncrier build --draft`` succeeds.

Usage::

    python scripts/validate-changelog.py

Exit 0 when all checks pass, 1 otherwise.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

CHANGELOG_DIR = Path("upcoming_changes")
SKIP_FILES = {"README.rst", "AGENTS.md"}
VALID_TYPES = (
    "new",
    "bugfix",
    "doc",
    "deprecation",
    "enhancements",
    "api",
    "maintenance",
)
FILENAME_PATTERN = re.compile(rf"^\d+\.({'|'.join(VALID_TYPES)})\.rst$")


def run_towncrier_draft() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["towncrier", "build", "--draft"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False, (
            "towncrier executable not found. Install towncrier in the current "
            "environment and re-run this script."
        )

    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    return result.returncode == 0, output


def validate_fragment_filenames() -> list[str]:
    warnings = []
    for path in sorted(CHANGELOG_DIR.glob("*.rst")):
        if path.name in SKIP_FILES:
            continue
        if not FILENAME_PATTERN.match(path.name):
            warnings.append(
                f"  {path.as_posix()}: does not match pattern <number>.<type>.rst"
            )
    return warnings


def main() -> int:
    failed_checks = 0

    print("Validating changelog fragments in upcoming_changes/...")

    towncrier_ok, towncrier_output = run_towncrier_draft()
    if towncrier_ok:
        print("towncrier --draft: OK")
    else:
        failed_checks += 1
        print("towncrier --draft: FAILED", file=sys.stderr)
        if towncrier_output:
            print(towncrier_output, file=sys.stderr)
        else:
            print(
                "towncrier did not produce output. Run 'towncrier build --draft' "
                "manually for more details.",
                file=sys.stderr,
            )

    filename_warnings = validate_fragment_filenames()
    if filename_warnings:
        failed_checks += 1
        print(
            f"Fragment filename validation: {len(filename_warnings)} warning(s):",
            file=sys.stderr,
        )
        print(
            "Rename each file to upcoming_changes/<number>.<type>.rst, where "
            f"<type> is one of: {', '.join(VALID_TYPES)}.",
            file=sys.stderr,
        )
        for warning in filename_warnings:
            print(warning, file=sys.stderr)
    else:
        count = sum(
            1 for path in CHANGELOG_DIR.glob("*.rst") if path.name not in SKIP_FILES
        )
        print(
            "Fragment filename validation: "
            f"All {count} fragments match expected pattern."
        )

    if failed_checks:
        print(f"{failed_checks} check(s) failed.", file=sys.stderr)
        return 1

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
