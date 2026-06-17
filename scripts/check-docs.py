#!/usr/bin/env python3
"""Pre-CI documentation validation — run what CI runs, locally.

Checks
------
1. **Fragment filenames** — verifies ``upcoming_changes/*.rst`` use the
   ``<number>.<type>.rst`` pattern.
2. **Towncrier draft** — runs ``towncrier build --draft`` to catch parse
   errors.
3. **Sphinx build** — runs ``sphinx-build -b html -d _build/doctrees
   -W --keep-going . _build/html`` in the ``doc/`` directory.
   Gallery examples are skipped by default for speed; use ``--full`` to
   restore them.
4. **Linkcheck** (optional) — runs ``sphinx-build -b linkcheck`` in the
   ``doc/`` directory.
5. **Doctest** (optional) — runs ``sphinx-build -b doctest`` in the
   ``doc/`` directory.

Usage
-----

.. code-block:: bash

    python scripts/check-docs.py            # fragments + sphinx (fast)
    python scripts/check-docs.py --quick    # fragments only
    python scripts/check-docs.py --full     # fragments + sphinx + gallery
    python scripts/check-docs.py --full --linkcheck  # full CI simulation

Exit 0 when all checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHANGELOG_DIR = REPO_ROOT / "upcoming_changes"
DOC_DIR = REPO_ROOT / "doc"
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


def _run(
    cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None
) -> tuple[bool, str]:
    """Run a command and return ``(success, combined_stdout_stderr)``."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=cwd,
            env=env,
        )
    except FileNotFoundError:
        return False, f"Command not found: {cmd[0]}"

    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    return result.returncode == 0, output


def _fragment_count() -> int:
    return sum(1 for p in CHANGELOG_DIR.glob("*.rst") if p.name not in SKIP_FILES)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_fragment_filenames() -> list[str]:
    """Validate fragment filenames match ``<number>.<type>.rst``."""
    warnings: list[str] = []
    for path in sorted(CHANGELOG_DIR.glob("*.rst")):
        if path.name in SKIP_FILES:
            continue
        if not FILENAME_PATTERN.match(path.name):
            warnings.append(
                f"  {path.name}: does not match pattern <number>.<type>.rst"
            )
    return warnings


def check_towncrier() -> tuple[bool, str]:
    """Run ``towncrier build --draft``."""
    ok, output = _run(["towncrier", "build", "--draft"], cwd=REPO_ROOT)
    if not ok and not output:
        output = (
            "towncrier did not produce output. "
            "Run 'towncrier build --draft' manually for more details."
        )
    return ok, output


def _sphinx_cmd(builder: str) -> list[str]:
    """Return a cross-platform sphinx-build command for ``builder``."""
    return [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        builder,
        "-d",
        "_build/doctrees",
        ".",
        f"_build/{builder}",
    ]


def check_sphinx(*, full: bool = False) -> tuple[bool, str]:
    """Run the CI-equivalent Sphinx doc build.

    By default sets ``HYPERSPY_FAST_CHECK=1`` so ``conf.py`` skips gallery
    execution.  Pass ``full=True`` to run the complete build.
    """
    env = {**os.environ}
    if not full:
        env["HYPERSPY_FAST_CHECK"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "html",
        "-d",
        "_build/doctrees",
        "-W",
        "--keep-going",
        ".",
        "_build/html",
    ]
    return _run(cmd, cwd=DOC_DIR, env=env)


def check_linkcheck() -> tuple[bool, str]:
    """Run ``sphinx-build -b linkcheck`` in the doc directory."""
    return _run(_sphinx_cmd("linkcheck"), cwd=DOC_DIR)


def check_doctest() -> tuple[bool, str]:
    """Run ``sphinx-build -b doctest`` in the doc directory."""
    return _run(_sphinx_cmd("doctest"), cwd=DOC_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Pre-CI documentation validation — run what CI runs, locally."),
    )
    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help=(
            "Fast check only: fragment filenames and towncrier draft "
            "(skip sphinx build)."
        ),
    )
    parser.add_argument(
        "--full",
        "-f",
        action="store_true",
        help=(
            "Run the full sphinx build including gallery example execution "
            "(slow — equivalent to CI)."
        ),
    )
    parser.add_argument(
        "--linkcheck",
        "-l",
        action="store_true",
        help="Also check external links (slow, requires network).",
    )
    parser.add_argument(
        "--doctest",
        "-d",
        action="store_true",
        help="Also run doctests in documentation source files.",
    )
    return parser


def _print_failure(label: str, output: str = "") -> None:
    print(f"  {label}: FAILED", file=sys.stderr)
    if output:
        for line in output.splitlines():
            stripped = line.strip()
            if stripped:
                print(f"    {stripped}", file=sys.stderr)


def main() -> int:
    args = make_parser().parse_args()
    failed = 0

    # ------------------------------------------------------------------
    # Always-run checks (fast)
    # ------------------------------------------------------------------
    print("check-docs: Validating changelog fragments...")

    # 1. Towncrier draft
    ok, output = check_towncrier()
    if ok:
        print("  towncrier --draft: OK")
    else:
        failed += 1
        _print_failure("towncrier --draft", output)

    # 2. Fragment filenames
    warnings = check_fragment_filenames()
    if warnings:
        failed += 1
        print(
            f"  Fragment filenames: {len(warnings)} warning(s):",
            file=sys.stderr,
        )
        print(
            "Rename each file to upcoming_changes/<number>.<type>.rst, where "
            f"<type> is one of: {', '.join(VALID_TYPES)}.",
            file=sys.stderr,
        )
        for w in warnings:
            print(w, file=sys.stderr)
    else:
        print(f"  Fragment filenames: All {_fragment_count()} fragments match pattern.")

    # ------------------------------------------------------------------
    # Sphinx build (CI-equivalent, skipped for --quick)
    # ------------------------------------------------------------------
    if args.quick:
        print("\ncheck-docs: Skipping Sphinx build (--quick).")
    else:
        if args.full:
            print(
                "\ncheck-docs: Building documentation "
                "(full CI-equivalent build, including gallery)..."
            )
        else:
            print(
                "\ncheck-docs: Building documentation "
                "(skipping gallery examples for speed — use --full for CI parity)..."
            )
        print(
            "  Command: cd doc && python -m sphinx -b html -d _build/doctrees "
            "-W --keep-going . _build/html"
        )
        ok, output = check_sphinx(full=args.full)
        if ok:
            print("  Sphinx build: OK")
        else:
            failed += 1
            _print_failure("Sphinx build", output)

    # ------------------------------------------------------------------
    # Optional: linkcheck
    # ------------------------------------------------------------------
    if args.linkcheck:
        print("\ncheck-docs: Checking external links...")
        ok, output = check_linkcheck()
        if ok:
            print("  linkcheck: OK")
        else:
            failed += 1
            _print_failure("linkcheck", output)

    # ------------------------------------------------------------------
    # Optional: doctest
    # ------------------------------------------------------------------
    if args.doctest:
        print("\ncheck-docs: Running doctests...")
        ok, output = check_doctest()
        if ok:
            print("  doctest: OK")
        else:
            failed += 1
            _print_failure("doctest", output)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if failed:
        print(f"\n{failed} check(s) failed.", file=sys.stderr)
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
