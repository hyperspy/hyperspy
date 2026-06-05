<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-23 | Updated: 2026-05-23 -->

# scripts

## Purpose
Development tooling scripts for pre-commit hooks and CI compliance checks. These scripts enforce project conventions (AI disclosure, changelog entries) and run in automated pipelines.

## Key Files

| File | Description |
|------|-------------|
| `check-ai-co-author.py` | Commit-msg pre-commit hook blocking `Co-authored-by:` trailers from AI tools; suggests `Assisted-by: <tool>:<model>` instead |
| `ci-check-ai-trailers.py` | CI check scanning PR commits for AI `Co-authored-by:` trailers; catches `--no-verify` bypasses of the pre-commit hook |
| `ci-check-changelog.py` | CI check requiring an `upcoming_changes/` entry when `hyperspy/` Python source is modified |

## For AI Agents

### Working In This Directory

- These scripts are standalone tooling — they don't import HyperSpy and don't depend on the project's Python environment beyond the standard library.
- The CI scripts (`ci-check-*.py`) take `BASE_REF HEAD_REF` arguments for `git log` / `git diff` in CI pipelines.
- `check-ai-co-author.py` is run as a `commit-msg` hook via `.pre-commit-config.yaml`.
- When adding new enforcement rules, follow the same pattern: a Python script that exits 0 on pass and prints actionable error messages on failure.

### Do NOT

- Import HyperSpy packages in these scripts — they must work in minimal CI environments.
- Add dependencies beyond the Python standard library.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
