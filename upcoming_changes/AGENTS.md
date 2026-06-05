<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-19 | Updated: 2026-05-19 -->

# upcoming_changes

## Purpose
Towncrier news fragments for the next release. Each file corresponds to one PR/issue and is assembled into `CHANGES.rst` at release time.

## Key Files

| File | Description |
|------|-------------|
| `README.rst` | Instructions for creating news fragments |

## For AI Agents

### Working In This Directory

Every user-facing change (new feature, bugfix, deprecation, API break, enhancement) requires a news fragment. Internal/maintenance changes use the `maintenance` type.

**Fragment filename format:** `<issue_number>.<type>.rst`

**Types:**
| Type | When to use |
|------|-------------|
| `new` | New user-facing feature or behaviour |
| `bugfix` | Bug fix |
| `doc` | Documentation improvement |
| `deprecation` | Feature deprecation |
| `enhancements` | Enhancement to existing functionality |
| `api` | Breaking API change |
| `maintenance` | Test suite, packaging, CI changes |

**Example fragment content** (`3601.bugfix.rst`):
```rst
Fix :meth:`~.signal.BaseSignal.map` incorrectly dropping axis calibration
when ``lazy=True`` and the input signal has non-uniform axes.
```

### Do NOT

- Write fragments for changes that have no user impact (e.g. refactoring internals with identical behaviour).
- Include the PR number in the text — Towncrier injects that automatically.
- Write multi-paragraph fragments; one concise sentence is the standard.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
