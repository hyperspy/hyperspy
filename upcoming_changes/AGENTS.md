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

## Common Mistakes (Anti-Patterns)

These are the most common mistakes that get caught by CI — avoid them:

### 1. Wrong cross-reference paths in fragment text
The changelog is rendered by Sphinx, so RST cross-references MUST resolve.
```rst
# ❌ WRONG — wrong module path, non-existent method
Fix :meth:`~.signal.map` crashing on empty data.

# ❌ WRONG — method doesn't exist, typo in name
Fix :meth:`~.signal.BaseSignal.mapp` raising TypeError.

# ✅ CORRECT — full path, actual method name
Fix :meth:`~.signal.BaseSignal.map` incorrectly dropping axis calibration
when ``lazy=True``.
```
**How to verify**: After writing a fragment with a cross-reference, check that
the object path actually exists in the codebase. For `:meth:` references, the
full path is `~.module.ClassName.method_name`. For `:doc:` references, the path
corresponds to a ``.rst`` file under ``doc/``.

### 2. Wrong fragment type
```rst
# ❌ WRONG — "enhancement" (singular) instead of "enhancements"
1234.enhancement.rst

# ❌ WRONG — "fix" is not a valid type
1234.fix.rst

# ✅ CORRECT
1234.bugfix.rst
1234.enhancements.rst
```
Valid types: ``new``, ``bugfix``, ``doc``, ``deprecation``, ``enhancements``, ``api``, ``maintenance``.

### 3. Multi-paragraph fragments for non-``new`` types
Fragments of type ``bugfix``, ``doc``, ``deprecation``, ``enhancements``, ``api``, ``maintenance``
should be a single concise sentence. Only ``new`` fragments may use multiple paragraphs
and code blocks.

### 4. Missing changelog entry for user-facing changes
Every user-facing change (new feature, bugfix, deprecation, API change, enhancement)
requires a fragment. Only purely internal refactoring with identical behaviour can
skip this requirement.

### 5. File naming errors
```rst
# ❌ WRONG — missing issue number before type
.bugfix.rst

# ❌ WRONG — extra dots
1234..bugfix.rst

# ✅ CORRECT
1234.bugfix.rst
```

### Local Validation Before Pushing
Run the validation script before committing changelog changes:

.. code-block:: bash

    python scripts/check-docs.py --quick

This catches filename errors and ``towncrier`` parse failures immediately,
without waiting for CI.

Before pushing, run the full check to also validate Sphinx cross-references:

.. code-block:: bash

    python scripts/check-docs.py
