

.. _coding_with_ai-label:

Coding with AI assistants
=========================

HyperSpy accepts contributions produced with the help of AI assistants,
as long as contributors review, understand, and take full responsibility
for the code they submit. All contributions — regardless of how they
were authored — must pass the same quality bar.

The cost asymmetry
^^^^^^^^^^^^^^^^^^

AI makes code production nearly free, but **review is still expensive**.
A contributor can generate thousands of lines in minutes, while a
maintainer still needs hours to review them properly. The review
process also requires far more expertise than code generation.

This guide exists to shift the burden back where it belongs:
contributors (and their AI tools) do the heavy lifting of explanation,
structuring, and verification, so reviewers only need to validate
design and test quality — not discover intent from raw code.

Choosing a project
^^^^^^^^^^^^^^^^^^

The right project size depends on your familiarity with HyperSpy.

+----------------------+------------------------+-----------------------------+
|                      | Small project          | Large / architectural       |
+======================+========================+=============================+
| **New contributor**  | Start here.  Submit    | Open an issue first.        |
|                      | small PRs to build     | Discuss the design with     |
|                      | trust and domain       | maintainers before writing  |
|                      | knowledge.             | code. If greenlit, deliver  |
|                      |                        | in staged, small PRs.       |
+----------------------+------------------------+-----------------------------+
| **Trusted             | Direct PR.  Same       | Pre-code design discussion  |
| contributor**        | quality bar.           | (issue or discussion) to    |
|                      |                        | socialize the approach.     |
|                      |                        | Staged delivery in          |
|                      |                        | reviewable chunks with a    |
|                      |                        | designated co-reviewer.     |
+----------------------+------------------------+-----------------------------+

Large changes should include a migration plan from day one:
deprecation path, backward compatibility, and documentation updates.

What makes a review light
^^^^^^^^^^^^^^^^^^^^^^^^^

The goal is for the reviewer to validate *design* and *test quality*
rather than *implementation correctness*. The contributor should
provide enough context that the reviewer never has to discover
intent from raw code.

For every PR, include:

**Change map** (for structural changes)
  A concise overview of what changed, what moved where, and why.
  This is not a changelog — it is a reviewer-facing map.  For example:
  "``axes.py`` split into ``data_axis.py``, ``uniform_axis.py``,
  ``navigation_axes.py`` because each axis type had grown into a
  self-contained concern."

**Test strategy note**
  What is tested, what is deliberately not tested, and why.
  For mechanical refactors: "all existing tests pass unchanged."

**Inline comments at non-obvious design choices**
  The root ``AGENTS.md`` checklist requires this.  A reviewer
  should never wonder *why* a particular approach was taken.

Large AI-assisted changes should be broken into reviewable,
well-documented pieces.  No code dumps.

Special case: parity and mirroring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some changes implement existing behavior for a new context
(e.g., making an eager-only method work with lazy signals).
These are ideal for AI assistance because:

- The **behavioral specification already exists** — the original
  implementation and its tests define correctness.
- **Tests verify parity directly** — the same test suite
  exercises both paths (e.g., parameterized over eager/lazy).
- **Review is proportional to the delta** — the reviewer
  validates the lazy-specific patterns (dask graph construction,
  ``.compute()`` placement, avoiding eager side-effects)
  and spot-checks representative methods.

The contributor must provide:

- Which methods were mirrored.
- Any deliberate deviations from the original, with rationale.
- Confirmation that tests exercise both eager and lazy paths.

AGENTS.md files
^^^^^^^^^^^^^^^

HyperSpy uses hierarchical ``AGENTS.md`` files to provide structured
context to AI coding assistants.  These files are **auto-generated**
from the codebase structure and should not be edited by hand except
as noted below.

Each ``AGENTS.md`` file consists of:

- A **generated header** with a timestamp — auto-updated on
  regeneration.
- **Auto-generated content**: directory purpose, key files,
  subdirectories table, AI agent instructions, and dependency
  information.
- A **MANUAL boundary** (``<!-- MANUAL: Any manually added notes
  below this line are preserved on regeneration -->``).

When to update
""""""""""""""""

**Do not edit generated content.**  Changes to generated sections
will be lost on regeneration.

Instead:

- **New files or subdirectories** that change the codebase
  structure: use the project's ``AGENTS.md`` regeneration tooling
  to regenerate the affected files.  The regeneration preserves all
  content below the ``MANUAL`` boundary.

- **Persistent notes** (conventions, gotchas, per-directory
  requirements not captured by auto-generation): add them below the
  ``MANUAL`` boundary line.  These notes survive regeneration.

- **Typos or factual errors in generated content**: trigger a
  regeneration instead of fixing them manually.

Agent completion checklist
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The root ``AGENTS.md`` contains a **completion checklist** in its
``MANUAL`` section.  Every AI-authored contribution should satisfy
all items on that checklist before being submitted for review.  The
checklist covers code quality, testing, changelog entries, commit
discipline, inline comment requirements, and HyperSpy-specific
conventions.

When reviewing AI-authored pull requests, verify that the checklist
items have been addressed.

For reviewers
^^^^^^^^^^^^^

When reviewing AI-assisted changes, focus on the substance:

**What to check:**
  - Design rationale — does the approach make sense?
  - Test adequacy — do the tests actually verify the claimed behavior?
  - Spot-checks at representative decision points — one file per
    architectural layer is usually enough.
  - Build and lint — mechanical checks should be automated.

**What can be skipped:**
  - Line-by-line correctness when tests provide strong coverage.
  - Mechanical transformations (rename, move, extract) that are
    covered by existing tests.
  - Repeated patterns already validated in a spot-check.

If the contributor has provided a good change map, test strategy
note, and inline comments, the review cost is proportional to design
complexity, not change volume.
