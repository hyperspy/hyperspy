

.. _coding_with_ai-label:

Coding with AI assistants
=========================

HyperSpy welcomes contributions authored with the help of AI coding
assistants, and maintains infrastructure to help them work effectively.
This section covers the conventions that both human and AI contributors
should follow when working on the repository.

AGENTS.md files
^^^^^^^^^^^^^^^

HyperSpy uses hierarchical ``AGENTS.md`` files to provide structured
context to AI assistants. These files are **auto-generated** from the
codebase structure and should not be edited by hand except as noted below.

Each ``AGENTS.md`` file consists of:

- A **generated header** with a timestamp — auto-updated on regeneration.
- **Auto-generated content**: directory purpose, key files, subdirectories
  table, AI agent instructions, and dependency information.
- A **MANUAL boundary** (``<!-- MANUAL: Any manually added notes below this
  line are preserved on regeneration -->``).

When to update
""""""""""""""""

**Do not edit generated content.** Changes to generated sections will be
lost on regeneration.

Instead:

- **New files or subdirectories** that change the codebase structure:
  use the project's ``AGENTS.md`` regeneration tooling to regenerate the
  affected files. The regeneration preserves all content below the
  ``MANUAL`` boundary.

- **Persistent notes** (conventions, gotchas, per-directory requirements
  not captured by auto-generation): add them below the ``MANUAL`` boundary
  line. These notes survive regeneration.

- **Typos or factual errors in generated content**: trigger a
  regeneration instead of fixing them manually.

Agent completion checklist
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The root ``AGENTS.md`` contains a **completion checklist** in its
``MANUAL`` section. Every AI-authored contribution should satisfy all
items on that checklist before being submitted for review. The checklist
covers code quality, testing, changelog entries, commit discipline,
and HyperSpy-specific conventions.

When reviewing AI-authored pull requests, verify that the checklist
items have been addressed.
