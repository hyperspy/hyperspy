

.. _coding_with_ai-label:

Coding with AI assistants
=========================

HyperSpy accepts contributions produced with the help of AI assistants,
as long as contributors review, understand, and take full responsibility
for the code they submit. All contributions — regardless of how they
were authored — must pass the same quality bar.

Your AI-assisted learning path
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This page covers AI governance: disclosure, project selection, and the
skill-building progression.  Before editing any files, your AI tool must
verify the development environment is ready.

**Open AGENTS.md in your AI tool**

  Direct your AI tool to the root
  `AGENTS.md <https://github.com/hyperspy/hyperspy/blob/main/AGENTS.md>`_
  with the instruction: *"Read this file, then run the checks in
  the AI Agent Setup section."*  Your tool will verify the editable
  install, activate pre-commit hooks, and confirm lint passes — or
  tell you what's missing and how to fix it.

**Read the rest of this page for AI governance**

  The sections below cover the ``Assisted-by:`` convention, how to
  choose a good first project, the skill-building progression, and
  what reviewers will expect from your contributions.

**Learn to verify AI output**

  AI tools make mistakes.  You are responsible for every line you
  submit, and to catch errors you need to understand the conventions
  the tool is following.  This is not busywork — it is what makes
  you effective as an AI supervisor and, eventually, as a reviewer
  of other AI-assisted PRs.

  - :ref:`Testing <testing-label>` — your AI will generate tests.
    Learn what makes a test trustworthy: the axis convention, the
    ``pytest.approx()`` pattern for floating-point comparisons, and
    why array dimensions should all differ in test fixtures.
  - The axis convention — documented in the root
    `AGENTS.md <https://github.com/hyperspy/hyperspy/blob/main/AGENTS.md>`_
    and the :ref:`User Guide <user_guide>`.  HyperSpy reverses
    dimensions relative to NumPy; understanding this is essential to
    verify any code that touches signal data or axes.
  - :ref:`Coding style <coding_style-label>` — the conventions your
    AI tool is expected to follow.  Reading this section before your
    first PR gives you a baseline to judge whether the tool is
    actually following them.
  - :ref:`Writing documentation <writing_documentation-label>` —
    when your PR adds public API or modifies behavior visible to users.

**Deepen as your PRs grow**

  Domain-specific sections of the developer guide exist for good
  reason.  When your PR touches plotting, lazy signals, extensions,
  or performance-critical paths, the corresponding guide pages will
  help you ask your AI tool the right questions and spot the right
  problems:

  - :ref:`Plotting <plotting-label>`
  - :doc:`/dev_guide/lazy_computations`
  - :ref:`Writing extensions <writing_extensions-label>`
  - :doc:`/dev_guide/speeding_up_code`
  - :ref:`Lazy imports <lazy_import-label>`

  The full developer guide is there to be read — the more you
  understand, the more effectively you can direct your AI tool and
  review contributions from others.

What counts as AI-assisted development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HyperSpy maintains ``AGENTS.md`` files so that AI tools can learn
and follow project conventions.  Copying code from a generic chatbot
that has no access to these files produces code that violates
HyperSpy conventions and places disproportionate burden on reviewers.

**Use project-aware tools**
  Coding agents and editor assistants that can browse the codebase,
  read ``AGENTS.md``, respect lint rules, run tests, and apply the
  completion checklist.  Examples: OpenCode, Claude Code, Cursor
  with project rules configured.

**Why generic chatbots don't work**
  A web-based chatbot (ChatGPT, Claude.ai, etc.) cannot see the
  repository.  It has no knowledge of HyperSpy's axis convention,
  signal methods, test patterns, or lint rules.  The resulting code
  will almost always violate project conventions, and the reviewer
  must catch every violation manually — defeating the purpose of
  the lightweight review practices described in this guide.

  Reviewers may reject contributions that consistently fail to
  follow project conventions, regardless of how they were produced.

The contributor is always responsible for the code.  If a tool
cannot follow HyperSpy's ``AGENTS.md`` conventions, the contributor
carries the full burden of learning and applying them manually.

If you cannot explain why a change was made, what it does, and how
it works, do not submit it.

Disclosing AI assistance
^^^^^^^^^^^^^^^^^^^^^^^^^^

When AI tools assist in producing a commit, HyperSpy requires the
``Assisted-by:`` git trailer.  This follows the convention adopted by
the Linux Kernel.

**Format**::

    Assisted-by: <tool-name>:<model-version>

**Examples**::

    Assisted-by: Claude:claude-sonnet-4-6
    Assisted-by: GitHub Copilot:gpt-4o
    Assisted-by: Cursor:claude-3.7-sonnet

The ``<tool-name>`` is the code-generating tool you are directly
interacting with — your editor's AI assistant (Claude Code, Cursor,
GitHub Copilot, OpenCode, etc.).  If you use an orchestration layer
or wrapper on top of that tool, attribute to the underlying tool,
not the wrapper.

**Why not Co-authored-by?**
  The ``Co-authored-by:`` trailer signals human co-authorship and is
  recognised by GitHub's contribution tracking.  Several AI tools add
  it to commits by default (Claude Code, VS Code with Copilot,
  Cursor).  HyperSpy **rejects** commits that use ``Co-authored-by:``
  for AI tools — a pre-commit hook blocks them, and reviewers may ask
  you to amend.

  If your AI tool adds ``Co-authored-by:`` automatically, strip it
  before committing.  Use ``Assisted-by:`` instead.

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

Every practice in this guide — small PRs, ``Assisted-by:`` disclosure,
change maps, reviewer discretion, and the skill-building path below —
is designed for a single purpose: **long-term sustainability**.  AI
makes code production nearly free; without these practices, reviewer
burnout is inevitable as AI-authored contributions multiply.  With
them, the community can absorb far more contributions while keeping
review quality high.

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

Building skills for the long term
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sustainability of AI-assisted development depends on contributors
who can direct, verify, and defend AI output — not just produce it.
Each contributor who reaches that level becomes a potential reviewer
for other AI-assisted PRs, multiplying the community's capacity.

**Start small — the whole protocol depends on it**

  A new contributor who submits a 2,000-line AI-generated PR breaks
  the feedback loop.  The reviewer must choose: invest hours teaching
  (unsustainable) or reject with minimal feedback (the contributor
  learns nothing).  Neither outcome builds the community.

  A 50-line PR, in contrast, is small enough for the reviewer to
  explain what the AI got wrong and why.  The contributor internalizes
  the convention, the next PR is better, and after a few cycles they
  begin reviewing others' work.  Starting small is not a restriction —
  it is the mechanism that makes the whole system compound.

**Skill-building progression**

  This is not a prohibition on using AI.  It is a recommended path
  to reach the point where AI amplifies your work without replacing
  your understanding:

  +-------------------+------------------------------------+---------------------------------------+
  | Stage             | What to build                      | AI's role                             |
  +===================+====================================+=======================================+
  | **Onboarding**    | One manual PR following every      | Use AI to explain code you don't      |
  |                   | convention from the checklist.     | understand and to catch mistakes in   |
  |                   | Learn the axis convention, the     | its own output — not to author your   |
  |                   | test patterns, and the review      | first submission.                     |
  |                   | workflow.                          |                                       |
  +-------------------+------------------------------------+---------------------------------------+
  | **Building trust**| Small features where you own the   | Generate tests, documentation, and    |
  |                   | design decisions.  Staged delivery | boilerplate.  You write or tightly    |
  |                   | in reviewable PRs.                 | direct the core logic.                |
  +-------------------+------------------------------------+---------------------------------------+
  | **Trusted**       | Architecture, large features,      | Full AI-assisted workflow.  You       |
  |                   | cross-cutting changes.             | self-regulate quality and review      |
  |                   |                                    | other AI-assisted PRs.                |
  +-------------------+------------------------------------+---------------------------------------+

  The goal is not to avoid AI — it is to reach the point where you
  can direct it, verify its output, and defend every line you submit.
  Teaching one contributor to use AI well creates a reviewer, not
  just a contributor.

**Before leaning on AI for core logic**

  These are not prerequisites for permission.  They are the
  foundations that make you effective — both as an AI-assisted
  contributor and as a reviewer when AI-assisted PRs multiply:

  - You have at least one merged PR where you followed the
    ``AGENTS.md`` checklist manually.
  - You can explain HyperSpy's navigation/signal axis convention
    and predict array shapes from signal type.
  - You know how to run the relevant test suite and interpret a
    failure.
  - You have read the completion checklist in the root
    ``AGENTS.md`` and used it on a manual PR.

  You cannot help another contributor debug AI-generated axis
  handling if you have not wrestled with it yourself.

What makes a review light
^^^^^^^^^^^^^^^^^^^^^^^^^

The goal is for the reviewer to validate *design* and *test quality*
rather than *implementation correctness*. The contributor should
provide enough context that the reviewer never has to discover
intent from raw code.

For every PR, include:

**Change map** (for structural changes)
  Include in the PR description. A concise overview of what changed,
  what moved where, and why. This is not a changelog — it is a
  reviewer-facing map. For example: "``axes.py`` split into
  ``data_axis.py``, ``uniform_axis.py``, ``navigation_axes.py``
  because each axis type had grown into a self-contained concern."

**Test strategy note**
  How the change was tested. For mechanical refactors: "all existing
  tests pass unchanged." For new functionality: what edge cases the
  tests cover.

**Inline comments at non-obvious design choices**
  The root ``AGENTS.md`` checklist requires this.  A reviewer
  should never wonder *why* a particular approach was taken.

Large AI-assisted changes should be broken into reviewable,
well-documented pieces.  No code dumps.

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

The root ``AGENTS.md`` contains an **agent setup** section and a
**completion checklist** in its ``MANUAL`` section.  Before starting
work, agents must verify that ``pre-commit`` hooks are installed and
passing.  Every AI-authored contribution should satisfy all checklist
items before being submitted for review.  The checklist covers code
quality, testing, changelog entries, commit discipline, AI attribution
conventions, inline comment requirements, and HyperSpy-specific
conventions.

When reviewing AI-authored pull requests, verify that the checklist
items have been addressed.

For reviewers
^^^^^^^^^^^^^

When reviewing AI-assisted changes, focus on the substance:

**What to check:**
  - Design rationale — does the approach make sense?
  - Test adequacy — do the tests actually verify the claimed behavior?
  - Spot-checks at representative decision points rather than
    exhaustive line-by-line reading.

**What can be skipped:**
  - Line-by-line correctness when tests provide strong coverage.
  - Mechanical transformations (rename, move, extract) that are
    covered by existing tests.
  - Repeated patterns already validated in a spot-check.

**Investing in new contributors**

  When a new contributor submits AI-assisted code they cannot fully
  explain, fixing it yourself is faster.  But teaching them why it is
  wrong — pointing to the specific convention, demonstrating the
  correct approach — is an investment in sustainability.  That
  contributor's next PR will be better, and eventually they will teach
  the next newcomer.  Multiply this across a growing contributor base
  and the community can absorb far more AI-assisted work without
  drowning in review.

  Concretely:
  - Ask "can you walk me through why this works?" instead of
    rejecting outright.
  - Point to the convention the code violates rather than rewriting
    it silently.
  - If the PR is too large to teach from, ask the contributor to
    split it into smaller pieces — return to "start small" above.

**Actions maintainers may take on AI-assisted contributions**

  Maintainers have discretion to handle AI-assisted contributions
  differently from human-only ones.  They may:

  - Ask the contributor to explain the prompts or process used to
    generate the code, to verify genuine understanding.
  - Suggest alternative prompts rather than inline code changes,
    when the issue is with the generation approach rather than the
    result.

  These are not penalties — they exist to keep review sustainable
  when code production cost drops to near zero.  A contributor who
  provides a clear change map, test strategy, and inline comments
  (as described above) is unlikely to trigger these actions.

**Recognizing review-efficient contributions**

  Some contributions touch many files but require surprisingly light
  review because correctness follows from existing structure or
  behavioral symmetry. These are ideal for AI assistance:

  **Parity implementations**
    Implementing existing behavior for a new context (e.g., making an
    eager-only method work with lazy signals, porting a 1D routine to
    2D). The original implementation and its tests already define
    correctness; review focuses on the delta — data-flow patterns,
    axis handling, deferred computation placement — rather than
    re-validating the algorithm. The contributor should state which
    methods were mirrored and any deliberate deviations.

  **Mechanical refactors**
    Renaming, module extraction, signature updates. When existing
    tests pass unchanged, the reviewer validates the transformation
    strategy and spot-checks representative files.

  **Pattern expansions**
    Adding many similar methods following an established pattern
    (e.g., new model components, I/O format plugins, statistical
    methods). Once the pattern is validated in one instance, the
    remaining additions are a consistency check — the reviewer
    spot-checks boundary conditions rather than each method.

If the contributor has provided a good change map, test strategy
note, and inline comments, the review cost is proportional to design
complexity, not change volume.

This is evolving
^^^^^^^^^^^^^^^^

AI coding tools and best practices are changing fast — we are all
learning continuously.  HyperSpy's policies will evolve with them.
If you have ideas for improving these guidelines, whether from your
own experience, another project's approach, or new research, please
share them through GitHub discussions, issues, or a pull request.
