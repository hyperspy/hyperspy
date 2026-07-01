.. _proposals-label:

HyperSpy Proposals
==================

HyperSpy uses a formal proposal process for significant changes to the ecosystem.
This ensures that the community can review and discuss the **approach** before
investing time in the **implementation**.

When is a proposal required?
----------------------------

The requirement for a proposal depends on the nature of the change and whether
AI tools were used to assist in its creation.

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Contribution type
     - Proposal required?
   * - **AI-assisted, non-trivial**
      - **Yes** — implementation PRs will not be reviewed without an accepted proposal.
   * - **AI-assisted, trivial**
      - No — the PR review is sufficient.
   * - **Human-only, non-trivial**
      - **Recommended** — not mandatory, but strongly encouraged for large changes.
   * - **Human-only, trivial**
      - No — the PR review is sufficient.

For more information on AI-assisted development, see :ref:`coding_with_ai-label`.

How to submit a proposal
------------------------

Proposals are submitted as pull requests to the `hyperspy/hyperspy-proposals <https://github.com/hyperspy/hyperspy-proposals>`_ repository.

1. **Create a markdown file** named ``<PR_NUMBER>-<short-slug>.md`` (e.g., ``0042-hspy-spec.md``).
   Use the PR number you will get when you open the PR — if unsure, use a placeholder and rename after.

2. **Start the file with a YAML metadata block**:

   .. code-block:: yaml

      ---
      proposal: 0042
      title: "hspy-spec — a metadata specification system for HyperSpy 3.0"
      type: Architecture          # Architecture | Feature | Bugfix | Process
      target_branch: hyperspy/hyperspy:RELEASE_next_major
      target_repos: [hyperspy/hyperspy, hyperspy/rosettasciio, hyperspy/hspy-spec]
      status: review              # review | accepted | implemented | superseded
      ai_assisted: true
      created: 2026-06-30
      ---

3. **Write the proposal**. Include the problem being solved, the proposed approach,
   affected repositories, breaking changes, and questions for the community.

4. **Open a PR** to the proposals repository with the markdown file.

How review works
----------------

- **Inline comments**: Reviewers comment on specific lines or paragraphs using standard GitHub PR reviews.
- **Consensus**: A proposal is accepted when maintainers of the affected repositories approve.
- **Deadline**: A feedback deadline (typically 2-4 weeks) should be set in the PR description.

After acceptance
----------------

1. **Merge the proposal PR**. The proposal is now accepted and lives in the repository permanently.
2. **Implement the changes** in the target repositories.
3. **Reference the proposal** in each implementation PR: "Implements `proposal 0042 <https://github.com/hyperspy/hyperspy-proposals/blob/main/0042-hspy-spec.md>`_."
4. **Update proposal status**: After the implementation is merged, update the proposal's metadata to ``status: implemented``.

AI-assisted plan-to-proposal workflow
-------------------------------------

When using an AI agent (like OpenCode) that generates execution plans, you can
transform these plans into human-friendly proposals.

Guidelines for transformation:

- **KEEP**: The core architectural decisions, data structure changes, and API signatures.
- **REMOVE**: Low-level implementation details (e.g., "edit line 42 of file X"),
  internal tool calls, and temporary workspace setup steps.
- **ADD**: High-level rationale, user-facing impact, and alternatives considered.
- **FORMAT**: Use clear headings, tables for complex comparisons, and code blocks for API examples.
- **LENGTH**: Aim for a document that a human can read and understand in 5-10 minutes.

Full details
------------

For the complete documentation on the proposals process, including licensing and
detailed metadata definitions, see the `README of the hyperspy-proposals repository <https://github.com/hyperspy/hyperspy-proposals>`_.
