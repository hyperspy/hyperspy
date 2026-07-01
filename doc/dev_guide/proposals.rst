.. _proposals-label:

HyperSpy Proposals
==================

HyperSpy uses a formal proposal process for significant changes to the
ecosystem. Proposals are submitted as pull requests to the
`hyperspy/hyperspy-proposals <https://github.com/hyperspy/hyperspy-proposals>`_
repository. See its
`README <https://github.com/hyperspy/hyperspy-proposals#readme>`_
for full details on when proposals are required, how to submit, how review
works, and what happens after acceptance.

For more information on AI-assisted development, see :ref:`coding_with_ai-label`.

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
