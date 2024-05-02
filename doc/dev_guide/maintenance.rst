.. _maintenance-label:

Maintenance
===========

GitHub Workflows
^^^^^^^^^^^^^^^^

`GitHub workflows <https://github.com/hyperspy/hyperspy/actions>`_ are used to:

- run the test suite
- build packages and upload to pypi and GitHub release
- build the documentation and check the links (external and cross-references)
- push the development documentation to the ``dev`` folder of https://github.com/hyperspy/hyperspy-doc

Some of these workflow need to access `GitHub "secrets" <https://docs.github.com/en/actions/security-guides/encrypted-secrets>`_,
which are private to the HyperSpy repository. The personal access token ``PAT_DOCUMENTATION`` is set to be able to push
the documentation built to the https://github.com/hyperspy/hyperspy-doc repository.

To reduce the risk that these "secrets" are made accessible publicly, for example, through the
injection of malicious code by third parties in one of the GitHub workflows used in the HyperSpy
organisation, the third party actions (those that are not provided by established trusted parties)
are pinned to the ``SHA`` of a specific commit, which is trusted not to contain malicious code.

Updating GitHub Actions
^^^^^^^^^^^^^^^^^^^^^^^

The workflows in the HyperSpy repository use GitHub actions provided by established trusted parties
and third parties. They are updated regularly by the
`dependabot <https://docs.github.com/en/code-security/dependabot/dependabot-version-updates/configuring-dependabot-version-updates>`_
in pull requests.

When updating a third party action, the action has to be pinned using the ``SHA`` of the commit of
the updated version and the corresponding code changes will need to be reviewed to verify that it
doesn't include malicious code.
