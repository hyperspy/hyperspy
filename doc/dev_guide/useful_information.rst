.. _useful_information-label:

Useful information
==================

.. _ai-integration-label:

AI Integration Tools
--------------------

HyperSpy includes tools to improve interaction with Large Language Models (LLMs) and AI coding assistants. These tools help AI systems understand HyperSpy's architecture, concepts, and workflows to provide better assistance to users.

llms.txt - LLM-Friendly Documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Following the `llms.txt specification <https://llmstxt.org/>`_, the ``llms.txt`` file in the repository root provides structured information about HyperSpy that AI systems can easily understand:

- **Project overview**: Clear explanation of HyperSpy's purpose and core concepts
- **Key concepts**: Navigation vs signal dimensions, signal types, lazy loading, etc.
- **Documentation links**: Direct links to essential user guides and API references  
- **Extension ecosystem**: Information about domain-specific packages (eXSpy, holoSpy, etc.)
- **Common workflows**: Typical usage patterns and examples

This file provides context automatically for repository-aware AI tools when working in the HyperSpy repository.

**When to use llms.txt vs online context files:**

- **Use llms.txt directly** for repository-aware AI tools and quick reference when working with the source code
- **Use online context files** (llms-ctx.txt) for chat-based AI interfaces or when you need comprehensive context

**The online context files provide:**

1. **Expanded explanations**: More detailed descriptions of key concepts and workflows
2. **Usage guidance**: Explicit instructions for AI systems on how to help HyperSpy users effectively  
3. **Code patterns**: Concrete examples of common import conventions and usage patterns
4. **Self-contained**: No dependency on external links being accessible to the AI system
5. **Version-specific**: Always matches the documentation version you're using

**Access online context files at:**

- Current stable: ``https://hyperspy.org/hyperspy-doc/current/llms-ctx.txt``
- Development: ``https://hyperspy.org/hyperspy-doc/dev/llms-ctx.txt``
- Specific version: ``https://hyperspy.org/hyperspy-doc/v2.4/llms-ctx.txt``

NEP 29 — Recommend Python and Numpy version support
---------------------------------------------------

Abstract
^^^^^^^^

`NEP 29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_
(NumPy Enhancement Proposals) recommends that all projects across the
Scientific Python ecosystem adopt a common “time window-based” policy for
support of Python and NumPy versions. Standardizing a recommendation for
project support of minimum Python and NumPy versions will improve downstream
project planning.

Implementation recommendation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This project supports:

* All minor versions of Python released 42 months prior to the project, and
  at minimum the two latest minor versions.
* All minor versions of ``numpy`` released in the 24 months prior to the project,
  and at minimum the last three minor versions.

In ``setup.py``, the ``python_requires`` variable should be set to the minimum
supported version of Python. All supported minor versions of Python should be
in the test matrix and have binary artifacts built for the release.

Minimum Python and NumPy version support should be adjusted upward on every
major and minor release, but never on a patch release.

Conda-forge packaging
---------------------

The feedstock for the conda package lives in the conda-forge organisation on
github: `conda-forge/hyperspy-feedstock <https://github.com/conda-forge/hyperspy-feedstock>`_.

Monitoring version distribution
-------------------------------

Download metrics are available from pypi and Anaconda cloud, but the reliability
of these numbers is poor for the following reason:

* hyperspy is distributed by other means: the
  `hyperspy-bundle <https://github.com/hyperspy/hyperspy-bundle>`_, or by
  various linux distribution (Arch-Linux, openSUSE)
* these packages may be used by continuous integration of other python libraries

However, distribution of downloaded versions can be useful to identify
issues, such as version pinning or library incompatibilities. Various services
processing the `pypi data <https://packaging.python.org/guides/analyzing-pypi-package-downloads/>`_
are available online:

* `pepy.tech <https://pepy.tech/project/hyperspy>`_
* `libraries.io <https://libraries.io/pypi/hyperspy/usage>`_
* `pypistats.org <https://pypistats.org/packages/hyperspy>`_

HTML Representations
--------------------

For use inside of jupyter notebooks, *html* representations are functions which allow for
more detailed data representations using snippets of populated HTML.

Hyperspy uses *jinja* and extends *dask's* *html* representations in many cases in
line with this PR: https://github.com/dask/dask/pull/8019
