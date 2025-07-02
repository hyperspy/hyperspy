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

Using generate_ai_context Function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HyperSpy provides a convenient function to generate AI context files directly from Python:

.. code-block:: python

    import hyperspy.api as hs
    
    # Generate basic context as a string
    context = hs.generate_ai_context()
    
    # Generate full context with web content
    full_context = hs.generate_ai_context(include_optional=True)
    
    # Save context directly to a file
    hs.generate_ai_context(output_file="hyperspy_context.txt")
    
    # Save full context to a file
    hs.generate_ai_context(include_optional=True, output_file="hyperspy_full_context.txt")

**Requirements:**

The function requires the ``llms_txt`` package to be installed:

.. code-block:: bash

    pip install llms_txt

If ``llms_txt`` is not installed, the function will raise an ``ImportError`` with installation instructions.

**When to use llms.txt vs generated context:**

- **Use llms.txt directly** for repository-aware AI tools, quick reference, or when AI systems can access external links
- **Use generated context files** for chat-based AI interfaces, offline scenarios, deep analysis tasks, or custom AI workflows

**Generated context files provide:**

1. **Expanded explanations**: More detailed descriptions of key concepts and workflows
2. **Usage guidance**: Explicit instructions for AI systems on how to help HyperSpy users effectively  
3. **Code patterns**: Concrete examples of common import conventions and usage patterns
4. **Self-contained**: No dependency on external links being accessible to the AI system
5. **Customizable**: Can be modified or extended for specific use cases

Example Usage Scenarios
^^^^^^^^^^^^^^^^^^^^^^^^

**Repository-aware AI Integration**: The ``llms.txt`` file provides context automatically when coding with AI tools that read repository files. No additional setup needed.

**Chat-based AI Development Session**:

.. code-block:: python

    import hyperspy.api as hs
    
    # Generate context and copy to clipboard or AI chat
    context = hs.generate_ai_context()
    # Copy 'context' to your AI chat interface

**Custom AI Assistant**:

.. code-block:: python

    import hyperspy.api as hs
    
    # Generate comprehensive context for AI applications
    hs.generate_ai_context(include_optional=True, output_file="complete_hyperspy_context.txt")


All necessary information is included without requiring external links.


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
