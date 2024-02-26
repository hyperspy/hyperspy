
.. _writing_documentation-label:

Writing documentation
=====================

Documentation comes in two parts: docstrings and user-guide documentation.

Docstrings
^^^^^^^^^^

Written at the start of a function, they give essential information
about how it should be used, such as which arguments can be passed to it and
what the syntax should be. The docstrings need to follow the `numpy
specification <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_, 
as shown in `this example
<https://numpydoc.readthedocs.io/en/latest/example.html>`_.

As a general rule, any code that is part of the public API (i.e. any function
or class that an end-user might access) should have a clear and comprehensive
docstring explaining how to use it. Private methods that are never intended to
be exposed to the end-user (usually a function or class starting with an underscore)
should still be documented to the extent that future developers can understand
what the function does.

To test code of "examples" section in the docstring, run:

.. code:: bash
    
    pytest --doctest-modules --ignore=hyperspy/tests


You can check whether your docstrings follow the convention by using the
``flake8-docstrings`` `extension <https://pypi.org/project/flake8-docstrings/>`_,
like this:

.. code:: bash

   # If not already installed, you need flake8 and flake8-docstrings
   pip install flake8 flake8-docstrings

   # Run flake8 on your file
   flake8 /path/to/your/file.py

   # Example output
   /path/to/your/file.py:46:1: D103 Missing docstring in public function
   /path/to/your/file.py:59:1: D205 1 blank line required between summary line and description


User-guide documentation
^^^^^^^^^^^^^^^^^^^^^^^^

A description of the functionality of the code and
how to use it with examples and links to the relevant code.

When writing both the docstrings and user guide documentation, it is useful to
have some data which the users can use themselves. Artificial
datasets for this purpose can be found in :mod:`~.api.data`.

Example codes in the user guide can be tested using
`doctest <https://docs.python.org/3/library/doctest.html>`_:

.. code:: bash
    
    pytest doc --doctest-modules --doctest-glob="*.rst" -v


Build the documentation
^^^^^^^^^^^^^^^^^^^^^^^

To check the output of what you wrote, you can build
the documentation by running the ``make`` command in the ``hyperspy/doc``
directory. For example ``make html`` will build the whole documentation in
html format. See the ``make`` command documentation for more details.

To install the documentation dependencies, run either

.. code-block:: bash    

    $ conda install hyperspy-dev

or

.. code-block:: bash

    $ pip install hyperspy[doc]


When writing documentation, the Python package `sphobjinv
<https://github.com/bskinn/sphobjinv>`_ can be useful for writing
cross-references. For example, to find how to write a cross-reference to
:meth:`~.api.signals.BaseSignal.set_signal_type`, use:


.. code-block:: bash

  $ sphobjinv suggest doc/_build/html/objects.inv set_signal_type -st 90


  Name                                                      Score
  ---------------------------------------------------------  -------
  :meth:`hyperspy.signal.BaseSignal.set_signal_type`      90

.. _versioned_documentation:

Hosting versioned documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Builds of the documentation for each minor and major release are hosted in the https://github.com/hyperspy/hyperspy-doc
repository and are used by the `version switcher <https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html>`_
of the documentation.

The ``"dev"`` version is updated automatically when pushing on the ``RELEASE_next_minor`` branch and the `"current"` (stable)
version is updated automatically when a tag is pushed.
When releasing a minor and major release, two manual steps are required:

1. in https://github.com/hyperspy/hyperspy-doc, copy the "current" stable documentation to a separate folder named with the corresponding version 
2. update the documentation version switch, in ``doc/_static/switcher.json``:

   - copy and paste the `"current"`` documentation entry
   - update the version in the "current" entry to match the version to be released, e.g. increment the minor or major digit
   - in the newly created entry, update the link to the folder created in step 1.
