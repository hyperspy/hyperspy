

.. _writing_documentation-label:

Writing documentation
=====================

Documentation comes in two parts: docstrings and user-guide documentation.

Docstrings
^^^^^^^^^^

Written at the start of a function and give essential information
about how it should be used, such as which arguments can be passed to it and
what the syntax should be. The docstrings need to follow the `numpy
specification <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT
.rst.txt>`_, as shown in `this example
<https://github.com/numpy/numpy/blob/master/doc/example.py>`_.

As a general rule, any code that is part of the public API (i.e. any function
or class that an end-user might access) should have a clear and comprehensive
docstring explaining how to use it. Private methods that are never intended to
be exposed to the end-user (usually a function or class starting with an underscore)
should still be documented to the extent that future developers can understand
what the function does.

You can check your docstrings follow the convention by using the
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
datasets for this purpose can be found in `hyperspy.datasets.artificial_data`.

Build the documentation
^^^^^^^^^^^^^^^^^^^^^^^

To check the output of what you wrote, you can build
the documentation by running the ``make`` command in the ``hyperspy/doc``
directory. For example ``make html`` will build the whole documentation in
html format. See the make command documentation for more details.

To install the documentation dependencies, run either

.. code-block:: bash

    $ conda install hyperspy-dev

or

.. code-block:: bash

    $ pip install hyperspy[build-doc]


When writing documentation, the Python package `sphobjinv
<https://github.com/bskinn/sphobjinv>`_ can be useful for writing
cross-references. For example, to find how to write a cross-reference to
:py:meth:`hyperspy.signal.BaseSignal.set_signal_type`, use:


.. code-block:: bash

  $ sphobjinv suggest doc/_build/html/objects.inv set_signal_type -st 90


  Name                                                      Score
  ---------------------------------------------------------  -------
  :py:meth:`hyperspy.signal.BaseSignal.set_signal_type`      90
