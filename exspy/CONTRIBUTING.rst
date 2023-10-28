.. _contributing_label:

Contributing
************

exSpy is meant to be a community maintained project. We welcome contributions
in the form of bug reports, documentation, code, feature requests, and more.
In the following we refer to some resources to help you make useful contributions.

Issues
======

The `issue tracker <https://github.com/hyperspy/exspy/issues>`_ can be used to
report bugs or propose new features. When reporting a bug, the following is
useful:

- give a minimal example demonstrating the bug,
- copy and paste the error traceback.

Pull Requests
=============

If you want to contribute to the exSpy source code, you can send us a
`pull request <https://github.com/hyperspy/exspy/pulls>`_. Small bug fixes or
corrections to the user guide are typically a good starting point. But don't
hesitate also for significant code contributions - if needed, we'll help you
to get the code ready to common standards.

Please refer to the
`HyperSpy developer guide <http://hyperspy.org/hyperspy-doc/current/dev_guide/intro.html>`_
in order to get started and for detailed contributing guidelines.

The :doc:`kikuchipy contributors guide <kikuchipy:dev/index>`, another HyperSpy
extension, also is a valuable resource that can get you started and provides useful
guidelines.

Reviewing
---------

As quality assurance, to improve the code, and to ensure a generalized
functionality, pull requests need to be thoroughly reviewed by at least one
other member of the development team before being merged.

Documentation
=============

The exSpy documentation consists of three elements:

- Docstrings following the `numpy standard
  <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_
  that document the functionality of individual methods on `GitHub
  <https://github.com/hyperspy/exspy>`_.
- The `documentation <https://exspy.readthedocs.io>`_ written using `Sphinx
  <https://www.sphinx-doc.org>`_ and hosted on `Read the Docs
  <https://exspy.readthedocs.io>`_. The source is part of the `GitHub repository
  <https://github.com/hyperspy/exspy/tree/main/doc>`_.
- Jupyter notebooks in the `exSpy demos repository
  <https://github.com/hyperspy/exspy-demos>`_ on GitHub that provide tutorials and example
  workflows.

Improving documentation is always welcome and a good way of starting out to learn the GitHub
functionality. You can contribute through pull requests to the respective repositories.

Code style
==========

exSpy follows `Style Guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`_
with `The Black Code style
<https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html>`_.

For `docstrings <https://www.python.org/dev/peps/pep-0257/>`_, we follow the `numpydoc
<https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_ standard.

Package imports should be structured into three blocks with blank lines between
them:

- standard libraries (like ``os`` and ``typing``),
- third party packages (like ``numpy`` and ``hyperspy``),
- and finally ``exspy`` imports.

Writing tests
=============

All functionality in exSpy is tested via the `pytest <https://docs.pytest.org>`_
framework. The tests reside in the ``test`` directory. Tests are short methods that call
functions in exSpy and compare resulting output values with known answers.
Please refer to the `HyperSpy development guide
<https://hyperspy.org/hyperspy-doc/current/dev_guide/testing.html>`_ for further
information on tests.

Releasing a new version
=======================

exSpy versioning follows `semantic versioning <https://semver.org/spec/v2.0.0.html>`_
and the version number is therefore a three-part number: MAJOR.MINOR.PATCH.
Each number will change depending on the type of changes according to the following:

- MAJOR increases when making incompatible API changes,
- MINOR increases when adding functionality in a backwards compatible manner, and
- PATCH increases when making backwards compatible bug fixes.

The process to release a new version that is pushed to `PyPI <https://pypi.org>`_ and
`Conda-Forge <https://conda-forge.org/>`_ is documented in the `Releasing guide
<https://github.com/hyperspy/exspy/blob/main/releasing_guide.md>`_.
