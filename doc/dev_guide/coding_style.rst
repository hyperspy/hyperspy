.. _ruff: https://docs.astral.sh/ruff
.. _coding_style-label:

Coding style
============

HyperSpy follows the Style Guide for Python Code - these are rules
for code consistency that you can read all about in the `Python Style Guide
<https://www.python.org/dev/peps/pep-0008/>`_. You can use the
`black <https://github.com/psf/black>`_ or `ruff`_ code formatter to automatically
fix the style of your code using pre-commit hooks.

Linting error can be suppressed in the code using the ``# noqa`` marker,
more information in the `ruff documentation <https://docs.astral.sh/ruff/linter/#error-suppression>`_.

Pre-commit hooks
================
Code linting and formatting is checked continuously using `ruff`_ pre-commit hooks.

These can be run locally by using `pre-commit <https://pre-commit.com>`__.
Alternatively, the comment ``pre-commit.ci autofix`` can be added to a PR to fix the formatting
using `pre-commit.ci <https://pre-commit.ci>`_.

Deprecations
============
HyperSpy follows `semantic versioning <https://semver.org>`_ where changes follow such that:

1. MAJOR version when you make incompatible API changes
2. MINOR version when you add functionality in a backward compatible manner
3. PATCH version when you make backward compatible bug fixes

This means that as little, ideally no, functionality should break between minor releases.
Deprecation warnings are raised whenever possible and feasible for functions/methods/properties/arguments,
so that users get a heads-up one (minor) release before something is removed or changes, with a possible
alternative to be used.

A deprecation decorator should be placed right above the object signature to be deprecated:

.. code-block:: python

    @deprecated(since=1.7.0, removal=2.0.0, alternative="bar")
    def foo(self, n):
        return n + 1

    @deprecated_argument(since=1.7.0, removal=2.0.0,name="x", alternative="y")
    def this_property(y):
        return y

This will update the docstring, and print a visible deprecation warning telling
the user to use the alternative function or argument.

These deprecation wrappers are inspired by those in ``kikuchipy``.
