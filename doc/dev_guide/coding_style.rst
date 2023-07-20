

.. _coding_style-label:

Coding style
============

HyperSpy follows the Style Guide for Python Code - these are rules
for code consistency that you can read all about in the `Python Style Guide
<https://www.python.org/dev/peps/pep-0008/>`_. You can use the
`black <https://github.com/psf/black>`_ code formatter to automatically
fix the style of your code. You can install and run ``black`` by:

.. code:: bash

   pip install black
   black /path/to/your/file.py

In Linux and MacOS you can run ``black`` automatically after each commit by
adding a ``post-commit`` file to ``.git/hook/`` with the following content:

.. code-block:: bash

    #!/bin/sh
    # From https://gist.github.com/temoto/6183235
    FILES=$(git diff HEAD^ HEAD --name-only --diff-filter=ACM | grep -e '\.py$')
    if [ -n "$FILES" ]; then
        for f in $FILES
        do
            # black correction
            black -v $f
            git add $f
        done
    #git commit -m "Automatic style corrections courtesy of black"
    GIT_COMMITTER_NAME="black" GIT_COMMITTER_EMAIL="black@email.com" git
    commit --author="black <black@email.com>" -m "Automatic style
    corrections courtesy of black"

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

These deprecation wrappers are inspired by those in ``kikuchipy``
