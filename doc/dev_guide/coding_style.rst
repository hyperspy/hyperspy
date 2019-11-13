

.. _coding_style-label:

Coding style
============

HyperSpy follows the Style Guide for Python Code - these are just some rules
for consistency that you can read all about in the `Python Style Guide
<https://www.python.org/dev/peps/pep-0008/>`_.

You can check your code with the `pep8 Code Checker
<https://pypi.python.org/pypi/pep8>`_.

Additionally you could use ``autopep8`` to fix the style of you code
automatically. In Linux and MacOS you can run ``autopep8`` automatically after
each commit by adding a ``post-commit`` file to ``.git/hook`` with the following
content:

.. code-block:: bash

    #!/bin/sh
    # From https://gist.github.com/temoto/6183235
    FILES=$(git diff HEAD^ HEAD --name-only --diff-filter=ACM | grep -e '\.py$')
    if [ -n "$FILES" ]; then
        for f in $FILES
        do
            # auto pep8 correction
            autopep8 --in-place -v --aggressive $f
            git add $f
        done
    #git commit -m "Automatic style corrections courtesy of autopep8"
    GIT_COMMITTER_NAME="autopep8" GIT_COMMITTER_EMAIL="autopep8@email.com" git
    commit --author="autopep8 <autopep8@email.com>" -m "Automatic style
    corrections courtesy of autopep8"
