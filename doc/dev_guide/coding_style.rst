

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
adding a ``post-commit`` file to ``.git/hook`` with the following content:

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
