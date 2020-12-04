

.. _using_github-label:

Using Git and GitHub
====================

For developing the code, the home of HyperSpy is on 
`GitHub <https://github.com>`_, and you'll see that
a lot of this guide boils down to properly use that platform. So, visit the
following link and poke around the code, issues, and pull requests: `HyperSpy
on GitHub <https://github.com/hyperspy/hyperspy>`_.

It is probably also worth visiting the `github.com <https://github.com/>`_
and to go through the `"boot camp" <https://help.github
.com/categories/bootcamp/>`_ to get a feel for the terminology.

In brief, to give you a hint on the terminology to search for and get
accustomed to, the contribution pattern is:

1. Setup git/github, if you don't have it yet.
2. Fork HyperSpy on GitHub.
3. Checkout your fork on your local machine.
4. Create a new branch locally, where you will make your changes.
5. Push the local changes to your own HyperSpy fork on GitHub.
6. Create a pull request (PR) to the official HyperSpy repository.

.. note::
  You cannot mess up the main HyperSpy project unless you have been
  promoted to write access and the dev-team. So when you're starting out be
  confident to play, get it wrong, and if it all goes wrong, you can always get
  a fresh install of HyperSpy!!

PS: If you choose to develop in Windows/Mac you may find `Github Desktop
<https://desktop.github.com>`_ useful.

Use Git and work in manageable branches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By now you will have had a look around GitHub -- but why is it so important?

Well, GitHub is the public forum in which we manage and discuss development of
the code. More importantly, it enables every developer to use `Git`, which is
an open source "version control" system. By version control, we mean that you
can separate out your contribution to the code into many versions (called
branches) and switch between them easily. Later, you can choose which version
you want to have integrated into HyperSpy. You can learn all about Git at
`git-scm <http://www.git-scm.com/about>`_!

It is very important to separate your contributions so
that each branch is a small advancement on the "master" code or on another
branch. In the end, each branch will have to be checked and reviewed by
someone else before it can be included -- so if it is too big, you will be
asked to split it up!

For personal use, before integrating things into the main HyperSpy code, you
can merge some together for your personal use. However, make sure each new
feature has it's own branch that is contributed through a separate pull
request!

Diagrammatically, you should be aiming for something like this:

.. figure:: images/branching_schematic.png


Semantic versioning and HyperSpy main branches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
HyperSpy versioning follows `semantic versioning <https://semver.org/spec/v2.0.0.html>`_ 
and the version number is therefore a three-part number: MAJOR.MINOR.PATCH.
Each number will change depending on the type of changes according to the following:

- MAJOR increases when making incompatible API changes,
- MINOR increases when adding functionality in a backwards compatible manner, and
- PATCH increases when making backwards compatible bug fixes.

The git repository of HyperSpy has 3 main branches matching the above pattern
and depending on the type of pull request, you will need to base your pull request
one one of the following branch:

- ``RELEASE_next_major`` to change the API in a not backward-compatible fashion,
- ``RELEASE_next_minor`` to add new features and improvement,
- ``RELEASE_next_patch`` for bug fixes.

The ``RELEASE_next_patch`` is merged daily in the ``RELEASE_next_minor`` by the github action
`Nightly Merge <https://github.com/hyperspy/hyperspy/actions>`_.


Changing base branch
^^^^^^^^^^^^^^^^^^^^

If you started your work in the wrong branch (typically on ``RELEASE_next_minor``
instead of ``RELEASE_next_patch`` and you are doing a bug fix), you can change the
base branch using ``git rebase --onto``, like this:

.. code:: bash

   $ git rebase --onto <NEW-BASE-BRANCH> <OLD-BASE-BRANCH> <YOUR-BRANCH>

For example, to rebase the ``bug_fix_branch`` branch from ``RELEASE_next_minor`` onto ``RELEASE_next_patch``:

.. code:: bash

   $ git rebase --onto RELEASE_next_patch RELEASE_next_minor bug_fix_branch

