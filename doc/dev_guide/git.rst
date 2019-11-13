

.. _using_github-label:

Using Git and GitHub
====================

For developing the code the home of HyperSpy is on github and you'll see that
a lot of this guide boils down to using that platform well. So visit the
following link and poke around the code, issues, and pull requests: `HyperSpy
on Github <https://github.com/hyperspy/hyperspy>`_.

It's probably also worth visiting the `Github <https://github.com/>`_ home page
and going through the `"boot camp" <https://help.github
.com/categories/bootcamp/>`_ to get a feel for the terminology.

In brief, to give you a hint on the terminology to search for, the contribution
pattern is:

1. Setup git/github if you don't have it.
2. Fork HyperSpy on github.
3. Checkout your fork on your local machine.
4. Create a new branch locally where you will make your changes.
5. Push the local changes to your own github fork.
6. Create a pull request (PR) to the official HyperSpy repository.

.. note::
  You cannot mess up the main HyperSpy project unless you have been
  promoted to write access and the dev-team. So when you're starting out be
  confident to play, get it wrong, and if it all goes wrong you can always get
  a fresh install of HyperSpy!!

PS: If you choose to develop in Windows/Mac you may find `Github Desktop
<https://desktop.github.com>`_ useful.

Use git and work in manageable branches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By now you'll have had a look around GitHub - but why's it so important?

Well GitHub is the public forum in which we manage and discuss development of
the code. More importantly, it enables every developer to utilise Git which is
an open source "version control" system that you can use on your laptop or
desktop. By version control we mean that you can separate out your contribution
to the code into many versions (called branches) and switch between them
easily. Later you can choose which version you want to have integrated into
HyperSpy.

You can learn all about Git `here <http://www.git-scm.com/about>`_!

The most important thing for you to do is to separate your contributions so
that each branch is small advancement on the "master" code or on another
branch. In the end each branch will have to be checked and reviewed by
someone else before it can be included - so if it's too big, you will be
asked to split it up!

For personal use, before integrating things into the main HyperSpy code, you
may want to use a few branches together. You can do that but just make sure
each new thing has it's own branch! You can merge some together for your
personal use.

Diagrammatically you should be aiming for something like this:

.. figure:: user_guide/images/branching_schematic.png
