Developer Guide
===============

This 5-step guide is intended to give people who want to start contributing their 
own tools to HyperSpy a foothold to kick-start the process. This is also the way
to start if you ultimately hope to become a member of the developer team.

We anticipate that many potential contributors and developers will be scientists
who may have a lot to offer in terms of expert knowledge but may have little
experience when it comes to working on a reasonably large open-source project
like HyperSpy. This guide is aimed at you - helping to reduce the barrier to make 
a contribution.

Before you start you should decide which platform (i.e. Linux, Windows, or Mac)
you are going to work in. All are possible and the advice below is the same it's
only the specifics that change.

1. Start using HyperSpy and understand it
-----------------------------------------

The best way to start understanding how HyperSpy works and to build a broad 
overview of the code as it stands is to use it -- so what are you waiting for?

`Download HyperSpy <https://hyperspy.org/download.html>`_

The user-guide also provides a good overview of all the parts of the code that
are currently implemented as well as much information about how everything works 
-- so read it well.

`HyerSpy User-Guide <www.hyperspy.org/hyperspy-doc/current/index.html>`_

For developing the code the home of hyperspy is on github and you'll see that
a lot of this guide boils down to using that platform well. so visit the link
below and poke around the code, issues, and pull requests.

`HyperSpy on Github <https://github.com/hyperspy/hyperspy>`_

it's probably also worth visiting the `Github <https://github.com/>`_ home page
and going through the "boot camp" to get a feel for the terminology.

In brief, to give you a hint on the terminology to search for, the contribution
pattern is:

    1. Setup git/github if you don't have it.
    2. Fork HyperSpy on github.
    3. Checkout your fork on your local machine.
    4. Create a new branch locally where you will make your changes.
    5. Push the local changes to your own github fork.
    6. Create a pull request to the official HyperSpy repository.

Note: You cannot mess up the main HyperSpy project unless you have been promoted
to write access and the dev-team. So when you're starting out be confident to
play, get it wrong, and if it all goes wrong you can always get a fresh install
of HyperSpy!!

PS: If you choose to develop in Windows/Mac you may find `Github Desktop <https://desktop.github.com>` useful.

2. Got a problem? -- ask!
-------------------------

Open source projects are all about community - we put in much effort to make
good tools available to all and most people are happy to help others start out. 
Everyone had to start at some point and the philosophy of these projects 
centres around the fact that we can do better by working together.

Much of the conversation happens in 'public' via online platforms. The main two
forums used by HyperSpy developers are:

`Gitter <https://gitter.im/hyperspy/hyperspy>`_ -- where we host a live chat-room in which people can ask questions and
discuss things in a relatively informal way.

`Github <https://github.com/hyperspy/hyperspy/issues>`_ -- the main repository for the source code also enables issues to be
raised in a way that means they're logged until dealt with. This is also a
good place to make a proposal for some new feature or tool that you want to
work on.


3. Pick your battles
--------------------

Work out what you want to contribute and break it down in to managable chunks.

You may have a very clear idea of what you want to contribute but if you're 
not sure where to start you can always look through the issues and pull requests 
on the `GitHub Page <https://github.com/hyperspy/hyperspy/>`_. You'll find that 
there are many known areas for development in the issues and a number of 
pull-requests are part finished projects just sitting there waiting for a keen
new contributor to come and learn by finishing.

When you've decided what you're going to work on - let people know using the 
online forums!

It may be that someone else is doing something similar and can help, it's also 
good to make sure that those working on related projects are pulling in the 
same direction.

4. Get good habits
------------------

There are 3 key points to get right when starting out as a contributor - keep 
work separated in managable sections, make sure that your code style is good,
and bear in mind that every new function you write will need a test and user
documentation!

Use git and work in managable branches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By now you'll have had a look around GitHub - but why's it so important?

Well GitHub is the public forum in which we manage and discuss development of
the code. More importantly, it enables every developer to utilise Git which is 
an open source "version control" system that you can use on your laptop or
desktop. By version control we mean that you can separate out your contribution
to the code into many versions (called branches) and switch between them easily.
Later you can choose which version you want to have integrated into HyperSpy.

You can learn all about Git `here <www.git-scm.com/about>`_!

The most important thing for you to do is to separate your contributions so that 
each branch is small advancement on the "master" code or on another branch. In
the end each branch will have to be checked and reviewed by someone else before
it can be included - so if it's too big, you will be asked to split it up!

For personal use, before integrating things into the main HyperSpy code, you may
want to use a few branches together. You can do that but just make sure each new
thing has it's own branch! You can merge some together for your personal use.

Diagramatically you should be aiming for something like this:

.. figure:: /user_guide/images/branching_schematic.eps


Get the style right
^^^^^^^^^^^^^^^^^^^

HyperSpy follows the Style Guide for Python Code - these are just some rules for
consistency that you can read all about in the `Python Style Guide <https://www.python.org/dev/peps/pep-0008/>`_.

You can check your code with the `pep8 Code Checker <https://pypi.python.org/pypi/pep8>`_.

Write tests & documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every new function that is writen in to HyperSpy needs to be tested and documented.

Tests -- these are short functions found in hyperspy/tests that call your functions
under some known conditions and check the outputs against known values. They should
depend on as few other features as possible so that when they break we know exactly
what caused it. Writing tests can seem laborious but you'll probaby soon find that
they're very important as they force you to sanity check all you do. When comparing
integers, it's fine to use ``==``. When comparing floats, be sure to use
``assert_almost_equal()``.

Documentation comes in two parts docstrings and user-guide documentation.

Docstrings -- written at the start of a function and give essential information
about how it should be used, such as which arguments can be passed to it and what
the syntax should be.

User-guide Documentation -- A description of the functionality of the code and how
to use it with examples and links to the relevant code.

5. Make your contribution
-------------------------

When you've got a branch that's ready to be incorporated in to the main code of
HyperSpy -- make a pull request on GitHub and wait for it to be reviewed and
discussed.
