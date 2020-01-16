﻿Introduction
=============

This guide is intended to give people who want to start contributing
to HyperSpy a foothold to kick-start the process.

We anticipate that many potential contributors and developers will be
scientists who may have a lot to offer in terms of expert knowledge but may
have little experience when it comes to working on a reasonably large
open-source project like HyperSpy. This guide is aimed at you -- helping to
reduce the barrier to make a contribution.

Getting started
---------------


1. Start using HyperSpy and understand it
-----------------------------------------

Probably you would not be interested in contributing to HyperSpy, if you were 
not already a user, but, just in case: the best way to start understanding how
HyperSpy works and to build a broad overview of the code as it stands is to
use it -- so what are you waiting for? `Install HyperSpy
<http://hyperspy.org/hyperspy-doc/current/user_guide/install.html>`_.

The `HyperSpy User-Guide <http://www.hyperspy.org/hyperspy-doc/current/index
.html>`_ also provides a good overview of all the parts of the code that
are currently implemented as well as much information about how everything
works -- so read it well.


2. Got a problem? -- ask!
-------------------------

Open source projects are all about community -- we put in much effort to make
good tools available to all and most people are happy to help others start out.
Everyone had to start at some point and the philosophy of these projects
centres around the fact that we can do better by working together.

Much of the conversation happens in 'public' via online platforms. The main two
forums used by HyperSpy developers are:

`Gitter <https://gitter.im/hyperspy/hyperspy>`_ -- where we host a live
chat-room in which people can ask questions and discuss things in a relatively
informal way.

`Github <https://github.com/hyperspy/hyperspy/issues>`_ -- the main repository
for the source code also enables issues to be raised in a way that means
they're logged until dealt with. This is also a good place to make a proposal
for some new feature or tool that you want to work on.


3. Contribute -- yes you can!
-----------------------------

You don't need to be a professional programmer to contribute to HyperSpy.
Indeed, there are many ways to contribute:

1. Just by asking a question in our
   `Gitter chat room <https://gitter.im/hyperspy/hyperspy>`_
   instead of sending a private email to the developers you are contributing to
   HyperSpy. Once you get more familiar with HyperSpy,  it will be awesome if 
   you could help others with their questions.
2. Issues reported in the
   `issues tracker <https://github.com/hyperspy/hyperspy/issues>`_
   are precious contributions.
3. `Pull request <https://github.com/hyperspy/hyperspy/pulls>`_ reviews are
   essential for the sustainability of open development software projects
   and HyperSpy is no exception. Therefore, reviews are highly appreciated.
   While you may need a good familiarity with
   the HyperSpy code base to review complex contributions,
   you can start by reviewing simpler ones such as documentation
   contributions or simple bug fixes.
4. Last but not least, you can contribute code in the form of
   documentation, bug fixes, enhancements or new features. That is the main
   topic of the rest of this guide.

4. Contributing code
--------------------

You may have a very clear idea of what you want to contribute, but if you're
not sure where to start, you can always look through the issues and pull
requests on the `GitHub Page <https://github.com/hyperspy/hyperspy/>`_.
You'll find that there are many known areas for development in the issues
and a number of pull-requests are partially finished projects just sitting 
there waiting for a keen new contributor to come and learn by finishing.

The documentation (let it be the docstrings,
guides or the website) is always in need of some care. Besides,
contributing to HyperSpy's documentation is a very good way to get
familiar with GitHub.

When you've decided what you're going to work on -- let people know using the
online forums! It may be that someone else is doing something similar and
can help.; it is
also good to make sure that those working on related projects are pulling in
the same direction.

There are 3 key points to get right when starting out as a contributor:

1. Work out what you want to contribute and break it down in to manageable
   chunks. Use :ref:`Git branches <using_github-label>` to keep work separated
   in manageable sections.
2. Make sure that your :ref:`code style <coding_style-label>` is good.
3. Bear in mind that every new function you write will need 
   :ref:`tests <testing-label>` and
   :ref:`user documentation <writing_documentation-label>`!
