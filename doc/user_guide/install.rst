
.. _install-label:

Installing HyperSpy
===================

The easiest way to install HyperSpy in Microsoft Windows is installing the
:ref:`HyperSpy Bundle <hyperspy-bundle>`.

For quick instructions on how to install HyperSpy in Linux, MacOs or Windows
using the `Anaconda Python distribution <http://docs.continuum.io/anaconda/>`_
see  :ref:`quick-anaconda-install`.

Those experienced with Python may like to
:ref:`install-with-python-installers` or :ref:`install-source`.

.. warning::

    Since version 0.8.4 HyperSpy only supports Python 3. If you need to install
    HyperSpy in Python 2.7 install HyperSpy 0.8.3.

.. _hyperspy-bundle:

HyperSpy Bundle for Microsoft Windows
-------------------------------------

.. versionadded:: 0.6

The easiest way to install HyperSpy in Windows is installing the HyperSpy
Bundle. This is a customised `WinPython <http://winpython.github.io/>`_
distribution that includes HyperSpy, all its dependencies and many other
scientific Python packages.

For details and download links go to https://github.com/hyperspy/hyperspy-bundle

.. _quick-anaconda-install:

Quick instructions to install HyperSpy using Anaconda (Linux, MacOs, Windows)
-----------------------------------------------------------------------------

Anaconda is recommended for the best performance (it is compiled using Intel
MKL libraries) and the easiest installation. The academic license is free.


#. Download and install
   `Anaconda <https://store.continuum.io/cshop/anaconda/>`_. If you are not
   familiar with Anaconda please refer to their
   `User Guide <https://docs.continuum.io/anaconda/>`_ for
   details.

#. Then install HyperSpy executing the following `conda` commands in the
   Anaconda Prompt, Linux/Mac Terminal or Microsoft Windows Command Prompt.
   (This depends on your OS and how you have installed Anaconda, see the
   `Anaconda User Guide <https://docs.continuum.io/anaconda/>`_) for
   details.

   .. code-block:: bash

       $ conda install hyperspy -c conda-forge

#.  (optional) Since HyperSpy v1.3 the
    `traitsui GUI elements <https://github.com/hyperspy/hyperspy_gui_traitsui>`_
    are not installed automatically (but the
    `Jupyter GUI elements <https://github.com/hyperspy/hyperspy_gui_ipywidgets>`_
    are). To install them:

    .. code-block:: bash

        $ conda install hyperspy-gui-traitsui -c conda-forge

.. note::
    Since version 0.8.4 HyperSpy only supports Python 3. If you need to
    install HyperSpy in Python 2.7 install version 0.8.3:

    .. code-block:: bash

        $ conda install traitsui
        $ pip install --upgrade hyperspy==0.8.3-1

To enable context-menu (right-click) startup in a chosen folder, install
`start_jupyter_cm <https://github.com/hyperspy/start_jupyter_cm>`_. (Currently
only available for Gnome and Windows, not MacOS.)


For more options and details read the rest of the documentation.


.. _install-with-python-installers:

Install using Python installers
-------------------------------

HyperSpy is listed in the `Python Package Index
<http://pypi.python.org/pypi>`_. Therefore, it can be automatically downloaded
and installed  `pip <http://pypi.python.org/pypi/pip>`_. You may need to
install pip for the following commands to run.

Install using `pip`:

.. code-block:: bash

    $ pip install hyperspy

.. warning::
    Since version 0.8.4 HyperSpy only supports Python 3. If you need to
    install HyperSpy in Python 2.7 install version 0.8.3:

    .. code-block:: bash

        $ pip install --upgrade hyperspy==0.8.3-1


pip installs automatically the strictly required libraries. However, for full
functionality you may need to install some other dependencies. To install with
full functionality:


.. code-block:: bash

    $ pip install hyperspy[all]

Alternatively you can select the extra functionalities required:

* ``learning`` to install required libraries for some machine learning features.
* ``gui-jupyter`` to install required libraries to use the
  `Jupyter widgets <http://ipywidgets.readthedocs.io/en/stable/>`_
  GUI elements.
* ``gui-traitsui`` to install required libraries to use the GUI elements based
  on `traitsui <http://docs.enthought.com/traitsui/>`_
* ``test`` to install required libraries to run HyperSpy's unit tests.
* ``mrcz`` to install the mrcz plugin.
* ``doc`` to install required libraries to build HyperSpy's documentation.
* ``speed`` install optional libraries that speed up some functionalities.

For example:

.. code-block:: bash

    $ pip install hyperspy[learning, gui-jupyter]

See also :ref:`install-dependencies`.

Finally, be aware that HyperSpy depends on a
number of libraries that usually need to be compiled and therefore installing
HyperSpy may require development tools. If the above does not work for you
remember that the easiest way to install HyperSpy is
:ref:`using Anaconda <quick-anaconda-install>`.


.. _install-binary:

Install from a binary
---------------------

We provide  binary distributions for Windows (`see the
Downloads section of the website <http://hyperspy.org/download.html>`_). To
install easily in other platforms see :ref:`install-with-python-installers`


.. _install-source:

Install from source
-------------------

.. _install-released-source:

Released version
^^^^^^^^^^^^^^^^

To install from source grab a tar.gz release and in Linux/Mac (requires to
:ref:`install-dependencies` manually):

.. code-block:: bash

    $ tar -xzf hyperspy.tar.gz
    $ cd hyperspy
    $ python setup.py install

You can also use a Python installer, e.g.

.. code-block:: bash

    $ pip install hyperspy.tar.gz

.. _install-dev:

Development version
^^^^^^^^^^^^^^^^^^^


To get the development version from our git repository you need to install `git
<http://git-scm.com//>`_. Then just do:

.. code-block:: bash

    $ git clone https://github.com/hyperspy/hyperspy.git

To install HyperSpy you could proceed like in :ref:`install-released-source`.
However, if you are installing from the development version most likely you
will prefer to install HyperSpy using  `pip <http://www.pip-installer.org>`_
development mode:


.. code-block:: bash

    $ cd hyperspy
    $ pip install -e ./

All required dependencies are automatically installed by pip. However, for
extra functionality you may need to install some extra dependencies, see
:ref:`install-dependencies`. Note the pip installer requires root to install,
so for Ubuntu:

.. code-block:: bash

    $ cd hyperspy
    $ sudo pip install -e ./

With development mode setup.py generates or updates git post-checkout hook,
which will cleanup the cythonized c files, cythonize it again and run
```build_ext --inplace``` after the next checkout.


..
    If using Arch Linux, the latest checkout of the master development branch
    can be installed through the AUR by installing the `hyperspy-git package
    <https://aur.archlinux.org/packages/hyperspy-git/>`_

.. _create-debian-binary:

Creating Debian/Ubuntu binaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can create binaries for Debian/Ubuntu from the source by running the
`release_debian` script

.. code-block:: bash

    $ ./release_debian

.. Warning::

    For this to work, the following packages must be installed in your system
    python-stdeb, debhelper, dpkg-dev and python-argparser are required.


.. _install-dependencies:

Installing the required libraries
---------------------------------


In addition to the libraries that are automatically installed when installing
HyperSpy using ``pip`` (see :ref:`install-with-python-installers`), if HyperSpy
is going to be installed from  source, Cython is also required. Also, to
compile the documentation sphinxcontrib-napoleon and sphinx_rtd_theme are
required.

In case some of the required libraries are not automatically installed when
installing from source in a conda environment, these can be obtained beforehand
by installing and removing hyperspy from that environment;

.. code-block:: bash
    $ conda install hyperspy
    $ conda remove hyperspy
    $ sudo pip install -e ./

.. _known-issues:

Known issues
------------

Windows
^^^^^^^

* If HyperSpy fails to start in Windows try installing the Microsoft Visual
  before reporting a bug.

* Concerning older installations with the "Hyperspy here" context menus: Due to
  a `Python bug <http://bugs.python.org/issue13276>`_ sometimes uninstalling
  HyperSpy does not uninstall the "Hyperspy here" entries in the context menu.
  Please run the following code in a Windows Terminal (command line prompt)
  with administrator rights to remove the entries manually:

  .. code-block:: bash

    $ uninstall_hyperspy_here


* If HyperSpy raises a MemoryError exception:

  * Install the 64bit version if you're using the 32bit one and you are running
    HyperSpy in a 64bit system.
  * Increase the available RAM by closing other applications or physically
    adding more RAM to your computer.
