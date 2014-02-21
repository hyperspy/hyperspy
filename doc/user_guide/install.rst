Installing Hyperspy
===================

For the easiest way to install Hyperspy in Windows 
:ref:`read this <quick-windows-install>`.

For the easiest way to install Hyperspy in MacOs 
:ref:`read this <quick-mac-install>`.

The easiest way to install Hyperspy in Ubuntu Linux is by downloading and
installing the deb file from the `Download section
<http://hyperspy.org/download.html>`_.

For installing in any other platform you can
:ref:`install-with-python-installers` or :ref:`install-source`. 

.. _quick-windows-install:

Quick instructions to install Hyperspy in Windows
-------------------------------------------------

Since version 0.6 Hyperspy is distributed in Windows with all the required
libraries and nothing else is required. 


.. _quick-mac-install:

Quick instructions to install Hyperspy in MacOs
-------------------------------------------------

#. Download and install `Canopy. <https://www.enthought.com/products/canopy/>`_
   Canopy/EPD is recommended for the best performance (it is compiled
   using Intel MKL libraries) and the easiest intallation (all the required
   libraries are included). The academic license is free. Since version 0.5
   Hyperspy runs (with limited functionality) in Canopy Free.

#. Open a new terminal and type: `easy_install hyperspy`

For more options and details read the rest of the documentation.


.. _install-with-python-installers:

Install using Python installers
-------------------------------
.. WARNING::
   Although it is possible to install hyperspy in Windows using the
   instructions of this section, installing using the Windows installer is
   recommended in this platform because the Python installers do not create
   entries in the ``Start Menu`` or the ``Context Menu``

Since version 4.1 Hyperspy is listed in the `Python Package Index
<http://pypi.python.org/pypi>`_. Therefore, it can be automatically downloaded
and installed using `distribute <http://pypi.python.org/pypi/distribute>`_ or
(our favourite) `pip <http://pypi.python.org/pypi/pip>`_. Depending on your
Python distribution, you might need to install at least one of these packages
manually.

Install using `pip`:

.. code-block:: bash

    $ pip install hyperspy

Install using `distribute` or `setuptools`:

.. code-block:: bash

    $ easy_install hyperspy

In any case, you must be sure to have all the dependencies installed, see
:ref:`install-dependencies`


.. _install-binary:
 
Install from a binary
---------------------

We provide  binary distributions for Ubuntu Linux and Windows (`see the
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

To install Hyperspy you could proceed like in :ref:`iinstall-released-source`.
However, if you are installing from the development version most likely you
will prefer to install Hyperspy using  `pip <http://www.pip-installer.org>`_
development mode: 


.. code-block:: bash

    $ cd hyperspy
    $ pip install -e ./
    
In any case, you must be sure to have all the dependencies installed, see
:ref:`install-dependencies`. Note the pip installer requires root to install,
so for Ubuntu:

.. code-block:: bash

    $ cd hyperspy
    $ sudo pip install -e ./

In addition there as a bug with the IPython version packaged in Ubuntu 12.10,
which can be fixed by upgrading it through pip:

.. code-block:: bash

    $ sudo pip install --upgrade ipython
 
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
    
    
When installing Hyperspy using Python installers or from source the Python
programming language and the following libraries must be installed in the
system: numpy, scipy, matplotlib (>= 1.2), ipython, traits and traitsui. For
full functionality it is recommended to also install h5py, mdp and
scikit-learn. In Windows Hyperspy uses the Ipython's QtConsole nd therefore Qt
and PyQt or PySide are also required.


In Debian/Ubuntu you can install the libraries as follows:

.. code-block:: bash

    $ sudo apt-get install python-numpy python-matplotlib ipython python-traits
    python-traitsui python-h5py python-mdp python-scikits-learn python-nose

.. _known-issues:

Known issues
------------

Windows
^^^^^^^

* Hyperspy does not work with an standard installation of the full version of 
  EPD 7.3-2 due to a problem of compilation in the h5py. The solution is to 
  uninstall Hyperspy and EPD and install a newer version of Hyperspy for
  Windows that, since version 0.6, does not require the installation of 
  any other Python libraries.
* If Hyperspy fails to start in Windows try installing the Microsoft Visual 
  C++ 2008 redistributable packages (
  `64 bit <http://www.microsoft.com/download/en/details.aspx?id=15336>`_ 
  or `32 bit <http://www.microsoft.com/download/en/details.aspx?id=29>`_)
  before reporting a bug.
* In some Windows machines an error is printed at the end of the installation
  and the entries in the context menu and the Start Menu are not installed 
  properly. In most cases the problem can be solved by restarting the computer
  and reinstalling Hyperspy.
* Due to a `Python bug <http://bugs.python.org/issue13276>`_ sometimes uninstalling
  Hyperspy does not uninstall the "Hyperspy Here" entries in the context menu.
  Please run the following code in a Windows Terminal with administrator rights 
  to remove the entries manually:
  
  .. code-block:: bash

    $ uninstall_hyperspy_here
* If Hyperspy raise a MemoryError exceptions:

  * Install the 64bit if you're using the 32bit one and you are running
    Hyperspy in a 64bit system.
  * Increase the available RAM but closing other applications or physically
    adding more RAM to your computer.

    















