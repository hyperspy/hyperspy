Installing Hyperspy
===================

The easiest way to install Hyperspy in all the platforms but Ubuntu Linux is by installing Python (see :ref:`install-dependencies`) and then :ref:`install-with-python-installers`. We also provide binary installers for Debian/Ubuntu and Windows. If we do not distribute a binary for your platform or installing from binary is not for you it is also very easy to :ref:`install-source` or to :ref:`install-dev`.

Quick instructions to install Hyperspy in Windows
-------------------------------------------------

It follows a recipe to install hyperspy in Windows. For more options and details read the rest of the documentation.

#. Download and install `EPD. <http://www.enthought.com/products/http://enthought.com/products/epd.php>`_ EPD is reccomended for the best performance (it is compiled using Intel MKL libraries) and the easiest intallation (all the required libraries are incluided). The academic license is free and it can be obtained `here. <http://www.enthought.com/products/edudownload.php>`_
#. (Maybe) Restart the computer
#. Download and install Hyperspy from the `Download sections <http://hyperspy.org/download.html>`_


.. _install-with-python-installers:

Install using Python installers
-------------------------------

Since version 4.1 Hyperspy is listed in the `Python Package Index <http://pypi.python.org/pypi>`_. Therefore, it can be automatically downloaded and installed using `setuptools <http://pypi.python.org/pypi/setuptools>`_, `distribute <http://pypi.python.org/pypi/distribute>`_ or (our favourite) `pip <http://pypi.python.org/pypi/pip>`_. However, depending on your Python distribution, you might need to install at least one of these packages manually. An extra advantage of using the python installers is that they should install the required libraries automatically.

Install using `pip`:

.. code-block:: bash

    $ pip install hyperspy

Install using `distribute` or `setuptools`:

.. code-block:: bash

    $ easy_install hyperspy

.. _install-binary:
 
Install from a binary
---------------------

We provide  binary distributions for Ubuntu Linux and Windows (`see the Downloads section of the website <http://hyperspy.org/download.html>`_). In Ubuntu the dependencies are installed automatically. In Windows it is necessary to previously install the required libraries, see :ref:`install-dependencies`. To install easily in other platforms see :ref:`install-with-python-installers`
    

.. _install-source:

Install from source
-------------------

.. _install-released-source:

Released version
^^^^^^^^^^^^^^^^

To install from source grab a tar.gz release and in Linux/Mac (requires to :ref:`install-dependencies` manually):

.. code-block:: bash

    $ tar -xzf hyperspy.tar.gz
    $ cd hyperspy
    $ python setup.py install
    
You can also use a Python installer, in which case the required libraries will also be automatically installed, e.g pip.

.. code-block:: bash

    $ pip install hyperspy.tar.gz

.. _install-dev:

Development version
^^^^^^^^^^^^^^^^^^^


To get the development version from our git repository you need to install `git <http://git-scm.com//>`_. Then just do:

.. code-block:: bash

    $ git clone https://github.com/hyperspy/hyperspy.git

To install Hyperspy you could proceed like in :ref:`iinstall-released-source`. However, if you are installing from the development version most likely you will prefer to install Hyperspy using  `pip <http://www.pip-installer.org>`_ development mode: 


.. code-block:: bash

    $ cd hyperspy
    $ pip install -e ./
    
In any case, you must be sure to have all the dependencies installed, see :ref:`install-dependencies`
 
.. _create-debian-binary: 
    
Creating Debian/Ubuntu binaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can create binaries for Debian/Ubuntu from the source by running the `release_debian` script

.. code-block:: bash

    $ ./release_debian
    
.. Warning::

    For this to work, the following packages must be installed in your system python-stdeb, debhelper, dpkg-dev and python-argparser are required.
    

.. _install-dependencies:

Installing the required libraries
---------------------------------

.. Warning::

    Read at least up to the second paragraph of this instruction before taking any action
    
    
Before installing Hyperspy Python and the following libraries be installed in the system: numpy, scipy, matplotlib, ipython, traits and traitsui. For full functionality it is reccomended to also install h5py, mdp and scikit-learn.

In Windows and Mac the easiest way to install these packages is by installing the `enthough python distribution <http://www.enthought.com/products/epd.php>`_ (EPD) that from version 0.7.1 comes with all the required libraries included by default. Please note that the academic version of EPD is free, `you can get it here. <http://www.enthought.com/products/epd_free.php>`_ .

If you use an Ubuntu binary to install Hyperspy all the dependencies should install automatically.















