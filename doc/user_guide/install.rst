Installing Hyperspy
===================

Most users will like to :ref:`install-binary`. At the moment we provide binary installers for Debian/Ubuntu and Windows. If we do not distribute a binary for your platform or installing from binary is not for you it is also very easy to :ref:`install-source` or to    :ref:`install-dev`.


.. _install-binary:
 
Install from a binary
---------------------

    We provide  binary distributions for Debian/Ubuntu Linux and Windows. In Debian and Ubuntu the dependencies are installed automatically. In Windows it is necessary to previously install the required libraries, see :ref:`install-dependencies`

.. _install-source:

Install from source
-------------------
.. Note::
    To date there is not any stable release of Hyperspy and therefore no tar.gz files are provided. Currently the only way to get Hyperspy is to :ref:`install-dev`



To install from source grab a tar.gz release and in Linux/Mac:

.. code-block:: bash

    $ tar -xzf hyperspy.tar.gz
    $ cd hyperspy
    $ python setup.py install

.. _install-dev:

Install the development version
-------------------------------


To get the development version from our git repository you need to install `git <http://git-scm.com//>`_. Then just do:

.. code-block:: bash

    $ git clone https://github.com/hyperspy/hyperspy.git

To install Hyperspy you could proceed like in :ref:`install-source`. However, if you are installing from the development version most likely you will prefer to install Hyperspy using  `pip <http://www.pip-installer.org>`_ development mode: 


.. code-block:: bash

    $ cd hyperspy
    $ pip install -e ./
    
In any case, like when installing from source you must be sure to have all the dependencies installed, see :ref:`install-dependencies`
 
.. _create-debian-binary: 
    
Creating Debian/Ubuntu binaries
-------------------------------

You can create binaries for Debian/Ubuntu running the `release_debian` script

.. code-block:: bash

    $ ./release_debian
    
.. Warning::

    For this to work, the following packages must be installed in your system python-stdeb, debhelper, dpkg-dev and python-argparser are required.
    
Creating windows binaries
-------------------------
    
To create a Windows binary run the `release_windows.bat` script in a windows machine.

.. _install-dependencies:

Installing the required libraries
---------------------------------

Before installing Hyperspy certain libraries have to be installed in the system. The easiest way to install these packages is by installing the `enthough python distribution <http://www.enthought.com/products/epd.php>`_ (EPD) that from version 0.7.1 comes with all the required libraries included by default. Please note that the academic version of EPD is free, `you can get it here. <http://www.enthought.com/products/edudownload.php>`_


Another option in Windows is to install `pythonxy <http://www.pythonxy.com/>`_.

If you use a Debian/Ubuntu binary to install Hyperspy all the dependencies should install automatically. Otherwise you must install the following packages (note that we use the Debian/Ubuntu package names): ``python-numpy``  ``python-scipy`` ``python-matplotlib`` ``ipython`` ``python-mdp`` ``python-netcdf`` ``python-h5py`` ``python-traits`` ``python-traitsui`` ``python-nose`` ``python-opencv`` ``python-chaco``.














