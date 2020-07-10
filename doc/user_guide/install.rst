
.. _install-label:

Installing HyperSpy
===================

The easiest way to install HyperSpy in Microsoft Windows is installing the
:ref:`HyperSpy Bundle <hyperspy-bundle>`.

For quick instructions on how to install HyperSpy in Linux, MacOs or Windows
using the `Anaconda Python distribution <http://docs.continuum.io/anaconda/>`_
see the :ref:`anaconda-install` section.

To enable context-menu (right-click) shortcut in a chosen folder, use the
`start_jupyter_cm <https://github.com/hyperspy/start_jupyter_cm>`_ library.

.. warning::

    Since version 0.8.4 HyperSpy only supports Python 3. If you need to install
    HyperSpy in Python 2.7 install HyperSpy 0.8.3.

.. _hyperspy-bundle:

HyperSpy Bundle for Microsoft Windows
-------------------------------------

The easiest way to install HyperSpy in Windows is installing the HyperSpy
Bundle. This is a customised `WinPython <http://winpython.github.io/>`_
distribution that includes HyperSpy, all its dependencies and many other
scientific Python packages.

For details and download links go to https://github.com/hyperspy/hyperspy-bundle

.. _anaconda-install:

Installation in an Anaconda/Miniconda distribution
--------------------------------------------------

Anaconda or Miniconda is recommended for the best performance (numpy is compiled
using the Intel MKL libraries) and the easiest installation. HyperSpy is
packaged in the `conda-forge <https://conda-forge.org/>`_ channel and can be
installed easily using the `conda <https://docs.conda.io/en/latest/>`_ package
manager:

#. Download and install
   `Anaconda <https://store.continuum.io/cshop/anaconda/>`_ if necessary.
   If you are not familiar with Anaconda please refer to their
   `User Guide <https://docs.continuum.io/anaconda/>`_ for details.

#. Then install HyperSpy executing the following 
   `conda <https://docs.conda.io/en/latest/>`_ commands in the
   Anaconda Prompt, Linux/Mac Terminal or Microsoft Windows Command Prompt.
   This depends on your OS and how you have installed Anaconda, see the
   `Anaconda User Guide <https://docs.continuum.io/anaconda/>`_ for
   details.

   .. code-block:: bash

       $ conda install hyperspy -c conda-forge

This will install also install the optional GUI packages ``hyperspy_gui_ipywidgets``
and ``hyperspy_gui_traitsui``. To install hyperspy without the GUI packages, use:


   .. code-block:: bash

       $ conda install hyperspy-base -c conda-forge

.. note::

    Using ``-c conda-forge`` is only necessary when the conda-forge is not
    already added to the conda configuration, see the 
    `conda-forge documentation <https://conda-forge.org/docs/user/introduction.html>`_
    for more details.


Further information
^^^^^^^^^^^^^^^^^^^

When installing packages, ``conda`` will verify that all requirements of `all`
packages installed in an environment are met. This can lead to situations where
a solution for dependencies resolution cannot be resolved or the solution may
include installing old or undesired versions of libraries. The requirements
depend on which libraries are already present in the environment as satisfying
their respective dependencies may be problematic. In such situation, possible
solutions are:

- use Miniconda instead of Anaconda, if you are installing a python
  distribution from scratch: Miniconda installs very few packages so satisfying
  all dependencies is relatively simple.
- install hyperspy in a `new environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.
  The following example illustrates creating a new environment named ``hspy_environment``,
  activating it and installing hyperspy in the new environment.

  .. code-block:: bash

      $ conda create -n hspy_environment
      $ conda activate hspy_environment
      $ conda install hyperspy -c conda-forge

  .. note::

      A consequence of installing hyperspy in a new environment is that you need
      to activate this environment using ``conda activate environment_name`` where
      ``environment_name`` is the name of the environment, however `shortcuts` can
      be created using different approaches:

      - Install `start_jupyter_cm <https://github.com/hyperspy/start_jupyter_cm>`_
        in the hyperspy environment.
      - Install `nb_conda_kernels <https://github.com/Anaconda-Platform/nb_conda_kernels>`_.
      - Create `IPython kernels for different environment <https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments>`_.

To learn more about the Anaconda eco-system:

- Choose between `Anaconda or Miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html#anaconda-or-miniconda>`_?
- Understanding `conda and pip <https://www.anaconda.com/blog/understanding-conda-and-pip>`_.

.. _install-with-pip:

Installation using pip
----------------------

HyperSpy is listed in the `Python Package Index
<http://pypi.python.org/pypi>`_. Therefore, it can be automatically downloaded
and installed  `pip <http://pypi.python.org/pypi/pip>`__. You may need to
install pip for the following commands to run.

To install all hyperspy functionalities, run:

.. code-block:: bash

    $ pip install hyperspy[all]

To install only the strictly required dependencies and limited functionalities,
use:

.. code-block:: bash

    $ pip install hyperspy

See the following list of selectors to select the installation of optional
dependencies required by specific functionalities:

* ``learning`` to install required libraries for some machine learning features,
* ``gui-jupyter`` to install required libraries to use the
  `Jupyter widgets <http://ipywidgets.readthedocs.io/en/stable/>`_
  GUI elementsm
* ``gui-traitsui`` to install required libraries to use the GUI elements based
  on `traitsui <http://docs.enthought.com/traitsui/>`_,
* ``mrcz`` to install the mrcz plugin,
* ``speed`` install optional libraries that speed up some functionalities,
* ``tests`` to install required libraries to run HyperSpy's unit tests,
* ``build-doc`` to install required libraries to build HyperSpy's documentation,
* ``dev`` to install all the above,
* ``all`` to install all the above expect the development requirements
  (``tests``, ``build-doc`` and ``dev``).

For example:

.. code-block:: bash

    $ pip install hyperspy[learning, gui-jupyter]

Finally, be aware that HyperSpy depends on a number of libraries that usually 
need to be compiled and therefore installing HyperSpy may require development
tools installed in the system. If the above does not work for you remember that
the easiest way to install HyperSpy is
:ref:`using Anaconda or Miniconda <anaconda-install>`.

.. _install-dev:

Install development version
---------------------------

Clone the hyperspy repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To get the development version from our git repository you need to install `git
<http://git-scm.com//>`_. Then just do:

.. code-block:: bash

    $ git clone https://github.com/hyperspy/hyperspy.git

.. Warning::

    When running hyperspy from a development version, it can happen that the
    dependency requirement changes in which you will need to keep this
    this requirement up to date (check dependency requirement in ``setup.py``)
    or run again the installation in development mode using ``pip`` as explained
    below.

Installation in a Anaconda/Minconda distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install the runtime and development dependencies requirements using conda:

.. code-block:: bash

    $ conda install hyperspy-base -c conda-forge --only-deps
    $ conda install hyperspy-dev -c conda-forge

The package ``hyperspy-dev`` will install the development dependencies required
for testing and building the documentation.

From the root folder of your hyperspy repository (folder containing the 
``setup.py`` file) run `pip <http://www.pip-installer.org>`_ in development mode:

.. code-block:: bash

    $ pip install -e . --no-deps

Installation in other (non-system) Python distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the root folder of your hyperspy repository (folder containing the 
``setup.py`` file) run `pip <http://www.pip-installer.org>`_ in development mode:

.. code-block:: bash

    $ pip install -e .[dev]

All required dependencies are automatically installed by pip. If you don't want
to install all dependencies and only install some of the optional dependencies,
use the corresponding selector as explained in the :ref:`install-with-pip` section

..
    If using Arch Linux, the latest checkout of the master development branch
    can be installed through the AUR by installing the `hyperspy-git package
    <https://aur.archlinux.org/packages/hyperspy-git/>`_

.. _create-debian-binary:

Installation in a system Python distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using a system Python distribution, it is recommanded to install the
dependencies using your system package manager.

From the root folder of your hyperspy repository (folder containing the 
``setup.py`` file) run `pip <http://www.pip-installer.org>`_ in development mode.

.. code-block:: bash

    $ pip install -e --user .[dev]

Creating Debian/Ubuntu binaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can create binaries for Debian/Ubuntu from the source by running the
`release_debian` script

.. code-block:: bash

    $ ./release_debian

.. Warning::

    For this to work, the following packages must be installed in your system
    python-stdeb, debhelper, dpkg-dev and python-argparser are required.

