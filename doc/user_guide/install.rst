
.. _install-label:

Installing HyperSpy
===================

The easiest way to install HyperSpy is to use the
:ref:`HyperSpy Bundle <hyperspy-bundle>`, which is available on Windows, MacOS
and Linux.

Alternatively, hyperspy can be installed in an existing python distribution,
read the :ref:`conda installation <anaconda-install>` and
:ref:`pip installation<install-with-pip>` sections for instructions.

.. note::

    To enable the context-menu (right-click) shortcut in a chosen folder, use
    the `start_jupyter_cm <https://github.com/hyperspy/start_jupyter_cm>`_ tool.

.. note::

    If you want to be notified about new releases, please *Watch (Releases only)* 
    the `hyperspy repository on GitHub <https://github.com/hyperspy/hyperspy/>`_ 
    (requires a `GitHub account <https://github.com/login>`_).

.. warning::

    Since version 0.8.4 HyperSpy only supports Python 3. If you need to install
    HyperSpy in Python 2.7 install HyperSpy 0.8.3.

.. _hyperspy-bundle:

HyperSpy Bundle
---------------

The HyperSpy bundle is very similar to the Anaconda distribution, and it includes:

  * HyperSpy
  * HyperSpyUI
  * `HyperSpy extensions <https://github.com/hyperspy/hyperspy-extensions-list>`_
  * context menu shortcut (right-click) to Jupyter Notebook, Qtconsole or JupyterLab

For instructions and download links go to https://github.com/hyperspy/hyperspy-bundle

Portable distribution (Windows only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A portable version of the `HyperSpy bundle <https://github.com/hyperspy/hyperspy-bundle>`_
based on the WinPython distribution is also available on Windows.


.. _anaconda-install:

Installation in an Anaconda/Miniconda distribution
--------------------------------------------------

HyperSpy is packaged in the `conda-forge <https://conda-forge.org/>`_ channel
and can be installed easily using the `conda <https://docs.conda.io/en/latest/>`_
package manager.

To install hyperspy run the following from the Anaconda Prompt on Windows or
from a Terminal on Linux and Mac.

   .. code-block:: bash

       $ conda install hyperspy -c conda-forge

This will install also install the optional GUI packages ``hyperspy_gui_ipywidgets``
and ``hyperspy_gui_traitsui``. To install hyperspy without the GUI packages, use:

   .. code-block:: bash

       $ conda install hyperspy-base -c conda-forge

.. note::

    Depending on how Anaconda has been installed, it is possible that the 
    ``conda`` command is not avaible from the Terminal, read the
    `Anaconda User Guide <https://docs.continuum.io/anaconda/>`_ for details.

.. note::

    Using ``-c conda-forge`` is only necessary when the ``conda-forge`` channel
    is not already added to the conda configuration, read the
    `conda-forge documentation <https://conda-forge.org/docs/user/introduction.html>`_
    for more details.

.. note::

    Depending on the packages installed in Anaconda, ``conda`` can be slow and
    in this case ``mamba`` can be used as an alternative of ``conda`` since the
    former is significantly faster. Read the
    `mamba documentation <https://github.com/mamba-org/mamba>`_ for instructions.

Further information
^^^^^^^^^^^^^^^^^^^

When installing packages, ``conda`` will verify that all requirements of `all`
packages installed in an environment are met. This can lead to situations where
a solution for dependencies resolution cannot be resolved or the solution may
include installing old or undesired versions of libraries. The requirements
depend on which libraries are already present in the environment as satisfying
their respective dependencies may be problematic. In such a situation, possible
solutions are:

- use Miniconda instead of Anaconda, if you are installing a python
  distribution from scratch: Miniconda only installs very few packages so satisfying
  all dependencies is simple.
- install hyperspy in a `new environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.
  The following example illustrates how to create a new environment named ``hspy_environment``,
  activate it and install hyperspy in the new environment.

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

* ``learning`` for some machine learning features,
* ``gui-jupyter`` to use the `Jupyter widgets <http://ipywidgets.readthedocs.io/en/stable/>`_
  GUI elements,
* ``gui-traitsui`` to use the GUI elements based on `traitsui <http://docs.enthought.com/traitsui/>`_,
* ``mrcz`` to read mrcz file,
* ``speed`` to speed up some functionalities,
* ``usid`` to read usid file,
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
:ref:`using the HyperSpy bundle <hyperspy-bundle>`.

.. _install-dev:

Rolling release Linux distributions
-----------------------------------

Due to the requirement of up to date versions for dependencies such as *numpy*,
*scipy*, etc., binary packages of HyperSpy are not provided for most linux
distributions and the installation via :ref:`Anaconda/Miniconda <anaconda-install>`
or :ref:`Pip <install-with-pip>` is recommended.

However, packages of the latest HyperSpy release and the related
GUI packages are maintained for the rolling release distributions 
**Arch-Linux** (in the `Arch User Repository 
<https://aur.archlinux.org/packages/python-hyperspy/>`_) (AUR) and 
**openSUSE** (`Community Package <https://software.opensuse.org/package/python-hyperspy>`_)
as ``python-hyperspy`` and ``python-hyperspy-gui-traitsui`` /
``python-hyperspy-gui-ipywidgets``.

A more up-to-date package that contains all updates to be included
in the next minor version release (likely including new features compared to
the stable release) is also available in the AUR as |python-hyperspy-git|_.

.. |python-hyperspy-git| replace:: ``python-hyperspy-git``
.. _python-hyperspy-git: https://aur.archlinux.org/packages/python-hyperspy-git 

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

