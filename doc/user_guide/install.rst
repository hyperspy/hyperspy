
.. _install-label:

Installing HyperSpy
===================

The easiest way to install HyperSpy is to use the
:ref:`HyperSpy Bundle <hyperspy-bundle>`, which is available on Windows, MacOS
and Linux.

Alternatively, HyperSpy can be installed in an existing python distribution,
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

The `HyperSpy <https://github.com/hyperspy/hyperspy-bundle>`__ bundle is very similar
to the Anaconda distribution, and it includes:

  * HyperSpy
  * HyperSpyUI
  * `HyperSpy extensions <https://github.com/hyperspy/hyperspy-extensions-list>`_
  * context `menu shortcut (right-click) <https://github.com/hyperspy/start_jupyter_cm>`_
    to Jupyter Notebook, Qtconsole or JupyterLab

.. raw:: html

    <div class="text-center">
        <a  class="downloadbutton"
            href="https://github.com/hyperspy/hyperspy-bundle/releases/latest">
                Download HyperSpy-bundle installer
        </a>
    </div>
    <br>

For instructions, see the `HyperSpy bundle <https://github.com/hyperspy/hyperspy-bundle>`__ repository.

Portable distribution (Windows only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A portable version of the `HyperSpy bundle <https://github.com/hyperspy/hyperspy-bundle>`__
based on the WinPython distribution is also available on Windows.

.. _anaconda-install:

Installation using conda
------------------------

`Conda <https://docs.conda.io/en/latest/>`_ is a package manager for Anaconda-like
distributions, such as the `Miniforge <https://github.com/conda-forge/miniforge>`_
or the `HyperSpy-bundle <https://github.com/hyperspy/hyperspy-bundle>`__.
Since HyperSpy is packaged in the `conda-forge <https://conda-forge.org/>`__ channel,
it can easily be installed using conda.

To install HyperSpy run the following from the Anaconda Prompt on Windows or
from a Terminal on Linux and Mac.

.. code-block:: bash

    $ conda install hyperspy -c conda-forge

This will also install the optional GUI packages ``hyperspy_gui_ipywidgets``
and ``hyperspy_gui_traitsui``. To install HyperSpy without the GUI packages, use:

.. code-block:: bash

    $ conda install hyperspy-base -c conda-forge

.. note::

    Depending on how Anaconda has been installed, it is possible that the
    ``conda`` command is not available from the Terminal, read the
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
- install HyperSpy in a `new environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.
  The following example illustrates how to create a new environment named ``hspy_environment``,
  activate it and install HyperSpy in the new environment.

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

- Choose between `Anaconda or Miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_?
- Understanding `conda and pip <https://www.anaconda.com/blog/understanding-conda-and-pip>`_.
- What is `conda-forge <https://conda-forge.org>`__.

.. _install-with-pip:

Installation using pip
----------------------

HyperSpy is listed in the `Python Package Index
<https://pypi.python.org/pypi>`_. Therefore, it can be automatically downloaded
and installed  `pip <https://pypi.python.org/pypi/pip>`__. You may need to
install pip for the following commands to run.

To install all of HyperSpy's functionalities, run:

.. code-block:: bash

    $ pip install hyperspy[all]

To install only the strictly required dependencies and limited functionalities,
use:

.. code-block:: bash

    $ pip install hyperspy

See the following list of selectors to select the installation of optional
dependencies required by specific functionalities:

* ``ipython`` for integration with the `ipython` terminal and parallel processing using `ipyparallel`,
* ``learning`` for some machine learning features,
* ``gui-jupyter`` to use the `Jupyter widgets <https://ipywidgets.readthedocs.io/en/stable/>`_
  GUI elements,
* ``gui-traitsui`` to use the GUI elements based on `traitsui <https://docs.enthought.com/traitsui/>`_,
* ``speed`` install numba and numexpr to speed up some functionalities,
* ``tests`` to install required libraries to run HyperSpy's unit tests,
* ``coverage`` to coverage statistics when running the tests,
* ``doc`` to install required libraries to build HyperSpy's documentation,
* ``dev`` to install all the above,
* ``all`` to install all the above except the development requirements
  (``tests``, ``doc`` and ``dev``).

For example:

.. code-block:: bash

    $ pip install hyperspy[learning, gui-jupyter]

Finally, be aware that HyperSpy depends on a number of libraries that usually
need to be compiled and therefore installing HyperSpy may require development
tools installed in the system. If the above does not work for you remember that
the easiest way to install HyperSpy is
:ref:`using the HyperSpy bundle <hyperspy-bundle>`.

.. _update-with-conda:

Update HyperSpy
---------------

Using conda
^^^^^^^^^^^

To update hyperspy to the latest release using conda:

.. code-block:: bash

    $ conda update hyperspy -c conda-forge

Using pip
^^^^^^^^^

To update hyperspy to the latest release using pip:

.. code-block:: bash

    $ pip install hyperspy --upgrade

Install specific version
------------------------

Using conda
^^^^^^^^^^^

To install a specific version of hyperspy (for example ``1.6.1``) using conda:

.. code-block:: bash

    $ conda install hyperspy=1.6.1 -c conda-forge

Using pip
^^^^^^^^^

To install a specific version of hyperspy (for example ``1.6.1``) using pip:

.. code-block:: bash

    $ pip install hyperspy==1.6.1


.. _install-rolling:

Rolling release Linux distributions
-----------------------------------

Due to the requirement of up to date versions for dependencies such as *numpy*,
*scipy*, etc., binary packages of HyperSpy are not provided for most linux
distributions and the installation via :ref:`Anaconda/Miniconda <anaconda-install>`
or :ref:`Pip <install-with-pip>` is recommended.

However, packages of the latest HyperSpy release and the related
GUI packages are maintained for the rolling release distributions
*Arch-Linux* (in the `Arch User Repository
<https://aur.archlinux.org/packages/python-hyperspy/>`_) (AUR) and
*openSUSE* (`Community Package <https://software.opensuse.org/package/python-hyperspy>`_)
as ``python-hyperspy`` and ``python-hyperspy-gui-traitsui``,
``python-hyperspy-gui-ipywidgets`` for the GUIs packages.

A more up-to-date package that contains all updates to be included
in the next minor version release (likely including new features compared to
the stable release) is also available in the AUR as |python-hyperspy-git|_.

.. |python-hyperspy-git| replace:: ``python-hyperspy-git``
.. _python-hyperspy-git: https://aur.archlinux.org/packages/python-hyperspy-git

.. _install-dev:

Install development version
---------------------------

Clone the hyperspy repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To get the development version from our git repository you need to install `git
<https://git-scm.com//>`_. Then just do:

.. code-block:: bash

    $ git clone https://github.com/hyperspy/hyperspy.git

.. Warning::

    When running hyperspy from a development version, it can happen that the
    dependency requirement changes in which you will need to keep this
    this requirement up to date (check dependency requirement in ``setup.py``)
    or run again the installation in development mode using ``pip`` as explained
    below.

Installation in a Anaconda/Miniconda distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Optionally, create an environment to separate your hyperspy installation from
other anaconda environments (`read more about environments here
<https://conda.io/projects/conda/en/latest/user-guide/concepts/environments.html>`_):

.. code-block:: bash

    $ conda create -n hspy_dev python # create an empty environment with latest python
    $ conda activate hspy_dev # activate environment

Install the runtime and development dependencies requirements using conda:

.. code-block:: bash

    $ conda install hyperspy-base -c conda-forge --only-deps # install hyperspy dependencies
    $ conda install hyperspy-dev -c conda-forge # install developer dependencies

The package ``hyperspy-dev`` will install the development dependencies required
for testing and building the documentation.

From the root folder of your hyperspy repository (folder containing the
``setup.py`` file) run `pip <https://pip.pypa.io/>`_ in development mode:

.. code-block:: bash

    $ pip install -e . --no-deps # install the currently checked-out branch of hyperspy

Installation in other (non-system) Python distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the root folder of your hyperspy repository (folder containing the
``setup.py`` file) run `pip <https://pip.pypa.io/>`_ in development mode:

.. code-block:: bash

    $ pip install -e .[dev]

All required dependencies are automatically installed by pip. If you don't want
to install all dependencies and only install some of the optional dependencies,
use the corresponding selector as explained in the :ref:`install-with-pip` section

..
    If using Arch Linux, the latest checkout of the master development branch
    can be installed through the AUR by installing the `hyperspy-git package
    <https://aur.archlinux.org/packages/hyperspy-git/>`_

Installation in a system Python distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using a system Python distribution, it is recommended to install the
dependencies using your system package manager.

From the root folder of your hyperspy repository (folder containing the
``setup.py`` file) run `pip <https://pip.pypa.io/>`_ in development mode.

.. code-block:: bash

    $ pip install -e --user .[dev]

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
