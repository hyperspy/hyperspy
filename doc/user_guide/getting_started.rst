Getting started
***************


.. _importing_hyperspy-label:

Starting Python in Windows
----------------------------
If you used the bundle installation you should be able to use the context menus
to get started. Right-click on the folder containing the data you wish to
analyse and select "Jupyter notebook here" or "Jupyter qtconsole here". We
recommend the former, since notebooks have many advantages over conventional
consoles, as will be illustrated in later sections. The examples in some later
sections assume Notebook operation. A new tab should appear in your default
browser listing the files in the selected folder. To start a python notebook
choose "Python 3" in the "New" drop-down menu at the top right of the page.
Another new tab will open which is your Notebook.

Starting Python in Linux and MacOS
------------------------------------

You can start IPython by opening a system terminal and executing ``ipython``,
(optionally followed by the "frontend": "qtconsole" for example). However, in
most cases, **the most agreeable way** to work with HyperSpy interactively
is using the `Jupyter Notebook <http://jupyter.org>`_ (previously known as
the IPython Notebook), which can be started as follows:

.. code-block:: bash

    $ jupyter notebook

Linux users may find it more convenient to start Jupyter/IPython from the
`file manager context menu <https://github.com/hyperspy/start_jupyter_cm>`_.
In either OS you can also start by `double-clicking a notebook file
<https://github.com/takluyver/nbopen>`_ if one already exists.

Starting HyperSpy in the notebook (or terminal)
-----------------------------------------------
Typically you will need to `set up IPython for interactive plotting with
matplotlib
<http://ipython.readthedocs.org/en/stable/interactive/plotting.html>`_ using
``%matplotlib`` (which is known as a 'Jupyter magic')
*before executing any plotting command*. So, typically, after starting
IPython, you can import HyperSpy and set up interactive matplotlib plotting by
executing the following two lines in the IPython terminal (In these docs we
normally use the general Python prompt symbol ``>>>`` but you will probably
see ``In [1]:`` etc.):

.. code-block:: python

   >>> %matplotlib qt
   >>> import hyperspy.api as hs

Note that to execute lines of code in the notebook you must press
``Shift+Return``. (For details about notebooks and their functionality try
the help menu in the notebook). Next, import two useful modules: numpy and
matplotlib.pyplot, as follows:

.. code-block:: python

   >>> import numpy as np
   >>> import matplotlib.pyplot as plt

The rest of the documentation will assume you have done this. It also assumes
that you have installed at least one of HyperSpy's GUI packages:
`jupyter widgets GUI <https://github.com/hyperspy/hyperspy_gui_ipywidgets>`_
and the
`traitsui GUI <https://github.com/hyperspy/hyperspy_gui_traitsui>`_.

Possible warnings when importing HyperSpy?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HyperSpy supports different GUIs and 
`matplotlib backends <https://matplotlib.org/tutorials/introductory/usage.html#backends>`_ 
which in specific cases can lead to warnings when importing HyperSpy. Most of the time 
there is nothing to worry about — the warnings simply inform you of several choices you have.
There may be several causes for a warning, for example:

* not all the GUIs packages are installed. If none is installed, we reccomend you to install
  at least the ``hyperspy-gui-ipywidgets`` package is your are planning to perform interactive
  data analysis in the Jupyter Notebook. Otherwise, you can simply disable the warning in
  :ref:`preferences <configuring-hyperspy-label>` as explained below.
* the ``hyperspy-gui-traitsui`` package is installed and you are using an incompatible matplotlib
  backend (e.g. ``notebook``, ``nbagg`` or ``widget``).
   * If you want to use the traitsui GUI, use the ``qt`` matplotlib backend instead.
   * Alternatively, if you prefer to use the ``notebook`` or ``widget`` matplotlib backend,
     and if you don't want to see the (harmless) warning, make sure that you have the
     ``hyperspy-gui-ipywidgets`` installed and disable the traitsui
     GUI in the :ref:`preferences <configuring-hyperspy-label>`.


By default, HyperSpy warns the user if one of the GUI packages is not installed.
These warnings can be turned off using the
:ref:`preferences <configuring-hyperspy-label>` GUI or programmatically as follows:

    .. code-block:: python

       >>> import hyperspy.api as hs
       >>> hs.preferences.GUIs.warn_if_guis_are_missing = False
       >>> hs.preferences.save()


.. versionchanged:: v1.3
    HyperSpy works with all matplotlib backends, including the ``notebook`` 
    (also called ``nbAgg``) backend that enables interactive plotting embedded 
    in the jupyter notebook.

.. warning::
        When using the qt4 backend in Python 2 the matplotlib magic must be
        executed after importing HyperSpy and qt must be the default HyperSpy
        backend.

.. NOTE::

    When running in a  headless system it is necessary to set the matplotlib
    backend appropiately to avoid a `cannot connect to X server` error, for
    example as follows:

    .. code-block:: python

       >>> import matplotlib
       >>> matplotlib.rcParams["backend"] = "Agg"
       >>> import hyperspy.api as hs


Getting help
------------

When using IPython, the documentation (docstring in Python jargon) can be
accessed by adding a question mark to the name of a function. e.g.:


.. code-block:: python

    >>> hs?
    >>> hs.load?
    >>> hs.signals?

This syntax is a shortcut to the standard way one of displaying the help
associated to a given functions (docstring in Python jargon) and it is one of
the many features of `IPython <http://ipython.scipy.org/moin/>`_, which is the
interactive python shell that HyperSpy uses under the hood.

Please note that the documentation of the code is a work in progress, so not
all the objects are documented yet.

Up-to-date documentation is always available in `the HyperSpy website.
<http://hyperspy.org/documentation.html>`_


Autocompletion
--------------

Another useful `IPython <http://ipython.scipy.org/moin/>`_ feature is the
autocompletion of commands and filenames using the tab and arrow keys. It is
highly recommended to read the `Ipython documentation
<http://ipython.scipy.org/moin/Documentation>`_ (specially their `Getting
started <http://ipython.org/ipython-doc/stable/interactive/tutorial.html>`_
section) for many more useful features that will boost your efficiency when
working with HyperSpy/Python interactively.


Loading data
------------

Once HyperSpy is running, to load from a supported file format (see
:ref:`supported-formats`) simply type:

.. code-block:: python

    >>> s = hs.load("filename")

.. HINT::

   The load function returns an object that contains data read from the file.
   We assign this object to the variable ``s`` but you can choose any (valid)
   variable name you like. for the filename, don\'t forget to include the
   quotation marks and the file extension.

If no argument is passed to the load function, a window will be raised that
allows to select a single file through your OS file manager, e.g.:

.. code-block:: python

    >>> # This raises the load user interface
    >>> s = hs.load()

It is also possible to load multiple files at once or even stack multiple
files. For more details read :ref:`loading_files`

"Loading" data from a numpy array
---------------------------------

HyperSpy can operate on any numpy array by assigning it to a BaseSignal class.
This is useful e.g. for loading data stored in a format that is not yet
supported by HyperSpy—supposing that they can be read with another Python
library—or to explore numpy arrays generated by other Python
libraries. Simply select the most appropriate signal from the
:py:mod:`~.signals` module and create a new instance by passing a numpy array
to the constructor e.g.

.. code-block:: python

    >>> my_np_array = np.random.random((10,20,100))
    >>> s = hs.signals.Signal1D(my_np_array)
    >>> s
    <Signal1D, title: , dimensions: (20, 10|100)>

The numpy array is stored in the :py:attr:`~.signal.BaseSignal.data` attribute
of the signal class.

.. _example-data-label:

Loading example data and data from online databases
---------------------------------------------------

HyperSpy is distributed with some example data that can be found in
`hs.datasets.example_signals`. The following example plots one of the example
signals:

.. code-block:: python

    >>> hs.datasets.example_signals.EDS_TEM_Spectrum().plot()

.. versionadded:: 1.4
    :py:mod:`~.datasets.artificial_data`

There are also artificial datasets, which are made to resemble real
experimental data.

.. code-block:: python

    >>> s = hs.datasets.artificial_data.get_core_loss_eels_signal()
    >>> s.plot()

.. _eelsdb-label:

The :py:func:`~.misc.eels.eelsdb.eelsdb` function in `hs.datasets` can
directly load spectra from `The EELS Database <http://eelsdb.eu>`_. For
example, the following loads all the boron trioxide spectra currently
available in the database:

.. code-block:: python

    >>> hs.datasets.eelsdb(formula="B2O3")
    [<EELSSpectrum, title: Boron oxide, dimensions: (|520)>,
     <EELSSpectrum, title: Boron oxide, dimensions: (|520)>]


.. _saving:

Saving Files
------------

The data can be saved to several file formats.  The format is specified by
the extension of the filename.

.. code-block:: python

    >>> # load the data
    >>> d = hs.load("example.tif")
    >>> # save the data as a tiff
    >>> d.save("example_processed.tif")
    >>> # save the data as a png
    >>> d.save("example_processed.png")
    >>> # save the data as an hspy file
    >>> d.save("example_processed.hspy")

Some file formats are much better at maintaining the information about
how you processed your data.  The preferred format in HyperSpy is hspy,
which is based on the HDF5 format.  This format keeps the most information
possible.

There are optional flags that may be passed to the save function. See
:ref:`saving_files` for more details.

Accessing and setting the metadata
----------------------------------

When loading a file HyperSpy stores all metadata in the BaseSignal
:py:attr:`~.signal.BaseSignal.original_metadata` attribute. In addition,
some of those metadata and any new metadata generated by HyperSpy are stored in
:py:attr:`~.signal.BaseSignal.metadata` attribute.


.. code-block:: python

   >>> s = hs.load("NbO2_Nb_M_David_Bach,_Wilfried_Sigle_217.msa")
   >>> s.metadata
   ├── original_filename = NbO2_Nb_M_David_Bach,_Wilfried_Sigle_217.msa
   ├── record_by = spectrum
   ├── signal_type = EELS
   └── title = NbO2_Nb_M_David_Bach,_Wilfried_Sigle_217

   >>> s.original_metadata
   ├── DATATYPE = XY
   ├── DATE =
   ├── FORMAT = EMSA/MAS Spectral Data File
   ├── NCOLUMNS = 1.0
   ├── NPOINTS = 1340.0
   ├── OFFSET = 120.0003
   ├── OWNER = eelsdatabase.net
   ├── SIGNALTYPE = ELS
   ├── TIME =
   ├── TITLE = NbO2_Nb_M_David_Bach,_Wilfried_Sigle_217
   ├── VERSION = 1.0
   ├── XPERCHAN = 0.5
   ├── XUNITS = eV
   └── YUNITS =

   >>> s.set_microscope_parameters(100, 10, 20)
   >>> s.metadata
   ├── TEM
   │   ├── EELS
   │   │   └── collection_angle = 20
   │   ├── beam_energy = 100
   │   └── convergence_angle = 10
   ├── original_filename = NbO2_Nb_M_David_Bach,_Wilfried_Sigle_217.msa
   ├── record_by = spectrum
   ├── signal_type = EELS
   └── title = NbO2_Nb_M_David_Bach,_Wilfried_Sigle_217

   >>> s.metadata.TEM.microscope = "STEM VG"
   >>> s.metadata
   ├── TEM
   │   ├── EELS
   │   │   └── collection_angle = 20
   │   ├── beam_energy = 100
   │   ├── convergence_angle = 10
   │   └── microscope = STEM VG
   ├── original_filename = NbO2_Nb_M_David_Bach,_Wilfried_Sigle_217.msa
   ├── record_by = spectrum
   ├── signal_type = EELS
   └── title = NbO2_Nb_M_David_Bach,_Wilfried_Sigle_217


.. _configuring-hyperspy-label:

Configuring HyperSpy
--------------------

The behaviour of HyperSpy can be customised using the
:py:class:`~.defaults_parser.Preferences` class. The easiest way to do it is by
calling the :meth:`gui` method:

.. code-block:: python

    >>> hs.preferences.gui()

This command should raise the Preferences user interface if one of the
hyperspy gui packages are installed and enabled:

.. _preferences_image:

.. figure::  images/preferences.png
   :align:   center

   Preferences user interface.

.. versionadded:: 1.3
    Possibility to enable/disable GUIs in the preferences.

It is also possible to set the preferences programmatically. For example,
to disable the traitsui GUI elements and save the changes to disk:

.. code-block:: python

    >>> hs.preferences.GUIs.enable_traitsui_gui = False
    >>> hs.preferences.save()
    >>> # if not saved, this setting will be used until the next jupyter kernel shutdown

.. versionchanged:: 1.3

   The following items were removed from preferences:
   ``General.default_export_format``, ``General.lazy``,
   ``Model.default_fitter``, ``Machine_learning.multiple_files``,
   ``Machine_learning.same_window``, ``Plot.default_style_to_compare_spectra``,
   ``Plot.plot_on_load``, ``Plot.pylab_inline``, ``EELS.fine_structure_width``,
   ``EELS.fine_structure_active``, ``EELS.fine_structure_smoothing``,
   ``EELS.synchronize_cl_with_ll``, ``EELS.preedge_safe_window_width``,
   ``EELS.min_distance_between_edges_for_fine_structure``.



.. _logger-label:

Messages log
------------

HyperSpy writes messages to the `Python logger
<https://docs.python.org/3/howto/logging.html#logging-basic-tutorial>`_. The
default log level is "WARNING", meaning that only warnings and more severe
event messages will be displayed. The default can be set in the
:ref:`preferences <configuring-hyperspy-label>`. Alternatively, it can be set
using :py:func:`~.logger.set_log_level` e.g.:

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> hs.set_log_level('INFO')
    >>> hs.load(r'my_file.dm3')
    INFO:hyperspy.io_plugins.digital_micrograph:DM version: 3
    INFO:hyperspy.io_plugins.digital_micrograph:size 4796607 B
    INFO:hyperspy.io_plugins.digital_micrograph:Is file Little endian? True
    INFO:hyperspy.io_plugins.digital_micrograph:Total tags in root group: 15
    <Signal2D, title: My file, dimensions: (|1024, 1024)
